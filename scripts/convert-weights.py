#!/usr/bin/env python3
"""
Convert openai/privacy-filter weights to pii-wasm's flat binary format.

Output file: `dist/pii-weights.bin` (by default). Parsed by the Stage 1
Zig runtime at `src/zig/wasm_exports.zig::parse_weights`.

Format v1 (little-endian throughout):

  Header — 16 bytes
    magic        4B   "PIIW"
    version      u32  = 1
    num_tensors  u32
    data_offset  u32  // byte offset from start of file where the data
                      // region begins. Equal to 64-align(16 + 96*N).

  Tensor entry — 104 bytes, one per tensor, packed back-to-back after header.
    name         char[64]  null-padded
    dtype        u32       0=f32 1=f16 2=bf16 3=i8 4=u8 5=i32
    ndim         u32
    shape        u32[4]    zero-padded for ndim<4
    data_offset  u64       from start of file
    data_size    u64       bytes

  Data region
    Each tensor's bytes, packed in the order tensors appear in the table.
    Each tensor start is 64-byte aligned (so SIMD loads are aligned).

Default subset for Phase B1 (~87 KB) is listed in `PHASE_B1_TENSORS`
below — classifier head + two RMSNorm weights. `--full` extracts the
whole 1.4B-param model (~2.8 GB bf16); use it only when Phase C/D
kernels are ready to consume it.
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
from pathlib import Path
from typing import Sequence


MAGIC = b"PIIW"
VERSION = 1
HEADER_SIZE = 16
ENTRY_SIZE = 104  # 64 (name) + 4 (dtype) + 4 (ndim) + 16 (shape) + 8 (data_off) + 8 (data_size)
NAME_FIELD = 64
ALIGN = 64

DTYPE_CODES = {
    "float32": 0,
    "float16": 1,
    "bfloat16": 2,
    "int8": 3,
    "uint8": 4,
    "int32": 5,
}


# Small subset sufficient to prove the end-to-end weight pipeline.
PHASE_B1_TENSORS: tuple[str, ...] = (
    "score.weight",
    "score.bias",
    "model.norm.weight",
    "model.layers.0.input_layernorm.weight",
)


def align_up(n: int, a: int) -> int:
    return (n + a - 1) & ~(a - 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="openai/privacy-filter",
                   help="HF Hub id or local path to the safetensors checkpoint.")
    p.add_argument("--output", type=Path, default=Path("dist/pii-weights.bin"),
                   help="Output binary path.")
    p.add_argument("--full", action="store_true",
                   help="Emit every tensor (1.4B params, ~2.8 GB). Default is a small subset for plumbing tests.")
    p.add_argument("--include", action="append", default=None,
                   help="Extra tensor name to include (on top of the default subset). Repeatable.")
    p.add_argument("--embed-rows", type=int, default=0,
                   help="If >0 and embed_tokens.weight is in the selection, truncate it to the first N rows. Lets tests exercise the embedding kernel without shipping 128 MB.")
    p.add_argument("--list", action="store_true",
                   help="Enumerate available tensor names and exit.")
    p.add_argument("--hash", type=Path, default=None,
                   help="Write sha256 of the output binary to this file (hex).")
    return p.parse_args()


def pack_name(name: str) -> bytes:
    b = name.encode("utf-8")
    if len(b) > NAME_FIELD:
        raise ValueError(f"tensor name {name!r} is {len(b)} bytes; limit is {NAME_FIELD}")
    return b + b"\x00" * (NAME_FIELD - len(b))


def dtype_code_from_torch(dtype: object) -> int:
    name = str(dtype).removeprefix("torch.")
    if name not in DTYPE_CODES:
        raise ValueError(f"unsupported dtype: {dtype!r}")
    return DTYPE_CODES[name]


def select_tensors(state_dict: dict, full: bool, extras: Sequence[str] | None) -> list[str]:
    if full:
        return list(state_dict.keys())
    missing = [name for name in PHASE_B1_TENSORS if name not in state_dict]
    if missing:
        raise ValueError(f"expected tensors not found in state_dict: {missing}")
    names = list(PHASE_B1_TENSORS)
    for extra in (extras or ()):
        if extra not in state_dict:
            raise ValueError(f"--include tensor not found in state_dict: {extra}")
        if extra not in names:
            names.append(extra)
    return names


def tensor_bytes(tensor: object) -> bytes:
    # Ensure contiguous, CPU, packed representation. bfloat16 has no
    # numpy analog, so we go through tensor.view(torch.uint16).numpy()
    # for bf16 specifically. For everything else, numpy().tobytes() works.
    import torch
    t = tensor.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy().tobytes()
    if t.dtype == torch.float16:
        return t.view(torch.uint16).numpy().tobytes()
    return t.numpy().tobytes()


def truncate_embedding(tensor: object, rows: int) -> object:
    import torch
    if rows <= 0:
        return tensor
    if tensor.dim() != 2:
        raise ValueError(f"--embed-rows only makes sense for 2-D tensors; got shape {tuple(tensor.shape)}")
    if rows > tensor.shape[0]:
        raise ValueError(f"--embed-rows {rows} exceeds tensor rows {tensor.shape[0]}")
    return tensor[:rows].contiguous()


def write_blob(
    state_dict: dict,
    names: Sequence[str],
    output: Path,
) -> tuple[int, str]:
    output.parent.mkdir(parents=True, exist_ok=True)

    n = len(names)
    table_end = HEADER_SIZE + ENTRY_SIZE * n
    data_offset = align_up(table_end, ALIGN)

    # First pass: compute tensor offsets + sizes.
    entries: list[tuple[str, int, tuple[int, ...], int, int, bytes]] = []
    cursor = data_offset
    for name in names:
        tensor = state_dict[name]
        dtype_code = dtype_code_from_torch(tensor.dtype)
        shape = tuple(tensor.shape)
        if len(shape) > 4:
            raise ValueError(f"tensor {name} has ndim={len(shape)}; max supported is 4")
        payload = tensor_bytes(tensor)
        aligned_cursor = align_up(cursor, ALIGN)
        entries.append((name, dtype_code, shape, aligned_cursor, len(payload), payload))
        cursor = aligned_cursor + len(payload)

    total_size = cursor

    # Second pass: write header, table, data (with padding).
    sha = hashlib.sha256()
    with output.open("wb") as f:
        header = MAGIC + struct.pack("<III", VERSION, n, data_offset)
        f.write(header)
        sha.update(header)

        for name, dtype_code, shape, data_off, data_size, _ in entries:
            padded_shape = list(shape) + [0] * (4 - len(shape))
            entry = pack_name(name) + struct.pack(
                "<II4IQQ",
                dtype_code,
                len(shape),
                padded_shape[0],
                padded_shape[1],
                padded_shape[2],
                padded_shape[3],
                data_off,
                data_size,
            )
            f.write(entry)
            sha.update(entry)

        # Pad to data_offset.
        pad = data_offset - (HEADER_SIZE + ENTRY_SIZE * n)
        if pad:
            z = b"\x00" * pad
            f.write(z)
            sha.update(z)

        # Write each tensor's bytes, padding to 64-byte align before each.
        written = data_offset
        for _, _, _, data_off, _, payload in entries:
            while written < data_off:
                z = b"\x00" * min(data_off - written, ALIGN)
                f.write(z)
                sha.update(z)
                written += len(z)
            f.write(payload)
            sha.update(payload)
            written += len(payload)

    return total_size, sha.hexdigest()


def main() -> int:
    args = parse_args()

    try:
        import torch  # noqa: F401
        from transformers import AutoModelForTokenClassification
    except ImportError as e:
        print(f"[convert-weights] missing dependency: {e}", file=sys.stderr)
        print("Install: pip install 'transformers>=5.6.0' torch safetensors", file=sys.stderr)
        return 1

    print(f"[convert-weights] loading {args.model} …", file=sys.stderr)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    state_dict = model.state_dict()

    if args.list:
        for name, tensor in sorted(state_dict.items()):
            print(f"{name}\t{tuple(tensor.shape)}\t{tensor.dtype}")
        return 0

    names = select_tensors(state_dict, args.full, args.include)

    # Apply any truncation to embed_tokens.weight. We mutate state_dict
    # in place because write_blob reads it by name; the original model
    # object keeps the untrimmed tensor for subsequent Python use.
    if args.embed_rows > 0 and "model.embed_tokens.weight" in names:
        state_dict = dict(state_dict)  # shallow copy, original untouched
        state_dict["model.embed_tokens.weight"] = truncate_embedding(
            state_dict["model.embed_tokens.weight"], args.embed_rows
        )
        print(
            f"[convert-weights] truncating embed_tokens.weight to first {args.embed_rows} rows",
            file=sys.stderr,
        )

    print(f"[convert-weights] writing {len(names)} tensors → {args.output}", file=sys.stderr)
    for name in names:
        t = state_dict[name]
        print(f"  {name:60s} shape={tuple(t.shape)} dtype={t.dtype}", file=sys.stderr)

    total_size, sha256 = write_blob(state_dict, names, args.output)
    print(f"[convert-weights] wrote {total_size} bytes ({total_size / 1024 / 1024:.2f} MiB)", file=sys.stderr)
    print(f"[convert-weights] sha256: {sha256}", file=sys.stderr)

    if args.hash:
        args.hash.write_text(sha256 + "\n")
        print(f"[convert-weights] hash → {args.hash}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
