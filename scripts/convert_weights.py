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
    # Custom: signed int4, blockwise along the last dim, block size 32,
    # per-block fp16 scale, no zero-point. Data layout per tensor is
    # [N, D/2] packed int4 (low nibble = even index) followed by
    # [N, D/32] fp16 scales, back-to-back.
    "int4_block32_sym": 6,
}
INT4_BLOCK_SIZE = 32


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
    p.add_argument("--quant", action="append", default=None, metavar="PATTERN",
                   help="Quantize tensors whose name contains PATTERN to int4-blockwise-32. Repeatable. Example: --quant mlp.experts.gate_up_proj --quant mlp.experts.down_proj")
    p.add_argument("--quant-transpose", action="append", default=None, metavar="PATTERN",
                   help="Like --quant but also transposes the last two dims before quantization. Needed for MoE expert weights stored as [E, D_in, D_out] upstream.")
    p.add_argument("--expert-keep", type=int, default=0, metavar="N",
                   help="For 3D tensors matching `mlp.experts.`, keep only the first N experts along dim 0.")
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


def quantize_int4_block32_sym(
    tensor: object, transpose_last: bool = False,
) -> tuple[bytes, tuple[int, ...]]:
    """
    Asymmetric uint4 blockwise quantization, matching ONNX MatMulNBits
    semantics: block size 32 along the last dim, per-block fp16 scale,
    per-block uint4 zero-point. Dequant: `value = (q - zp) * scale`.

    Data layout written into one blob (back-to-back):
      [..., N, D/2]          uint4 weights (low nibble = even d, high = odd)
      [..., N, D/32]         fp16 scales
      [..., N, ceil(D/32/2)] uint4 zero-points packed 2-per-byte

    Returns (packed_bytes, logical_shape).
    """
    import torch
    t = tensor.detach().contiguous().cpu().to(torch.float32)
    if transpose_last:
        if t.dim() < 2:
            raise ValueError(f"transpose_last requires >=2 dims; got {tuple(t.shape)}")
        t = t.transpose(-1, -2).contiguous()
    shape = tuple(t.shape)
    if len(shape) < 2:
        raise ValueError(f"int4-block32 needs >=2 dims; got {shape}")
    D = shape[-1]
    if D % INT4_BLOCK_SIZE != 0:
        raise ValueError(f"last dim {D} not divisible by {INT4_BLOCK_SIZE}")

    leading = t.shape[:-1]
    t2 = t.reshape(-1, D)                                # [R, D]
    R = t2.shape[0]
    n_blocks = D // INT4_BLOCK_SIZE
    blocks = t2.view(R, n_blocks, INT4_BLOCK_SIZE)

    block_min = blocks.amin(dim=-1, keepdim=True)
    block_max = blocks.amax(dim=-1, keepdim=True)
    # Asymmetric range 0..15. scale = (max-min)/15; zp = round(-min/scale).
    span = (block_max - block_min).clamp_min(1e-8)
    scale = span / 15.0
    zp_f = (-block_min / scale).round().clamp(0, 15)
    q = (blocks / scale + zp_f).round().clamp(0, 15).to(torch.uint8)

    # Pack weight nibbles: low = even d, high = odd d.
    q_flat = q.reshape(R, D)
    low = q_flat[:, 0::2]
    high = q_flat[:, 1::2]
    packed = (low | (high << 4)).to(torch.uint8).contiguous()              # [R, D/2] u8

    scale_fp16 = scale.squeeze(-1).to(torch.float16).reshape(R, n_blocks)

    # Pack zero-points: block 0 → low nibble of byte 0, block 1 → high nibble
    # of byte 0, block 2 → low of byte 1, etc. For odd n_blocks, the trailing
    # byte's high nibble is unused (spec-compliant, decoder ignores it).
    zp_flat = zp_f.squeeze(-1).reshape(R, n_blocks).to(torch.uint8)
    zp_bytes_per_row = (n_blocks + 1) // 2
    zp_packed = torch.zeros(R, zp_bytes_per_row, dtype=torch.uint8)
    zp_packed[:, : n_blocks // 2] = zp_flat[:, 0::2] | (zp_flat[:, 1::2] << 4)
    if n_blocks % 2 == 1:
        zp_packed[:, -1] = zp_flat[:, -1]

    int4_bytes = packed.numpy().tobytes()
    scale_bytes = scale_fp16.view(torch.uint16).numpy().tobytes()
    zp_bytes = zp_packed.numpy().tobytes()
    return int4_bytes + scale_bytes + zp_bytes, shape


def should_quantize(name: str, patterns: Sequence[str] | None) -> bool:
    # Match on suffix so "mlp.experts.gate_up_proj" doesn't accidentally
    # pick up "mlp.experts.gate_up_proj_bias" (a 2-D tensor that can't be
    # int4-blockwise-quantized along 32-wide blocks).
    if not patterns:
        return False
    return any(name.endswith(p) for p in patterns)


def should_transpose_quant(name: str, patterns: Sequence[str] | None) -> bool:
    if not patterns:
        return False
    return any(name.endswith(p) for p in patterns)


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
    quant_patterns: Sequence[str] | None = None,
    quant_transpose_patterns: Sequence[str] | None = None,
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
        if should_transpose_quant(name, quant_transpose_patterns):
            payload, shape = quantize_int4_block32_sym(tensor, transpose_last=True)
            dtype_code = DTYPE_CODES["int4_block32_sym"]
        elif should_quantize(name, quant_patterns):
            payload, shape = quantize_int4_block32_sym(tensor, transpose_last=False)
            dtype_code = DTYPE_CODES["int4_block32_sym"]
        else:
            dtype_code = dtype_code_from_torch(tensor.dtype)
            shape = tuple(tensor.shape)
            payload = tensor_bytes(tensor)
        if len(shape) > 4:
            raise ValueError(f"tensor {name} has ndim={len(shape)}; max supported is 4")
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

    # Expert truncation: slice MoE tensors along dim 0 to first N experts.
    # Covers both the 3-D expert weight tensors (`mlp.experts.*`) AND the
    # 2-D / 1-D router parameters (`mlp.router.*`). The router's output
    # dimension MUST match the blob's expert count — otherwise the router
    # emits indices that our blob can't serve.
    if args.expert_keep > 0:
        state_dict = dict(state_dict)
        for n_name in names:
            is_expert = "mlp.experts." in n_name and state_dict[n_name].dim() >= 2
            is_router = "mlp.router." in n_name and state_dict[n_name].dim() >= 1
            if is_expert or is_router:
                t = state_dict[n_name]
                if t.shape[0] > args.expert_keep:
                    state_dict[n_name] = t[: args.expert_keep].contiguous()
                    print(
                        f"[convert-weights] truncating {n_name} along dim 0 to first {args.expert_keep}",
                        file=sys.stderr,
                    )

    total_size, sha256 = write_blob(
        state_dict, names, args.output, args.quant, args.quant_transpose,
    )
    print(f"[convert-weights] wrote {total_size} bytes ({total_size / 1024 / 1024:.2f} MiB)", file=sys.stderr)
    print(f"[convert-weights] sha256: {sha256}", file=sys.stderr)

    if args.hash:
        args.hash.write_text(sha256 + "\n")
        print(f"[convert-weights] hash → {args.hash}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
