#!/usr/bin/env python3
"""
Regenerate kernel parity fixtures + matching weight blob under tests/fixtures/.

Fixtures are committed. This script exists so anyone can reproduce the
bytes from first principles against a known PyTorch version + seed. If
a PyTorch upgrade shifts bf16 rounding or RNG state, the regenerated
fixtures won't match the committed ones; update the committed files and
the `baseline_matches` in `tests/fixtures/manifest.json` together.

Run from repo root:

    python3 scripts/gen-kernel-fixtures.py

Expected PyTorch: 2.4.x or newer (needs torch.nn.functional.rms_norm).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
SEED = 42

T = 8
D = 640
N_CLASSES = 33  # matches score.weight / score.bias
V_TEST = 1024   # truncated embed table — full vocab is 200 064, too big for tests
INT4_BLOCK = 32


# Share the quantizer with convert_weights.py so blob + fixture byte
# patterns stay identical under any tweak.
def _load_convert_weights():
    path = REPO_ROOT / "scripts" / "convert_weights.py"
    spec = importlib.util.spec_from_file_location("convert_weights", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bf16_bytes(tensor: torch.Tensor) -> bytes:
    assert tensor.dtype == torch.bfloat16
    return tensor.detach().contiguous().cpu().view(torch.uint16).numpy().tobytes()


def write(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    print(f"  wrote {path.relative_to(REPO_ROOT)}  ({len(data)} bytes)", file=sys.stderr)


def dequantize_int4block(packed: bytes, n: int, d: int) -> torch.Tensor:
    """
    Decode the bytes produced by `convert_weights.quantize_int4_block32_sym`
    back to an fp32 [N, D] tensor. Reference path for computing expected
    outputs — exactly what the WASM kernel computes on-the-fly.
    """
    import numpy as np
    block = INT4_BLOCK
    n_blocks = d // block
    int4_bytes = n * d // 2
    # Split packed bytes.
    int4_flat = np.frombuffer(packed[:int4_bytes], dtype=np.uint8).reshape(n, d // 2)
    scale_u16 = np.frombuffer(packed[int4_bytes:], dtype=np.uint16).reshape(n, n_blocks)
    scales = torch.from_numpy(scale_u16.view(np.uint16).copy()).view(torch.float16).to(torch.float32)

    # Unpack nibbles: low (even d) and high (odd d). Sign-extend 4 bits.
    lo = int4_flat & 0x0F
    hi = (int4_flat >> 4) & 0x0F
    # Sign-extend: treat as int8 via two's complement of a nibble.
    lo = np.where(lo & 0x08, lo.astype(np.int16) - 16, lo.astype(np.int16)).astype(np.int8)
    hi = np.where(hi & 0x08, hi.astype(np.int16) - 16, hi.astype(np.int16)).astype(np.int8)
    q = np.empty((n, d), dtype=np.int8)
    q[:, 0::2] = lo
    q[:, 1::2] = hi
    q_fp = torch.from_numpy(q).to(torch.float32)     # [N, D]
    q_blocks = q_fp.view(n, n_blocks, block)
    dequant = q_blocks * scales.unsqueeze(-1)
    return dequant.view(n, d)


def main() -> int:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    print("[gen] loading openai/privacy-filter …", file=sys.stderr)
    from transformers import AutoModelForTokenClassification
    model = AutoModelForTokenClassification.from_pretrained("openai/privacy-filter")
    sd = model.state_dict()

    cw = _load_convert_weights()

    # ----- 1. Weight blob (5 tensors, with embed table truncated to V_TEST rows) -----
    print("[gen] building weight blob via scripts/convert_weights.py …", file=sys.stderr)
    convert = REPO_ROOT / "scripts" / "convert_weights.py"
    output_blob = FIXTURES_DIR / "pii-weights.bin"
    cmd = [
        sys.executable, str(convert),
        "--output", str(output_blob),
        "--include", "model.embed_tokens.weight",
        "--embed-rows", str(V_TEST),
    ]
    subprocess.check_call(cmd)

    # ----- 2. RMSNorm fixture -----
    torch.manual_seed(SEED)
    x_rms = torch.randn(T, D, dtype=torch.bfloat16)
    gamma = sd["model.norm.weight"].to(torch.bfloat16)
    y_rms = F.rms_norm(x_rms, [D], weight=gamma, eps=1e-5)
    write(FIXTURES_DIR / "rms_x.bf16", bf16_bytes(x_rms))
    write(FIXTURES_DIR / "rms_y.bf16", bf16_bytes(y_rms.to(torch.bfloat16)))

    # ----- 3. Matmul fixture (classifier head, bf16 path) -----
    torch.manual_seed(SEED + 1)
    x_mm = torch.randn(T, D, dtype=torch.bfloat16)
    w_bf = sd["score.weight"].to(torch.bfloat16)    # [N, D]
    b_bf = sd["score.bias"].to(torch.bfloat16)      # [N]
    y_mm = F.linear(x_mm, w_bf, b_bf)
    write(FIXTURES_DIR / "matmul_x.bf16", bf16_bytes(x_mm))
    write(FIXTURES_DIR / "matmul_y.bf16", bf16_bytes(y_mm.to(torch.bfloat16)))

    # ----- 4. Embed fixture -----
    g = torch.Generator().manual_seed(SEED + 2)
    ids = torch.randint(0, V_TEST, (T,), generator=g, dtype=torch.int32)
    embed = sd["model.embed_tokens.weight"][:V_TEST].to(torch.bfloat16)
    y_embed = embed[ids.long()]
    write(FIXTURES_DIR / "embed_ids.i32", ids.numpy().tobytes())
    write(FIXTURES_DIR / "embed_out.bf16", bf16_bytes(y_embed))

    # ----- 5. Int4-blockwise matmul fixture (quantized score.weight) -----
    torch.manual_seed(SEED + 3)
    x_i4 = torch.randn(T, D, dtype=torch.bfloat16)
    # Quantize W using the same code convert_weights.py will use in prod.
    # Source fp32 to get a clean quant (not twice-rounded through bf16).
    w_fp = sd["score.weight"].to(torch.float32)
    packed, w_shape = cw.quantize_int4_block32_sym(w_fp)
    assert w_shape == (N_CLASSES, D)
    # Reference expected output: dequantize W exactly, then F.linear in fp32.
    w_dequant = dequantize_int4block(packed, N_CLASSES, D)           # [N, D] fp32
    x_fp = x_i4.to(torch.float32)
    b_fp = sd["score.bias"].to(torch.float32)
    y_i4 = F.linear(x_fp, w_dequant, b_fp).to(torch.bfloat16)
    write(FIXTURES_DIR / "int4_matmul_x.bf16", bf16_bytes(x_i4))
    write(FIXTURES_DIR / "int4_matmul_w.i4block", packed)
    write(FIXTURES_DIR / "int4_matmul_bias.bf16", bf16_bytes(sd["score.bias"].to(torch.bfloat16)))
    write(FIXTURES_DIR / "int4_matmul_y.bf16", bf16_bytes(y_i4))

    print("[gen] done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
