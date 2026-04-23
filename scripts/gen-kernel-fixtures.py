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

import struct
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


def bf16_bytes(tensor: torch.Tensor) -> bytes:
    assert tensor.dtype == torch.bfloat16
    return tensor.detach().contiguous().cpu().view(torch.uint16).numpy().tobytes()


def write(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    print(f"  wrote {path.relative_to(REPO_ROOT)}  ({len(data)} bytes)", file=sys.stderr)


def main() -> int:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    print("[gen] loading openai/privacy-filter …", file=sys.stderr)
    from transformers import AutoModelForTokenClassification
    model = AutoModelForTokenClassification.from_pretrained("openai/privacy-filter")
    sd = model.state_dict()

    # ----- 1. Weight blob (5 tensors, with embed table truncated to V_TEST rows) -----
    print("[gen] building weight blob via scripts/convert-weights.py …", file=sys.stderr)
    convert = REPO_ROOT / "scripts" / "convert-weights.py"
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
    # PyTorch rms_norm runs in bf16 if inputs are bf16; reference kernel
    # upcasts sum-of-squares to f32 internally either way.
    y_rms = F.rms_norm(x_rms, [D], weight=gamma, eps=1e-5)
    write(FIXTURES_DIR / "rms_x.bf16", bf16_bytes(x_rms))
    write(FIXTURES_DIR / "rms_y.bf16", bf16_bytes(y_rms.to(torch.bfloat16)))

    # ----- 3. Matmul fixture (classifier head) -----
    torch.manual_seed(SEED + 1)
    x_mm = torch.randn(T, D, dtype=torch.bfloat16)
    w = sd["score.weight"].to(torch.bfloat16)         # [N, D]
    b = sd["score.bias"].to(torch.bfloat16)           # [N]
    # F.linear(x, W, b) → x @ W.T + b, matching our kernel signature.
    y_mm = F.linear(x_mm, w, b)
    write(FIXTURES_DIR / "matmul_x.bf16", bf16_bytes(x_mm))
    write(FIXTURES_DIR / "matmul_y.bf16", bf16_bytes(y_mm.to(torch.bfloat16)))

    # ----- 4. Embed fixture -----
    g = torch.Generator().manual_seed(SEED + 2)
    ids = torch.randint(0, V_TEST, (T,), generator=g, dtype=torch.int32)
    embed = sd["model.embed_tokens.weight"][:V_TEST].to(torch.bfloat16)  # [V_TEST, D]
    y_embed = embed[ids.long()]
    write(FIXTURES_DIR / "embed_ids.i32", ids.numpy().tobytes())
    write(FIXTURES_DIR / "embed_out.bf16", bf16_bytes(y_embed))

    print("[gen] done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
