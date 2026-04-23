#!/usr/bin/env python3
"""
Build the full-model e2e parity fixture.

Generates three files under the scratch dir:

  <scratch>/pii-weights-full.bin        — all 140 tensors, experts in
                                           int4 blockwise (transposed).
  <scratch>/full_input_ids.i32          — tokenized sample input.
  <scratch>/full_logits.bf16            — PyTorch reference logits
                                           [T, num_classes] bf16.

Designed to be consumed by `scripts/verify-full-parity.mjs`. Both sides
agree on the scratch dir via --scratch-dir.

    python3 scripts/gen-full-parity-fixture.py --scratch-dir /tmp/pii-wasm-e2e

This is a heavy fixture: the full blob is ~800 MB (all 128 experts,
bf16 weights everywhere except the int4 expert projections). Not for
CI. Run once locally, re-run after a weight / quantization change.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_TEXT = (
    "Please email me at jane.doe@example.com about invoice 4511 "
    "or call (555) 123-4567. My mailing address is 123 Main Street, "
    "Springfield, IL 62704."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scratch-dir", type=Path, required=True,
                   help="Directory to write the blob + fixtures into.")
    p.add_argument("--sample", default=SAMPLE_TEXT,
                   help="Text to tokenize + run through the model.")
    p.add_argument("--skip-blob", action="store_true",
                   help="Skip regenerating the weight blob (assume already present).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.scratch_dir.mkdir(parents=True, exist_ok=True)

    blob_path = args.scratch_dir / "pii-weights-full.bin"
    ids_path = args.scratch_dir / "full_input_ids.i32"
    logits_path = args.scratch_dir / "full_logits.bf16"

    if not args.skip_blob:
        print(f"[full-parity] building full blob → {blob_path}", file=sys.stderr)
        convert = REPO_ROOT / "scripts" / "convert_weights.py"
        cmd = [
            sys.executable, str(convert),
            "--full",
            "--output", str(blob_path),
            "--quant-transpose", "mlp.experts.gate_up_proj",
            "--quant-transpose", "mlp.experts.down_proj",
        ]
        subprocess.check_call(cmd)
    else:
        print(f"[full-parity] reusing existing blob at {blob_path}", file=sys.stderr)

    print("[full-parity] loading openai/privacy-filter for reference pass …", file=sys.stderr)
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    model = AutoModelForTokenClassification.from_pretrained("openai/privacy-filter").eval()
    tokenizer = AutoTokenizer.from_pretrained("openai/privacy-filter")

    enc = tokenizer(args.sample, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]                         # [1, T]
    attention_mask = enc.get("attention_mask")
    T = input_ids.shape[1]
    print(f"[full-parity] sample tokenized to T={T}", file=sys.stderr)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits.squeeze(0).contiguous().to(torch.bfloat16)

    input_ids.squeeze(0).to(torch.int32).numpy().tofile(ids_path)
    logits.view(torch.uint16).numpy().tofile(logits_path)
    print(f"[full-parity] wrote {ids_path}  ({T * 4} bytes)", file=sys.stderr)
    print(f"[full-parity] wrote {logits_path}  ({T * 33 * 2} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
