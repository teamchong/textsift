#!/usr/bin/env python3
"""
Generate PyTorch reference fixtures for textsift conformance.

For each input in `inputs.json`, run the openai/privacy-filter model via
HuggingFace transformers (the canonical PyTorch reference), record the
per-token argmax tag and the resulting BIOES-decoded spans, and write the
combined output to `fixtures.json`.

`fixtures.json` is committed to git; the JS-side test
(`tests/native/integration/pytorch-parity.test.js`) loads it and asserts
textsift's WASM/native backends produce the same per-token tags.

Run once when the model changes; otherwise the committed fixtures are
treated as the ground truth.

Usage:
    python -m venv .venv && source .venv/bin/activate
    pip install torch transformers
    python tests/conformance/pytorch/generate-fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_NAME = "openai/privacy-filter"
HERE = Path(__file__).resolve().parent
INPUTS_FILE = HERE / "inputs.json"
OUT_FILE = HERE / "fixtures.json"


def bioes_to_spans(tags: list[str], offsets: list[tuple[int, int]]) -> list[dict]:
    """Convert per-token BIOES tags + (start, end) offsets to span list."""
    spans: list[dict] = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag == "O" or tag.startswith("E-") or tag.startswith("I-"):
            i += 1
            continue
        if tag.startswith("S-"):
            label = tag[2:]
            start, end = offsets[i]
            if start != end:
                spans.append({"label": label, "start": start, "end": end})
            i += 1
            continue
        if tag.startswith("B-"):
            label = tag[2:]
            start = offsets[i][0]
            j = i + 1
            while j < len(tags) and tags[j] in (f"I-{label}", f"E-{label}"):
                j += 1
            end = offsets[j - 1][1] if j > i else offsets[i][1]
            spans.append({"label": label, "start": start, "end": end})
            i = j
            continue
        i += 1
    return spans


def main() -> None:
    inputs: list[str] = json.loads(INPUTS_FILE.read_text())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME).eval()
    id2label = model.config.id2label

    results = []
    for text in inputs:
        encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        offsets = encoded.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            logits = model(**encoded).logits[0]
        argmax_ids = logits.argmax(dim=-1).tolist()
        tags = [id2label[i] for i in argmax_ids]
        spans = bioes_to_spans(tags, offsets)
        results.append(
            {
                "text": text,
                "token_ids": encoded["input_ids"][0].tolist(),
                "offsets": offsets,
                "argmax_tags": tags,
                "spans": spans,
            }
        )

    OUT_FILE.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {len(results)} fixtures to {OUT_FILE}")


if __name__ == "__main__":
    main()
