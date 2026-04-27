#!/usr/bin/env python3
"""
Generate ONNX-Runtime reference fixtures for textsift conformance.

Uses the SAME `model_q4f16.onnx` file textsift loads (downloaded from
HuggingFace), runs it through ONNX Runtime (the canonical reference for
how this model behaves at int4+fp16 quantization), and applies the SAME
Viterbi+biases decoder textsift uses (loaded from the model's
`viterbi_calibration.json`). The resulting spans are committed as
`fixtures.json`; the JS-side test (`tests/native/integration/
pytorch-parity.test.js`) loads them and asserts textsift's
re-implementation produces identical spans.

Why ORT q4f16 + canonical Viterbi (not PyTorch fp32 + argmax):

- textsift loads `model_q4f16.onnx`. PyTorch's full-precision pipeline
  is a different *model artefact* — small q4f16 quantization deltas can
  flip argmax on borderline tokens, which is real-but-not-textsift's-
  fault divergence.

- The model card specifies Viterbi+biases as the canonical decoder
  (`opf._core.decoding.VITERBI_BIAS_KEYS`, mirrored in textsift's
  `model/calibration.ts`). Pure argmax can produce BIOES-invalid
  sequences; both ORT-side and textsift-side need the same legality-
  enforcing decoder for an apples-to-apples comparison.

By matching both the model file and the decoder, any remaining
divergence is a textsift kernel/wiring bug — exactly what conformance
tests exist to catch.

Run once when the upstream model changes; otherwise the committed
`fixtures.json` is the ground truth.

Usage:
    python -m venv .venv && source .venv/bin/activate
    pip install onnxruntime transformers huggingface-hub numpy
    python tests/conformance/pytorch/generate-fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

MODEL_REPO = "openai/privacy-filter"
ONNX_PATH = "onnx/model_q4f16.onnx"
HERE = Path(__file__).resolve().parent
INPUTS_FILE = HERE / "inputs.json"
OUT_FILE = HERE / "fixtures.json"

NEG_INF = -1e30


def load_id2label() -> list[str]:
    """Pull config.json directly to get id2label without instantiating
    the full bf16 PyTorch model (which torch can't move to numpy)."""
    cfg_path = hf_hub_download(MODEL_REPO, "config.json")
    cfg = json.loads(Path(cfg_path).read_text())
    id2label = cfg["id2label"]
    out = [""] * len(id2label)
    for k, v in id2label.items():
        out[int(k)] = v
    return out


def load_calibration() -> dict[str, float]:
    cal_path = hf_hub_download(MODEL_REPO, "viterbi_calibration.json")
    biases = json.loads(Path(cal_path).read_text())["operating_points"]["default"]["biases"]
    return {
        "background_stay": biases["transition_bias_background_stay"],
        "background_to_start": biases["transition_bias_background_to_start"],
        "inside_to_continue": biases["transition_bias_inside_to_continue"],
        "inside_to_end": biases["transition_bias_inside_to_end"],
        "end_to_background": biases["transition_bias_end_to_background"],
        "end_to_start": biases["transition_bias_end_to_start"],
    }


def build_decoder(label_set: list[str], cal: dict[str, float]):
    """Build the BIOES Viterbi decoder used by textsift.

    Layout: tag id 0 = O. Span labels start at 1, each contributes 4
    consecutive ids (B, I, E, S). This matches `bioesTagOf(label, boundary)`
    in `inference/viterbi.ts` and the on-disk class layout in the ONNX
    output.
    """
    span_labels = [lbl for lbl in label_set if lbl != "O"]
    # BIOES tag id schema must match the model's class output. Inspect
    # the class names; they're already in BIOES form (O, B-x, I-x, E-x, S-x).
    # Group by label.
    groups: dict[str, dict[str, int]] = {}
    for tag_id, name in enumerate(label_set):
        if name == "O":
            continue
        prefix, _, lbl = name.partition("-")
        groups.setdefault(lbl, {})[prefix] = tag_id

    C = len(label_set)
    O_ID = label_set.index("O")

    transitions = np.full((C, C), NEG_INF, dtype=np.float32)
    # transitions[to, from] is the bias added on `from → to`.
    transitions[O_ID, O_ID] = cal["background_stay"]
    for lbl, ids in groups.items():
        B, I, E, S = ids.get("B"), ids.get("I"), ids.get("E"), ids.get("S")
        if None in (B, I, E, S):
            continue
        # O → B / O → S
        transitions[B, O_ID] = cal["background_to_start"]
        transitions[S, O_ID] = cal["background_to_start"]
        # B → I / B → E
        transitions[I, B] = cal["inside_to_continue"]
        transitions[E, B] = cal["inside_to_end"]
        # I → I / I → E
        transitions[I, I] = cal["inside_to_continue"]
        transitions[E, I] = cal["inside_to_end"]
        # E → O / S → O
        transitions[O_ID, E] = cal["end_to_background"]
        transitions[O_ID, S] = cal["end_to_background"]
        # E/S → B/S of any label (entity-immediately-followed-by-entity).
        for lbl2, ids2 in groups.items():
            B2, S2 = ids2.get("B"), ids2.get("S")
            if B2 is None or S2 is None:
                continue
            transitions[B2, E] = cal["end_to_start"]
            transitions[S2, E] = cal["end_to_start"]
            transitions[B2, S] = cal["end_to_start"]
            transitions[S2, S] = cal["end_to_start"]

    start_scores = np.full(C, NEG_INF, dtype=np.float32)
    start_scores[O_ID] = 0.0
    for ids in groups.values():
        if "B" in ids:
            start_scores[ids["B"]] = 0.0
        if "S" in ids:
            start_scores[ids["S"]] = 0.0

    return transitions, start_scores, groups, O_ID


def viterbi_decode(
    logits: np.ndarray,  # (T, C)
    transitions: np.ndarray,  # (C, C) — transitions[to, from]
    start_scores: np.ndarray,  # (C,)
) -> np.ndarray:
    T, C = logits.shape
    alpha = np.full((T, C), NEG_INF, dtype=np.float32)
    back = np.zeros((T, C), dtype=np.int32)

    alpha[0] = start_scores + logits[0]
    for t in range(1, T):
        # For each `to` j, find best `from` i.
        scores = alpha[t - 1][None, :] + transitions  # (C-to, C-from)
        best_from = scores.argmax(axis=1)
        best_val = scores[np.arange(C), best_from]
        alpha[t] = best_val + logits[t]
        back[t] = best_from

    tags = np.zeros(T, dtype=np.int32)
    tags[T - 1] = int(alpha[T - 1].argmax())
    for t in range(T - 1, 0, -1):
        tags[t - 1] = back[t, tags[t]]
    return tags


def trim_leading_ws(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start] in (" ", "\t", "\n", "\r"):
        start += 1
    return start, end


def tags_to_spans(
    tags: np.ndarray, offsets: list[tuple[int, int]], label_set: list[str], text: str
) -> list[dict]:
    spans: list[dict] = []
    i = 0
    while i < len(tags):
        name = label_set[int(tags[i])]
        if name == "O" or name.startswith("E-") or name.startswith("I-"):
            i += 1
            continue
        prefix, _, lbl = name.partition("-")
        if prefix == "S":
            start, end = offsets[i]
            start, end = trim_leading_ws(text, start, end)
            if start != end:
                spans.append({"label": lbl, "start": start, "end": end})
            i += 1
            continue
        if prefix == "B":
            start = offsets[i][0]
            j = i + 1
            while j < len(tags):
                tag_name = label_set[int(tags[j])]
                if tag_name in (f"I-{lbl}", f"E-{lbl}"):
                    j += 1
                    continue
                break
            end = offsets[j - 1][1] if j > i else offsets[i][1]
            start, end = trim_leading_ws(text, start, end)
            if start != end:
                spans.append({"label": lbl, "start": start, "end": end})
            i = j
            continue
        i += 1
    return spans


def main() -> None:
    inputs: list[str] = json.loads(INPUTS_FILE.read_text())

    onnx_path = hf_hub_download(MODEL_REPO, ONNX_PATH)
    # Pull the .onnx_data shards alongside; ORT loads them by sibling-path.
    for shard in ("onnx/model_q4f16.onnx_data",):
        try:
            hf_hub_download(MODEL_REPO, shard)
        except Exception:
            pass
    sess = ort.InferenceSession(onnx_path)
    tok = AutoTokenizer.from_pretrained(MODEL_REPO)
    label_set = load_id2label()
    cal = load_calibration()
    transitions, start_scores, _, _ = build_decoder(label_set, cal)

    results = []
    for text in inputs:
        enc = tok(text, return_tensors="np", return_offsets_mapping=True)
        offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
        out = sess.run(
            None,
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            },
        )
        logits = out[0][0]  # (T, C)
        tags = viterbi_decode(logits, transitions, start_scores)
        spans = tags_to_spans(tags, offsets, label_set, text)
        results.append(
            {
                "text": text,
                "token_ids": enc["input_ids"][0].tolist(),
                "offsets": [list(o) for o in offsets],
                "tags": [label_set[int(t)] for t in tags],
                "spans": spans,
            }
        )

    OUT_FILE.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {len(results)} fixtures to {OUT_FILE}")


if __name__ == "__main__":
    main()
