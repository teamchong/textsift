# PyTorch reference fixtures

This directory holds the ground-truth PyTorch outputs the JS-side
conformance test compares textsift against. The fixtures are
**committed to git**; the Python regenerator only runs when the
upstream model changes.

## Files

- `inputs.json` — the fixed list of input strings to evaluate.
- `generate-fixtures.py` — runs `openai/privacy-filter` via PyTorch /
  HuggingFace transformers, dumps per-token argmax tags + spans.
- `fixtures.json` — generated output, committed.
- `../../native/integration/pytorch-parity.test.js` — Node-side test
  that loads `fixtures.json` and asserts textsift's spans match.

## Why a separate fixture file (vs running PyTorch in CI)

PyTorch + transformers is a multi-GB pip install and is slow to set
up on every CI run. Generating fixtures once and committing them
gives us a deterministic, fast comparison without dragging Python
into the JS test loop.

The model is frozen on HuggingFace, so the fixtures stay valid until
someone explicitly re-runs the generator with a newer revision.

## Regenerate

```sh
python -m venv .venv && source .venv/bin/activate
pip install torch transformers
python tests/conformance/pytorch/generate-fixtures.py
```

Commit the new `fixtures.json` and explain why in the commit message.

## Pinning the model revision

Both the Python generator and the JS test load `openai/privacy-filter`
without a revision pin — they get whatever HuggingFace serves as
`main`. If the model is ever updated in a way that breaks parity,
either pin a revision in `generate-fixtures.py` and document the SHA
here, or accept the new outputs by regenerating.

## Known divergence (not yet covered by this suite)

When tags 0..len(input) end up containing an orphan `E-<label>` from
a *different* span than the one being decoded (e.g. the q4f16
argmax over `Send the wire to bank account 9876543210 routing
011000015 by Friday.` produces a stray `E-account_number` for
`015` with no preceding `B-`), textsift's Viterbi can suppress the
upstream legitimate span (the actual bank account `9876543210`),
while a reference run over the same logits decodes it correctly.

The current `inputs.json` deliberately avoids that pattern so the
suite reflects the typical case. To investigate the divergence,
revert the bank-account input to the longer
`Send the wire to bank account 9876543210 routing 011000015 by
Friday.` form and trace logits / Viterbi state.
