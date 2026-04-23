# Stage 3 gate — NO-GO

Measurement run for the MoE-compression go/no-go (Stage 3 gate in
`docs/roadmap.md`). Executed 2026-04-23 on `openai/privacy-filter`.
Raw output artifact: `measurements/enron-50k.npz`.

## Setup

- Model: `openai/privacy-filter` from HuggingFace Hub, BF16 safetensors.
- transformers: 5.6.0 (5.5.3 doesn't know the architecture yet;
  5.6.0 added it).
- Device: MPS (Apple Silicon).
- Corpus: `LLM-PBE/enron-email` (HF Hub), first 2 000 emails with
  length ≥ 100 chars, whitespace-flattened to one sample per line.
  Stored as `data/enron.txt` (3.3 MB).
- Budget: `--max-tokens 50000`. The run stopped at 50 124 tokens
  across 127 samples (emails are ~400 BPE tokens on average).
- Per-layer activation counter: 128 slots (matches
  `config.num_local_experts`).
- Script: `scripts/measure-experts.py`.

Two bugs in the script were fixed during the run:

1. Counter was sized to a hardcoded 1024; switched to reading
   `config.num_local_experts` from the model config.
2. The router's forward output is a 3-tuple
   `(logits [T, n_exp] f32, routing_weights [T, top_k] f32,
   selected_indices [T, top_k] i64)`. The hook was binning `out[1]`
   (float routing weights) and silently casting to int — every
   activation collapsed into bin 0, producing a spurious GO. The
   fix scans the tuple for an int-dtype tensor and bins that.

## Results

### Activation frequency per layer (Enron, 50 124 tokens × 4 top-k = 200 496 activations per layer)

| Layer | top-20% share | experts > 10% | experts < 0.1% |
|-------|---------------|---------------|----------------|
| 0     | 44.6%         | 0             | 2              |
| 1     | 42.9%         | 0             | 1              |
| 2     | 58.0%         | 0             | 1              |
| 3     | 48.0%         | 0             | 0              |
| 4     | 50.8%         | 0             | 0              |
| 5     | 52.0%         | 0             | 0              |
| 6     | 50.6%         | 0             | 0              |
| 7     | 53.7%         | 0             | 0              |

- Top-20% of experts (26 of 128 per layer) absorb 43–58% of
  activations. Mild tail; far below the ≥ 70% skew that makes
  pruning worthwhile.
- **No expert gets more than 10% of the activations anywhere** —
  no hot kernels to preserve bit-exact.
- **Only zero to two experts per layer get < 0.1%** — no cold tail
  to drop. Dropping the bottom 20% would cut ~8–12% of routed
  capacity, redistributed across ~100 other experts.

### Pairwise expert cosine similarity (weight-based, corpus-independent)

Computed over `experts.gate_up_proj ⊕ experts.down_proj`, flattened
per expert, row-normalised.

| Layer | median | p90   | max off-diag |
|-------|--------|-------|--------------|
| 0     | 0.013  | 0.020 | 0.034        |
| 1     | 0.012  | 0.018 | 0.036        |
| 2     | 0.009  | 0.015 | 0.032        |
| 3     | 0.014  | 0.024 | 0.048        |
| 4     | 0.012  | 0.019 | 0.040        |
| 5     | 0.012  | 0.018 | 0.050        |
| 6     | 0.003  | 0.007 | 0.040        |
| 7     | 0.003  | 0.007 | 0.033        |

The random-baseline cosine for independent vectors in ~1 M-dim space
is ~0.001–0.03. These experts sit squarely in that range — they are
essentially orthogonal, as MoE design intends.

## Interpretation

Both compression axes are ruled out:

- **K-means clustering:** needs cosine > ~0.6 among cold experts to
  cluster without major loss. Observed: order-of-magnitude lower than
  required.
- **Cold-expert pruning:** needs a long heavy tail (top-20% ≥ 70%).
  Observed: top-20% = 50% ± 8%. Nearly uniform.

This is consistent with the model's training: sparse-MoE checkpoints
ship a load-balancing auxiliary loss that explicitly penalises
concentrated routing. The result is a "dense in disguise" MoE —
small active compute, but no redundancy to compress post-hoc.

Hadamard rotation + int3 residual alone is still technically
available but yields ~1.3× reduction (~580 MB). It doesn't fit
Cloudflare Workers' 128 MB cap, and in the browser the 772 MB
Stage-0 weights already cache cleanly, so the engineering cost
isn't justified.

## Decision

Stage 3 deferred indefinitely. Revisit only if:

- A retraining-based path (distillation / MoE-to-dense collapse with
  LoRA fine-tune) enters scope. Currently out-of-scope per project
  rules (`.claude/CLAUDE.md` — "Don't propose training-based
  solutions").
- Upstream publishes a smaller `openai/privacy-filter` variant.
- Cloudflare raises the Workers memory cap high enough for ~580 MB.

## Reproducing

```bash
pip install 'transformers>=5.6.0' torch numpy
mkdir -p data/enron measurements

# Corpus (936 MB; first 2k emails are enough)
curl -L -o data/enron/enron_email_all.jsonl \
  https://huggingface.co/datasets/LLM-PBE/enron-email/resolve/main/enron_email_all.jsonl

python3 -c "
import json
with open('data/enron/enron_email_all.jsonl') as fin, open('data/enron.txt','w') as fout:
    n=0
    for line in fin:
        if n>=2000: break
        d=json.loads(line); t=d.get('text','')
        if not isinstance(t,str): continue
        flat=' '.join(t.split())
        if len(flat)<100: continue
        fout.write(flat+'\n'); n+=1
"

python3 scripts/measure-experts.py \
    --corpus data/enron.txt \
    --max-tokens 50000 \
    --output measurements/enron-50k.npz
```

Expected exit code: 3 (NO-GO). Expected stdout: the two tables above.
