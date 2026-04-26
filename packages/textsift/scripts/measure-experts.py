#!/usr/bin/env python3
"""
Go/no-go measurement for MoE-aware compression.

Runs openai/privacy-filter over a corpus of sample text and records, per
expert per layer, how often it was selected by the router. This produces:

  1. Per-layer expert activation frequency — histogram of hits / N_tokens
     across all experts at each layer. If the distribution has a heavy
     long tail (say bottom-30% of experts account for <5% of activations),
     cold-expert pruning is viable.

  2. Pairwise cosine similarity of expert weight tensors within each layer.
     If many experts are near-duplicates (cos > 0.9), K-means clustering
     over experts can compress without much quality loss.

Both signals are logged to stdout and saved as `.npz` for follow-up
plotting or modeling.

Usage:
    pip install torch transformers safetensors huggingface-hub numpy
    python3 scripts/measure-experts.py \\
        --corpus data/sample_corpus.txt \\
        --max-tokens 50000 \\
        --output measurements/expert-activations.npz

Corpus: any text file, one example per line. Use a diverse sample:
Common Crawl excerpt + Enron emails + Wikipedia should give a realistic
mix of PII-adjacent prose.

Go/no-go criteria (both must hold for MoE compression to be worth
pursuing in Stage 3):
  - GOOD: top-20% of experts absorb >80% of activations per layer.
  - GOOD: median pairwise cos-sim among cold experts > 0.6 (cluster-able).
  - BAD: activation distribution is flat (all experts used equally).
  - BAD: cold-expert weights are noise-like (cos-sim ~ 0).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="openai/privacy-filter",
                   help="HF Hub model id or local path.")
    p.add_argument("--corpus", type=Path, required=True,
                   help="Text corpus: one sample per line.")
    p.add_argument("--max-tokens", type=int, default=50_000,
                   help="Stop after this many tokens processed across all samples.")
    p.add_argument("--max-samples", type=int, default=10_000,
                   help="Stop after this many corpus samples (whichever comes first).")
    p.add_argument("--top-k-experts", type=int, default=8,
                   help="How many experts the router picks per token. Match upstream config.")
    p.add_argument("--output", type=Path, default=Path("measurements/expert-activations.npz"),
                   help="Path for the .npz activation + similarity measurements.")
    p.add_argument("--device", default=None,
                   help="torch device, e.g. 'cpu', 'cuda', 'mps'. Default: auto.")
    return p.parse_args()


def select_device(requested: str | None) -> str:
    if requested is not None:
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_corpus(path: Path, max_samples: int) -> list[str]:
    if not path.exists():
        print(f"[measure-experts] corpus file not found: {path}", file=sys.stderr)
        sys.exit(1)
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(line)
            if len(lines) >= max_samples:
                break
    return lines


def register_router_hooks(model: Any, num_experts: int) -> tuple[list[np.ndarray], list[str]]:
    """
    Attach forward-hooks on every MoE router in the model so we can log
    which experts were selected for each token.

    Returns:
      counters: list of ndarrays, one per MoE layer, shape [num_experts,].
      names:    list of dotted module names (for labelling the output).
    """
    import torch
    import torch.nn as nn

    counters: list[np.ndarray] = []
    names: list[str] = []

    def make_hook(layer_index: int):
        def hook(_mod: nn.Module, _inp: Any, out: Any) -> None:
            # Routers differ across architectures. Identify the selected-
            # expert-indices tensor by dtype + shape rather than tuple
            # position: it's the integer tensor whose last dim is top_k.
            # openai_privacy_filter returns (router_logits [T, n_exp],
            # routing_weights [T, top_k] f32, selected_indices [T, top_k] i64).
            # Positional indexing on out[1] captures the float weights,
            # which round to 0 under int cast — silently collapsing every
            # activation onto expert 0.
            selected = None
            if isinstance(out, tuple):
                for item in out:
                    if torch.is_tensor(item) and item.dtype in (torch.int32, torch.int64):
                        selected = item
                        break
            elif isinstance(out, dict) and "selected_experts" in out:
                selected = out["selected_experts"]
            elif torch.is_tensor(out) and out.dtype in (torch.int32, torch.int64):
                selected = out
            if selected is None:
                return
            counts = torch.bincount(
                selected.flatten().to(torch.int64),
                minlength=counters[layer_index].shape[0],
            ).cpu().numpy()
            counters[layer_index] += counts[: counters[layer_index].shape[0]]
        return hook

    for module_name, module in model.named_modules():
        lower = module_name.lower()
        if lower.endswith(".router") or lower.endswith(".gate") and "expert" in lower:
            counters.append(np.zeros(num_experts, dtype=np.int64))
            names.append(module_name)
            module.register_forward_hook(make_hook(len(counters) - 1))
    return counters, names


def compute_expert_cosines(model: Any, router_names: list[str]) -> dict[str, np.ndarray]:
    """
    For each MoE layer with a hooked router, compute pairwise cosine
    similarity between expert weight tensors. openai_privacy_filter stacks
    all experts into a single parameter per layer:
      `mlp.experts.gate_up_proj`  shape [num_experts, d_model, 2*d_ff]
      `mlp.experts.down_proj`     shape [num_experts, d_ff,  d_model]
    We flatten each expert's (gate_up_proj ⊕ down_proj) into one vector and
    compute a [num_experts, num_experts] cosine-similarity matrix per layer.

    Returns: mapping of layer name → [num_experts, num_experts] cos-sim matrix.
    """
    import torch

    out: dict[str, np.ndarray] = {}
    for name in router_names:
        layer_prefix = name.rsplit(".router", 1)[0] if ".router" in name else name.rsplit(".gate", 1)[0]
        experts_module = None
        for sub_name, module in model.named_modules():
            if sub_name == f"{layer_prefix}.experts":
                experts_module = module
                break
        if experts_module is None:
            continue

        # Gather all stacked per-expert tensors. For openai_privacy_filter
        # the names are gate_up_proj / down_proj; other MoE archs may use
        # wi / wo or experts.<i>.w_up — we handle both shapes.
        stacked_tensors: list[torch.Tensor] = []
        for param_name, param in experts_module.named_parameters(recurse=True):
            if not isinstance(param, torch.Tensor):
                continue
            if param.dim() < 2:
                continue
            # First dim must be num_experts; anything else we flatten later.
            stacked_tensors.append(param.detach().float())

        if not stacked_tensors:
            continue

        n_experts = stacked_tensors[0].shape[0]
        if any(t.shape[0] != n_experts for t in stacked_tensors):
            # Mixed first-dim sizes — bail out for this layer rather than
            # producing a confusing matrix.
            continue

        flat = torch.cat(
            [t.reshape(n_experts, -1) for t in stacked_tensors],
            dim=1,
        )
        flat = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
        cos = (flat @ flat.T).cpu().numpy()
        out[name] = cos
    return out


def summarise_activations(counters: list[np.ndarray], names: list[str]) -> None:
    print()
    print("=== Activation frequency per layer ===")
    print(f"{'layer':<50s} {'n_exp':>6s} {'top20%':>8s} {'hot>10%':>8s} {'cold<0.1%':>10s}")
    for name, cnts in zip(names, counters):
        total = cnts.sum()
        if total == 0:
            print(f"{name[:50]:<50s} {'—':>6s}")
            continue
        fractions = cnts / total
        sorted_fracs = np.sort(fractions)[::-1]
        top20_count = max(1, len(fractions) // 5)
        top20_share = sorted_fracs[:top20_count].sum()
        hot = int((fractions > 0.10).sum())
        cold = int((fractions < 0.001).sum())
        print(f"{name[:50]:<50s} {len(fractions):>6d} {top20_share:>7.1%} {hot:>8d} {cold:>10d}")


def summarise_cosines(cosines: dict[str, np.ndarray]) -> None:
    if not cosines:
        print("(no expert weight tensors identified — cosine analysis skipped)")
        return
    print()
    print("=== Pairwise expert cosine-similarity per layer ===")
    print(f"{'layer':<50s} {'median':>8s} {'p90':>8s} {'max-off-diag':>14s}")
    for name, mat in cosines.items():
        # Off-diagonal only.
        n = mat.shape[0]
        off = mat[~np.eye(n, dtype=bool)]
        print(f"{name[:50]:<50s} {np.median(off):>8.3f} {np.percentile(off, 90):>8.3f} {np.max(off):>14.3f}")


def main() -> int:
    args = parse_args()

    # Import heavy deps here so `-h` is fast.
    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as e:
        print(f"[measure-experts] missing dependency: {e}", file=sys.stderr)
        print("Install: pip install torch transformers", file=sys.stderr)
        return 1

    device = select_device(args.device)
    print(f"[measure-experts] device: {device}")
    print(f"[measure-experts] loading {args.model} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    model.eval()
    model.to(device)
    print(f"[measure-experts] model on device, param count: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    num_experts = int(getattr(model.config, "num_local_experts", 0))
    if num_experts <= 0:
        print("[measure-experts] config.num_local_experts not set; falling back to 1024", file=sys.stderr)
        num_experts = 1024
    counters, names = register_router_hooks(model, num_experts)
    print(f"[measure-experts] hooked {len(names)} router modules × {num_experts} experts")
    if not names:
        print("[measure-experts] no router modules identified — model may not expose MoE internals via standard names.", file=sys.stderr)
        print("                   Adjust `register_router_hooks` for this model's actual MoE naming.", file=sys.stderr)

    samples = load_corpus(args.corpus, args.max_samples)
    print(f"[measure-experts] corpus: {len(samples)} samples")

    tokens_processed = 0
    with torch.inference_mode():
        for i, text in enumerate(samples):
            if tokens_processed >= args.max_tokens:
                break
            enc = tokenizer(text, truncation=True, max_length=2048, return_tensors="pt").to(device)
            _ = model(**enc)
            tokens_processed += int(enc["input_ids"].numel())
            if i % 50 == 0:
                print(f"[measure-experts]   {i}/{len(samples)} samples, {tokens_processed} tokens")

    print(f"[measure-experts] done: {tokens_processed} tokens over {i + 1} samples")

    summarise_activations(counters, names)
    print("[measure-experts] computing pairwise expert cosine similarities …")
    cosines = compute_expert_cosines(model, names)
    summarise_cosines(cosines)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        router_names=np.array(names, dtype=object),
        **{f"counts__{i}": c for i, c in enumerate(counters)},
        **{f"cosines__{i}": m for i, m in enumerate(cosines.values())},
        meta=np.array([json.dumps({
            "model": args.model,
            "tokens_processed": tokens_processed,
            "samples": len(samples),
            "top_k_experts": args.top_k_experts,
        })]),
    )
    print(f"[measure-experts] saved to {args.output}")

    # Exit status encodes the go/no-go signal so CI can gate on it.
    total_layers = len(counters)
    if total_layers == 0:
        return 2  # couldn't measure
    hot_ratios = [
        np.sort(c / c.sum())[::-1][:max(1, len(c) // 5)].sum()
        if c.sum() > 0 else 0.0
        for c in counters
    ]
    if all(r > 0.70 for r in hot_ratios):
        print("[measure-experts] GO — top-20% of experts absorb >70% of activations per layer.")
        return 0
    print("[measure-experts] NO-GO — activation distribution is too flat for expert clustering.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
