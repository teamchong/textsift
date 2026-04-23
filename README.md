# pii-wasm

**Browser-native PII detection and redaction, based on `openai/privacy-filter`.**

Runs entirely on the user's device via WebAssembly (universal) or WebGPU (fast path on capable browsers). No network calls to external inference services. PII text never leaves the browser.

## Why

OpenAI released `openai/privacy-filter` (April 20, 2026) — a 1.5B-parameter / 50M-active MoE token classifier for PII detection, Apache-2.0 licensed. It's marketed as browser-runnable; OpenAI's own Python SDK (`opf`) runs server-side.

The gap: no one has shipped a browser-native inference engine for it yet. That's what this repo is.

## Status

Scaffold — public API designed end-to-end, backend pending.

Roadmap tracked in `docs/roadmap.md`:

- Go/no-go measurement: MoE expert activation-frequency distribution (`scripts/measure-experts.py`).
- Stage 0 backend: `transformers.js` baseline.
- Stage 1 backend: Zig + WASM + SIMD128.
- Stage 2 backend: WGSL.
- Viterbi CRF decoder in JS (matches upstream calibration artifacts).
- BIOES → span reconstruction.
- Chunking for long inputs.
- Browser demo + browser-extension wrapper.

## Public API

```ts
import { PrivacyFilter } from "@yourorg/pii-wasm";

const filter = await PrivacyFilter.create({
  onProgress: (p) => console.log(`${p.stage}: ${p.loaded}/${p.total}`),
});

const result = await filter.redact("Alice was born on 1990-01-02.");
//  result.redactedText = "[private_person] was born on [private_date]."
//  result.spans = [
//    { label: "private_person", start: 0,  end: 5,  text: "Alice",      marker: "[private_person]" },
//    { label: "private_date",   start: 18, end: 28, text: "1990-01-02", marker: "[private_date]"   },
//  ]
//  result.summary = { private_person: 1, private_date: 1 }
```

Custom markers, per-category enabling, batch input, and streaming are part of the design — see `src/js/types.ts`.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  PrivacyFilter (public API)                             │
│    create() / redact() / detect() / redactBatch()       │
└────────────┬────────────────────────────────────────────┘
             │
     ┌───────┴────────┐
     │                │
     ▼                ▼
┌──────────┐    ┌──────────┐
│ Backend  │    │ Backend  │   WebGPU if available,
│ WebGPU   │    │ WASM     │   WASM otherwise.
└────┬─────┘    └────┬─────┘
     │               │
     └───────┬───────┘
             │
             ▼
    ┌────────────────┐
    │ Forward pass   │   Produces per-token logits
    │ (weights from  │   over 33 classes (background + 8 spans × BIOES).
    │  OPFS cache)   │
    └────────┬───────┘
             │
             ▼
    ┌────────────────────────┐
    │ Viterbi CRF decoder    │   Uses loaded calibration.json
    │ (JS-side)              │   to resolve transition biases.
    └────────┬───────────────┘
             │ best tag sequence
             ▼
    ┌────────────────────────┐
    │ BIOES → span merger    │   Character-level spans.
    └────────┬───────────────┘
             │
             ▼
    ┌────────────────────────┐
    │ Redaction applicator   │   Replace spans with markers.
    └────────────────────────┘
```

The backend interface is the only thing that changes between WebGPU and WASM paths. Everything else (tokenization, Viterbi, span merging, redaction application) is shared JavaScript.

## Research plan

Three-stage cost reduction, each independently validatable:

**Stage 0 — `transformers.js` baseline.**
Working end-to-end browser inference. Benchmark memory + latency. Establishes the number to beat.

**Stage 1 — Zig + WASM + SIMD128 custom inference.**
Hand-rolled int4 MoE matmul + attention + router. Target: ~2× faster than transformers.js WASM backend; 30-50% lower memory; universal browser support.

**Stage 2 — WebGPU path with hand-tuned WGSL.**
Custom WGSL matmul + MoE dispatch. Target: ~2× faster than transformers.js WebGPU backend; preferred when `navigator.gpu` is available.

**Stage 3 — MoE-specific compression (novel).**

MoE models have structural redundancy a dense model doesn't: 384 experts per layer, many rarely activated. Stackable techniques:

| Technique | Expected shrink | Notes |
|---|---|---|
| Int4 baseline quantization | 4× | Standard, minimal accuracy loss |
| Expert-cluster codebook (K-means over experts) | 3-5× on experts | Novel; repurposes PLE-codebook pattern for MoE |
| Cold-expert pruning (drop low-activation experts) | 1.5-2× | Wanda-style, MoE-aware |
| Hadamard rotation + int3 residual | 1.3× | QuIP#-style incoherence processing |

Stacked: theoretical ~20-50× reduction vs fp16. Target ~150 MB on disk → fits Cloudflare Workers (128 MB memory cap).

**Go/no-go for Stage 3:** `scripts/measure-experts.py` measures per-expert activation frequency on the calibration corpus. Goal: confirm long-tail distribution that justifies cluster + prune. ~2 hours of Python; binary signal.

## Non-goals

- **Not** a wrapper around the OpenAI Python CLI.
- **Not** an HTTP API to a remote inference service.
- **Not** a multi-tenant server product.
- **Not** a fine-tuning tool.

This library does ONE thing: run privacy-filter inference in the browser, as efficiently as possible, with a clean API.

## License

Apache 2.0 — matches upstream.

## Upstream

- Model: https://huggingface.co/openai/privacy-filter
- Reference implementation: https://github.com/openai/privacy-filter
