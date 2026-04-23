# Roadmap

This is the execution plan for `@yourorg/pii-wasm`. Stage 0 ships a working
library by wrapping transformers.js behind the public API. Stages 1–3
replace internals with progressively faster and more compact inference
paths. The public API doesn't change between stages.

## Stage 0 — transformers.js baseline

**Goal:** a working end-to-end `PrivacyFilter` that passes the same
inputs through the same output pipeline as upstream.

- [x] Public API surface (`src/js/types.ts`, `src/js/index.ts`).
- [x] Viterbi CRF decoder in JS (`src/js/inference/viterbi.ts`).
- [x] BIOES → span conversion (`src/js/inference/spans.ts`).
- [x] Chunking for long inputs (`src/js/inference/chunking.ts`).
- [x] Redaction applicator with flexible marker strategy (`src/js/inference/redact.ts`).
- [x] Calibration JSON loader (`src/js/model/calibration.ts`).
- [ ] Model loader (`src/js/model/loader.ts`) — OPFS + HuggingFace Hub fetch.
- [ ] Tokenizer wrapper (`src/js/model/tokenizer.ts`) — delegates to
      `@huggingface/transformers` tokenizer utilities.
- [ ] Transformers.js backend wiring (`src/js/backends/transformers-js.ts`):
      extract raw per-token logits from `AutoModelForTokenClassification`,
      not the pre-decoded span list from the higher-level pipeline.
- [ ] End-to-end test: tokenize → forward → Viterbi → spans → redact,
      on 5 sample inputs; compare against `opf` CLI output on the same
      inputs (byte-identical redacted text + span tuples).
- [ ] Demo page (`demo/index.html`).

**Ship gate:** parity with `opf` CLI redaction on the upstream sample
set. Benchmark: memory + latency per 1k-token input on Chrome + Firefox.

## Stage 1 — Zig + WASM + SIMD128

**Goal:** beat transformers.js WASM backend on speed + memory.

Target: 1.5–2× faster, 30–50% lower peak memory.

- [ ] Zig build config (`build.zig`) producing a `wasm32-freestanding`
      module with `simd128` + `relaxed_simd` CPU features.
- [ ] Int4 matmul with `@Vector` FMA chains.
- [ ] MoE router + sparse expert dispatch.
- [ ] GQA attention kernel.
- [ ] Layer norm, activation, embedding lookup.
- [ ] C-ABI exports (`src/zig/wasm_exports.zig`).
- [ ] JS bridge (`src/js/backends/wasm.ts`).
- [ ] Cross-backend conformance: WASM output matches transformers.js
      within 1e-5 relative error.
- [ ] Benchmark vs Stage 0.

## Stage 2 — WebGPU with hand-tuned WGSL (deferred)

**Deferred until post-shipping customer demand justifies it. The WASM
path from Stage 1 is the minimum viable product.**

Rationale: GPU advantage for this model (50M active, single-pass
classifier) is ~3× over CPU, not the 30-100× seen with large
autoregressive models. Stage 1 alone delivers 200-400 ms per typical
input — usable for every product shape we plan to ship (form
submission, browser extension, ETL pre-processing). WGSL doubles the
maintenance surface (conformance tests, browser-compat branches) for a
polish win that doesn't unlock new use cases.

Revisit if:
- A customer has a long-document redaction workload (10k+ tokens per
  request) where N²-attention on CPU becomes a real bottleneck.
- Batch throughput (hundreds of documents / second) becomes a
  real requirement. Desktop GPU shines at batched inference.

Rough design if we end up building it:
- [ ] WGSL kernels: int4 matmul, MoE dispatch, attention, norm.
- [ ] JS backend (`src/js/backends/webgpu.ts`).
- [ ] Subgroup-free reductions so Firefox works.
- [ ] Benchmark vs Stage 1.

## Stage 3 — MoE-specific compression

**Goal:** shrink the model below Cloudflare Workers' 128 MB memory cap.

**Gate:** `scripts/measure-experts.py` must return GO.

- [ ] Run go/no-go measurement script on openai/privacy-filter.
- [ ] If GO: K-means clustering over experts per layer, K ∈ {8, 16, 32, 64}.
      Measure BIOES F1 delta vs uncompressed at each K.
- [ ] Cold-expert pruning: drop experts below activation threshold,
      remeasure F1.
- [ ] Hadamard rotation + int3 residual (QuIP#-style).
- [ ] Compressed-model binary format: routing table + shared-codebook
      experts + preserved-bit-exact hot experts.
- [ ] Cloudflare Worker deployment: model fits under 128 MB, WASM
      backend uses compressed format.

## Stage 4 — Distribution

- [ ] npm publish.
- [ ] Browser extension (PII warning on form submit).
- [ ] Docs site (Tufte-style minimal, hosted on GitHub Pages).
- [ ] Blog post with benchmarks vs transformers.js + Presidio.

## Out of scope

- Multi-tenant HTTP API (we're a library, not a service).
- Server-side Node-cluster deployments (users run their own).
- Fine-tuning tooling (defer to upstream `opf` CLI).
- Non-English PII categories (upstream model is English-first).
