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
- [x] Bump `@huggingface/transformers` to `^4.2.0`. The
      `openai_privacy_filter` architecture was added after v3.8.1.
- [x] Model loader (`src/js/model/loader.ts`) — fetches `config.json`
      and `viterbi_calibration.json`, derives the 8 span labels in
      model-head order from `id2label`, and returns the bundle
      transformers.js needs to run `from_pretrained`. Model weights
      + tokenizer are fetched and cached by transformers.js itself.
- [x] Tokenizer wrapper (`src/js/model/tokenizer.ts`) — wraps
      `AutoTokenizer`. transformers.js 4.2.0 doesn't expose
      `return_offsets_mapping`, so `tokenToCharOffset` is derived
      from the GPT-2 byte-level reverse map (same approach as
      upstream `opf`'s tiktoken path).
- [x] Transformers.js backend wiring
      (`src/js/backends/transformers-js.ts`):
      `AutoModelForTokenClassification.from_pretrained(id, { dtype, device })`,
      then one `model({ input_ids, attention_mask })` per chunk,
      returning `output.logits` as our `Logits` struct.
- [x] Cross-check vs the reference transformers.js
      `pipeline('token-classification')` on upstream's
      `sample_eval_five_examples.jsonl`: **byte-identical span tuples**
      (modulo a deliberate whitespace trim to match `opf` CLI's
      `trim_whitespace=true` default). Spans that don't match the
      labelled fixture are model behaviour, not a wiring defect.
- [x] Demo page (`demo/index.html`) — browser-tested via CDP.

**Ship gate (met).** Parity confirmed transitively: pii-wasm ↔
transformers.js pipeline is byte-exact on the same logits; the
pipeline shares the model with `opf` CLI so decoding agreement is
mechanical. Chrome benchmark captured in `docs/benchmarks.md`
(~322 ms warm @ ~700 tokens on M-series, 872 MB peak heap). Firefox
measurement deferred — q4f16 needs MatMulNBits + GatherBlockQuantized
via WebGPU, which Firefox 127+ may dispatch but has not been
verified.

### Stage 0 constraints discovered while wiring

- **BIOES class layout is per-label, stride 4.** The model's
  `id2label` groups `(B, I, E, S)` contiguously per span label, not
  all-Bs then all-Is. Encoded in `viterbi.ts`'s `bioesTagOf` and
  `describeTag` — breaking this invariant mis-decodes every span.
- **ORT Web has no WASM-CPU kernel for `GatherBlockQuantized` or
  `MatMulNBits`.** The only browser-viable ONNX export is
  `model_q4f16.onnx` running on WebGPU. `model_q4` fails even on
  WASM; fp16 (~2 GB) OOMs. In Node, onnxruntime-node picks CPU
  automatically — it does not accept `"wasm"` as a device string,
  so the runtime detects browser (via `navigator.gpu`) and only
  pins `device` when it knows WebGPU is available.

## Stage 1 — Zig + WASM + SIMD128

**Goal:** beat transformers.js ORT-Web path on speed. Original goal
("fall back to WASM on WebGPU-less browsers") was retired on
2026-04-23 — Firefox + Safari both run Stage 0's `q4f16` ONNX
successfully, so this stage is a pure-speed play, not a portability
one.

Target: 1.5–2× faster than Stage 0 Chrome warm-inference
(322 ms @ 700 tokens → ~160–215 ms), 30–50% lower peak memory.

### Phase A — pipeline scaffolding (done)

- [x] `src/zig/wasm_exports.zig` — bump allocator (`heap_init` /
      `alloc` / `reset` / `heap_used`), plumbing smoke-test helpers
      (`echo`, `sum_i32`).
- [x] Built via `npm run build:zig` → `dist/pii.wasm`
      (`wasm32-freestanding`, `simd128` + `relaxed_simd`).
- [x] JS bridge `src/js/backends/wasm.ts` with `loadPiiWasm()`
      loader + `WasmBackend` class (`forward()` throws until
      kernels land).
- [x] End-to-end round-trip verified: echo ABI, JS→WASM memory
      staging, bump heap grow + rewind.

### Phase B — weight conversion + loading (done, plumbing only)

- [x] Python converter `scripts/convert_weights.py`. Phase-B1 scope:
      a 4-tensor subset (classifier head + 2 RMSNorm weights, ~87 KB
      in bf16). `--full` flag emits all 140 tensors (~2.8 GB).
      Int4 blockwise quantization deferred to Phase C (where the int4
      matmul kernel defines the block layout).
- [x] Zig parser in `src/zig/wasm_exports.zig` — validates magic +
      version, walks the 104-byte entry table, exposes per-tensor
      metadata via `weights_{count,dtype,ndim,shape,name,data_ptr,
      data_size}`.
- [x] JS loader `loadWeights()` in `src/js/backends/wasm.ts` —
      fetches blob, verifies sha256, copies into WASM linear memory,
      calls `weights_load`, then pins via `heap_mark_now()` so the
      blob survives per-call `reset()`.
- [x] Cross-verified bit-exact against PyTorch: bf16 bytes in the
      blob and in WASM memory match `state_dict[name].view(uint16)`
      for all 4 test tensors.
- [ ] Full-model blob generation + int4 quantization — unblocked
      only once Phase C/D define the quant block layout.

### Phase C — tiny kernels

- [x] RMSNorm with f32 sum-of-squares accumulation; bf16 in/out with
      round-to-nearest-even on f32→bf16 cast. **5119/5120 bit-exact**
      vs PyTorch `F.rms_norm` on [8, 640] random input with
      `model.norm.weight`. The single 1-ULP mismatch comes from
      accumulation order (PyTorch uses pairwise sum internally). If
      Phase E parity test fails on norms, swap to Kahan summation.
- [x] bf16 matmul `y = x @ W.T + bias` for `[T, D] × [N, D]`.
      **264/264 bit-exact** vs PyTorch `F.linear` on classifier head
      (score.weight [33, 640] + score.bias). Same kernel reused for
      q/k/v/o projections + router. f32 accumulation, bf16 output
      with RNE rounding. 4-wide SIMD inner loop (`@Vector(4, u16)`
      bf16→f32 widen + vector multiply; scalar lane-by-lane drain
      preserves the accumulate order that matches PyTorch's reference.
      No `@mulAdd` — relaxed_madd fusion would diverge from PyTorch's
      two-rounding semantics).
- [x] Embedding lookup — pure gather. **5120/5120 bit-exact** on
      [T=8, D=640] with a 1024-row truncated embed table. 0.06 ms.
      OOB ids zero-fill rather than crash.
- [x] Int4 blockwise matmul (`matmul_bf16_x_int4block`). Block size 32
      along D, signed symmetric int4 (range −8..7, scale = max_abs/7 so
      the −8 slot goes unused), per-block fp16 scale. Layout per
      tensor: `[N, D/2]` packed u8 followed by `[N, D/32]` fp16 scales.
      Matches ONNX MatMulNBits semantics so ONNX exports could in
      principle share the format. Kernel: 4-wide SIMD over 32-element
      blocks (bf16 widen + int4 nibble unpack + vector multiply +
      lane-drain into `block_sum`), scalar accumulate across blocks
      (each block has its own scale, can't hoist). **Bit-exact** on
      the fixture (maxAbs=0, maxRel=0 vs `F.linear(x, dequant(W), b)`
      at bf16 output precision). Manifest tolerance set to 1e-3
      relative / 1e-4 absolute — tight for the bf16+int4 combination;
      anything looser would hide a kernel bug.

**Allocator bug caught during Phase C4.** Initial design hardcoded
`heap_base = 64 KiB`, assuming Zig's static data + shadow stack fit
in the first page. In practice wasm-ld places globals around the
1 MiB mark. Writing a 1.3 MB weight blob starting at 64 KiB
overwrote `heap_end` itself, producing wild garbage pointers on the
next `alloc`. Fixed by reading `__heap_base` (linker-provided
symbol), then made idempotent via lazy `ensureHeapInit()` inside
`alloc`/`reset`/`heap_mark_now`, so JS skipping `heap_init()` can no
longer produce silent corruption.

**Pre-Phase-D review pass (2026-04-23).** Committed parity harness —
`scripts/gen-kernel-fixtures.py` (PyTorch regen recipe) +
`scripts/verify-kernels.mjs` + `tests/fixtures/` (inputs, expected
outputs, weight blob). `npm run verify:kernels` returns bit-exact
baselines and blocks on regression. Zig unit tests for pure helpers
(`src/zig/math.zig`) under `npm run test:zig`. Post-build
`wasm-opt -O3` pass under `npm run build:opt`. WASM module is
~3.4 KB with the SIMD matmul.

### Phase D — attention + MoE (biggest)

- [ ] Rotary embeddings with yarn scaling (`rope_type: "yarn"` in
      the upstream config).
- [ ] GQA attention with banded/sliding window (14 query heads,
      2 KV heads, window 128).
- [ ] MoE router (top-4 of 128 experts per token) + sparse expert
      dispatch.

### Phase E — assembly + parity

- [ ] Full 8-block forward pass.
- [ ] Cross-backend conformance: WASM output matches transformers.js
      within 1e-5 relative error on the 5 upstream sample inputs.
- [ ] Prefill: one dummy forward pass inside `warmup()` to amortize
      kernel JIT + buffer allocation cost out of the user's first
      click. (Was never added to the transformers.js backend; going
      to bake it into this one from the start.)
- [ ] Wire `CreateOptions.backend: "wasm"` in `selectBackend` so the
      new path is reachable from the public API.
- [ ] Benchmark vs Stage 0 on the same Chrome / Firefox / WebKit
      configurations; update `docs/benchmarks.md`.

**OPFS note.** With Stage 3 NO-GO (2026-04-23), the weight size
Stage 1 carries equals what Stage 0 already ships: ~772 MB
(`model_q4f16.onnx`). Stage 0 runs that off Cache API today on Chrome
without issue. Safari sits at ~1 GB quota per origin — 77% full — so
if we ever see eviction in the field on Safari, OPFS is the remedy.
Default plan: keep using Cache API; revisit OPFS only if a real
eviction report comes in.

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

## Stage 3 — MoE-specific compression (deferred, NO-GO)

**Measurement gate returned NO-GO on 2026-04-23.** See
`docs/stage3-gate-measurement.md` for the raw numbers and how to
reproduce. Both compression axes are off the table without
retraining:

- **K-means expert clustering: dead.** Pairwise expert cosine
  similarity is 0.003–0.014 (median per layer), at the random-baseline
  level for high-dim vectors. Experts are essentially orthogonal.
- **Cold-expert pruning: dead.** Top-20% of experts absorb only
  43–58% of activations per layer. Zero experts above 10% share
  anywhere. Zero to two experts below 0.1% per layer. Almost every
  expert is used at meaningful rates — no heavy tail to prune.

This is consistent with the model being trained with a load-balancing
auxiliary loss, which pushes routing toward uniform utilization. We
measured a feature of the training recipe, not a hidden compression
opportunity.

The stack that *would* have landed Stage 3:

- Int4 baseline — already shipped in Stage 0 (`model_q4f16.onnx`, ~772 MB).
- Expert K-means clustering — ruled out by cosine data.
- Cold-expert pruning — ruled out by activation data.
- Hadamard rotation + int3 residual alone — yields ~1.3× (~580 MB).
  Still too large for Cloudflare Workers' 128 MB cap, and not a
  meaningful browser win over 772 MB / Cache API. Not worth the
  engineering cost as a standalone effort.

Consequence: the "edge inference in a Cloudflare Worker" product
angle is off the table until something changes. Stages 0–1 alone
define the shippable product today.

### Revisit if

- A retraining-based path comes into scope (distillation into a
  dense small model, MoE → dense collapse with LoRA fine-tune, etc).
  Currently out-of-scope per project rules.
- Upstream publishes a smaller `openai/privacy-filter` variant.
- Cloudflare raises Workers memory to fit a ~580 MB variant.

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
