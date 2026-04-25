# Roadmap

This is the execution plan for `textsift`. Stage 0 ships a working
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

**Ship gate (met).** Parity confirmed transitively: textsift ↔
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
- [x] Built via `npm run build:zig` → `dist/textsift.wasm`
      (`wasm32-freestanding`, `simd128` + `relaxed_simd`).
- [x] JS bridge `src/js/backends/wasm.ts` with `loadTextsift()`
      loader + `WasmBackend` class (`forward()` throws until
      kernels land).
- [x] End-to-end round-trip verified: echo ABI, JS→WASM memory
      staging, bump heap grow + rewind.

### Phase B — weight conversion + loading (superseded)

Originally scoped around a custom `pii-weights.bin` format (Python
converter + Zig parser + JS loader). That format was retired 2026-04-23
in favour of reading upstream's `onnx/model_q4f16.onnx` + `.onnx_data`
directly (see "ONNX pivot" below). Converter scripts, Zig parser
exports, and the full-blob generation step were removed. The committed
truncated `tests/fixtures/pii-weights.bin` remains as frozen per-kernel
fixture data only; `scripts/verify-kernels.mjs` parses it in JS.

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
      with RNE rounding. 4-wide SIMD inner loop with four independent
      `@Vector(4, f32)` accumulators unrolled 16-wide — breaks the
      serial add chain, horizontal-reduces at the end. **~17 GFLOPS**
      measured in Node on M-series across the Phase-D shapes. Up from
      2.7 GFLOPS before the multi-accumulator rewrite. No `@mulAdd` —
      relaxed_madd fusion has implementation-defined rounding which
      would add drift on top of the accumulator-order drift.
- [x] Embedding lookup — pure gather. **5120/5120 bit-exact** on
      [T=8, D=640] with a 1024-row truncated embed table. 0.06 ms.
      OOB ids zero-fill rather than crash.
- [x] Int4 blockwise matmul (`matmul_bf16_x_int4block`). Block size 32
      along D, signed symmetric int4 (range −8..7, scale = max_abs/7 so
      the −8 slot goes unused), per-block fp16 scale. Layout per
      tensor: `[N, D/2]` packed u8 followed by `[N, D/32]` fp16 scales.
      Matches ONNX MatMulNBits semantics so ONNX exports could in
      principle share the format. Kernel: wide 16-byte int4 load per
      block + sign-extended nibble unpack via `(i8 << 4) >>_s 4` +
      8 compile-time `@shuffle` masks to carve out 4-lane slices.
      TR=4 outer tiling across T amortizes int4 decode over 4 rows of
      X sharing the same W row — eight vector accumulators in the tile
      (2 per T row). **Bit-exact** on the fixture (maxAbs=0, maxRel=0
      vs `F.linear(x, dequant(W), b)` at bf16 output precision).
      Manifest tolerance set to 1e-3 relative / 1e-4 absolute — tight
      for the bf16+int4 combination; anything looser would hide a
      kernel bug. **~13 GFLOPS** measured (up from 3.3 pre-TR).

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

### Phase D — attention + MoE (done)

- [x] Rotary embeddings with yarn scaling (`rope_type: "yarn"`).
      `src/js/inference/rope.ts` computes inv_freq (NTK-by-parts ramp
      between interpolation and extrapolation) + attention_scaling
      (`0.1·log(factor)+1`) + cos/sin tables as bf16 — bit-identical
      to PyTorch's layer output. `rope_apply` in Zig does the
      interleaved rotation using PyTorch's eager 3-rounding bf16
      semantics (upcast → mul → round → upcast → combine → round),
      which is noisier than a single-rounding chain but bit-exact
      against the reference.
- [x] GQA attention with banded/sliding window (14 query heads,
      2 KV heads, window 128). `banded_attention` kernel scores only
      cells inside `[i−128, i+128]` (O(T·window), not O(T²));
      attention sinks folded into softmax before AV combine.
      `attention_forward` TS composition wires Q/K/V projections
      + RoPE + `head_dim^-0.25` scale + banded attention + O proj.
      Parity maxAbs 0.25, rms 0.01 vs real layer-0 weights.
- [x] MoE router + sparse expert dispatch. `matmul_bf16_out_f32`
      for fp32 router logits, `topk_partial_f32`, `softmax_f32`.
      Expert dispatch is expert-major: inverts routing, batches
      tokens per expert, runs int4 matmul (bf16 x → int4 W → f32 out)
      → SwiGLU-with-clamp → int4 matmul (f32 x → int4 W → f32 out),
      scatter-adds weighted into an f32 accumulator, scales by
      `num_experts_per_tok` and rounds to bf16. Parity maxRel 7e-3.
- [x] Block + model forward composition. `blockForward` stitches
      norm + attn + residual + norm + MLP + residual. `modelForward`
      runs embed + N blocks (ping-pong buffers) + final norm +
      classifier head. `WasmBackend.forward` wires this into the
      public `InferenceBackend` contract with auto-detected layer
      count and expert count. `selectBackend({ backend: "wasm",
      wasmWeightsUrl })` reaches it. 1-layer-truncated e2e parity
      vs PyTorch: rms 0.014 on final logits.

### Phase E — assembly + parity (done)

- [x] Full 8-block forward pass — `modelForward` runs embed +
      8 blocks + final norm + classifier through the int4 kernels.
- [x] Prefill: `WasmBackend.warmup()` runs one dummy T=16 forward so
      V8 JITs every hot kernel and the bump heap hits steady state.
- [x] Public API wiring — `CreateOptions.backend: "wasm"` routes
      through `selectBackend` to `WasmBackend` with the shared
      `LoadedModelBundle.modelSource`. No per-backend options.
- [x] Cross-backend conformance scaffold — `tests/browser/smoke.spec.ts`
      spins up both `PrivacyFilter` instances in a real Chromium and
      asserts span-for-span equality on a representative input.
      (Actual run with the 772 MB model is a manual `npm run test`;
      not wired into default CI yet.)
- [x] Benchmark vs Stage 0 — `tests/browser/bench.{html,spec.ts}`
      measures warm forward medians at multiple T; numbers in
      `docs/benchmarks.md`.

### Phase F — browser harness (done)

- [x] Playwright config + browser tests (`smoke.spec.ts`,
      `bench.spec.ts`, `public-api.html` + spec) running under
      Chromium with `--enable-unsafe-webgpu --use-angle=metal` so
      the transformers.js WebGPU path is real, not swiftshader.
- [x] Inline `dist/textsift.wasm` as a Uint8Array via
      `scripts/inline-wasm.mjs` — the JS bundle ships the .wasm bytes
      so there's no separate HTTP round-trip and no URL-resolution
      quirk when the library is re-bundled.
- [x] TR=4 outer tiling in bf16 matmul (~23 GFLOPS on T≥16 shapes,
      up from 17 GFLOPS pre-tiling). Same change lifted the int4
      matmul from 13 → ~20 GFLOPS.

### Phase G — ONNX pivot (done 2026-04-23)

Replace the custom `pii-weights.bin` path with direct reading of
`openai/privacy-filter`'s upstream `onnx/model_q4f16.onnx` +
`.onnx_data`. Both backends now share exactly one HTTP download
(~772 MB) and one cache entry.

- [x] Minimal ONNX protobuf decoder in TS
      (`src/js/model/onnx-reader.ts`, no external deps). Parses the
      initializer list; handles inline `raw_data` / `float_data` and
      external `(location, offset, length)` pointers into the
      `.onnx_data` sidecar.
- [x] Int4 matmul kernels extended to asymmetric uint4 with per-block
      uint4 zero-point buffer — matches ONNX MatMulNBits exactly.
      `w_zp_ptr = 0` preserves the old symmetric-decode path.
- [x] New `embed_lookup_int4` Zig kernel for ONNX
      GatherBlockQuantized semantics (uint4 packed embed table +
      fp16 scales + uint4 zp, block size 32 along D).
- [x] `loadOnnxWeights()` in `src/js/backends/wasm.ts` fetches both
      ONNX files, parses the graph, and pins each tensor into WASM
      memory in the dtype/layout the kernels want. Preprocessing at
      load time: f16→bf16 for biases/norms, f32→fp16 for router
      scales, f32→bf16 for router bias, and XOR-0x88 +
      synthesised-0x88 zp buffer for QMoE signed-int4 experts
      (reinterpret as unsigned+zp=8, bit-identical dequant).
- [x] Attention Q/K/V/O, MoE router, and classifier head all moved
      off bf16 matmul onto int4 matmul. Embed table moved onto
      `embed_lookup_int4`. No int4-dequant-at-load inflation — the
      772 MB ONNX stays 772 MB in WASM memory.
- [x] `CreateOptions.wasmWeightsUrl` / `wasmWeightsSha256` dropped
      from the public API; backend derives the URL from
      `bundle.modelSource`.
- [x] Cleanup — `scripts/convert_weights.py`,
      `scripts/gen-kernel-fixtures.py`,
      `scripts/gen-full-parity-fixture.py`, and the
      blob-dependent verify/bench scripts removed. Zig blob-parser
      exports (`weights_load` et al.) + `readU32LE`/`readU64LE`
      helpers removed. Composition fixture specs pruned from
      `tests/fixtures/manifest.json` (per-kernel fixtures stay).
      `WeightDType.I4_BLOCK32_SYM` removed. dist/textsift.wasm: 25 KB.

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
