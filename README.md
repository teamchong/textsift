# textsift

**PII detection and redaction for the browser and Node, powered by `openai/privacy-filter`.**

Runs entirely on the user's device — WebGPU on capable browsers, WebAssembly + SIMD128 everywhere else. No backend, no network calls to inference services, no text ever leaves the device.

> The only npm package that runs `openai/privacy-filter` client-side with both GPU and CPU paths.

[**Docs**](https://teamchong.github.io/textsift/) · [**Quickstart**](https://teamchong.github.io/textsift/quickstart/) · [**API**](https://teamchong.github.io/textsift/api/) · [**Benchmarks**](https://teamchong.github.io/textsift/benchmarks/)

## Install

```sh
npm install textsift
```

## Use

```ts
import { PrivacyFilter } from "textsift";

const filter = await PrivacyFilter.create();

const result = await filter.redact(
  "Hi, my name is John Smith and my email is john@example.com.",
);

// result.redactedText
//   "Hi, my name is [private_person] and my email is [private_email]."

// result.spans
//   [ { label: "private_person", start: 15, end: 25, ... },
//     { label: "private_email",  start: 43, end: 59, ... } ]
```

Detection without applying redactions:

```ts
const { spans, containsPii } = await filter.detect(text);
```

Batch inputs, custom markers, per-category enabling — see the [API reference](https://teamchong.github.io/textsift/api/).

## Why

OpenAI released [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) on 2026-04-20 — a 1.5B-parameter MoE (50M active) bidirectional token classifier for PII detection. Apache 2.0. State-of-the-art on PII-Masking-300k (96% F1).

The official SDK is Python. `transformers.js` runs it on WebGPU but fails on CPU — ORT-Web's WASM bundle is missing `GatherBlockQuantized` and `MatMulNBits` kernels. **textsift is the only package that runs this model end-to-end in the browser on both GPU and CPU**, with a single public API.

## Performance

M-series MacBook / Chromium 147, median of 5 runs after 2 warmups. Full table in [/benchmarks](https://teamchong.github.io/textsift/benchmarks/):

| | tjs (WebGPU) | **textsift (WebGPU)** | speedup |
|---|---:|---:|---:|
| Short input | 29 ms | **7 ms** | **4.1×** |
| Medium input | 39 ms | **12 ms** | **3.2×** |
| Long input | 57 ms | **25 ms** | **2.3×** |
| Second-visit warmup | 11.2 s | **1.1 s** | **10×** |

Cold-start parity on first visit (both download ~770 MB from HF CDN); repeat visits hit our OPFS cache.

## Backends

Three interchangeable engines, same public API:

| Backend | Path | When |
|---|---|---|
| `auto` *(default)* | transformers.js, WebGPU via ORT-Web | Drop-in, works everywhere tjs works |
| `webgpu` | Hand-tuned WGSL compute shaders | **Fastest.** Modern Chromium / Safari / Firefox with WebGPU + `shader-f16` |
| `wasm` | Zig + SIMD128, compiled to WebAssembly | Universal fallback. Only working CPU path for this model |

```ts
const gpu = await PrivacyFilter.create({ backend: "webgpu" });   // fastest
const wasm = await PrivacyFilter.create({ backend: "wasm" });    // universal
const auto = await PrivacyFilter.create({ backend: "auto" });    // tjs baseline
```

All three produce byte-identical spans on the same input. See [/backends](https://teamchong.github.io/textsift/backends/).

## Architecture

```
 ┌─────────────────────────────────────┐
 │ PrivacyFilter (public API)          │
 │ create / redact / detect / dispose  │
 └────────────┬────────────────────────┘
              │
              ▼
 ┌─────────────────────────────────────┐
 │ Tokenizer → Chunking → Backend      │
 │ → Viterbi CRF → BIOES merge         │
 │ → Redaction applicator              │
 └────────────┬────────────────────────┘
              │
              ▼
   ┌──────┬──────┬──────┐
   │ tjs  │ wgpu │ wasm │  backends (interchangeable)
   └──────┴──────┴──────┘
              │
              ▼
      ONNX weights (770 MB)
      ↑ cached in OPFS ↑
```

Full architecture details in [/architecture](https://teamchong.github.io/textsift/architecture/).

## Caveats

`openai/privacy-filter` is a detection aid, **not an anonymization guarantee**. No dedicated SSN / passport label. English-first (Japanese ~88% F1, other languages untested). Short text under-contextualizes.

See the [caveats page](https://teamchong.github.io/textsift/caveats/) and OpenAI's [model card](https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf) before treating redacted output as compliance-safe.

## Development

```sh
npm install
npm run build           # Zig → WASM, JS bundle, .d.ts
npm run typecheck
npm run test            # playwright browser tests
```

Source layout:

- `src/js/` — public API, backends, inference composition (Viterbi, chunking, redaction).
- `src/zig/` — Zig kernels (int4 matmul, banded attention, RoPE, QMoE dispatch, …) compiled to WASM.
- `src/js/backends/webgpu.ts` — WGSL shaders for the Stage-2 GPU backend.
- `docs/` — engineering notes (roadmap, benchmarks, measurement artefacts).
- `docs-site/` — Astro + Starlight user docs, deployed to GitHub Pages.
- `tests/browser/` — Playwright parity + benchmark tests.

## License

Apache 2.0, matching the upstream model.
