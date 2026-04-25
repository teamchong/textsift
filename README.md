# textsift

**PII detection and redaction for the browser, Node, and edge runtimes — powered by `openai/privacy-filter`.**

Runs entirely on the user's device. WebGPU on capable browsers, WebAssembly + SIMD128 everywhere else. No backend, no network calls to inference services, no text ever leaves the device.

[**Docs**](https://teamchong.github.io/textsift/) · [**Quickstart**](https://teamchong.github.io/textsift/quickstart/) · [**Playground**](https://teamchong.github.io/textsift/playground/) · [**API**](https://teamchong.github.io/textsift/api/)

## Two packages

```sh
# Lean: 76 KB gzipped, native o200k-style BPE tokenizer, no transformers.js dep.
# Recommended for browser / edge / proxy apps that want the smallest bundle.
npm install textsift-core

# Full: 226 KB gzipped, adds a transformers.js fallback for runtimes
# without WebGPU and without SIMD-capable WASM. Drop-in for the
# old single-package install.
npm install textsift
```

Both expose the same `PrivacyFilter` API. Same model, same spans, same docs.

## Use

```ts
import { PrivacyFilter } from "textsift-core"; // or "textsift"

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

The official SDK is Python. `transformers.js` runs it on WebGPU but fails on CPU — ORT-Web's WASM bundle is missing `GatherBlockQuantized` and `MatMulNBits` kernels. **textsift is the only package that runs this model end-to-end client-side on both GPU and CPU**, with a single public API.

## Performance

M-series MacBook / Chromium 147, median of 5 runs after 2 warmups:

| | tjs (WebGPU) | **textsift (WebGPU)** | speedup |
|---|---:|---:|---:|
| Short input | 29 ms | **7 ms** | **4.1×** |
| Medium input | 39 ms | **12 ms** | **3.2×** |
| Long input | 57 ms | **25 ms** | **2.3×** |
| Second-visit warmup | 11.2 s | **1.1 s** | **10×** |

Cold-start parity on first visit (both download ~770 MB from HF CDN); repeat visits hit our OPFS cache.

## Backends

Three interchangeable engines behind one API. `textsift-core` ships the first two; `textsift` adds the third for compatibility with WebGPU-less and SIMD-less runtimes.

| Backend | Engine | Where |
|---|---|---|
| `webgpu` | Hand-tuned WGSL compute shaders | Modern Chromium / Safari / Firefox with WebGPU + `shader-f16` |
| `wasm` | Zig + SIMD128, multi-thread when COOP/COEP set | Universal fallback. Only working CPU path for this model |
| `auto` (transformers.js) | ORT-Web (umbrella `textsift` only) | Final fallback when neither path is viable |

```ts
const gpu = await PrivacyFilter.create({ backend: "webgpu" });   // fastest
const wasm = await PrivacyFilter.create({ backend: "wasm" });    // universal
const auto = await PrivacyFilter.create();                        // pick best
```

All paths produce byte-identical spans on the same input.

## Repo layout (monorepo)

```
packages/
  textsift-core/     ← lean engine: tokenizer + WebGPU + WASM backends
    src/js/          ← public API, viterbi, chunking, redaction, native BPE tokenizer
    src/zig/         ← Zig kernels → WASM
    src/c/           ← FMA shim for relaxed_simd
  textsift/          ← umbrella: depends on textsift-core + @huggingface/transformers
    src/             ← thin wrapper, transformers.js fallback backend
docs-site/           ← Astro + Starlight docs site (textsift.teamchong.github.io)
tests/browser/       ← Playwright parity + benchmark tests
docs/                ← engineering notes (roadmap, benchmarks)
```

## Development

```sh
npm install                # workspace bootstrap
npm run build              # builds both packages (zig → wasm, bundle, .d.ts)
npm run typecheck          # strict, noUncheckedIndexedAccess on
npm run test               # playwright browser tests
```

The tokenizer-conformance test (`tests/browser/tokenizer-conformance.spec.ts`) verifies the native BPE tokenizer produces byte-for-byte identical token-id sequences to AutoTokenizer across a 46-case corpus (English, Unicode, code, edge whitespace, special tokens).

## Caveats

`openai/privacy-filter` is a detection aid, **not an anonymization guarantee**. English-first (Japanese ~88% F1, other languages untested). Short text under-contextualizes.

See the [caveats page](https://teamchong.github.io/textsift/caveats/) and OpenAI's [model card](https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf) before treating redacted output as compliance-safe.

## License

Apache 2.0, matching the upstream model.
