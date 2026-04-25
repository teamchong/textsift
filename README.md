# textsift

> **Personal learning project.** I built this to teach myself WebGPU compute shaders, Zig→WASM with SIMD intrinsics, and the o200k-style BPE tokenizer pipeline. The code works and the tests pass, but treat it as such — there's no SLA, no roadmap commitment, no team behind it. PRs and bug reports welcome; "production support" is not.

PII detection and redaction that runs [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) on the user's device. WebGPU when available; Zig + SIMD128 WASM otherwise. Apache 2.0.

[**Docs**](https://teamchong.github.io/textsift/) · [**Quickstart**](https://teamchong.github.io/textsift/quickstart/) · [**Playground**](https://teamchong.github.io/textsift/playground/) · [**API**](https://teamchong.github.io/textsift/api/)

## What this is

A npm package. Two flavours, same `PrivacyFilter` API:

```sh
npm install textsift-core   # 76 KB gzipped, no runtime deps
npm install textsift        # 226 KB gzipped, adds a transformers.js fallback backend
```

The model is OpenAI's; the value here is packaging:

- A native o200k-style BPE tokenizer in pure TypeScript (no `@huggingface/transformers` dep on `textsift-core`).
- Two custom backends — WGSL for WebGPU and Zig-compiled SIMD WASM — that produce byte-identical span output to transformers.js's WebGPU path.
- A WASM path that loads `model_q4f16.onnx` at all. transformers.js's WASM EP doesn't, because ORT-Web's WASM bundle has no `MatMulNBits` / `GatherBlockQuantized` kernel.
- Persistent OPFS caching of the 770 MB model weights, configured by default.
- A streaming overload of `detect()` for AI-proxy / LLM-output-filtering use cases — pass an `AsyncIterable<string>` instead of a `string`. O(N) over a stream of N tokens, vs O(N²) for the naive "call detect after every chunk" pattern.

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

Detect-only:

```ts
const { spans, containsPii } = await filter.detect(text);
```

Streaming detect (proxy use case) — same `detect()`, just pass an async source:

```ts
async function* llmStream() {
  for await (const chunk of openai.chat.completions.create({ stream: true, ... })) {
    yield chunk.choices[0]?.delta?.content ?? "";
  }
}

const handle = filter.detect(llmStream());

// Iterate spans as they become detectable...
for await (const span of handle.spanStream) {
  if (span.label === "secret" && span.confidence > 0.9) abort();
}

// ...and/or await the final result.
const result = await handle.result;
```

Batch inputs, custom markers, per-category enabling — see the [API reference](https://teamchong.github.io/textsift/api/).

## Measured numbers (M3 Pro, Chromium 147)

Forward-pass latency, median of 5 runs:

| Input length | textsift (WebGPU) | textsift (WASM ST) | tjs (WebGPU default) |
|---|---:|---:|---:|
| ~12 tokens | 7 ms | 66 ms | 29 ms |
| ~25 tokens | 12 ms | 158 ms | 39 ms |
| ~80 tokens | 25 ms | 342 ms | 57 ms |

Cold start (defaults — both packages can be configured otherwise):

| | First visit | Second visit |
|---|---:|---:|
| tjs (Cache API default) | 13.3 s | 11.2 s |
| textsift (OPFS default) | 14.1 s | 1.1 s |

The second-visit gap is a default-configuration comparison, not an architectural ceiling. transformers.js with a custom OPFS cache plugged in can match textsift on second-visit warmup. The factual claim is "textsift caches large models persistently with no configuration", not "tjs can't cache".

These will look different on your hardware. They're a snapshot.

## Repo layout (npm workspaces monorepo)

```
packages/
  textsift-core/     ← lean engine: tokenizer + WebGPU + WASM backends
    src/js/          ← public API, viterbi, chunking, redaction, native BPE tokenizer
    src/zig/         ← Zig kernels → WASM
    src/c/           ← FMA shim for relaxed_simd
  textsift/          ← umbrella: depends on textsift-core + @huggingface/transformers
    src/             ← thin wrapper, transformers.js fallback backend
docs-site/           ← Astro + Starlight docs site
tests/browser/       ← Playwright tests, including tokenizer-conformance + stream
```

## Development

```sh
npm install                # workspace bootstrap
npm run build              # builds both packages (zig → wasm, bundle, .d.ts)
npm run typecheck          # strict, noUncheckedIndexedAccess on
npm run test               # all playwright tests
```

The tokenizer-conformance test (`tests/browser/tokenizer-conformance.spec.ts`) verifies the native BPE tokenizer produces token-id sequences identical to AutoTokenizer across 46 cases (English, Unicode, code, edge whitespace, special tokens). The stream test (`tests/browser/stream.spec.ts`) verifies streaming detection yields the same spans as a single batch `detect()` call.

## Caveats

`openai/privacy-filter` is a detection aid, not an anonymization guarantee. English-first (Japanese ~88% F1, other languages untested). Short text under-contextualizes.

Read the [caveats page](https://teamchong.github.io/textsift/caveats/) and OpenAI's [model card](https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf) before treating output as compliance-safe.

## License

Apache 2.0, matching the upstream model.
