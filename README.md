# textsift

> **Personal learning project.** I built this to teach myself WebGPU compute shaders, Zig→WASM with SIMD intrinsics, and the o200k-style BPE tokenizer pipeline. The code works and the tests pass, but treat it as such — there's no SLA, no roadmap commitment, no team behind it. PRs and bug reports welcome; "production support" is not.

PII detection and redaction that runs [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) on the user's device. WebGPU when available; Zig + SIMD128 WASM otherwise. Apache 2.0.

[**Docs**](https://teamchong.github.io/textsift/) · [**Quickstart**](https://teamchong.github.io/textsift/quickstart/) · [**Playground**](https://teamchong.github.io/textsift/playground/) · [**API**](https://teamchong.github.io/textsift/api/)

## What this is

One npm package, two entry points so browsers never bundle native code:

```sh
npm install textsift
```

```ts
// Browser / Node-via-WASM — pure WebGPU + WASM, no native binary.
import { PrivacyFilter } from "textsift/browser";

// Node native — NAPI binding (in progress, see issue #79). Throws today.
import { PrivacyFilter } from "textsift";
```

Bundlers (Vite/Webpack/esbuild/etc.) resolve `textsift/browser` and never touch the native entry. Node code that wants the fastest path resolves `textsift` and gets the native binding (once #79 lands).

The model is OpenAI's; the value here is packaging:

- A native o200k-style BPE tokenizer in pure TypeScript (no `@huggingface/transformers` runtime dependency).
- Two custom backends — WGSL for WebGPU and Zig-compiled SIMD WASM — that produce byte-identical span output.
- A WASM path that loads `model_q4f16.onnx` at all (ORT-Web's WASM EP doesn't, because its WASM bundle has no `MatMulNBits` / `GatherBlockQuantized` kernel).
- Persistent OPFS caching of the 770 MB model weights, configured by default.
- A streaming overload of `detect()` and `redact()` for AI-proxy / LLM-output-filtering use cases — pass an `AsyncIterable<string>` instead of a `string`. O(N) over a stream of N tokens, vs O(N²) for the naive "call detect after every chunk" pattern.
- Custom rule engine (regex + match-fn) that merges with model spans. Built-in `"secrets"` preset covers JWT, GitHub PAT, AWS, Slack, OpenAI/Anthropic/Google/Stripe keys, and PEM private-key headers.

## Use

```ts
import { PrivacyFilter } from "textsift/browser";

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

Streaming detect / redact (proxy use case) — same `detect()` / `redact()`, just pass an async source:

```ts
async function* llmStream() {
  for await (const chunk of openai.chat.completions.create({ stream: true, ... })) {
    yield chunk.choices[0]?.delta?.content ?? "";
  }
}

// Detect — iterate spans as they become detectable
const det = filter.detect(llmStream());
for await (const span of det.spanStream) {
  if (span.label === "secret" && span.confidence > 0.9) abort();
}
const detFinal = await det.result;

// Redact — pipe redacted text downstream as it becomes safe to emit.
const red = filter.redact(llmStream());
for await (const piece of red.textStream) {
  await downstreamWriter.write(piece);
}
const redFinal = await red.result;
```

Built-in secrets preset:

```ts
const filter = await PrivacyFilter.create({ presets: ["secrets"] });
// Detects JWT, GitHub PAT, AWS access keys, Slack tokens + webhooks,
// OpenAI/Anthropic/Google API keys, Stripe keys + webhook secrets,
// npm tokens, PEM private-key headers. All severity "block".
```

Batch inputs, custom markers, per-category enabling — see the [API reference](https://teamchong.github.io/textsift/api/).

## Measured numbers (M3 Pro, Chromium 147)

Steady-state per-forward latency, median of 5 runs:

| Input length | textsift (WebGPU) | textsift (WASM MT) | tjs (WebGPU) |
|---|---:|---:|---:|
| ~7 tokens | **8.9 ms** | 29.0 ms | 32.7 ms |
| ~25 tokens | **11.8 ms** | 44.6 ms | 38.5 ms |
| ~80 tokens | **22.0 ms** | 95.9 ms | 56.4 ms |

Sustained throughput (30-forward loop, tok/s):

| Input length | textsift (WebGPU) | textsift (WASM MT) | tjs (WebGPU) |
|---|---:|---:|---:|
| ~7 tokens | **801** | 249 | 249 |
| ~25 tokens | **2068** | 558 | 644 |
| ~80 tokens | **3644** | 840 | 1396 |

WebGPU is 2.6–3.7× faster than transformers.js on both metrics across every input size. textsift WASM is the only working CPU option for this q4f16 model — tjs has no working WASM path because ORT-Web's WASM EP lacks the int4 contrib kernels.

We don't claim a cold-start speedup. See [benchmarks](https://teamchong.github.io/textsift/benchmarks/) for the rationale; in short, the difference between OPFS (default here) and Cache API (default in tjs) is a storage choice, not an inference-engine one.

These numbers will look different on your hardware.

## Repo layout (npm workspaces monorepo)

```
packages/
  textsift/
    src/
      browser/         ← public API, viterbi, chunking, redaction, native BPE tokenizer
      zig/             ← Zig kernels → WASM
      c/               ← FMA shim for relaxed_simd
      index.ts         ← Node native entry (NAPI binding for #79)
    scripts/           ← inline-wasm.mjs, serve-coi.py, etc.
docs-site/             ← Astro + Starlight docs site
tests/browser/         ← Playwright tests
  helpers/             ← bench-only TransformersJsBackend (not shipped)
```

## Development

```sh
npm install                # workspace bootstrap
npm run build              # zig → wasm, bundle, .d.ts
npm run typecheck          # strict, noUncheckedIndexedAccess on
npm run test               # all playwright tests
```

## Caveats

`openai/privacy-filter` is a detection aid, not an anonymization guarantee. English-first (Japanese ~88% F1, other languages untested). Short text under-contextualizes.

Read the [caveats page](https://teamchong.github.io/textsift/caveats/) and OpenAI's [model card](https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf) before treating output as compliance-safe.

## License

Apache 2.0, matching the upstream model.
