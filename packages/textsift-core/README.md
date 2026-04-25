# textsift-core

PII detection + redaction for browser, Node, and edge runtimes. 76 KB gzipped, no runtime dependencies.

The lean half of the [textsift](https://www.npmjs.com/package/textsift) split. Custom WebGPU + WASM backends, native o200k-style BPE tokenizer.

```sh
npm install textsift-core
```

```ts
import { PrivacyFilter } from "textsift-core";

const filter = await PrivacyFilter.create();
const result = await filter.redact(
  "Hi John Smith, your email john@example.com is on file.",
);
// result.redactedText
//   "Hi [private_person], your email [private_email] is on file."
```

`PrivacyFilter.create()` picks WebGPU when an adapter with `shader-f16` is available, otherwise the custom Zig+SIMD WASM backend. Force a path with `{ backend: "webgpu" | "wasm" }`.

## When `textsift-core` vs `textsift`

`textsift-core` ships only the custom backends. If you want a transformers.js fallback for runtimes without WebGPU and without SIMD-capable WASM, install [`textsift`](https://www.npmjs.com/package/textsift) instead — same API, ~226 KB gzipped.

| Scenario | Pick |
|---|---|
| Browser app on modern Chromium / Safari 18+ | `textsift-core` |
| Server-side Node 20+ | `textsift-core` |
| Edge worker, Cloudflare Workers, Vercel Edge | `textsift-core` |
| AI proxy that needs to ship everywhere | `textsift-core` |
| Browser that may not have WebGPU OR cross-origin-isolation for SIMD-WASM | `textsift` |
| You're already using transformers.js elsewhere and want a drop-in replacement | `textsift` |

## Public API

```ts
PrivacyFilter.create(opts?): Promise<PrivacyFilter>
filter.redact(text, opts?): Promise<RedactResult>
filter.detect(text, opts?): Promise<DetectResult>
filter.redactBatch(inputs, opts?): Promise<RedactResult[]>
filter.startStream(opts?): Promise<DetectStreamSession>  // incremental, O(N)
filter.dispose(): void

// Standalone tokenizer (token counting, custom chunking, etc.)
Tokenizer.fromBundle(bundle): Promise<Tokenizer>
tokenizer.encode(text): EncodeResult

// Storage
getCachedModelInfo(): Promise<CachedModelInfo>
clearCachedModel(): Promise<{ removed, bytes }>

// Direct backend access
WasmBackend, WebGpuBackend, ModelLoader
```

Full API reference: [teamchong.github.io/textsift/api/](https://teamchong.github.io/textsift/api/).

## Caveats

`openai/privacy-filter` is a detection aid, not an anonymization guarantee. Read the [caveats page](https://teamchong.github.io/textsift/caveats/) before treating output as compliance-safe.

## License

Apache 2.0.
