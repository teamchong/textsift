# textsift-core

**Lean PII detection + redaction. ~76 KB gzipped. No transformers.js dependency.**

The minimal client-side engine for [openai/privacy-filter](https://huggingface.co/openai/privacy-filter). Custom WebGPU + WASM backends, native o200k-style BPE tokenizer, zero runtime dependencies.

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

`PrivacyFilter.create()` picks WebGPU when an adapter with `shader-f16` is available, otherwise falls back to the custom Zig+SIMD WASM backend. Force a path with `{ backend: "webgpu" | "wasm" }`.

## When to pick `textsift-core` vs `textsift`

| Scenario | Pick |
|---|---|
| Browser app with modern Chromium / Safari 18+ users | **`textsift-core`** |
| Server-side Node 20+ with optional WebGPU | **`textsift-core`** |
| Edge worker / Cloudflare Workers / Vercel Edge | **`textsift-core`** |
| AI proxy that runs everywhere and needs minimum bundle | **`textsift-core`** |
| Browser app that needs to support Firefox without `dom.webgpu.enabled` | [`textsift`](https://www.npmjs.com/package/textsift) |
| Drop-in replacement for an existing transformers.js setup | [`textsift`](https://www.npmjs.com/package/textsift) |

## Public API

```ts
// Main API
PrivacyFilter.create(opts?: CreateOptions): Promise<PrivacyFilter>
filter.redact(text, opts?): Promise<RedactResult>
filter.detect(text, opts?): Promise<DetectResult>
filter.redactBatch(inputs, opts?): Promise<RedactResult[]>
filter.dispose(): void

// Tokenizer (also useful standalone)
Tokenizer.fromBundle(bundle): Promise<Tokenizer>
tokenizer.encode(text): EncodeResult

// Storage
getCachedModelInfo(): Promise<CachedModelInfo>
clearCachedModel(): Promise<{ removed, bytes }>

// Advanced
WasmBackend, WebGpuBackend, ModelLoader
```

Full docs at [teamchong.github.io/textsift](https://teamchong.github.io/textsift/).

## What's NOT in textsift-core

The transformers.js auto-fallback backend. If you need that — typically because you're targeting browsers without WebGPU and without cross-origin isolation for SIMD-capable WASM — install [`textsift`](https://www.npmjs.com/package/textsift) instead. Same API, ~226 KB gzipped.

## License

Apache 2.0, matching the upstream model.
