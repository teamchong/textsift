# textsift

**PII detection + redaction with auto backend selection. ~226 KB gzipped.**

Wraps [`textsift-core`](https://www.npmjs.com/package/textsift-core) and adds a transformers.js-based fallback so `PrivacyFilter.create()` works on any browser or Node runtime out of the box.

```sh
npm install textsift
```

```ts
import { PrivacyFilter } from "textsift";

const filter = await PrivacyFilter.create();   // picks the fastest viable backend
const result = await filter.redact(
  "Hi John Smith, your email john@example.com is on file.",
);
```

## When to pick `textsift` vs `textsift-core`

| Scenario | Pick |
|---|---|
| You don't know what backend your users have | **`textsift`** |
| You want to support Firefox without `dom.webgpu.enabled` | **`textsift`** |
| You want a drop-in replacement for transformers.js | **`textsift`** |
| Browser app on modern Chromium / Safari 18+ | [`textsift-core`](https://www.npmjs.com/package/textsift-core) (3× smaller) |
| Server-side Node 20+ | [`textsift-core`](https://www.npmjs.com/package/textsift-core) |

## How auto-fallback works

`PrivacyFilter.create()` with no `backend` option picks:

1. **WebGPU** (custom WGSL kernels) — if a WebGPU adapter with `shader-f16` is available
2. **WASM** (custom Zig + SIMD128, multi-thread when COOP/COEP set) — otherwise
3. **transformers.js** (this package's contribution) — final fallback for browsers without WebGPU

Force a specific backend with `{ backend: "webgpu" | "wasm" | "auto" }`.

## Public API

Identical to [`textsift-core`](https://www.npmjs.com/package/textsift-core) — every export is re-exported here, plus `TransformersJsBackend` for callers who want to instantiate it directly.

## License

Apache 2.0, matching the upstream model.
