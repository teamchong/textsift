# textsift

> **Personal learning project.** Treat as such — no SLA, no roadmap commitment. See the [main README](https://github.com/teamchong/textsift) for context.

Wraps [`textsift-core`](https://www.npmjs.com/package/textsift-core) and adds a transformers.js fallback backend for runtimes that can't run WebGPU or SIMD-capable WASM. 226 KB gzipped (vs 76 KB for the lean `textsift-core`).

```sh
npm install textsift
```

```ts
import { PrivacyFilter } from "textsift";

const filter = await PrivacyFilter.create();
const result = await filter.redact(
  "Hi John Smith, your email john@example.com is on file.",
);
```

Same `PrivacyFilter` API as `textsift-core` — every export from core is re-exported here, plus `TransformersJsBackend` for callers who want to instantiate the fallback directly.

## When `textsift` vs `textsift-core`

| Scenario | Pick |
|---|---|
| Browser that may not have WebGPU or COOP/COEP cross-origin-isolation | `textsift` |
| Already using transformers.js elsewhere, want a drop-in replacement | `textsift` |
| Browser app on modern Chromium / Safari 18+ where WebGPU is available | [`textsift-core`](https://www.npmjs.com/package/textsift-core) (3× smaller) |
| Node 20+ server-side | [`textsift-core`](https://www.npmjs.com/package/textsift-core) |

## How auto-fallback works

`PrivacyFilter.create()` with no `backend` option picks:

1. WebGPU (custom WGSL kernels) if a WebGPU adapter with `shader-f16` is available, OR
2. WASM (custom Zig + SIMD128) if SharedArrayBuffer is available (cross-origin isolation), OR
3. transformers.js — this package's contribution — as the final fallback.

Force a specific path with `{ backend: "webgpu" | "wasm" }`.

## Public API

Identical to `textsift-core` plus:

```ts
import { TransformersJsBackend } from "textsift";
// for callers who want to instantiate the fallback directly
```

## License

Apache 2.0.
