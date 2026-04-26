# textsift

> **Personal learning project.** Treat as such — no SLA, no roadmap commitment. See the [main README](https://github.com/teamchong/textsift) for context.

PII detection + redaction running [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) on the user's device. Custom WGSL kernels for WebGPU and Zig+SIMD128 WASM kernels for everywhere else. **No transformers.js dependency.**

```sh
npm install textsift
```

Two entry points so browsers never bundle native code:

```ts
// Browser / Node-via-WASM (today)
import { PrivacyFilter } from "textsift/browser";

// Node native NAPI binding (issue #79 — PrivacyFilter throws today;
// kernels done, Metal-direct on macOS is ~1.9× faster than browser
// at T=32).
import { PrivacyFilter } from "textsift";
```

```ts
const filter = await PrivacyFilter.create();
const { redactedText } = await filter.redact(
  "Hi John Smith, your email john@example.com is on file.",
);
```

## Why two entry points

Bundlers (Vite/Webpack/esbuild/etc.) resolve `textsift/browser` and pull in only the WASM/WebGPU code path. The native NAPI binding lives at the bare `textsift` import, so a Node CLI / server can use it without forcing browser code into anything else's bundle.

## Public API

See the [API reference](https://teamchong.github.io/textsift/api/). Highlights:

- `PrivacyFilter.create({ backend, modelSource, markers, enabledCategories, rules, presets })`
- `filter.detect(text | AsyncIterable<string>)` — batch returns a Promise; streaming returns a sync handle with `spanStream` + `result`
- `filter.redact(text | AsyncIterable<string>)` — same shape; streaming surfaces `textStream` of safe-to-emit pieces
- `presets: ["secrets"]` enables JWT, GitHub PAT, AWS, Slack, OpenAI/Anthropic/Google/Stripe keys, and PEM private-key headers (all severity `"block"`)
- Custom `rules` (regex or function) merge with model spans

## License

Apache 2.0.
