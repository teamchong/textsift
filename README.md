# textsift

> **Personal learning project.** I built this to teach myself WebGPU compute shaders, Zig→WASM with SIMD intrinsics, and the o200k-style BPE tokenizer pipeline. The code works and the tests pass, but treat it as such — there's no SLA, no roadmap commitment, no team behind it. PRs and bug reports welcome; "production support" is not.

PII detection and redaction that runs [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) on the user's device. Per-platform GPU fast paths (Metal on macOS, Vulkan on Linux, Dawn on Windows, WebGPU in browsers); Zig + SIMD128 WASM as the no-GPU fallback. Apache 2.0.

[**Docs**](https://teamchong.github.io/textsift/) · [**Quickstart**](https://teamchong.github.io/textsift/quickstart/) · [**Playground**](https://teamchong.github.io/textsift/playground/) · [**API**](https://teamchong.github.io/textsift/api/) · [**Intro video**](https://teamchong.github.io/textsift/intro.mp4)

> Architecture walkthrough — [▶ play](https://teamchong.github.io/textsift/intro.mp4)

## What this is

One npm package, two entry points + a CLI:

```sh
npm install textsift
```

```ts
// Browser / Node-via-WASM — pure WebGPU + WASM, no native binary.
import { PrivacyFilter } from "textsift/browser";

// Node native — auto-picks the platform's GPU fast path (Metal on macOS,
// Vulkan on Linux, Dawn on Windows). Falls back to WASM if no GPU.
import { PrivacyFilter } from "textsift";
```

```sh
# Same engine as a CLI — no install, no browser, no clipboard dance
echo "Hi Alice, alice@example.com" | npx textsift redact
npx textsift table customers.csv --header --mode synth > clean.csv
npx textsift detect log.txt --jsonl | jq 'select(.label == "private_email")'
TEXTSIFT_OFFLINE=1 npx textsift redact file.txt   # CI: fail if not pre-cached
npx textsift download                              # pre-warm in CI
npx textsift cache info                            # show cache location + size
```

```yaml
# Or as a pre-commit hook — block commits that contain PII
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/teamchong/textsift
    rev: v0.1.0
    hooks:
      - id: textsift-pii-scan
```

```yaml
# Or as a GitHub Action — block PRs that introduce PII; findings
# show up inline + in the repo's Security tab via SARIF.
# .github/workflows/pii.yml
- uses: teamchong/textsift@v0.1.0
  with:
    sarif-output: textsift.sarif
- uses: github/codeql-action/upload-sarif@v3
  with: { sarif_file: textsift.sarif, category: textsift }
```

Bundlers (Vite/Webpack/esbuild/etc.) resolve `textsift/browser` and never touch the native entry. Node code resolves `textsift` and gets the platform-native binding via `optionalDependencies`.

The model is OpenAI's; the value here is packaging:

- A native o200k-style BPE tokenizer in pure TypeScript. If you're not already shipping `@huggingface/transformers` for other models, that's a real bundle-size win.
- Per-platform native GPU backends — hand-written MSL on macOS, hand-written GLSL→SPIR-V on Linux, Tint→D3D12 on Windows — plus WGSL for browser WebGPU. All produce byte-identical span output.
- A WASM CPU path (Zig + SIMD128) that loads `model_q4f16.onnx` directly. The transformers.js / ORT-Web stack can't load this model on CPU because ORT-Web's WASM bundle lacks `MatMulNBits` / `GatherBlockQuantized` — different runtimes (onnxruntime-node, web-llm, etc.) can in principle, but no JS ecosystem alternative ships out-of-the-box.
- Persistent OPFS caching of the 770 MB model weights in browsers (filesystem cache in Node), configured by default.
- Streaming overloads of `detect()` and `redact()` — pass an `AsyncIterable<string>` to abort an LLM stream the moment a credit card / API key appears, render redacted text progressively as it arrives, or front a model gateway (Cloudflare Worker style) that has to forward chunk-by-chunk.
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

Streaming detect / redact — abort an LLM stream when PII appears, render progressively, or proxy chunk-by-chunk. Same `detect()` / `redact()`, just pass an async source:

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

Faker mode — emit realistic fakes instead of `[private_email]` markers (so downstream validators / templates / pipelines still see PII-shaped data):

```ts
import { PrivacyFilter, markerPresets } from "textsift";

const filter = await PrivacyFilter.create({ markers: markerPresets.faker() });
await filter.redact("Hi Alice, email alice@example.com, phone +1-555-0123");
// → "Hi Alice Anderson, email alice.anderson@example.com, phone +1-555-0100"
//   Same input text → same fake within the filter's lifetime
//   (so "Alice" appearing twice yields "Alice Anderson" both times)
```

Tabular data — classify which CSV / DB columns contain PII, or redact a whole table in one call:

```ts
const rows = [
  ["id", "name",         "email",             "amount"],
  ["1",  "Alice Carter", "alice@example.com", "100"],
  ["2",  "Bob Davis",    "bob@example.com",   "250"],
];

// Audit: which columns have PII?
const cols = await filter.classifyColumns(rows, { headerRow: true });
// → [{ index:0, label:null }, { index:1, label:"private_person", confidence:1 },
//    { index:2, label:"private_email", confidence:1 }, { index:3, label:null }]

// Pipeline: redact in one of three modes
const safe = await filter.redactTable(rows, {
  headerRow: true,
  mode: "synth",   // "redact" | "synth" | "drop_column"
});
// mode "synth" gives you Tonic.ai-style fake-but-realistic output;
// "drop_column" omits PII columns entirely; "redact" uses [label] markers.
```

Batch inputs, custom markers, per-category enabling — see the [API reference](https://teamchong.github.io/textsift/api/).

## Measured numbers

Per-forward latency, median of 5–10 runs, synthetic-weight bench at production model dimensions.

**Browser (M3 Pro, Chromium 147):**

| Input length | textsift (WebGPU) | textsift (WASM MT) | tjs (WebGPU) |
|---|---:|---:|---:|
| ~7 tokens | **8.9 ms** | 29.0 ms | 32.7 ms |
| ~25 tokens | **11.8 ms** | 44.6 ms | 38.5 ms |
| ~80 tokens | **22.0 ms** | 95.9 ms | 56.4 ms |

textsift WebGPU is 2.6–3.7× faster than transformers.js across every input length.

**Node native — macOS (M2 Pro, Metal-direct):**

| T   | textsift native | tjs CPU equivalent |
|----:|----------------:|-------------------:|
|  7  | **5.2 ms**      | ~30 ms             |
| 32  | **10.8 ms**     | ~40 ms             |
| 80  | **23.8 ms**     | ~95 ms             |

Hand-written MSL beats Tint's WGSL→MSL codegen by ~1.9× on the same hardware.

**Node native — Linux (Intel Iris Xe, Vulkan-direct):**

| T   | textsift native | ONNX Runtime Node CPU |
|----:|----------------:|----------------------:|
| 32  | **28 ms**       | ~800 ms (**28×** slower) |

The Linux story is the real differentiator: GPU-accelerated PII detection on Intel iGPU / AMD APU / non-NVIDIA hardware **without CUDA, without ROCm, without driver dance**. `npm install textsift` ships a vendored Vulkan-direct binary that talks to whatever Mesa-supported GPU is there.

**Cold start:** we don't claim a speedup over transformers.js. See [benchmarks](https://teamchong.github.io/textsift/benchmarks/) for the rationale; the OPFS-vs-Cache-API gap is a storage choice, not an inference-engine one.

These numbers will look different on your hardware.

## Repo layout (npm workspaces monorepo)

```
packages/
  textsift/
    src/
      browser/         ← public API, viterbi, chunking, redaction, native BPE tokenizer
      zig/             ← Zig kernels → WASM
      c/               ← FMA shim for relaxed_simd
      native/          ← Node-native backends (Metal / Vulkan / Dawn) + NAPI bindings
        metal/         ← Mac: Obj-C bridge + hand-written MSL kernels
        vulkan/        ← Linux: C bridge + hand-written GLSL → SPIR-V kernels
        dawn/          ← Windows: Dawn C++ via Tint
        shaders/       ← canonical WGSL kernels (single source of truth)
      index.ts         ← Node native entry (auto-picks platform GPU + WASM fallback)
    scripts/           ← inline-wasm.mjs, build-native.sh, serve-coi.py, etc.
docs-site/             ← Astro + Starlight docs site
tests/browser/         ← Playwright tests
tests/native/          ← Node native conformance + bench + integration tests
.github/workflows/     ← test / release / bench across linux/darwin/windows
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
