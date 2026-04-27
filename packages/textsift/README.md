# textsift

> **Personal learning project.** Treat as such — no SLA, no roadmap commitment. See the [main README](https://github.com/teamchong/textsift) for context.

PII detection + redaction running [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) on the user's device. Per-platform GPU backends (Metal on macOS, Vulkan on Linux, Dawn on Windows, WGSL in browsers); Zig + SIMD128 WASM as the no-GPU fallback. The native BPE tokenizer is pure TS, so apps that don't already ship `@huggingface/transformers` save a multi-MB dep.

```sh
npm install textsift
```

Two entry points so browsers never bundle native code:

```ts
// Browser / Node-via-WASM
import { PrivacyFilter } from "textsift/browser";

// Node — auto-picks the platform-native fast path (Metal on macOS,
// Vulkan on Linux, Dawn on Windows) and falls back to WASM if no GPU.
import { PrivacyFilter } from "textsift";
```

```ts
const filter = await PrivacyFilter.create();
const { redactedText } = await filter.redact(
  "Hi John Smith, your email john@example.com is on file.",
);
```

## Why this matters on Linux

If you're on Linux and want GPU-accelerated PII filtering in Node, the realistic options today are bad:

| Path | Setup | T=32 latency on a typical iGPU box |
|---|---|---:|
| ONNX Runtime Node CPU | `npm i onnxruntime-node`, write your own inference loop | ~600–800 ms |
| transformers.js (Node) | `npm i @xenova/transformers`, no GPU on Node so WASM | ~80–100 ms |
| PyTorch CPU | `pip install torch transformers safetensors`, write inference | ~150–500 ms |
| PyTorch CUDA | NVIDIA GPU + driver + cuda-toolkit + matched torch wheel | n/a (no NVIDIA on most laptops) |
| **textsift native** | **`npm install textsift`** | **~28 ms** |

On the same Linux box (Intel Iris Xe, Mesa Vulkan), textsift native is **22–28× faster** than ORT Node CPU because it talks Vulkan directly with hand-written GLSL→SPIR-V kernels — no CUDA, no driver dance, no model conversion.

End-to-end on a 122-character input with 4 PII spans: `redact()` returns in ~50–75 ms.

## Per-platform fast paths

| Platform | Backend | What it uses |
|---|---|---|
| macOS arm64/x64 | Metal-direct | Hand-written MSL kernels via Obj-C bridge |
| Linux x86_64/arm64 | Vulkan-direct | Hand-written GLSL → SPIR-V via glslangValidator |
| Windows x86_64 | Dawn-direct | Tint → D3D12 via statically-linked Google Dawn |
| (any platform, no GPU) | WASM fallback | Zig + SIMD128 in WebAssembly |

Each platform's `.node` binary is built with comptime-gated Zig code so it only contains the relevant backend — Mac binaries don't ship Vulkan code, Windows binaries don't ship Obj-C, etc. npm picks the right `optionalDependencies` subpackage at install time (`textsift-{linux-x64,linux-arm64,darwin-x64,darwin-arm64,windows-x64}`).

## Linux prereqs (one-time)

For the GPU fast path on Linux, you need a Vulkan loader. Most distros ship one in their default packages:

```sh
# Ubuntu/Debian
sudo apt install -y libvulkan1 mesa-vulkan-drivers

# Fedora/RHEL
sudo dnf install -y vulkan-loader mesa-vulkan-drivers

# Arch
sudo pacman -S vulkan-icd-loader vulkan-mesa-layers
```

If Vulkan isn't available, `import { PrivacyFilter } from "textsift"` automatically falls back to the WASM CPU path — same API, slower runtime (still faster than ORT Node CPU thanks to Zig SIMD128 kernels).

## Why two entry points

Bundlers (Vite/Webpack/esbuild/etc.) resolve `textsift/browser` and pull in only the WASM/WebGPU code path. The native NAPI binding lives at the bare `textsift` import, so a Node CLI / server can use it without forcing browser code into anything else's bundle.

## Public API

See the [API reference](https://teamchong.github.io/textsift/api/). Highlights:

- `PrivacyFilter.create({ backend, modelSource, markers, enabledCategories, rules, presets, minConfidence, cacheDir, modelPath, offline })`
- `filter.detect(text | AsyncIterable<string>)` — batch returns a Promise; streaming returns a sync handle with `spanStream` + `result`
- `filter.redact(text | AsyncIterable<string>)` — same shape; streaming surfaces `textStream` of safe-to-emit pieces
- `filter.classifyColumns(rows, { headerRow, sampleSize })` — per-column PII classification for tabular data
- `filter.redactTable(rows, { mode })` — `"redact"` / `"synth"` / `"drop_column"` for one-shot CSV cleaning
- `presets: ["secrets"]` enables JWT, GitHub PAT, AWS, Slack, OpenAI/Anthropic/Google/Stripe keys, and PEM private-key headers (all severity `"block"`)
- `markerPresets.faker()` — realistic-looking fake values instead of `[label]` markers (consistent across mentions)
- Custom `rules` (regex or function) merge with model spans
- SARIF v2.1.0 export at `textsift/sarif` for GitHub Code Scanning / similar consumers

## Other surfaces

Same engine, four surfaces total:
- `npx textsift` — [CLI](https://teamchong.github.io/textsift/cli/) (`redact`, `detect`, `table`, `classify`, `download`, `cache`).
- [Pre-commit hook](https://teamchong.github.io/textsift/precommit/) — block commits containing PII.
- [GitHub Action](https://teamchong.github.io/textsift/github-action/) — `uses: teamchong/textsift@v1` with PR annotations + Security-tab integration via SARIF.

## License

Apache 2.0.
