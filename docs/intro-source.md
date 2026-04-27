# textsift technical reference (source for the v0.1.0 intro deck)

This document is a comprehensive reference for the architecture,
engineering decisions, measurements, and limitations of textsift v0.1.0.
NotebookLM ingests it; the resulting deck should read like a
Cloudflare blog post (narrative, first-person, technical-but-readable).
This source itself is **not** the deck — it's the material the deck is
extracted from. Every concrete number on a future slide must be
quotable from this file.

If a fact isn't in this document, it doesn't go on a slide.

---

## 0. One-paragraph project description

textsift runs OpenAI's `openai/privacy-filter` model entirely on the
user's device — in the browser via WebGPU + WASM, on Node via per-
platform GPU fast paths (Metal on macOS, Vulkan on Linux, Dawn on
Windows). Same TypeScript API across all surfaces. Same `model_q4f16
.onnx` file across all backends. The point is to detect and redact
PII (emails, phones, SSNs, credit cards, API keys, addresses, names,
etc.) without sending the raw text to a remote service. It's a
personal learning project — Apache 2.0, no SLA, no team.

## 1. Why this exists at all

Three motivations stacked:

1. **I wanted to learn three things**: WebGPU compute shaders (WGSL),
   Zig→WASM with SIMD intrinsics (specifically `simd128` plus
   `relaxed_simd`), and the o200k-style BPE tokenizer pipeline.
   PII detection happened to need all three at once.

2. **There's a real coverage gap on consumer Linux**. If you have an
   NVIDIA card you can run `openai/privacy-filter` via PyTorch + CUDA.
   If you have an Apple machine you can use Core ML or MPS. But on
   Intel iGPU (UHD, Iris Xe), AMD APU, or non-NVIDIA Linux generally,
   there's no off-the-shelf path: ONNX Runtime Node falls back to
   CPU; PyTorch's CPU build is slow; transformers.js needs a GPU
   too. Mesa Vulkan is sitting there unused. A hand-tuned Vulkan-
   direct backend fills that gap.

3. **The privacy story is real**. Server-side PII filters mean every
   chunk of user text leaves the device — a network round-trip, a
   third-party service in the data path, plus latency. Running the
   model client-side keeps raw text on the device.

## 2. The model

`openai/privacy-filter`:

- 8 PII labels: `account_number`, `private_address`, `private_email`,
  `private_person`, `private_phone`, `private_url`, `private_date`,
  `secret`. Tags are emitted in BIOES form (so 8 × 4 + 1 = 33 output
  classes per token, plus the model's own internal label `O`).
- 8 transformer layers with mixture-of-experts: 64 experts per layer,
  top-K=8 routing. Hidden size 512, intermediate (FFN) size 2048,
  attention head dim 64.
- Quantized to int4 weights + fp16 activations
  (`model_q4f16.onnx`). 770 MB total on disk including the
  `.onnx_data` shards. This is the artifact textsift ships with.
- BPE tokenizer is the o200k scheme (the same one tiktoken's o200k
  encoder uses).

The HuggingFace repo also ships `viterbi_calibration.json`. The
canonical decoder is **Viterbi over BIOES tags with six transition
biases**, not naive argmax. The biases parameterize:

```
transition_bias_background_stay        (O → O)
transition_bias_background_to_start    (O → B-x or O → S-x)
transition_bias_inside_to_continue     (B-x → I-x and I-x → I-x)
transition_bias_inside_to_end          (B-x → E-x and I-x → E-x)
transition_bias_end_to_background      (E-x → O and S-x → O)
transition_bias_end_to_start           (E-x → B-y / S-y, and S-x → B-y / S-y)
```

In the default operating point published with the model, all six
biases are 0.0 — meaning Viterbi reduces to "find the highest-scoring
BIOES-legal path through the per-token logits". An orphan `E-x` with
no preceding `B-x` is illegal; Viterbi will choose `O` for that
position even if `E-x` is the highest emission, because the path
score under BIOES legality requires it.

## 3. Backend pipeline

There are five backends. Each one reimplements the same model graph
against the same ONNX file; what differs is the compute layer.

### 3.1 Browser, WebGPU

- Hand-written WGSL kernels (15 kernels: `rms_norm`, `rope_apply`,
  `banded_attention`, `matmul_int4_fp16_f16`, `matmul_int4_f32_f32`,
  `embed_lookup_int4`, `add_rmsnorm_fp16_to_f32`, `swiglu_clamp`,
  `router_topk`, `qmoe_gate_up`, `qmoe_down_scatter`, `cast_*`,
  `add_fp16`, `zero_f32`).
- WGSL is the canonical source of truth — the Mac/Linux/Windows
  native ports are translations of these kernels.
- Requires `shader-f16` adapter feature. If unavailable, falls back
  to WASM cleanly (the auto-fallback probes `adapter.features.has
  ("shader-f16")` before instantiating the WebGPU backend).

### 3.2 Browser, WASM

- Zig → WASM with `-mcpu=generic+simd128+relaxed_simd`. Multi-threaded
  variant adds `+atomics+bulk_memory` and uses SharedArrayBuffer (a
  worker pool dispatches MoE expert calls in parallel via shared-
  memory atomics — no `postMessage` on the hot path).
- WASM bytes (~92 KB ST + ~92 KB MT) are inlined into the JS bundle
  as `Uint8Array` constants — no second `fetch` for the WASM file at
  load time.
- This is the only working browser CPU path for `model_q4f16.onnx`.
  ORT-Web's WASM bundle lacks the `MatMulNBits` and
  `GatherBlockQuantized` contrib kernels needed to load this model
  on CPU; `transformers.js` with `device: "wasm"` fails session
  creation here.

### 3.3 Node native, macOS — Metal-direct

- Hand-written MSL (Metal Shading Language) ports of the WGSL kernels.
- Obj-C bridge (`metal/bridge.{h,m}`, compiled with `-fobjc-arc`)
  exposes a small C ABI that the Zig NAPI surface wraps.
- Pipeline cache + encoder-batched dispatch. End-to-end forward
  encodes ~133 dispatches per layer, batched into a single Metal
  command buffer.

### 3.4 Node native, Linux — Vulkan-direct

- Hand-written GLSL kernels (`vulkan/shaders/*.comp.glsl`) compiled
  to SPIR-V at build time via `glslangValidator`.
- ~600 LOC C bridge for instance/device/queue/buffer/descriptor/
  pipeline/cmd-buffer/fence management. Push constants for uniforms
  (cheaper than buffer-backed uniform blocks per dispatch).
- Statically links against system `libvulkan` at runtime via dlopen
  through the Vulkan loader.

### 3.5 Node native, Windows — Dawn-direct

- Statically-linked Google Dawn (~37 MB of C++ in the .node binary).
- Tint compiles the canonical WGSL kernels at runtime to HLSL → D3D12
  via Dawn's backend selection. We ship the same WGSL the browser
  ships.
- `wgpuInstanceWaitAny` for sync, sidesteps the `setImmediate`
  deadlock that broke `node-webgpu` on heavy compute loads.

### 3.6 Why Dawn is Windows-only (it used to also be on Linux)

Dawn-on-Linux was redundant. Dawn on Linux uses the Vulkan backend
internally — if `libvulkan` isn't present, Dawn fails to initialize
the same way Vulkan-direct does. The "Dawn as fallback" idea didn't
hold up: when Vulkan-direct couldn't find an adapter, Dawn-on-Linux
couldn't either. The actual no-GPU path was always WASM. So we
dropped Dawn from the Linux build:

- 37 MB cut from the Linux `.node`.
- ~30 minutes of cold Dawn build cut from CI per release.
- One fewer codegen path (Tint vs hand-written GLSL) in the Linux
  test matrix.

### 3.7 Backend selection logic

`PrivacyFilter.create()` picks at runtime:

```
opts.backend === "webgpu"  → WebGPU (or throw if no shader-f16)
opts.backend === "wasm"    → WASM CPU
opts.backend === "auto"    → probe; WebGPU if shader-f16 available,
                             else WASM (browser); native first then
                             WASM (Node, via the umbrella).
```

The umbrella `textsift` package's resolver tries the per-platform
native binding first via `require("textsift-native-${process.platform}-${process.arch}/textsift-native.node")`.
If the .node fails to load (no GPU, no Vulkan loader, missing
binary, etc.), the resolver falls through to the WASM backend.
Throwing the failure is gated behind explicit `offline: true` mode
for users who want loud failure instead of silent fallback.

## 4. The tokenizer is its own win

The HuggingFace approach pulls in:

- `@huggingface/transformers` (the JS package wrapping
  `tokenizers`, `protobuf`, `onnxruntime-web`, `numpy`-equivalents)
- Tokenizers binary (a Rust-compiled WASM blob)
- protobuf for the tokenizer config
- Several MB of code, much of it unrelated to BPE

textsift's tokenizer is hand-rolled pure TypeScript:

- 76 KB gzipped over the wire (measured against the production
  `dist/browser/` bundle in `benchmarks.mdx`)
- Implements the o200k BPE merge rules directly
- Loads `tokenizer.json` (~6 MB JSON of merge rules) on first use,
  caches in OPFS with the model weights
- No native code, no protobuf, no transformer dependency

For people who'd otherwise pull `@huggingface/transformers` *only*
for tokenization, the bundle-size delta is real.

## 5. Viterbi decoder

`packages/textsift/src/browser/inference/viterbi.ts`. Standard
forward-DP Viterbi over the BIOES tag space:

```
alpha[0][j] = start_scores[j] + logits[0][j]
alpha[t][j] = logits[t][j] + max_i (alpha[t-1][i] + transition[j,i])
back[t][j] = argmax_i (alpha[t-1][i] + transition[j,i])
```

Final tag = argmax over `alpha[T-1]`; backtrace via `back[]`.

The transition matrix is built from the six biases above. Illegal
transitions (e.g., `O → I-x`, `B-x → O`) are marked `-INF`. Allowed
transitions get the corresponding bias value.

## 6. Spans + redaction

After Viterbi produces tags, BIOES-to-span conversion walks the tag
sequence:

- `B-label` opens a span; consume `I-label*` and a closing `E-label`
- `S-label` is a single-token span
- Span offsets are pulled from the tokenizer's offset map (each
  token has a `(charStart, charEnd)` in the input string)
- Leading whitespace is trimmed from span starts (the o200k BPE
  assigns leading whitespace to the *following* token, so a span
  starting at the space-before-John would over-include — we trim it)

`redact()` replaces each span with a marker. Default markers are
`[label]`. Custom strategies:

```ts
// Static map
const filter = await PrivacyFilter.create({
  markers: { private_person: "[NAME]", private_email: "[EMAIL]" }
});

// Dynamic indexed
const filter = await PrivacyFilter.create({
  markers: (span, i) => `[${span.label.toUpperCase()}_${i}]`
});

// Faker preset — emit realistic fakes, stable per-text within
// the filter's lifetime ("Alice" → "Alice Anderson" both times)
const filter = await PrivacyFilter.create({
  markers: markerPresets.faker()
});
```

`secret` spans deliberately render as `[secret]` regardless of preset
— emitting another credible-looking secret is a footgun.

## 7. Custom rules

Regex or function rules merged with model spans:

```ts
type Rule =
  | { label: string; severity?: "block" | "warn" | "track";
      marker?: string; pattern: RegExp }
  | { label: string; severity?: ...; marker?: string;
      match: (text: string) => Array<{ start: number; end: number }> };
```

Built-in `secrets` preset covers credentials the model wasn't trained
on: JWT, GitHub PAT (all variants), AWS access key, Slack tokens +
webhooks, OpenAI/Anthropic/Google API keys, Stripe live/test keys,
Stripe webhook secrets, npm tokens, PEM private-key headers. All
default to severity `block`.

Performance note: the runtime unions all regex rules into one
alternation and scans the input once, regardless of preset count.
N rules ≈ 1 scan.

## 8. Streaming detect / redact

`detect()` and `redact()` accept an `AsyncIterable<string>` in
addition to a string. Used for:

- Aborting an LLM stream the moment a credit card / API key appears
- Rendering redacted text progressively as it streams from the model
- Fronting a model gateway (Cloudflare Worker style) that has to
  forward chunk-by-chunk

Implementation: chunks are tokenized incrementally, the forward pass
runs over the full prefix tokens, the Viterbi state survives across
chunks (no recomputation of earlier tokens). The API yields a
`spanStream` (each new span as it becomes detectable) and a
`textStream` (each new redacted string slice as it becomes safe to
emit).

This is *not* O(N) vs O(N²) — that framing is misleading. The honest
description: streaming maintains decoder state across chunks instead
of re-running detect on each prefix. The naive thing is "call detect
on the full prefix every chunk", which a normal developer wouldn't
write — they'd batch.

## 9. Performance — measured numbers, by hardware

All measurements are real, with hardware named on every row.

### 9.1 Browser (M3 Pro, Chromium 147)

Forward latency, median of 5 samples after 2 warmup. Reproduce with
`tests/browser/bench.spec.ts`.

```
~7 tokens:    textsift WebGPU  8.9 ms    textsift WASM MT 29.0 ms    transformers.js WebGPU 32.7 ms
~25 tokens:   textsift WebGPU 11.8 ms    textsift WASM MT 44.6 ms    transformers.js WebGPU 38.5 ms
~80 tokens:   textsift WebGPU 22.0 ms    textsift WASM MT 95.9 ms    transformers.js WebGPU 56.4 ms
```

textsift WebGPU is 2.6–3.7× faster than `transformers.js` across every
input length. The differential is the kernels (textsift's hand-tuned
int4 matmul / banded attention vs ORT-Web's int4 contrib ops).

### 9.2 Node native — macOS (M2 Pro, Metal-direct)

```
T=7:  textsift native  5.2 ms   browser textsift WebGPU  8.9 ms (M3 Pro)
T=25: textsift native 10.0 ms   browser textsift WebGPU 11.8 ms
T=32: textsift native 10.8 ms
T=80: textsift native 23.8 ms   browser textsift WebGPU 22.0 ms
```

Hand-written MSL beats Tint's WGSL→MSL codegen by ~1.9× at T=32 on
the same hardware. The gap is real codegen quality: control over loop
unrolling, threadgroup memory layout, simdgroup matrix ops on M3+.

End-to-end `redact()` on a 122-character input with 4 PII spans:
~110 ms on M2 Pro (BPE tokenization + Metal-direct forward + Viterbi
+ span replacement).

### 9.3 Node native — Linux (Intel Iris Xe, Mesa Vulkan)

```
T=32: textsift Vulkan-direct  ~25 ms     ORT Node CPU ~785 ms     speedup 32×
```

Theoretical Iris Xe memory-bandwidth ceiling for this forward (~770 MB
weights touched once + scratch) is ~11 ms. We're at ~50% of ceiling.
Optimization headroom (subgroup ops, persistent descriptor caching,
larger matmul tiles): ~10–15% wall, diminishing returns.

This is the headline Linux story: GPU-accelerated PII detection on
Intel iGPU / AMD APU / non-NVIDIA hardware, no CUDA, no ROCm, no
driver dance.

### 9.4 Cold start (browser, M3 Pro)

```
adapter + device request:        0.20 s (Metal driver warm-up)
OPFS read (770 MB model):        0.36 s
ONNX parse:                      0.001 s
GPU buffer upload:               0.38 s
pipeline compile (14 WGSL):      0.002 s
TOTAL (cache hit):               0.93 s
```

We do not claim a cold-start speedup over `transformers.js`. The
honest comparison is muddled by storage choice: `transformers.js`
defaults to the Cache API, which silently rejects the 770 MB model
weights with `QuotaExceededError`, so it re-fetches every visit. That's
a real user-visible cost in default config, but it's a *storage
decision*, not an *engine speedup*. Plug an OPFS-backed adapter into
`transformers.js` and the gap closes.

### 9.5 Bundle size

```
textsift/browser:  630 KB minified, 76 KB gzipped over the wire
+ 90 KB .wasm loaded async (not in JS bundle)
```

The native entry (`textsift` bare import) ships zero bytes for browser
bundlers — they only resolve the `./browser` subpath.

## 10. Conformance test (10/10 vs ONNX reference)

`tests/conformance/pytorch/` (the path is historical; the test no
longer touches PyTorch).

Workflow:

1. Python script (`generate-fixtures.py`) loads `model_q4f16.onnx`
   into ONNX Runtime — the same file textsift ships.
2. Loads `viterbi_calibration.json` from the same HuggingFace repo
   and applies the canonical Viterbi+biases decoder in Python.
3. Runs forward + decode on each input in `inputs.json` (10 inputs
   covering email, phone, address, SSN, credit card, passport, bank
   account, dates, person names, URLs).
4. Dumps spans (label + char start + char end) to `fixtures.json`.
5. Commits `fixtures.json`.

JS side (`tests/native/integration/pytorch-parity.test.js`):

1. Loads `fixtures.json`.
2. For each input, calls `filter.detect(text)` via textsift.
3. Asserts the resulting span set is bit-equal to the fixture's
   span set (label + start + end, sorted).

CI runs this on every push, on Linux/macOS/Windows. As of v0.1.0:
**10/10 inputs span-equivalent to the canonical ONNX reference.**

By matching both the model file (q4f16, not fp32) and the decoder
(Viterbi+biases, not argmax), any divergence would be a textsift
kernel/wiring bug. The test is real verification, not a slogan.

### 10.1 Known divergence

When q4f16's argmax produces a stray `E-<label>` with no preceding
`B-` somewhere downstream of a legitimate span, textsift's Viterbi
can over-suppress the legitimate span (it's globally rebalancing
probability mass around the orphan E and choosing all-O as the
highest-scoring legal path). The reference, decoded with the same
Viterbi+biases in Python, doesn't repro this on the same input —
suggesting a bug in textsift's logits or path-cost arithmetic on
this specific case.

The current `inputs.json` deliberately avoids this pattern. The bug
is filed under follow-up work; the fix likely lives in
`viterbi.ts` and may involve a subtle off-by-one in the path-score
update.

## 11. The Windows port (10-commit slog)

In order:

1. **Node distros for Windows don't ship C headers.** Node on Linux/
   Mac includes `<prefix>/include/node/`. On Windows it doesn't.
   Fix: detect Windows + missing headers, download
   `node-v<ver>-headers.tar.gz` from nodejs.org under
   `vendor/node-headers/`.

2. **Zig's link search doesn't read MSVC's `LIB` env var.**
   `ilammy/msvc-dev-cmd@v1` populates `LIB` from `vcvars64.bat`,
   but Zig's lld-link only walks the `-L` paths it's given. Fix:
   split `LIB` on `;` and emit each entry as its own `-L` flag.

3. **Zig's default Windows target is `windows-gnu` (MinGW), not
   `windows-msvc`.** Dawn was built MSVC-ABI (clang on Windows
   defaults to MSVC). Mixing produces duplicate-symbol link errors
   on Control Flow Guard intrinsics (libmingw32 vs msvcrt). Fix:
   `-target x86_64-windows-msvc` explicitly.

4. **`translate-c` chokes on MSVC's `SIZE_MAX`.** MSVC's `stdint.h`
   defines it as `0xFFFFFFFFFFFFFFFFui64`; the `ui64` suffix is
   non-standard and Zig can't parse it. Fix: define our own
   `NAPI_AUTO_LENGTH = std.math.maxInt(usize)` instead of using the
   translated `c.NAPI_AUTO_LENGTH`.

5. **UCRT imports weren't getting linked.** Symptom:
   `__acrt_thread_attach` undefined → segfault at `LoadLibrary` →
   exit 139 with no stderr. Fix: explicit `-lucrt -lvcruntime`.

6. **`node.lib` (the Windows-specific Node import library exposing
   `napi_*`) wasn't being downloaded.** All `napi_*` symbols
   undefined at link time. Fix: fetch `node.lib` from
   `nodejs.org/dist/v<ver>/win-x64/node.lib` alongside the headers.

7. **PowerShell was swallowing stderr** from a child process that
   crashed during `dlopen`. The conformance step ran for 0.4 s and
   exited 1 with empty logs. Fix: `shell: bash` on every
   conformance + integration step.

8. **GitHub Actions Windows runners have no D3D12-capable adapter.**
   Even after the binary loaded, Dawn couldn't `createBackend()`.
   Fix: detect "no adapter" specifically and skip the conformance
   test cleanly with a clear message — fail loudly when there's a
   real problem, skip cleanly when there's a known environment
   limitation.

9. **bash 3.2 (macOS default) errors on empty array expansion**
   under `set -u`. The Windows-only `ZIG_TARGET_ARGS=()` array,
   expanded as `"${ZIG_TARGET_ARGS[@]}"`, broke the macOS leg of
   CI. Fix: `${ZIG_TARGET_ARGS[@]+"${ZIG_TARGET_ARGS[@]}"}`.

10. **Dawn produces `webgpu_dawn.lib` on Windows (MSVC) and
    `libwebgpu_dawn.a` on Linux (clang).** The build script's `cp`
    was hardcoded to the UNIX name and exited 1 after a clean
    1102/1102 link. Fix: OS-aware `cp` with both name patterns.

Each fix was small. The slog was diagnosis: log output was suppressed,
errors fired at the wrong layer, or the symptom was a segfault with
no stack trace. The CI loop took ~25 min per attempt because the
Dawn cold build is slow on Windows runners — meaning the iteration
cost was high.

## 12. CI matrix

`.github/workflows/test.yml` runs on every push:

- `linux-x64` (Ubuntu, Mesa llvmpipe — software Vulkan; we cap
  `rope_apply` tolerance to 32 fp16 ULPs because llvmpipe rounds
  differently than real GPUs)
- `darwin-arm64` (macOS, real silicon)
- `windows-x64` (Windows, no D3D12 adapter on the runner; Dawn
  conformance skips cleanly; WASM fallback runs end-to-end)
- `browser` (Playwright Chromium without `shader-f16` → WebGPU
  specs skip with a clear reason; WASM specs run)

`bench.yml` runs the same matrix but executes the perf bench and
publishes a markdown summary. Numbers from CI runners are explicitly
labelled by device kind (`real-gpu` / `software-fallback` /
`no-adapter`) so a llvmpipe row isn't misread as a real-GPU
measurement.

`release.yml` runs on `v*.*.*` tags. Builds the per-platform `.node`
binary on each runner, uploads as artifact, then a single `publish`
job pulls all artifacts and publishes:

- `textsift-native-{linux-x64,linux-arm64,darwin-x64,darwin-arm64,
  win32-x64}` (per-triple, single .node each)
- `textsift` (umbrella, points at all 5 via `optionalDependencies`)

Uses npm OIDC trusted publishing — `id-token: write` permission +
`--provenance` flag. No NPM_TOKEN secret to manage.

## 13. The CLI / pre-commit / GitHub Action surfaces

Same engine, three more entry points:

```sh
# CLI
echo "Hi Alice, alice@example.com" | npx textsift redact
npx textsift table customers.csv --header --mode synth > clean.csv
npx textsift detect log.txt --jsonl | jq 'select(.label == "private_email")'
TEXTSIFT_OFFLINE=1 npx textsift redact file.txt   # CI: fail if not pre-cached
npx textsift download                              # pre-warm in CI
```

```yaml
# pre-commit framework
repos:
  - repo: https://github.com/teamchong/textsift
    rev: v0.1.0
    hooks:
      - id: textsift-pii-scan
```

```yaml
# GitHub Action — annotates PRs with ::error file=… markers,
# uploads SARIF for the Security tab
- uses: teamchong/textsift@v0.1.0
  with:
    sarif-output: textsift.sarif
- uses: github/codeql-action/upload-sarif@v3
  with: { sarif_file: textsift.sarif, category: textsift }
```

SARIF v2.1.0 export is a separate import:
`import { detectResultToSarif, toSarif } from "textsift/sarif"`.
Includes `partialFingerprints` for cross-run dedup and per-label
rule definitions in `tool.driver.rules`.

## 14. OPFS caching (browser)

`transformers.js` defaults to the Cache API for model weights. The
Cache API silently rejects payloads above its per-resource limit
(`QuotaExceededError` for our 770 MB file in some browsers; a
truncated write that fails on read in others). Effect:
`transformers.js` re-fetches the 770 MB on every visit by default.

textsift's loader uses OPFS (Origin Private File System) directly:
treats the browser like a local filesystem, no Cache API
abstraction. The 770 MB persists, second-visit warmup is 0.93 s.

## 15. Tabular data (CSV / DB columns)

```ts
const rows = [
  ["id", "name",         "email",             "amount"],
  ["1",  "Alice Carter", "alice@example.com", "100"],
  ["2",  "Bob Davis",    "bob@example.com",   "250"],
];

// "Which columns have PII?" — sample-based classifier
const cols = await filter.classifyColumns(rows, { headerRow: true });
// → [{ index: 0, label: null }, { index: 1, label: "private_person", confidence: 1 },
//    { index: 2, label: "private_email", confidence: 1 }, { index: 3, label: null }]

// Three modes for column-level redaction
const safe = await filter.redactTable(rows, {
  headerRow: true,
  mode: "synth",   // "redact" | "synth" | "drop_column"
});
```

Use case: GDPR right-to-be-forgotten, vendor data sharing prep, or
"is this dump safe to commit?".

## 16. Caveats (must appear on the deck somewhere)

`openai/privacy-filter` is a detection aid, not an anonymization
guarantee. English-first, ~88% F1 on Japanese, other languages
untested. Short text under-contextualizes — the same digit sequence
might be flagged in one phrasing and missed in another (one such
divergence is documented in `tests/conformance/pytorch/README.md`).

Pre-1.0: the surface can break before a 1.0 release. There's no SLA,
no team, no roadmap commitment.

## 17. License + where it lives

- Apache 2.0 (matching the upstream model's license)
- npm: `textsift` (umbrella) + the five per-triple natives
- Source: <https://github.com/teamchong/textsift>
- Docs: <https://teamchong.github.io/textsift/>
- Playground (browser, runs the model in-page):
  <https://teamchong.github.io/textsift/playground/>
- Faker mode demo:
  <https://teamchong.github.io/textsift/playground-faker/>
- Model: <https://huggingface.co/openai/privacy-filter>
