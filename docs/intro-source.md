# textsift: notes on building a client-side PII filter

This is the source document for the architecture intro video. It's
written deliberately as the kind of post a Cloudflare engineer would
publish — first-person, honest about tradeoffs, every number tied to
specific hardware. NotebookLM ingests this and generates the slide
deck. Anything not in this document **must not appear in the deck.**

## Why this exists

I wanted to teach myself three things at once: WebGPU compute shaders,
Zig→WASM with SIMD intrinsics, and the o200k-style BPE tokenizer
pipeline. PII detection happened to be a good excuse — there's a
shipped model (`openai/privacy-filter`), there's a clear privacy story
(don't send raw text to a remote service just to redact it), and the
inference fits in 770 MB of int4+fp16 weights, which is on the edge of
what a browser can host.

There's also a real coverage gap: on consumer Linux without an NVIDIA
GPU (Intel iGPU, AMD APU), there's no good off-the-shelf path to run
this model fast. ONNX Runtime Node falls back to CPU. PyTorch needs
CUDA. Mesa Vulkan is sitting there unused. Filling that gap with a
hand-tuned Vulkan backend is part of the project's shape.

## The model

`openai/privacy-filter` is a small token-classification model with 8
PII labels: `account_number`, `private_address`, `private_email`,
`private_person`, `private_phone`, `private_url`, `private_date`, and
`secret`. The model card specifies a Viterbi decoder with calibrated
transition biases (`viterbi_calibration.json` ships alongside the
weights) — pure argmax over per-token logits is *not* the canonical
decoder.

We use the `model_q4f16.onnx` artifact: 4-bit weights, fp16
activations, ~770 MB total. Same file across all backends.

## Architecture: many backends, one API

There's a single TypeScript surface (`PrivacyFilter.create()`,
`detect()`, `redact()`). Behind it, the runtime picks the fastest
available backend for the host:

- **Browser, GPU available**: WebGPU with hand-written WGSL kernels.
- **Browser, no shader-f16**: WASM CPU (Zig + SIMD128 + relaxed_simd).
- **Node on macOS**: Metal-direct, hand-written MSL via an Obj-C
  bridge.
- **Node on Linux**: Vulkan-direct, hand-written GLSL → SPIR-V via
  `glslangValidator`.
- **Node on Windows**: Dawn-direct (statically-linked Google Dawn,
  Tint compiles WGSL → HLSL → D3D12 at runtime).
- **Node, no GPU**: WASM CPU fallback through the same API.

These five paths reimplement the same model graph against the same
ONNX file. The native binary for each platform is wired up via npm
`optionalDependencies` — the umbrella `textsift` package is JS-only;
each `textsift-native-<triple>` subpackage ships a single `.node`
binary; npm picks the matching one at install time.

## The tokenizer is its own story

The HuggingFace path here is `@huggingface/transformers`, which pulls
in `tokenizers`, `protobuf`, `numpy` (transitively, via Pyodide-ish
abstractions), and a long tail of dependencies. For a project whose
whole point is "small bundle, runs in the browser", that's a fight.

So the tokenizer is hand-rolled: a pure-TypeScript implementation of
the o200k BPE scheme, no native code, no protobuf, no transformer
package. It's 76 KB gzipped over the wire. That's measured against
our `dist/browser/` bundle in `benchmarks.mdx`, not extrapolated.

## The Viterbi decoder

The model is a token classifier emitting BIOES tags (`O`, `B-label`,
`I-label`, `E-label`, `S-label`). Pure argmax over per-token logits
can produce invalid sequences (e.g. an orphan `E-` with no preceding
`B-`). The model authors ship a Viterbi calibration JSON with six
transition biases that score the BIOES legality constraints; the
canonical pipeline applies Viterbi with those biases, not naive
argmax.

textsift mirrors that calibration. The conformance test
(`tests/conformance/pytorch/`) loads the same `model_q4f16.onnx` file
into ONNX Runtime, applies the same Viterbi+biases decoder in
Python, and dumps the resulting spans. The JS test then runs textsift
on the same inputs and asserts the spans match. As of v0.1.0:
**10/10 fixtures match the canonical ONNX reference**, across
Linux/macOS/Windows in CI.

## The Linux story

The Vulkan-direct backend was the hardest piece. The interesting
result: on an Intel Iris Xe (Mesa Vulkan), at T=32 tokens, the
forward pass takes ~25 ms via Vulkan-direct. The same workload via
ONNX Runtime Node CPU takes ~785 ms on the same machine. That's a 32×
gap, and it's all about avoiding the CPU fallback when there's a GPU
sitting there.

Theoretical memory-bandwidth ceiling for the same forward (~770 MB of
weights touched once + scratch) is roughly 11 ms on Iris Xe. We're at
~50% of ceiling — there's headroom in subgroup ops, persistent
descriptor caching, and larger matmul tiles, each worth maybe 5–10 %.
Not low-hanging fruit.

## The Mac story

On an M2 Pro, the Metal-direct backend at T=32 measures 10.8 ms vs
22 ms for Chrome's own WebGPU stack on the same machine — about 1.9×
faster. Dawn (Chrome's WebGPU implementation) does WGSL → MSL via the
Tint compiler. Hand-written MSL controls loop unrolling, threadgroup
memory layout, and simdgroup matrix ops directly, and that's where
the 1.9× comes from. Same model weights, same input.

End-to-end `redact()` on a 122-character input with 4 PII spans:
~110 ms on the user's M2 Pro. That includes BPE tokenization, the
forward pass, Viterbi decode, span extraction, and string
substitution.

## The Windows port

Dawn-direct on Windows was a 10-commit slog. The order of breakage
was:

1. Node distros for Windows don't ship C headers — fetch them from
   nodejs.org.
2. Zig's link search doesn't read MSVC's `LIB` env var — translate it
   to explicit `-L` flags.
3. Zig's default Windows target is `windows-gnu` (MinGW) but Dawn was
   built MSVC-ABI; force `-target x86_64-windows-msvc`.
4. `translate-c` chokes on MSVC's `SIZE_MAX` (the `0xFFFFFFFFFFFFFFFFui64`
   suffix is non-standard) — define `NAPI_AUTO_LENGTH` ourselves
   instead of using the translated macro.
5. UCRT imports weren't getting linked — `__acrt_thread_attach`
   undefined → segfault at `LoadLibrary` → no output, just exit 139.
   Add `-lucrt -lvcruntime`.
6. `node.lib` (the Windows-specific Node import library exposing
   `napi_*`) wasn't being downloaded — fetch it from nodejs.org's
   `/win-x64/` bin distribution.
7. PowerShell was swallowing stderr from a child process that
   crashed during `dlopen`; switching the conformance step to bash
   surfaced the real error.

Each one was a small fix; the slog was figuring out which thing was
actually broken.

## What CI runs

GitHub Actions matrix on every push:

- `linux-x64` (Ubuntu, Mesa llvmpipe — software Vulkan; we cap
  rope_apply tolerance to 32 fp16 ULPs because llvmpipe rounds
  differently than real GPUs)
- `darwin-arm64` (macOS, real silicon)
- `windows-x64` (Windows, no D3D12 adapter on the runner — Dawn
  conformance skips cleanly with a clear "no adapter" message; the
  WASM fallback path runs end-to-end)
- `browser` (Playwright, Chromium without `shader-f16` → WebGPU specs
  skip with a clear reason; WASM specs run)

The conformance test (10/10 vs ONNX reference) runs on all three OSes
through the WASM fallback path on jobs that don't have a real GPU.

## Caveats

`openai/privacy-filter` is a detection aid, not an anonymization
guarantee. English-first, ~88% F1 on Japanese, other languages
untested. Short text under-contextualizes — the same digit sequence
might be flagged in one phrasing and missed in another (one such
divergence is documented in `tests/conformance/pytorch/README.md`).

## Where it lives

- npm: `textsift` (umbrella) + `textsift-native-{linux-x64,
  linux-arm64,darwin-x64,darwin-arm64,win32-x64}` (per-triple natives)
- Docs: <https://teamchong.github.io/textsift/>
- Playground: <https://teamchong.github.io/textsift/playground/>
- Source: <https://github.com/teamchong/textsift>
- Model: <https://huggingface.co/openai/privacy-filter>

Apache 2.0, matching the upstream model.
