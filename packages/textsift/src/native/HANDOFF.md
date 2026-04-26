# Native binding — Linux handoff

Last update: 2026-04-26.

This is a handoff doc for whoever picks up the cross-platform native binding work (issue #79). Mac fast-path is done. Linux and Windows are the remaining work, plus one Dawn-Node hang investigation that needs a non-Mac machine to triage.

## What's done

### Kernel layer (all platforms via WGSL)
All 15 WGSL compute kernels live at `packages/textsift/src/native/shaders/*.wgsl`. They are byte-equal to the browser fixture via two independent harnesses:
- `node tests/native/conformance/all.test.js` — runs the 15 kernels through wgpu-native (Naga codegen) and compares against fixtures dumped from Chromium.
- `node tests/native/metal/conformance-all.js` — runs the 15 hand-written MSL kernels (Mac only) via the metal-direct backend.

Both pass 15/15.

### Mac fast-path (Metal-direct)
- `packages/textsift/src/native/metal/shaders.metal` — hand-written MSL ports of all 15 kernels.
- `packages/textsift/src/native/metal/bridge.{h,m}` — Obj-C → C bridge (compiled with `-fobjc-arc`).
- `packages/textsift/src/native/metal_backend.zig` — Zig wrapper with pipeline cache + encoder-batched dispatch.
- `packages/textsift/src/native/napi.zig` — `metal*` NAPI surface: `metalCreateBackend`, `metalCreateBuffer`, `metalCreateEmptyBuffer`, `metalReleaseBuffer`, `metalReadBuffer`, `metalWriteBuffer`, `metalDispatchOneShot`, `metalBeginEncoder`, `metalEnqueueDispatch`, `metalSubmitAndReadback`.
- `tests/native/forward-metal.js` — end-to-end synthetic-weight forward via Metal-direct.

Bench (M2 Pro):

| T  | Metal-direct | wgpu-native | Browser WebGPU | tjs WebGPU |
|---:|-------------:|------------:|---------------:|-----------:|
|  7 | 5.3 ms       | —           | 8.9 ms         | 32.7 ms    |
| 25 | 10.0 ms      | —           | 11.8 ms        | 38.5 ms    |
| 32 | **11.6 ms**  | 22.0 ms     | 22.0 ms        | —          |
| 80 | **25.6 ms**  | 43.3 ms     | —              | 56.4 ms    |

**Reproduce:**
```sh
cd packages/textsift && bash scripts/build-native.sh
T=32 node tests/native/forward-metal.js
T=80 node tests/native/forward-metal.js
```

### wgpu-native (cross-platform but slow)
- `packages/textsift/src/native/wgpu_init.zig` — wgpu-native v29 wrapper (Mozilla wgpu Rust crate compiled to a C library).
- Vendored prebuilts at `packages/textsift/vendor/wgpu-native/` per platform.
- All 15 kernels conform via Naga's WGSL→MSL/SPIR-V/HLSL codegen.
- Naga produces measurably slower MSL than Tint — wgpu-native is the floor, not the goal.

### Dawn investigation (the open question)
Google's Dawn project ships a Node binding via the npm `webgpu` package (maintained by the Dawn team, last published 2026-03-27). Cross-platform prebuilts: darwin-universal, linux-x64, linux-arm64, win32-x64. ~71 MB tarball.

Three test files at `tests/native/dawn/`:
- `smoke-rms-norm.js` — single dispatch via Dawn, byte-equal vs browser fixture. **Works.**
- `smoke-multi.js` — 5-dispatch pass via Dawn. **Works.**
- `forward-dawn.js` — full 130-dispatch-per-pass forward via Dawn. **Hangs reliably on Mac after 0–7 successful iterations.**

Best Dawn data point captured before the hang resurfaced: T=32 = **16.9 ms median** (closer to browser than to Metal-direct, but a single fluke run). The hang is in `await readBuf.mapAsync()` — the submit is accepted but Dawn never reports the buffer mappable.

## What needs doing on Linux

### 1. Verify whether the Dawn hang is Mac-specific

Run `node tests/native/dawn/forward-dawn.js` on real Linux GPU. Expected outcomes:

- **If it works on Linux** → ship Dawn for Linux/Windows. Mac stays Metal-direct (faster). wgpu-native code can be removed from the user-facing path.
- **If it also hangs on Linux** → file an upstream issue at https://github.com/dawn-gpu/node-webgpu/issues with the repro. Either ship wgpu-native as the Linux/Windows path (slower but proven), or write Vulkan-direct kernels (analogous to Metal-direct) for ~1.45× speedup over Dawn.

The hang investigation findings are at the top of `tests/native/dawn/forward-dawn.js` — every fix that *didn't* work on Mac is documented there so you don't repeat it.

### 2. Cross-platform `.node` distribution

Once a Linux/Windows path is chosen:

- GitHub Actions matrix to build `packages/textsift/dist/textsift-native.node` per platform.
- `optionalDependencies` packaging pattern (same as `napi-rs` / `better-sqlite3`):
  - `@textsift/native-darwin-arm64` — Mac arm64 prebuild (Metal-direct).
  - `@textsift/native-darwin-x64` — Mac x64 prebuild.
  - `@textsift/native-linux-x64` — Linux x64.
  - `@textsift/native-linux-arm64` — Linux arm64.
  - `@textsift/native-win32-x64` — Windows x64.
- npm picks the right one at install. Source-build fallback for unsupported platforms.

If the Linux/Windows path is Dawn (npm `webgpu`), no platform-specific `.node` is needed for those — the `webgpu` package already ships prebuilts. The optionalDependencies pattern is then only needed for the Mac Metal-direct binary.

### 3. PrivacyFilter wiring on the native entry

Currently `import { PrivacyFilter } from "textsift"` throws — kernel layer is done but the high-level API isn't wired through. Required:

- Reuse `src/browser/` for tokenizer + Viterbi + span extraction (pure TS, no DOM deps confirmed in `WebGpuBackend.forward()` — only `navigator.gpu` and `fetch` are used).
- Replace OPFS model storage with filesystem cache at `$XDG_CACHE_HOME/textsift/<sha>/model_q4f16.onnx` (default `~/.cache/textsift/`). Browser path stays OPFS.
- Backend selection: darwin → Metal-direct, others → Dawn (or wgpu-native per (1) above).
- Parse ONNX once at warmup, upload weights to GPU buffers, then forward() per request.

The forward orchestration in `tests/native/forward-metal.js` and `tests/native/dawn/forward-dawn.js` is the working dispatch sequence — mirror it in the production code.

## Decision matrix

| Linux Dawn outcome | Backends shipped | Linux/Win speed | Eng cost |
|---|---|---|---|
| Works | Mac=Metal, others=Dawn | ≈ browser | low (just package) |
| Hangs, ship floor | Mac=Metal, others=wgpu-native | ~2× browser slower | low |
| Hangs, write Vulkan | Mac=Metal, Linux=Vulkan, Win=Dawn-or-D3D | ~1.45× browser faster | high (15 SPIR-V kernels + Linux GPU CI) |

## Reproduction commands

```sh
# Build the native .node (Mac only — needs Linux equivalent)
cd packages/textsift && bash scripts/build-native.sh

# Conformance: all 15 kernels byte-equal vs browser fixture
node tests/native/conformance/all.test.js                  # via wgpu-native
node tests/native/metal/conformance-all.js                 # via Metal-direct (Mac)

# Bench: end-to-end forward
T=32 node tests/native/forward.js                          # wgpu-native
T=32 node tests/native/forward-metal.js                    # Metal-direct (Mac)
T=8  node tests/native/dawn/forward-dawn.js                # Dawn — currently hangs

# Dawn smoke (proves the basic path works)
node tests/native/dawn/smoke-rms-norm.js
node tests/native/dawn/smoke-multi.js
```

## Architectural notes

- The browser `WebGpuBackend` at `packages/textsift/src/browser/backends/webgpu.ts` uses the same dispatch sequence we mirror in the Node forward files. Eventually we should run that exact code in Node by exposing Dawn's `GPU` instance as `globalThis.navigator.gpu` — but only after the Dawn hang is fixed.
- Naga rejects `ptr<storage, ...>` as function arguments while Tint accepts it. The 5 affected kernels (`embed_lookup_int4`, `matmul_int4_fp16_f16`, `matmul_int4_f32_f32`, `qmoe_gate_up`, `qmoe_down_scatter`) have the int4 access helpers inlined for Naga compatibility. Dawn uses Tint, so the inlining isn't strictly needed there but doesn't hurt.
- The `webgpu` npm package's `mapAsync` is implemented via Dawn's `AsyncRunner.cpp` which uses `setImmediate` to pump `wgpuInstanceProcessEvents`. The hang on Mac is likely either a binding bug or a Metal backend issue under heavy command-buffer load. The fix belongs in the binding (Dawn upstream), not in user-side timing yields. See https://github.com/webgpu-native/webgpu-headers/issues/199 for the upstream WGPUFuture work that should eventually replace this pump.
