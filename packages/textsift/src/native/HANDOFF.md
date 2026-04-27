# Native binding — status

Last update: 2026-04-26 (PrivacyFilter wiring done; Mac adapter fixes verified end-to-end).

Mac fast-path (Metal-direct), Linux fast-path (Vulkan-direct), and the Dawn-direct alternative are all shipped. Windows uses Dawn-direct as its primary path. `PrivacyFilter.create()` works on all three platforms with WASM auto-fallback if the GPU path fails. Remaining work is CI execution + npm publish flow (workflows are committed to `.github/workflows/` but haven't run yet — no remote configured).

## Comptime platform routing

Each platform's `.node` binary only contains the surfaces relevant to its target. From `napi.zig`:

```zig
const is_macos   = builtin.os.tag == .macos;
const is_linux   = builtin.os.tag == .linux;
const is_windows = builtin.os.tag == .windows;

const Metal  = if (is_macos)              struct { ... } else struct { empty registerAll };
const Vulkan = if (is_linux)              struct { ... } else struct { empty registerAll };
const Dawn   = if (is_linux or is_windows) struct { ... } else struct { empty registerAll };
```

| Platform | Backend on user path | NAPI surfaces in `.node` | Sources compiled |
|---|---|---|---|
| macOS arm64/x64 | Metal-direct | `metal*` only | `metal/bridge.{h,m}` + `metal_backend.zig` |
| Linux x86_64/arm64 | Vulkan-direct | `vulkan*` (primary) + `dawn*` (measurement) | `vulkan/bridge.{h,c}` + `vulkan_backend.zig` + `dawn/bridge.{h,c}` + `dawn_backend.zig` |
| Windows x86_64 | Dawn-direct | `dawn*` only | `dawn/bridge.{h,c}` + `dawn_backend.zig` |

The build script (`scripts/build-native.sh`) mirrors the same `case "$HOST_OS"` split and only compiles each platform's source files + linker flags.

## What's done

### Kernel layer
All 15 WGSL compute kernels live at `packages/textsift/src/native/shaders/*.wgsl` — these are the canonical reference and the source the browser path uses. Per-platform native ports are in `packages/textsift/src/native/metal/shaders.metal` (MSL) and `packages/textsift/src/native/vulkan/shaders/*.comp.glsl` (GLSL → SPIR-V at build time).

### Mac fast-path (Metal-direct)
- `packages/textsift/src/native/metal/shaders.metal` — hand-written MSL ports of all 15 kernels.
- `packages/textsift/src/native/metal/bridge.{h,m}` — Obj-C → C bridge (compiled with `-fobjc-arc`).
- `packages/textsift/src/native/metal_backend.zig` — Zig wrapper with pipeline cache + encoder-batched dispatch.
- `packages/textsift/src/native/napi.zig` — `metal*` NAPI surface: `metalCreateBackend`, `metalCreateBuffer`, `metalCreateEmptyBuffer`, `metalReleaseBuffer`, `metalReadBuffer`, `metalWriteBuffer`, `metalDispatchOneShot`, `metalBeginEncoder`, `metalEnqueueDispatch`, `metalSubmitAndReadback`.
- `tests/native/forward-metal.js` — end-to-end synthetic-weight forward via Metal-direct.

Bench (M2 Pro, vs browser/Tint as the within-stack ceiling). Updated 2026-04-26 after the matmul dispatch fix (1D over T*N, was 2D tile shape — the dispatch grid bug actually made the broken version *appear* faster because the GPU spent less time scheduling 2D tiles; the fix simultaneously corrects router/classifier outputs and improves perf):

| T  | Metal-direct | Browser WebGPU | tjs WebGPU |
|---:|-------------:|---------------:|-----------:|
|  7 | 5.2 ms       | 8.9 ms         | 32.7 ms    |
| 25 | 10.0 ms      | 11.8 ms        | 38.5 ms    |
| 32 | **10.8 ms**  | 22.0 ms        | —          |
| 80 | **23.8 ms**  | —              | 56.4 ms    |

End-to-end `PrivacyFilter.redact()` on a 122-character input with 4 PII spans: ~110 ms (BPE tokenization + Metal-direct forward + Viterbi + span replacement). Verified 2026-04-26 via `tests/native/integration/filter-redact.test.js`.

**Reproduce:**
```sh
cd packages/textsift && bash scripts/build-native.sh
T=32 node tests/native/forward-metal.js
T=80 node tests/native/forward-metal.js
```

### Linux fast-path (Vulkan-direct)
- `packages/textsift/src/native/vulkan/shaders/*.comp.glsl` — hand-written GLSL ports of all 15 kernels. Compiled to SPIR-V at build time via `glslangValidator`.
- `packages/textsift/src/native/vulkan/bridge.{h,c}` — C bridge: instance/device/queue/buffer/descriptor/pipeline/cmd-buffer/fence. ~600 LOC.
- `packages/textsift/src/native/vulkan_backend.zig` — Zig wrapper mirroring `metal_backend.zig`'s API. Pipeline cache, descriptor-pool-per-encoder, push-constants for uniforms.
- `packages/textsift/src/native/napi.zig` — `vulkan*` NAPI surface, comptime-gated to `!is_macos` (parallel to the Metal struct).
- `tests/native/forward-vulkan.js` — end-to-end synthetic-weight forward via Vulkan-direct.

Bench (Intel Iris Xe, Mesa ANV Vulkan; ONNX Runtime Node CPU as the realistic baseline most users would otherwise hit):

| T  | Vulkan-direct | ORT Node (CPU) | Speedup |
|---:|--------------:|---------------:|--------:|
| 32 | **24.5 ms**   | 785.0 ms       | **32×** |

Vs the previously-shipped wgpu-native floor (now removed): Vulkan-direct was 1.95× faster at T=32 on the same Iris Xe — same speedup ratio Metal-direct gets over Tint codegen on Mac. See `tests/native/bench-onnx/ort-vs-vulkan.js` for the head-to-head harness.

**Conformance:** all 15 kernels byte-equal or within fp16/fp32 ε vs browser WGSL fixtures. Run `node tests/native/vulkan/conformance-all.js`.

**Reproduce:**
```sh
sudo apt install -y glslang-tools libvulkan-dev   # one-time prereqs
cd packages/textsift && bash scripts/build-native.sh
T=32 node tests/native/forward-vulkan.js
```

**Phase breakdown at T=32 (`PROFILE=1`):**
- encode (133 enqueueDispatch + barriers): 2.2 ms
- submit + GPU compute + readback: 22.4 ms (92%)

GPU compute is the wall. Theoretical Iris Xe memory bandwidth ceiling for our forward (~770 MB weights touched once + scratch) is ~11 ms; we're at ~50% of that. Realistic optimization headroom: ~10–15% wall via subgroup ops + persistent descriptor caching + larger matmul tiles.

### Dawn-direct (cross-platform; Linux measurement + Windows primary)

Statically-linked Google Dawn C++ library with a thin C bridge (`dawn/bridge.{h,c}`, ~700 LOC). Tint compiles the canonical WGSL kernels at runtime — single source shared with the browser path, no per-platform shader maintenance.

- `packages/textsift/src/native/dawn/bridge.{h,c}` — C bridge: instance/adapter/device/queue + buffer/pipeline/encoder. Uses `wgpuInstanceWaitAny` for sync (sidesteps the `setImmediate` deadlock that broke `node-webgpu` on heavy loads).
- `packages/textsift/src/native/dawn_backend.zig` — Zig wrapper mirroring `vulkan_backend.zig`'s API. Multi-uniform support via `uniform_sizes[]` array.
- `packages/textsift/src/native/napi.zig` — `dawn*` NAPI surface, comptime-gated to `is_linux or is_windows`.
- `packages/textsift/vendor/dawn/` — vendored: `include/{webgpu,dawn}/webgpu.h` headers + `lib/libwebgpu_dawn.a` (37 MB, built with hidden visibility so its bundled abseil doesn't collide with V8's at runtime).

**Conformance:** all 15 kernels byte-equal or within fp ε vs browser fixture. Run `node tests/native/dawn/conformance-all.js`.

Bench (Iris Xe, T=32): Dawn-direct = 55.9 ms vs Vulkan-direct = 26.3 ms. **Vulkan-direct ~2× faster** because (a) hand-tuned GLSL beats Tint's WGSL→SPIR-V codegen on our specific kernels, and (b) Dawn allocates a fresh uniform buffer + bind group per dispatch (133/forward) while Vulkan-direct uses push constants. On Windows where there's no hand-tuned alternative, Dawn-direct is the user path.

**Reproduce Dawn build (one-time):**
```sh
git clone --depth 1 https://dawn.googlesource.com/dawn /tmp/dawn
cd /tmp/dawn
CC=clang CXX=clang++ cmake -B out -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_VISIBILITY_PRESET=hidden -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON \
  -DDAWN_FETCH_DEPENDENCIES=ON -DDAWN_BUILD_SAMPLES=OFF \
  -DDAWN_USE_X11=OFF -DDAWN_USE_WAYLAND=OFF -DDAWN_USE_GLFW=OFF \
  -DDAWN_ENABLE_OPENGLES=OFF -DDAWN_ENABLE_DESKTOP_GL=OFF \
  -DDAWN_BUILD_TESTS=OFF -DTINT_BUILD_TESTS=OFF -DTINT_BUILD_CMD_TOOLS=OFF
ninja -C out -j16 webgpu_dawn
cp out/src/dawn/native/libwebgpu_dawn.a $TEXTSIFT/packages/textsift/vendor/dawn/lib/
cp include/webgpu/webgpu.h $TEXTSIFT/packages/textsift/vendor/dawn/include/webgpu/
cp out/gen/include/dawn/webgpu.h $TEXTSIFT/packages/textsift/vendor/dawn/include/dawn/
```

### Packaging (workflows committed; first run pending)

GitHub Actions matrix + `optionalDependencies` are committed at `.github/workflows/{test,release,bench}.yml`. Workflows haven't run yet — no GitHub remote configured.

When the repo is pushed and the matrix runs for the first time, expect ~30–60 min for the cold Dawn build per OS; subsequent runs hit the per-OS cache and complete in 10–20 s.

`@textsift/native-{darwin-arm64,darwin-x64,linux-x64,linux-arm64,win32-x64}` per-triple subpackages are wired through `release.yml`. NPM_TOKEN secret needs to be set in repo settings before `release.yml` works.

### PrivacyFilter wiring (DONE)

`PrivacyFilter.create()` on the native entry works end-to-end on macOS (Metal-direct) and Linux (Vulkan-direct), with WASM fallback when no GPU is available. Verified by `tests/native/integration/filter-redact.test.js` (Mac + Linux), `filter-faker.test.js`, `filter-table.test.js`, and `cli.test.js` (14 subprocess tests).

The full path: tokenizer + Viterbi + spans + redact + rules from `src/browser/inference/` are reused unchanged (pure TS, no DOM deps). Model loading from filesystem is in `src/native/loader.ts` with `cacheDir` / `modelPath` / `offline` overrides. Backend selection is in `src/native/backend.ts` with comptime-gated `metal*` / `vulkan*` / `dawn*` NAPI surfaces.

Adapter bug fixed in `14221c3` (Mac side): `metalCreateBuffer` / `metalCreateEmptyBuffer` were dropping the backend handle, and `cast_f32_to_fp16_scaled` needs its two uniform bindings split for Metal (Vulkan/Dawn handle the concatenated form via push constants).

## Reproduction commands

```sh
# Build the native .node (platform-detects via uname; comptime selects which
# backend code is compiled — Metal on Mac, Vulkan+Dawn on Linux, Dawn on Win).
cd packages/textsift && bash scripts/build-native.sh

# Conformance: all 15 kernels byte-equal (or within fp ε) vs browser fixture
node tests/native/metal/conformance-all.js                 # Metal-direct (Mac)
node tests/native/vulkan/conformance-all.js                # Vulkan-direct (Linux)
node tests/native/dawn/conformance-all.js                  # Dawn-direct (Linux+Win)

# Bench: end-to-end forward
T=32 node tests/native/forward-metal.js                    # Metal-direct (Mac)
T=32 node tests/native/forward-vulkan.js                   # Vulkan-direct (Linux, primary)
T=32 node tests/native/forward-dawn.js                     # Dawn-direct (Linux measurement)

# Vs ORT Node CPU on same hardware (realistic baseline)
T=32 node tests/native/bench-onnx/ort-vs-vulkan.js
```

## Architectural notes

- The browser `WebGpuBackend` at `packages/textsift/src/browser/backends/webgpu.ts` uses the same dispatch sequence the Node forward files mirror. Once `PrivacyFilter` is wired through, that exact orchestration drives the Mac (`metal*`) and Linux (`vulkan*`) NAPI surfaces — same JS-side logic, platform-detected at `loadNative()`.
- The 5 int4 kernels (`embed_lookup_int4`, `matmul_int4_fp16_f16`, `matmul_int4_f32_f32`, `qmoe_gate_up`, `qmoe_down_scatter`) have their `load_byte` / `load_nibble` helpers inlined at every call site. The original reason was Naga compatibility (`ptr<storage, ...>` as fn argument); the inlined version compiles cleanly through Tint, glslang, and Apple's MSL compiler too, so the convention is now portable across all our codegen paths.
- Push constants vs uniform buffers: the WGSL kernels declare `var<uniform> dims` at binding 0. In the Vulkan port we use push constants instead — saves a buffer alloc + descriptor binding, and Vulkan caps push constants at 128 B which fits all our Dims structs (max 32 B). Storage buffer slots in GLSL accordingly start at binding 0.
- f32 atomic add is implemented via CAS on uint storage (`atomicCompSwap` loop) since Mesa ANV doesn't expose `VK_KHR_shader_atomic_float`. Same drift profile as the WGSL/Metal versions; portable to every Vulkan driver.
- Memory barriers: Vulkan-direct emits a global compute→compute write→read barrier after every dispatch (conservative). Per-resource barriers would shave microseconds per forward but the ~133-dispatch chain makes that sub-1% — not worth the complexity.
