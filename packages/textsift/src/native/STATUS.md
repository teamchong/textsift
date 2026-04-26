# Native binding status (issue #79)

Snapshot end-of-session 2026-04-25.

## Done

### Build pipeline
- Zig (`packages/textsift/src/native/{napi,wgpu_init}.zig`) → `.node` shared library via `scripts/build-native.sh`
- `wgpu-native` v29 vendored per-platform via `scripts/fetch-wgpu-native.sh`
- `npm run build:native` does both end-to-end

### NAPI surface
- `getAdapterInfo()` → adapter info object
- `getDeviceInfo()` → adapter + granted limits
- `roundtripBuffer(bytes)` → host→GPU→host validation
- `dispatchDouble(Float32Array)` → first compute dispatch
- `matmulF32(a, b, m, n, k)` → multi-buffer dispatch with uniform
- `dispatchByName(name, uniform, extras, inputs, output, dispatch)` → one-shot generic dispatch
- `dispatchRmsnorm(...)` → typed RMSNorm wrapper
- **`createBackend()` / `destroyBackend(handle)`** → persistent instance/adapter/device/queue
- **`backendDispatch(handle, ...)`** → fast generic dispatch (no per-call adapter init)
- **`benchDispatch(handle, ..., iters)`** → setup-once / run-N timing harness; returns Float64Array

### Conformance
**ALL 15 shaders pass conformance vs browser WebGPU dump:**

```
rms_norm                  byte-equal
zero_f32                  byte-equal
cast_fp16_to_f32          byte-equal
cast_f32_to_fp16_scaled   byte-equal
add_fp16                  byte-equal
swiglu_clamp              byte-equal
rope_apply                byte-equal
matmul_int4_fp16_f16      byte-equal   ← workhorse (Q/K/V/O proj)
matmul_int4_f32_f32       byte-equal   ← classifier head
embed_lookup_int4         byte-equal
add_rmsnorm_fp16_to_f32   byte-equal   ← fused residual+norm
router_topk               byte-equal
banded_attention          byte-equal   ← largest kernel
qmoe_gate_up              byte-equal
qmoe_down_scatter         within 8e-7 relative drift   (atomicCAS commit order)
```

Reproduce: `npx playwright test conformance/dump-fixtures.spec.ts && node tests/native/conformance/all.test.js`

### Per-dispatch microbench

Two regimes — per-call latency (`chain=1`) and per-100-dispatch chunk (`chain=100`, closer to a real forward pass that issues ~133 dispatches per submit):

| Regime | Geomean native / browser | Interpretation |
|---|---|---|
| `chain=1` | 0.84× | native is **1.20× faster** per dispatch (low overhead wins) |
| `chain=100` | 1.25× | native is **1.25× slower** per 100-dispatch chunk |

Why the divergence: at `chain=1`, native wins by skipping `wgpuDevicePoll(wait=true)` (which adds ~1.27 ms unconditional latency). At `chain=100`, the per-call overhead amortizes and Naga's MSL codegen vs Tint's MSL codegen starts to matter — Naga produces slower MSL for some simple kernels (rms_norm 1.66×, basic casts ~1.5×) while matching or beating Tint on the heavy ones (qmoe pair 0.97×, embed_lookup 0.87×).

**End-to-end projection:** real forward has ~133 dispatches per submit but with *different* shaders (each `setPipeline` may add cost). Realistic estimate: native is **roughly even** with browser end-to-end. Combined with browser textsift WebGPU being 2.6–3.7× faster than transformers.js, native projects to **~2–3× faster than tjs end-to-end** (down from my earlier 3× extrapolation that was based on the misleading `chain=1` win).

Reproduce:
```sh
npx playwright test conformance/bench-shaders.spec.ts                              # chain=1
node tests/native/bench/shaders-bench.js                                            # chain=1
CHAIN=100 npx playwright test conformance/bench-shaders.spec.ts && \
  CHAIN=100 node tests/native/bench/shaders-bench.js                                # chain=100
```

## Architecture

```
packages/textsift/src/native/
├── napi.zig         ← NAPI bindings (Zig calls into Node-API C ABI)
├── wgpu_init.zig    ← wgpu-native wrappers, dispatchKernel, Backend
└── shaders/         ← 15 .wgsl files, single source of truth
                       (browser shaders.ts is generated from these via
                        scripts/gen-shaders-ts.mjs)
```

WGSL files are loaded via `@embedFile` at compile time (native) and via the generated `shaders.ts` (browser). Edit a `.wgsl`, then `npm run build:native` and `node packages/textsift/scripts/gen-shaders-ts.mjs` to refresh both consumers.

## Naga-vs-Tint shader compatibility

`load_byte` / `load_nibble` helpers in `int4_access.wgsl` were inlined at every call site because Naga (wgpu-native) rejects `ptr<storage, ...>` as a function argument while Tint (browser Dawn) accepts it. The math simplifies cleanly:

- `load_nibble(arr, n)` → `(arr[n >> 3] >> ((n & 7) * 4)) & 0xF`
- `load_byte(arr, b)` → `(arr[b >> 2] >> ((b & 3) * 8)) & 0xFF`

Affected shaders: `embed_lookup_int4`, `matmul_int4_fp16_f16`, `matmul_int4_f32_f32`, `qmoe_gate_up`, `qmoe_down_scatter`. Browser path verified still byte-equal — same math, no helper indirection.

## Foundation laid for next session

`wgpu_init.zig` now exports:
- `createPersistentBuffer(backend, bytes)` → `c.WGPUBuffer` — uploads via staged copy, returns the buffer handle. Use for model weights at warmup.
- `releasePersistentBuffer(buf)` — release on dispose.
- `dispatchOnBackendMixed(...)` — accepts a tagged-union `StorageBinding` per input, so the caller can pass either inline bytes (transient activations) OR a persistent buffer handle (weights). Same function-call shape as `dispatchOnBackend`, just doesn't re-upload weights every dispatch.

What's missing to make this usable from JS:
- NAPI bindings: `napiCreateBuffer(handle, bytes) → BigInt`, `napiReleaseBuffer(handle, ptr)`, `napiDispatchByBuffers(handle, name, uniform, extras, mixed_inputs[], output, dispatch)`. Each input entry in `mixed_inputs` is either `{binding, bytes}` (uploaded fresh) or `{binding, bufPtr, byteLen}` (persistent).
- JS orchestration that calls these.

## Not done — next session

### End-to-end forward — DONE (synthetic weights)

`tests/native/forward.js` ports the full `WebGpuBackend.forward()`
to JS, calling the new encoder-batched API:
  beginEncoder → ~140 enqueueDispatch → submitAndReadback

Pipeline cache on the backend means each shader compiles once;
all subsequent dispatches reuse the cached pipeline + bgl.

**Measured (M3 Pro, T=80, synthetic weights at production dims):**

| Backend | Forward latency |
|---|---:|
| transformers.js WebGPU | 56.4 ms (from benchmarks.mdx) |
| **Native textsift** | **43.9 ms** |
| Browser textsift WebGPU | 22.0 ms (from benchmarks.mdx) |

**Native is 1.28× faster than transformers.js end-to-end.** Browser
textsift remains 2× faster than native — Naga's MSL codegen is
slower than Tint's per kernel (consistent with the chain bench).

Iteration history:
- 143.7 ms — per-dispatch readback (each shader call mapped + read host-side)
- 50.2 ms — encoder-batched: one submit + one readback per forward
- **22.0 ms (T=32) / 43.9 ms (T=80) — pipeline cache: each shader compiles once on the backend, reused per dispatch**

Reproduce: `T=80 node tests/native/forward.js` (or any T).

### Original forward pass
The 15 shaders work in isolation. They are not yet wired into a `forward(tokenIds, attentionMask) → logits` entry point. Doing that requires replicating the orchestration in `packages/textsift/src/browser/backends/webgpu.ts:460` (the ~3000-line `WebGpuBackend.forward()`):

1. Allocate / pre-upload weight buffers per layer (~770 MB ONNX)
2. Create transient activation buffers (ping-pong between layers)
3. Per layer, in order:
   - Q/K/V projections (3× `matmul_int4_fp16_f16`)
   - `rope_apply` on Q and K
   - `banded_attention(Q, K, V, sinks, mask)`
   - O projection (1× `matmul_int4_fp16_f16`)
   - `add_rmsnorm_fp16_to_f32` (residual + norm)
   - `router_topk`
   - `qmoe_gate_up` → `swiglu_clamp` → `qmoe_down_scatter`
4. Final `rms_norm` + classifier head matmul
5. Read back logits

Two implementation paths:

**A) Zig orchestration** — proper end state. The whole forward pass lives in `wgpu_init.zig` as a single function; one `wgpuQueueSubmit` per forward; one `mapAsync` per forward. Best perf since encode + submit + map costs amortize across all ~100 dispatches.

**B) JS orchestration** — call `backendDispatch` from JS for each shader, hold intermediate buffer bytes in JS. Faster to write, slower to run (every dispatch crosses the NAPI boundary; readbacks between shaders).

Recommendation: A. ~2-4 days of careful work + parity testing against browser logits.

### Bench against transformers.js
Once forward is wired, bench `native.forward(tokenIds)` median latency against browser WebGPU and against tjs. Predicted from per-dispatch ratio:

- Browser textsift WebGPU vs tjs: 2.6–3.7× faster (proven; see `benchmarks.mdx`)
- Native vs browser: 1.20× per-dispatch microbench (proven this session)
- Predicted native vs tjs end-to-end: **~3× faster** (multiplicative)

The 1.20× per-dispatch win mainly comes from removing JS/V8 round-trip overhead in the dispatch loop. At forward scale (100 dispatches per call), this advantage scales linearly.

### CI matrix
Cross-platform builds for npm distribution:
- macos-arm64 (current dev host — works)
- macos-x86_64
- linux-x86_64 (glibc + musl)
- linux-aarch64
- windows-x86_64

GitHub Actions matrix + per-platform `optionalDependencies` package. Standard napi-rs / prebuildify pattern, but adapted to Zig's build script.
