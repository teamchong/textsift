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
Native vs browser, same WGSL, same fixture, same warmup+iters protocol:

```
geomean native/browser across 15 shaders: 0.84x
→ native is 1.20× faster than browser per dispatch
```

Win came from skipping `wgpuDevicePoll(wait=true)` between submit and `mapAsync` (added ~1.27 ms unconditional latency). The mapAsync callback already implies GPU completion; `processEvents` just dispatches it.

Reproduce: `npx playwright test conformance/bench-shaders.spec.ts && node tests/native/bench/shaders-bench.js`

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

## Not done — next session

### End-to-end forward pass
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
