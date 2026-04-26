// C-callable Dawn-direct bridge. Implementation in bridge.c. Called
// from Zig via @cImport. Mirrors the shape of vulkan/bridge.h so the
// Zig wrapper layer is structurally identical between Vulkan-direct
// and Dawn-direct backends.
//
// Differences from vulkan/bridge.h:
//   - Pipelines accept WGSL source instead of pre-compiled SPIR-V.
//     Dawn's Tint compiles WGSL → SPIR-V/MSL/HLSL at runtime.
//   - "uniform_data" parameter feeds a uniform buffer (Dawn lacks push
//     constants in the WebGPU spec); the bridge manages an internal
//     uniform pool to keep dispatch latency low.
//   - Async ops use Dawn's WGPUFuture API; the bridge uses
//     wgpuInstanceWaitAny with a sentinel timeout to make them sync.

#ifndef TEXTSIFT_DAWN_BRIDGE_H
#define TEXTSIFT_DAWN_BRIDGE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TsDawnBackendImpl* TsDawnBackend;
typedef struct TsDawnBufferImpl* TsDawnBuffer;
typedef struct TsDawnPipelineImpl* TsDawnPipeline;
typedef struct TsDawnEncoderImpl* TsDawnEncoder;

// ── backend lifecycle ──
TsDawnBackend ts_dawn_create_backend(char* err_out, size_t err_cap);
void ts_dawn_destroy_backend(TsDawnBackend b);
const char* ts_dawn_device_name(TsDawnBackend b);

// ── pipelines ──
// Build a compute pipeline from raw WGSL source.
//   num_storage_buffers = number of `var<storage, ...>` SSBO bindings.
//   num_uniforms        = number of `var<uniform>` bindings (most kernels = 1).
//   uniform_sizes       = byte size of each uniform block, in WGSL binding
//                         order. Array of length num_uniforms; ignored if
//                         num_uniforms == 0.
//   entry_point         = WGSL entry fn name (typically "main").
//
// Dawn uses auto-layout, so the WGSL's `@binding(N)` indices determine
// the bind-group layout. The bridge's job is just to allocate the
// matching uniform buffers per dispatch and bind them.
TsDawnPipeline ts_dawn_pipeline_create(
    TsDawnBackend b,
    const char* wgsl_source, size_t wgsl_len,
    const char* entry_point,
    uint32_t num_storage_buffers,
    uint32_t num_uniforms,
    const uint32_t* uniform_sizes,
    char* err_out, size_t err_cap);
void ts_dawn_pipeline_release(TsDawnBackend b, TsDawnPipeline p);

// ── buffers ──
TsDawnBuffer ts_dawn_buffer_create(TsDawnBackend b, size_t length, const void* init_data);
void ts_dawn_buffer_release(TsDawnBackend b, TsDawnBuffer buf);
void ts_dawn_buffer_write(TsDawnBackend b, TsDawnBuffer buf, size_t offset, const void* data, size_t len);
void ts_dawn_buffer_read(TsDawnBackend b, TsDawnBuffer buf, size_t offset, void* dst, size_t len);

// ── encoder ──
TsDawnEncoder ts_dawn_encoder_begin(TsDawnBackend b);
void ts_dawn_encoder_release(TsDawnEncoder e);

// One dispatch: bind storage buffers in slot order (matching the
// WGSL `@binding(N)` indices), write `uniform_data` to a fresh
// (or pooled) uniform buffer, dispatch, then encode.
void ts_dawn_encoder_dispatch(
    TsDawnEncoder e,
    TsDawnPipeline p,
    const TsDawnBuffer* bindings, uint32_t num_bindings,
    const void* uniform_data, uint32_t uniform_size,
    uint32_t gx, uint32_t gy, uint32_t gz);

// End the compute pass, finish the command buffer, submit, and wait
// for the queue to drain. After this returns, buffer reads see the
// dispatch results.
void ts_dawn_encoder_submit_and_wait(TsDawnEncoder e);

#ifdef __cplusplus
}
#endif

#endif
