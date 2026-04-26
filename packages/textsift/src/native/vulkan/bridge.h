// C-callable Vulkan-direct bridge. Implementation in bridge.c. Called
// from Zig via @cImport. Mirrors the shape of metal/bridge.h so the
// Zig wrapper layer can be platform-agnostic at the API surface.
//
// Conventions:
//  - Opaque handles (struct *) so Zig doesn't see Vulkan internals.
//  - Errors surface as NULL returns + a caller-provided err_out buffer
//    (same pattern as metal/bridge.h).
//  - Buffers are HOST_VISIBLE | HOST_COHERENT. Iris Xe's UMA makes
//    DEVICE_LOCAL+HOST_VISIBLE single-pool — no staging copies needed.
//  - Pipelines hold their VkPipelineLayout and VkDescriptorSetLayout;
//    they're cached by name in the Zig layer.
//  - Each dispatch allocates a fresh VkDescriptorSet from the encoder's
//    pool. Pool resets on encoder release. ~133 dispatches/forward
//    means we never blow the pool (sized at 1024 sets / 8192 binds).
//  - Push constants carry the per-dispatch uniform (≤128 B). Replaces
//    the WGSL `var<uniform>` binding.
//  - Memory barriers: bridge emits a COMPUTE→COMPUTE write→read
//    barrier after every dispatch automatically. Conservative but
//    correct. Tuning this is a Phase-9 optimization.

#ifndef TEXTSIFT_VULKAN_BRIDGE_H
#define TEXTSIFT_VULKAN_BRIDGE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TsVkBackendImpl* TsVkBackend;
typedef struct TsVkBufferImpl* TsVkBuffer;
typedef struct TsVkPipelineImpl* TsVkPipeline;
typedef struct TsVkEncoderImpl* TsVkEncoder;

// ── backend lifecycle ──
// Picks the highest-priority physical device (DISCRETE > INTEGRATED > CPU)
// supporting compute + the f16/16-bit-storage feature set.
// Returns NULL on failure; err_out (≥1024 B) carries the diagnostic.
TsVkBackend ts_vk_create_backend(char* err_out, size_t err_cap);
void ts_vk_destroy_backend(TsVkBackend b);
const char* ts_vk_device_name(TsVkBackend b);

// ── pipelines ──
// Build a compute pipeline from raw SPIR-V bytes (already compiled at
// build time via glslangValidator). num_storage_buffers = number of
// `layout(set=0, binding=K)` SSBO slots the kernel uses.
// push_constant_size = bytes of `layout(push_constant)` block (0 if
// the kernel uses no push constants). entry_point is "main" by GLSL
// convention.
TsVkPipeline ts_vk_pipeline_create(
    TsVkBackend b,
    const uint32_t* spv, size_t spv_len_bytes,
    const char* entry_point,
    uint32_t num_storage_buffers,
    uint32_t push_constant_size,
    char* err_out, size_t err_cap);
void ts_vk_pipeline_release(TsVkBackend b, TsVkPipeline p);

// ── buffers ──
// Allocate a HOST_VISIBLE | HOST_COHERENT buffer. If init_data is
// non-NULL it's memcpy'd in on creation. Buffer is bound in storage
// usage (USAGE_STORAGE_BUFFER | USAGE_TRANSFER_SRC | USAGE_TRANSFER_DST).
TsVkBuffer ts_vk_buffer_create(TsVkBackend b, size_t length, const void* init_data);
void ts_vk_buffer_release(TsVkBackend b, TsVkBuffer buf);
void ts_vk_buffer_write(TsVkBuffer buf, size_t offset, const void* data, size_t len);
void ts_vk_buffer_read(TsVkBuffer buf, size_t offset, void* dst, size_t len);
size_t ts_vk_buffer_length(TsVkBuffer buf);

// ── encoder (one command buffer + one descriptor pool) ──
// Begin recording. Returns NULL on cmd-buffer alloc / begin failure.
TsVkEncoder ts_vk_encoder_begin(TsVkBackend b);
// Discard without submitting (error path).
void ts_vk_encoder_release(TsVkEncoder e);

// Per-dispatch state: pipeline, optional push constants, buffer bindings
// for slots 0..num_storage_buffers-1. After dispatch, the bridge auto-
// emits a compute→compute write→read barrier.
//
// `bindings` is an array of TsVkBuffer handles, length = num_storage_buffers
// from the pipeline. They MUST be in binding order. NULL entries are an
// error (every slot must be bound).
//
// `push_data` is the push-constant block bytes (size must match the
// pipeline's push_constant_size). NULL only allowed if push_constant_size==0.
void ts_vk_encoder_dispatch(
    TsVkEncoder e,
    TsVkPipeline p,
    const TsVkBuffer* bindings, uint32_t num_bindings,
    const void* push_data, uint32_t push_data_len,
    uint32_t gx, uint32_t gy, uint32_t gz);

// End recording, submit, fence-wait, then release the encoder. Buffer
// reads after this return are guaranteed to see the dispatch results.
void ts_vk_encoder_submit_and_wait(TsVkEncoder e);

#ifdef __cplusplus
}
#endif

#endif
