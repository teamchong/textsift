// C-callable Metal bridge. Implementation in bridge.m (Objective-C);
// called from Zig via @cImport. Each function wraps the corresponding
// Metal Obj-C call into a plain-C ABI handle (opaque pointers as void*)
// so Zig can manage Metal resources without knowing about the
// Objective-C runtime.

#ifndef TEXTSIFT_METAL_BRIDGE_H
#define TEXTSIFT_METAL_BRIDGE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles. Each is an autoreleased Obj-C object retained on
// creation; release via the matching ts_metal_*_release.
typedef void* TsMetalDevice;
typedef void* TsMetalQueue;
typedef void* TsMetalLibrary;
typedef void* TsMetalPipeline;
typedef void* TsMetalBuffer;
typedef void* TsMetalCommandBuffer;
typedef void* TsMetalEncoder;

// ── device + queue ──
TsMetalDevice ts_metal_create_device(void);
void ts_metal_device_release(TsMetalDevice d);
const char* ts_metal_device_name(TsMetalDevice d);
TsMetalQueue ts_metal_device_queue(TsMetalDevice d);
void ts_metal_queue_release(TsMetalQueue q);

// ── library + compute pipelines ──
// Compile MSL source into a library. Returns NULL on compile failure;
// `err_out` (must be at least 1024 bytes) is filled with the compiler
// diagnostic.
TsMetalLibrary ts_metal_library_from_source(
    TsMetalDevice d, const char* msl_source, size_t source_len,
    char* err_out, size_t err_cap);
void ts_metal_library_release(TsMetalLibrary lib);

// Get a compute pipeline state for entry-point `name` from `lib`.
TsMetalPipeline ts_metal_pipeline_for(
    TsMetalDevice d, TsMetalLibrary lib, const char* entry_name,
    char* err_out, size_t err_cap);
void ts_metal_pipeline_release(TsMetalPipeline p);
size_t ts_metal_pipeline_threadgroup_size(TsMetalPipeline p);

// ── buffers ──
// Allocate a buffer with `length` bytes. If `init_data` is non-NULL,
// it's copied into the buffer; otherwise the buffer is zeroed by the
// driver. Storage mode = shared so we can read/write from CPU+GPU.
TsMetalBuffer ts_metal_buffer_create(
    TsMetalDevice d, size_t length, const void* init_data);
void ts_metal_buffer_release(TsMetalBuffer b);
void ts_metal_buffer_write(TsMetalBuffer b, size_t offset, const void* data, size_t len);
void ts_metal_buffer_read(TsMetalBuffer b, size_t offset, void* dst, size_t len);
size_t ts_metal_buffer_length(TsMetalBuffer b);

// ── command encoding ──
TsMetalCommandBuffer ts_metal_queue_command_buffer(TsMetalQueue q);
void ts_metal_command_buffer_release(TsMetalCommandBuffer cb);
TsMetalEncoder ts_metal_command_buffer_compute_encoder(TsMetalCommandBuffer cb);
void ts_metal_encoder_set_pipeline(TsMetalEncoder enc, TsMetalPipeline p);
// Bind buffer at `index` (Metal uses indices 0..N for buffer slots).
// `offset` is byte offset into the buffer.
void ts_metal_encoder_set_buffer(TsMetalEncoder enc, TsMetalBuffer b, size_t offset, uint32_t index);
// Bind raw bytes (uniform data ≤ 4 KiB recommended). Avoids needing
// a separate buffer for tiny per-dispatch uniforms.
void ts_metal_encoder_set_bytes(TsMetalEncoder enc, const void* data, size_t len, uint32_t index);
// Dispatch a 3D grid of threads, with the given threadgroup size.
void ts_metal_encoder_dispatch(
    TsMetalEncoder enc,
    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    uint32_t tg_x, uint32_t tg_y, uint32_t tg_z);
void ts_metal_encoder_end(TsMetalEncoder enc);
void ts_metal_command_buffer_commit(TsMetalCommandBuffer cb);
// Block until the command buffer finishes executing on the GPU.
void ts_metal_command_buffer_wait(TsMetalCommandBuffer cb);

// Copy `length` bytes from `src` (offset `src_off`) to `dst` (offset 0).
// Recorded into the command buffer; happens after dispatch.
void ts_metal_encoder_blit_copy(
    TsMetalCommandBuffer cb,
    TsMetalBuffer src, size_t src_off,
    TsMetalBuffer dst, size_t dst_off,
    size_t length);

#ifdef __cplusplus
}
#endif

#endif
