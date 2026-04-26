// Vulkan-direct backend (Linux/Windows). Hand-tuned GLSL kernels via a
// thin C bridge (see vulkan/bridge.h, vulkan/bridge.c). Kernels live
// in vulkan/shaders/*.comp.glsl and pre-compile to SPIR-V at build
// time via glslangValidator (see scripts/build-native.sh).
//
// Why hand-tuned: we control loop unrolling, workgroup memory layout,
// and storage buffer access patterns directly. ~1.95× faster than
// Tint/Naga's WGSL→SPIR-V on Iris Xe at T=32 (mirror of Mac's Metal-
// direct gain over Tint's WGSL→MSL).
//
// Public API mirrors metal_backend.zig 1:1 so JS-side orchestration is
// identical between platforms.

const std = @import("std");

pub const cb = @cImport({
    @cInclude("vulkan/bridge.h");
});

pub const VulkanError = error{
    BackendCreateFailed,
    PipelineCreateFailed,
    BufferCreateFailed,
    EncoderCreateFailed,
    UnknownKernel,
    BindingMismatch,
    OutOfMemory,
};

// Per-shader metadata: SPIR-V bytes (embedded at build time), number
// of storage-buffer slots the kernel uses (= descriptor set bindings
// 0..N-1), and push-constant block size (≤128 B; replaces WGSL's
// uniform binding).
pub const PipelineMeta = struct {
    spv: []const u8,
    num_storage_buffers: u32,
    push_constant_size: u32,
};

const ShaderEntry = struct { name: []const u8, meta: PipelineMeta };

// All 15 kernels. SPIR-V is compiled from src/native/vulkan/shaders/*.comp.glsl
// at build time (build-native.sh runs glslangValidator) and @embedFile'd here.
//
// Binding count notes:
//   - WGSL declares uniforms at binding(0) and SSBOs at binding(1..N+1).
//     We collapsed uniforms into push constants, so SSBO slots start at
//     binding(0). num_storage_buffers below counts SSBOs only.
//   - cast_f32_to_fp16_scaled merges Dims+Scale into one 32-byte push block.
//   - banded_attention / qmoe_*'s Dims structs are 32 bytes (8 u32 fields).
pub const SHADERS = [_]ShaderEntry{
    .{ .name = "rms_norm", .meta = .{
        .spv = @embedFile("vulkan/shaders/rms_norm.comp.spv"),
        .num_storage_buffers = 3, .push_constant_size = 16,
    } },
    .{ .name = "zero_f32", .meta = .{
        .spv = @embedFile("vulkan/shaders/zero_f32.comp.spv"),
        .num_storage_buffers = 1, .push_constant_size = 16,
    } },
    .{ .name = "cast_fp16_to_f32", .meta = .{
        .spv = @embedFile("vulkan/shaders/cast_fp16_to_f32.comp.spv"),
        .num_storage_buffers = 2, .push_constant_size = 16,
    } },
    .{ .name = "cast_f32_to_fp16_scaled", .meta = .{
        .spv = @embedFile("vulkan/shaders/cast_f32_to_fp16_scaled.comp.spv"),
        .num_storage_buffers = 2, .push_constant_size = 32,
    } },
    .{ .name = "add_fp16", .meta = .{
        .spv = @embedFile("vulkan/shaders/add_fp16.comp.spv"),
        .num_storage_buffers = 3, .push_constant_size = 16,
    } },
    .{ .name = "swiglu_clamp", .meta = .{
        .spv = @embedFile("vulkan/shaders/swiglu_clamp.comp.spv"),
        .num_storage_buffers = 2, .push_constant_size = 16,
    } },
    .{ .name = "rope_apply", .meta = .{
        .spv = @embedFile("vulkan/shaders/rope_apply.comp.spv"),
        .num_storage_buffers = 3, .push_constant_size = 16,
    } },
    .{ .name = "matmul_int4_fp16_f16", .meta = .{
        .spv = @embedFile("vulkan/shaders/matmul_int4_fp16_f16.comp.spv"),
        .num_storage_buffers = 6, .push_constant_size = 16,
    } },
    .{ .name = "matmul_int4_f32_f32", .meta = .{
        .spv = @embedFile("vulkan/shaders/matmul_int4_f32_f32.comp.spv"),
        .num_storage_buffers = 6, .push_constant_size = 16,
    } },
    .{ .name = "embed_lookup_int4", .meta = .{
        .spv = @embedFile("vulkan/shaders/embed_lookup_int4.comp.spv"),
        .num_storage_buffers = 5, .push_constant_size = 16,
    } },
    .{ .name = "add_rmsnorm_fp16_to_f32", .meta = .{
        .spv = @embedFile("vulkan/shaders/add_rmsnorm_fp16_to_f32.comp.spv"),
        .num_storage_buffers = 5, .push_constant_size = 16,
    } },
    .{ .name = "router_topk", .meta = .{
        .spv = @embedFile("vulkan/shaders/router_topk.comp.spv"),
        .num_storage_buffers = 3, .push_constant_size = 16,
    } },
    .{ .name = "banded_attention", .meta = .{
        .spv = @embedFile("vulkan/shaders/banded_attention.comp.spv"),
        .num_storage_buffers = 6, .push_constant_size = 32,
    } },
    .{ .name = "qmoe_gate_up", .meta = .{
        .spv = @embedFile("vulkan/shaders/qmoe_gate_up.comp.spv"),
        .num_storage_buffers = 7, .push_constant_size = 32,
    } },
    .{ .name = "qmoe_down_scatter", .meta = .{
        .spv = @embedFile("vulkan/shaders/qmoe_down_scatter.comp.spv"),
        .num_storage_buffers = 8, .push_constant_size = 32,
    } },
};

fn shaderMeta(name: []const u8) ?PipelineMeta {
    for (SHADERS) |entry| {
        if (std.mem.eql(u8, entry.name, name)) return entry.meta;
    }
    return null;
}

pub const Backend = struct {
    handle: cb.TsVkBackend,
    pipelines: std.StringHashMap(cb.TsVkPipeline),
};

pub fn createBackend() VulkanError!*Backend {
    var err_buf: [1024]u8 = undefined;
    @memset(&err_buf, 0);
    const h = cb.ts_vk_create_backend(&err_buf, err_buf.len) orelse {
        std.debug.print("vulkan: create_backend failed: {s}\n", .{std.mem.sliceTo(&err_buf, 0)});
        return VulkanError.BackendCreateFailed;
    };

    const b = std.heap.c_allocator.create(Backend) catch {
        cb.ts_vk_destroy_backend(h);
        return VulkanError.OutOfMemory;
    };
    b.* = .{
        .handle = h,
        .pipelines = std.StringHashMap(cb.TsVkPipeline).init(std.heap.c_allocator),
    };
    return b;
}

pub fn destroyBackend(b: *Backend) void {
    var it = b.pipelines.iterator();
    while (it.next()) |entry| {
        cb.ts_vk_pipeline_release(b.handle, entry.value_ptr.*);
        std.heap.c_allocator.free(entry.key_ptr.*);
    }
    b.pipelines.deinit();
    cb.ts_vk_destroy_backend(b.handle);
    std.heap.c_allocator.destroy(b);
}

pub fn deviceName(b: *Backend) [*:0]const u8 {
    return @ptrCast(cb.ts_vk_device_name(b.handle));
}

const PipeAndMeta = struct {
    p: cb.TsVkPipeline,
    meta: PipelineMeta,
};

fn getOrCreatePipeline(b: *Backend, name: []const u8) VulkanError!PipeAndMeta {
    const meta = shaderMeta(name) orelse return VulkanError.UnknownKernel;

    if (b.pipelines.get(name)) |p| return .{ .p = p, .meta = meta };

    var err_buf: [1024]u8 = undefined;
    @memset(&err_buf, 0);

    // SPIR-V requires uint32_t alignment for vkCreateShaderModule, but
    // @embedFile in Zig 0.15 doesn't guarantee any specific alignment —
    // the bytes land at whatever offset the linker picks in .rodata,
    // which is fine for some shaders and breaks for others. Copy into
    // a u32-aligned scratch buffer (allocator.alloc(u32, ...) is
    // naturally 4-byte aligned) before handing to the bridge.
    const word_count = (meta.spv.len + 3) / 4;
    const spv_words = std.heap.c_allocator.alloc(u32, word_count) catch return VulkanError.OutOfMemory;
    defer std.heap.c_allocator.free(spv_words);
    const spv_bytes = std.mem.sliceAsBytes(spv_words);
    @memcpy(spv_bytes[0..meta.spv.len], meta.spv);

    const p = cb.ts_vk_pipeline_create(
        b.handle,
        spv_words.ptr,
        meta.spv.len,
        "main",
        meta.num_storage_buffers,
        meta.push_constant_size,
        &err_buf,
        err_buf.len,
    ) orelse {
        std.debug.print("vulkan: pipeline '{s}' failed: {s}\n", .{ name, std.mem.sliceTo(&err_buf, 0) });
        return VulkanError.PipelineCreateFailed;
    };

    const key = std.heap.c_allocator.dupe(u8, name) catch {
        cb.ts_vk_pipeline_release(b.handle, p);
        return VulkanError.OutOfMemory;
    };
    b.pipelines.put(key, p) catch {
        std.heap.c_allocator.free(key);
        cb.ts_vk_pipeline_release(b.handle, p);
        return VulkanError.OutOfMemory;
    };
    return .{ .p = p, .meta = meta };
}

pub fn createBuffer(b: *Backend, bytes: []const u8) VulkanError!cb.TsVkBuffer {
    const buf = cb.ts_vk_buffer_create(b.handle, bytes.len, bytes.ptr) orelse
        return VulkanError.BufferCreateFailed;
    return buf;
}

pub fn createEmptyBuffer(b: *Backend, len: usize) VulkanError!cb.TsVkBuffer {
    const buf = cb.ts_vk_buffer_create(b.handle, len, null) orelse
        return VulkanError.BufferCreateFailed;
    return buf;
}

pub fn releaseBuffer(b: *Backend, buf: cb.TsVkBuffer) void {
    cb.ts_vk_buffer_release(b.handle, buf);
}

pub fn writeBuffer(buf: cb.TsVkBuffer, offset: usize, data: []const u8) void {
    cb.ts_vk_buffer_write(buf, offset, data.ptr, data.len);
}

pub fn readBuffer(buf: cb.TsVkBuffer, offset: usize, out: []u8) void {
    cb.ts_vk_buffer_read(buf, offset, out.ptr, out.len);
}

/// One-shot dispatch: encode + submit + wait. Used for conformance
/// tests where each kernel is measured in isolation. For real forward,
/// use `beginEncoder` / `enqueueOnEncoder` / `submitAndReadback` to
/// batch all kernels under one command buffer.
pub fn dispatchOneShot(
    b: *Backend,
    name: []const u8,
    bindings: []const cb.TsVkBuffer,
    push_data: []const u8,
    grid: [3]u32,
) VulkanError!void {
    const pipe = try getOrCreatePipeline(b, name);
    if (bindings.len != pipe.meta.num_storage_buffers) return VulkanError.BindingMismatch;
    if (push_data.len != pipe.meta.push_constant_size) return VulkanError.BindingMismatch;

    const e = cb.ts_vk_encoder_begin(b.handle) orelse return VulkanError.EncoderCreateFailed;
    cb.ts_vk_encoder_dispatch(
        e,
        pipe.p,
        bindings.ptr,
        @intCast(bindings.len),
        if (push_data.len > 0) push_data.ptr else null,
        @intCast(push_data.len),
        grid[0], grid[1], grid[2],
    );
    cb.ts_vk_encoder_submit_and_wait(e);
}

/// Encoder-batched dispatch handle. One Vulkan command buffer + one
/// descriptor pool accumulate every kernel call until submit.
/// Equivalent to metal_backend.zig's Encoder.
pub const Encoder = struct {
    backend: *Backend,
    handle: cb.TsVkEncoder,
};

pub fn beginEncoder(b: *Backend) VulkanError!*Encoder {
    const e = cb.ts_vk_encoder_begin(b.handle) orelse return VulkanError.EncoderCreateFailed;
    const wrap = std.heap.c_allocator.create(Encoder) catch {
        cb.ts_vk_encoder_release(e);
        return VulkanError.OutOfMemory;
    };
    wrap.* = .{ .backend = b, .handle = e };
    return wrap;
}

pub fn enqueueOnEncoder(
    e: *Encoder,
    name: []const u8,
    bindings: []const cb.TsVkBuffer,
    push_data: []const u8,
    grid: [3]u32,
) VulkanError!void {
    const pipe = try getOrCreatePipeline(e.backend, name);
    if (bindings.len != pipe.meta.num_storage_buffers) return VulkanError.BindingMismatch;
    if (push_data.len != pipe.meta.push_constant_size) return VulkanError.BindingMismatch;

    cb.ts_vk_encoder_dispatch(
        e.handle,
        pipe.p,
        bindings.ptr,
        @intCast(bindings.len),
        if (push_data.len > 0) push_data.ptr else null,
        @intCast(push_data.len),
        grid[0], grid[1], grid[2],
    );
}

/// End the encoder, submit, fence-wait, then read back `out.len` bytes
/// from `out_buf` starting at `offset`. Releases the encoder handle.
pub fn submitAndReadback(
    e: *Encoder,
    out_buf: cb.TsVkBuffer,
    offset: usize,
    out: []u8,
) void {
    cb.ts_vk_encoder_submit_and_wait(e.handle);
    cb.ts_vk_buffer_read(out_buf, offset, out.ptr, out.len);
    std.heap.c_allocator.destroy(e);
}
