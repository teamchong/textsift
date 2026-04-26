// Metal-direct backend. Bypasses wgpu-native + Naga entirely on
// macOS — uses the Metal Obj-C API via a thin C bridge (see
// metal/bridge.h, metal/bridge.m). Hand-tuned MSL kernels live in
// metal/shaders.metal and are compiled into a single MTLLibrary
// at backend creation time.
//
// Why bother: Naga's WGSL→MSL output is measurably slower than
// browser Tint's. With hand-written MSL we control loop unrolling,
// threadgroup memory layout, and (on M3+) simdgroup matrix ops.
// Realistic target: match-or-beat browser textsift WebGPU on the
// same hardware.

const std = @import("std");

pub const cb = @cImport({
    @cInclude("bridge.h");
});

const std_c = @cImport({
    @cInclude("stdio.h");
});

pub const MetalError = error{
    DeviceCreateFailed,
    QueueCreateFailed,
    LibraryCompileFailed,
    PipelineCreateFailed,
    BufferCreateFailed,
    EncoderCreateFailed,
    UnknownKernel,
    OutOfMemory,
};

pub const MSL_SOURCE = @embedFile("metal/shaders.metal");

pub const Backend = struct {
    device: cb.TsMetalDevice,
    queue: cb.TsMetalQueue,
    library: cb.TsMetalLibrary,
    pipelines: std.StringHashMap(cb.TsMetalPipeline),
};

pub fn createBackend() MetalError!*Backend {
    const dev = cb.ts_metal_create_device() orelse return MetalError.DeviceCreateFailed;
    errdefer cb.ts_metal_device_release(dev);

    const q = cb.ts_metal_device_queue(dev) orelse return MetalError.QueueCreateFailed;
    errdefer cb.ts_metal_queue_release(q);

    var err_buf: [2048]u8 = undefined;
    @memset(&err_buf, 0);
    const lib = cb.ts_metal_library_from_source(
        dev,
        MSL_SOURCE.ptr,
        MSL_SOURCE.len,
        &err_buf,
        err_buf.len,
    ) orelse {
        _ = std_c.fprintf(std_c.stderr(), "metal: MSL library compile failed: %s\n", &err_buf);
        return MetalError.LibraryCompileFailed;
    };
    errdefer cb.ts_metal_library_release(lib);

    const b = std.heap.c_allocator.create(Backend) catch return MetalError.OutOfMemory;
    b.* = .{
        .device = dev,
        .queue = q,
        .library = lib,
        .pipelines = std.StringHashMap(cb.TsMetalPipeline).init(std.heap.c_allocator),
    };
    return b;
}

pub fn destroyBackend(b: *Backend) void {
    var it = b.pipelines.iterator();
    while (it.next()) |entry| {
        cb.ts_metal_pipeline_release(entry.value_ptr.*);
        std.heap.c_allocator.free(entry.key_ptr.*);
    }
    b.pipelines.deinit();
    cb.ts_metal_library_release(b.library);
    cb.ts_metal_queue_release(b.queue);
    cb.ts_metal_device_release(b.device);
    std.heap.c_allocator.destroy(b);
}

pub fn deviceName(b: *Backend) [*:0]const u8 {
    return @ptrCast(cb.ts_metal_device_name(b.device));
}

fn getOrCompilePipeline(b: *Backend, name: []const u8) MetalError!cb.TsMetalPipeline {
    if (b.pipelines.get(name)) |p| return p;
    var name_buf: [128]u8 = undefined;
    if (name.len + 1 > name_buf.len) return MetalError.UnknownKernel;
    @memcpy(name_buf[0..name.len], name);
    name_buf[name.len] = 0;

    var err: [1024]u8 = undefined;
    @memset(&err, 0);
    const p = cb.ts_metal_pipeline_for(b.device, b.library, &name_buf, &err, err.len) orelse {
        _ = std_c.fprintf(std_c.stderr(), "metal: pipeline '%s' creation failed: %s\n", &name_buf, &err);
        return MetalError.PipelineCreateFailed;
    };

    const key = std.heap.c_allocator.dupe(u8, name) catch {
        cb.ts_metal_pipeline_release(p);
        return MetalError.OutOfMemory;
    };
    b.pipelines.put(key, p) catch {
        std.heap.c_allocator.free(key);
        cb.ts_metal_pipeline_release(p);
        return MetalError.OutOfMemory;
    };
    return p;
}

pub fn createBuffer(b: *Backend, bytes: []const u8) MetalError!cb.TsMetalBuffer {
    const buf = cb.ts_metal_buffer_create(b.device, bytes.len, bytes.ptr) orelse
        return MetalError.BufferCreateFailed;
    return buf;
}

pub fn createEmptyBuffer(b: *Backend, len: usize) MetalError!cb.TsMetalBuffer {
    const buf = cb.ts_metal_buffer_create(b.device, len, null) orelse
        return MetalError.BufferCreateFailed;
    return buf;
}

pub fn releaseBuffer(buf: cb.TsMetalBuffer) void {
    cb.ts_metal_buffer_release(buf);
}

pub fn writeBuffer(buf: cb.TsMetalBuffer, offset: usize, data: []const u8) void {
    cb.ts_metal_buffer_write(buf, offset, data.ptr, data.len);
}

pub fn readBuffer(buf: cb.TsMetalBuffer, offset: usize, out: []u8) void {
    cb.ts_metal_buffer_read(buf, offset, out.ptr, out.len);
}

/// Per-binding: either inline bytes (set as setBytes) or a buffer.
/// All bindings get an `index` matching the MSL `[[buffer(N)]]` slot.
pub const Binding = union(enum) {
    bytes: struct { index: u32, bytes: []const u8 },
    buffer: struct { index: u32, buf: cb.TsMetalBuffer, offset: u32 = 0 },
};

/// One-shot dispatch: encode a single kernel + commit + wait. Used
/// for conformance tests where we want each kernel measured in
/// isolation. For real forward, use `enqueueOnCommandBuffer` + a
/// shared command buffer that batches everything.
pub fn dispatchOneShot(
    b: *Backend,
    name: []const u8,
    bindings: []const Binding,
    grid: [3]u32,
    threadgroup: [3]u32,
) MetalError!void {
    const p = try getOrCompilePipeline(b, name);
    const cmd = cb.ts_metal_queue_command_buffer(b.queue) orelse return MetalError.EncoderCreateFailed;
    defer cb.ts_metal_command_buffer_release(cmd);
    const enc = cb.ts_metal_command_buffer_compute_encoder(cmd) orelse return MetalError.EncoderCreateFailed;
    cb.ts_metal_encoder_set_pipeline(enc, p);
    for (bindings) |bnd| {
        switch (bnd) {
            .bytes => |bb| cb.ts_metal_encoder_set_bytes(enc, bb.bytes.ptr, bb.bytes.len, bb.index),
            .buffer => |bb| cb.ts_metal_encoder_set_buffer(enc, bb.buf, bb.offset, bb.index),
        }
    }
    cb.ts_metal_encoder_dispatch(enc, grid[0], grid[1], grid[2], threadgroup[0], threadgroup[1], threadgroup[2]);
    cb.ts_metal_encoder_end(enc);
    cb.ts_metal_command_buffer_commit(cmd);
    cb.ts_metal_command_buffer_wait(cmd);
}
