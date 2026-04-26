// Dawn-direct backend (cross-platform). Statically links Google's Dawn
// C++ library via the C bridge in dawn/bridge.{h,c}. Tint compiles the
// canonical WGSL kernels at runtime — no separate GLSL/MSL/HLSL ports.
//
// Why this exists alongside Vulkan-direct on Linux: it lets us measure
// honestly how Tint's WGSL→SPIR-V codegen compares to our hand-written
// GLSL→SPIR-V on the same kernels, same hardware. On the user path,
// hand-tuned wins; this backend is the reference for "what would
// shared-with-browser WGSL get us."
//
// Public API mirrors metal_backend.zig and vulkan_backend.zig 1:1 so
// JS-side orchestration is identical between platforms.

const std = @import("std");

pub const cb = @cImport({
    @cInclude("dawn/bridge.h");
});

pub const DawnError = error{
    BackendCreateFailed,
    PipelineCreateFailed,
    BufferCreateFailed,
    EncoderCreateFailed,
    UnknownKernel,
    BindingMismatch,
    OutOfMemory,
};

// Per-shader metadata. Dawn takes WGSL source directly, so we just
// @embedFile the canonical .wgsl files (same ones the browser uses).
//
// `uniform_sizes` is an array of byte sizes for each `var<uniform>`
// binding the WGSL declares (in binding order). Most kernels have
// exactly one uniform (Dims, 16 or 32 B); `cast_f32_to_fp16_scaled`
// has two (Dims + Scale).
pub const PipelineMeta = struct {
    wgsl: []const u8,
    num_storage_buffers: u32,
    uniform_sizes: []const u32,
};

const ShaderEntry = struct { name: []const u8, meta: PipelineMeta };

pub const SHADERS = [_]ShaderEntry{
    .{ .name = "rms_norm", .meta = .{
        .wgsl = @embedFile("shaders/rms_norm.wgsl"),
        .num_storage_buffers = 3, .uniform_sizes = &.{16},
    } },
    .{ .name = "zero_f32", .meta = .{
        .wgsl = @embedFile("shaders/zero_f32.wgsl"),
        .num_storage_buffers = 1, .uniform_sizes = &.{16},
    } },
    .{ .name = "cast_fp16_to_f32", .meta = .{
        .wgsl = @embedFile("shaders/cast_fp16_to_f32.wgsl"),
        .num_storage_buffers = 2, .uniform_sizes = &.{16},
    } },
    .{ .name = "cast_f32_to_fp16_scaled", .meta = .{
        .wgsl = @embedFile("shaders/cast_f32_to_fp16_scaled.wgsl"),
        // Two uniforms in WGSL: Dims (binding 0, 16 B) + Scale (binding 1, 16 B).
        .num_storage_buffers = 2, .uniform_sizes = &.{ 16, 16 },
    } },
    .{ .name = "add_fp16", .meta = .{
        .wgsl = @embedFile("shaders/add_fp16.wgsl"),
        .num_storage_buffers = 3, .uniform_sizes = &.{16},
    } },
    .{ .name = "swiglu_clamp", .meta = .{
        .wgsl = @embedFile("shaders/swiglu_clamp.wgsl"),
        .num_storage_buffers = 2, .uniform_sizes = &.{16},
    } },
    .{ .name = "rope_apply", .meta = .{
        .wgsl = @embedFile("shaders/rope_apply.wgsl"),
        .num_storage_buffers = 3, .uniform_sizes = &.{16},
    } },
    .{ .name = "matmul_int4_fp16_f16", .meta = .{
        .wgsl = @embedFile("shaders/matmul_int4_fp16_f16.wgsl"),
        .num_storage_buffers = 6, .uniform_sizes = &.{16},
    } },
    .{ .name = "matmul_int4_f32_f32", .meta = .{
        .wgsl = @embedFile("shaders/matmul_int4_f32_f32.wgsl"),
        .num_storage_buffers = 6, .uniform_sizes = &.{16},
    } },
    .{ .name = "embed_lookup_int4", .meta = .{
        .wgsl = @embedFile("shaders/embed_lookup_int4.wgsl"),
        .num_storage_buffers = 5, .uniform_sizes = &.{16},
    } },
    .{ .name = "add_rmsnorm_fp16_to_f32", .meta = .{
        .wgsl = @embedFile("shaders/add_rmsnorm_fp16_to_f32.wgsl"),
        .num_storage_buffers = 5, .uniform_sizes = &.{16},
    } },
    .{ .name = "router_topk", .meta = .{
        .wgsl = @embedFile("shaders/router_topk.wgsl"),
        .num_storage_buffers = 3, .uniform_sizes = &.{16},
    } },
    .{ .name = "banded_attention", .meta = .{
        .wgsl = @embedFile("shaders/banded_attention.wgsl"),
        .num_storage_buffers = 6, .uniform_sizes = &.{32},
    } },
    .{ .name = "qmoe_gate_up", .meta = .{
        .wgsl = @embedFile("shaders/qmoe_gate_up.wgsl"),
        .num_storage_buffers = 7, .uniform_sizes = &.{32},
    } },
    .{ .name = "qmoe_down_scatter", .meta = .{
        .wgsl = @embedFile("shaders/qmoe_down_scatter.wgsl"),
        .num_storage_buffers = 8, .uniform_sizes = &.{32},
    } },
};

fn shaderMeta(name: []const u8) ?PipelineMeta {
    for (SHADERS) |entry| {
        if (std.mem.eql(u8, entry.name, name)) return entry.meta;
    }
    return null;
}

fn totalUniformSize(sizes: []const u32) u32 {
    var sum: u32 = 0;
    for (sizes) |s| sum += s;
    return sum;
}

pub const Backend = struct {
    handle: cb.TsDawnBackend,
    pipelines: std.StringHashMap(cb.TsDawnPipeline),
};

pub fn createBackend() DawnError!*Backend {
    var err_buf: [1024]u8 = undefined;
    @memset(&err_buf, 0);
    const h = cb.ts_dawn_create_backend(&err_buf, err_buf.len) orelse {
        std.debug.print("dawn: create_backend failed: {s}\n", .{std.mem.sliceTo(&err_buf, 0)});
        return DawnError.BackendCreateFailed;
    };

    const b = std.heap.c_allocator.create(Backend) catch {
        cb.ts_dawn_destroy_backend(h);
        return DawnError.OutOfMemory;
    };
    b.* = .{
        .handle = h,
        .pipelines = std.StringHashMap(cb.TsDawnPipeline).init(std.heap.c_allocator),
    };
    return b;
}

pub fn destroyBackend(b: *Backend) void {
    var it = b.pipelines.iterator();
    while (it.next()) |entry| {
        cb.ts_dawn_pipeline_release(b.handle, entry.value_ptr.*);
        std.heap.c_allocator.free(entry.key_ptr.*);
    }
    b.pipelines.deinit();
    cb.ts_dawn_destroy_backend(b.handle);
    std.heap.c_allocator.destroy(b);
}

pub fn deviceName(b: *Backend) [*:0]const u8 {
    return @ptrCast(cb.ts_dawn_device_name(b.handle));
}

const PipeAndMeta = struct {
    p: cb.TsDawnPipeline,
    meta: PipelineMeta,
};

fn getOrCreatePipeline(b: *Backend, name: []const u8) DawnError!PipeAndMeta {
    const meta = shaderMeta(name) orelse return DawnError.UnknownKernel;

    if (b.pipelines.get(name)) |p| return .{ .p = p, .meta = meta };

    var err_buf: [1024]u8 = undefined;
    @memset(&err_buf, 0);
    const p = cb.ts_dawn_pipeline_create(
        b.handle,
        meta.wgsl.ptr,
        meta.wgsl.len,
        "main",
        meta.num_storage_buffers,
        @intCast(meta.uniform_sizes.len),
        meta.uniform_sizes.ptr,
        &err_buf,
        err_buf.len,
    ) orelse {
        std.debug.print("dawn: pipeline '{s}' failed: {s}\n", .{ name, std.mem.sliceTo(&err_buf, 0) });
        return DawnError.PipelineCreateFailed;
    };

    const key = std.heap.c_allocator.dupe(u8, name) catch {
        cb.ts_dawn_pipeline_release(b.handle, p);
        return DawnError.OutOfMemory;
    };
    b.pipelines.put(key, p) catch {
        std.heap.c_allocator.free(key);
        cb.ts_dawn_pipeline_release(b.handle, p);
        return DawnError.OutOfMemory;
    };
    return .{ .p = p, .meta = meta };
}

pub fn createBuffer(b: *Backend, bytes: []const u8) DawnError!cb.TsDawnBuffer {
    const buf = cb.ts_dawn_buffer_create(b.handle, bytes.len, bytes.ptr) orelse
        return DawnError.BufferCreateFailed;
    return buf;
}

pub fn createEmptyBuffer(b: *Backend, len: usize) DawnError!cb.TsDawnBuffer {
    const buf = cb.ts_dawn_buffer_create(b.handle, len, null) orelse
        return DawnError.BufferCreateFailed;
    return buf;
}

pub fn releaseBuffer(b: *Backend, buf: cb.TsDawnBuffer) void {
    cb.ts_dawn_buffer_release(b.handle, buf);
}

pub fn writeBuffer(b: *Backend, buf: cb.TsDawnBuffer, offset: usize, data: []const u8) void {
    cb.ts_dawn_buffer_write(b.handle, buf, offset, data.ptr, data.len);
}

pub fn readBuffer(b: *Backend, buf: cb.TsDawnBuffer, offset: usize, out: []u8) void {
    cb.ts_dawn_buffer_read(b.handle, buf, offset, out.ptr, out.len);
}

pub fn dispatchOneShot(
    b: *Backend,
    name: []const u8,
    bindings: []const cb.TsDawnBuffer,
    uniform_data: []const u8,
    grid: [3]u32,
) DawnError!void {
    const pipe = try getOrCreatePipeline(b, name);
    if (bindings.len != pipe.meta.num_storage_buffers) return DawnError.BindingMismatch;
    if (uniform_data.len != totalUniformSize(pipe.meta.uniform_sizes)) return DawnError.BindingMismatch;

    const e = cb.ts_dawn_encoder_begin(b.handle) orelse return DawnError.EncoderCreateFailed;
    cb.ts_dawn_encoder_dispatch(
        e,
        pipe.p,
        bindings.ptr,
        @intCast(bindings.len),
        if (uniform_data.len > 0) uniform_data.ptr else null,
        @intCast(uniform_data.len),
        grid[0], grid[1], grid[2],
    );
    cb.ts_dawn_encoder_submit_and_wait(e);
}

pub const Encoder = struct {
    backend: *Backend,
    handle: cb.TsDawnEncoder,
};

pub fn beginEncoder(b: *Backend) DawnError!*Encoder {
    const e = cb.ts_dawn_encoder_begin(b.handle) orelse return DawnError.EncoderCreateFailed;
    const wrap = std.heap.c_allocator.create(Encoder) catch {
        cb.ts_dawn_encoder_release(e);
        return DawnError.OutOfMemory;
    };
    wrap.* = .{ .backend = b, .handle = e };
    return wrap;
}

pub fn enqueueOnEncoder(
    e: *Encoder,
    name: []const u8,
    bindings: []const cb.TsDawnBuffer,
    uniform_data: []const u8,
    grid: [3]u32,
) DawnError!void {
    const pipe = try getOrCreatePipeline(e.backend, name);
    if (bindings.len != pipe.meta.num_storage_buffers) return DawnError.BindingMismatch;
    if (uniform_data.len != totalUniformSize(pipe.meta.uniform_sizes)) return DawnError.BindingMismatch;

    cb.ts_dawn_encoder_dispatch(
        e.handle,
        pipe.p,
        bindings.ptr,
        @intCast(bindings.len),
        if (uniform_data.len > 0) uniform_data.ptr else null,
        @intCast(uniform_data.len),
        grid[0], grid[1], grid[2],
    );
}

pub fn submitAndReadback(
    e: *Encoder,
    out_buf: cb.TsDawnBuffer,
    offset: usize,
    out: []u8,
) void {
    cb.ts_dawn_encoder_submit_and_wait(e.handle);
    cb.ts_dawn_buffer_read(e.backend.handle, out_buf, offset, out.ptr, out.len);
    std.heap.c_allocator.destroy(e);
}
