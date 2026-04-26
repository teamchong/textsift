// Minimal wgpu-native bring-up: create instance, request an adapter,
// pull adapter info. Surfaced to JS as `getAdapterInfo()` so the JS
// side can validate the GPU is available before doing any inference
// work. If no adapter is found, throws — the caller is expected to
// catch that and fall through to `textsift/browser` (per the contract
// in src/index.ts).

const std = @import("std");

pub const c = @cImport({
    @cInclude("webgpu/webgpu.h");
    @cInclude("webgpu/wgpu.h");
});

const std_c = @cImport({
    @cInclude("stdio.h");
    @cInclude("time.h");
});

fn nanos() i128 {
    var ts: std_c.struct_timespec = undefined;
    _ = std_c.clock_gettime(std_c.CLOCK_MONOTONIC, &ts);
    return @as(i128, ts.tv_sec) * 1_000_000_000 + @as(i128, ts.tv_nsec);
}

pub const AdapterInfo = struct {
    vendor: []const u8,
    architecture: []const u8,
    device: []const u8,
    description: []const u8,
    backend_type: u32,
    adapter_type: u32,
};

pub const WgpuError = error{
    InstanceCreateFailed,
    AdapterRequestFailed,
    AdapterUnavailable,
    AdapterInfoFailed,
    ShaderF16Unavailable,
    DeviceRequestFailed,
    DeviceUnavailable,
    DeviceLimitsFailed,
    BufferCreateFailed,
    BufferMapFailed,
    BufferRangeFailed,
    ShaderModuleFailed,
    ComputePipelineFailed,
    BindGroupFailed,
};

pub const DeviceInfo = struct {
    adapter: AdapterInfo,
    max_storage_buffer_binding_size: u64,
    max_buffer_size: u64,
    max_storage_buffers_per_shader_stage: u32,
    max_compute_workgroup_storage_size: u32,
    max_compute_invocations_per_workgroup: u32,
    max_compute_workgroup_size_x: u32,
};

const RequestState = struct {
    // 0 isn't a valid WGPURequestAdapterStatus value (the enum starts
    // at _Success = 0x1), so it serves as a "callback hasn't run yet"
    // sentinel without name-collision with the upstream header.
    status: c.WGPURequestAdapterStatus = 0,
    adapter: c.WGPUAdapter = null,
    fired: bool = false,
};

const DeviceState = struct {
    status: c.WGPURequestDeviceStatus = 0,
    device: c.WGPUDevice = null,
    fired: bool = false,
};

const MapState = struct {
    status: c.WGPUMapAsyncStatus = 0,
    fired: bool = false,
};

fn onBufferMap(
    status: c.WGPUMapAsyncStatus,
    _: c.WGPUStringView,
    userdata1: ?*anyopaque,
    _: ?*anyopaque,
) callconv(.c) void {
    const state: *MapState = @ptrCast(@alignCast(userdata1.?));
    state.status = status;
    state.fired = true;
}

fn onAdapter(
    status: c.WGPURequestAdapterStatus,
    adapter: c.WGPUAdapter,
    _: c.WGPUStringView, // message diagnostic — wgpu logs it elsewhere
    userdata1: ?*anyopaque,
    _: ?*anyopaque,
) callconv(.c) void {
    const state: *RequestState = @ptrCast(@alignCast(userdata1.?));
    state.status = status;
    state.adapter = adapter;
    state.fired = true;
}

fn onDevice(
    status: c.WGPURequestDeviceStatus,
    device: c.WGPUDevice,
    _: c.WGPUStringView,
    userdata1: ?*anyopaque,
    _: ?*anyopaque,
) callconv(.c) void {
    const state: *DeviceState = @ptrCast(@alignCast(userdata1.?));
    state.status = status;
    state.device = device;
    state.fired = true;
}

/// Create a wgpu instance, request a high-perf adapter, return both
/// the instance + adapter handles. Caller owns release of both.
pub fn createInstanceAndAdapter() WgpuError!struct {
    instance: c.WGPUInstance,
    adapter: c.WGPUAdapter,
} {
    const instance = c.wgpuCreateInstance(null);
    if (instance == null) return WgpuError.InstanceCreateFailed;

    var state = RequestState{};
    const cb_info = c.WGPURequestAdapterCallbackInfo{
        .nextInChain = null,
        // wgpu-native v29 doesn't implement WaitAnyOnly — poll via
        // ProcessEvents until the callback fires.
        .mode = c.WGPUCallbackMode_AllowProcessEvents,
        .callback = onAdapter,
        .userdata1 = &state,
        .userdata2 = null,
    };
    _ = c.wgpuInstanceRequestAdapter(instance, null, cb_info);

    // Bound the spin so a broken driver can't lock the calling thread.
    // 1000 iters × wgpuInstanceProcessEvents is more than enough for
    // an adapter-request that already has the answer in hand.
    var spins: u32 = 0;
    while (!state.fired and spins < 1000) : (spins += 1) {
        c.wgpuInstanceProcessEvents(instance);
    }
    if (!state.fired or state.status != c.WGPURequestAdapterStatus_Success or state.adapter == null) {
        c.wgpuInstanceRelease(instance);
        return WgpuError.AdapterUnavailable;
    }

    return .{ .instance = instance, .adapter = state.adapter };
}

fn stringViewToSlice(sv: c.WGPUStringView) []const u8 {
    if (sv.data == null or sv.length == 0) return "";
    return sv.data[0..sv.length];
}

/// Pull adapter info into an `AdapterInfo` allocated on the caller-
/// supplied allocator. The wgpu-side strings are duped so the caller
/// can free the wgpu adapter immediately after.
pub fn getAdapterInfo(adapter: c.WGPUAdapter, allocator: std.mem.Allocator) WgpuError!AdapterInfo {
    var info: c.WGPUAdapterInfo = undefined;
    @memset(@as([*]u8, @ptrCast(&info))[0..@sizeOf(c.WGPUAdapterInfo)], 0);
    const status = c.wgpuAdapterGetInfo(adapter, &info);
    if (status != c.WGPUStatus_Success) return WgpuError.AdapterInfoFailed;
    defer c.wgpuAdapterInfoFreeMembers(info);

    return AdapterInfo{
        .vendor = allocator.dupe(u8, stringViewToSlice(info.vendor)) catch "",
        .architecture = allocator.dupe(u8, stringViewToSlice(info.architecture)) catch "",
        .device = allocator.dupe(u8, stringViewToSlice(info.device)) catch "",
        .description = allocator.dupe(u8, stringViewToSlice(info.description)) catch "",
        .backend_type = info.backendType,
        .adapter_type = info.adapterType,
    };
}

/// Verify the adapter exposes shader-f16 (the model's int4-fp16
/// matmul kernels need it; the browser path enforces the same).
pub fn requireShaderF16(adapter: c.WGPUAdapter) WgpuError!void {
    if (c.wgpuAdapterHasFeature(adapter, c.WGPUFeatureName_ShaderF16) == 0) {
        return WgpuError.ShaderF16Unavailable;
    }
}

// Per-thread storage for the most recent uncaptured device error.
// Naga's default handler panics, which crashes the Node process; we
// install our own callback that just records the message so the
// caller can fail gracefully and surface it to the test runner.
var last_device_error_buf: [4096]u8 = undefined;
var last_device_error_len: usize = 0;

fn onUncapturedError(
    _: ?*const c.WGPUDevice,
    _: c.WGPUErrorType,
    message: c.WGPUStringView,
    _: ?*anyopaque,
    _: ?*anyopaque,
) callconv(.c) void {
    const msg_len = if (message.length == std.math.maxInt(usize))
        std.mem.len(@as([*:0]const u8, @ptrCast(message.data orelse return)))
    else
        message.length;
    const copy_len = @min(msg_len, last_device_error_buf.len);
    if (message.data) |d| {
        @memcpy(last_device_error_buf[0..copy_len], d[0..copy_len]);
    }
    last_device_error_len = copy_len;
    // Surface to stderr via libc so the test runner sees the
    // actual WGSL validation error (avoids Zig stdlib API churn).
    _ = std_c.fprintf(std_c.stderr(), "wgpu validation error: %.*s\n", @as(c_int, @intCast(copy_len)), last_device_error_buf[0..copy_len].ptr);
}

/// Request a device with the same limit clamps the browser
/// WebGpuBackend uses (so a kernel that fits in the browser fits in
/// the native binding too). Returns the device handle; caller owns
/// release.
pub fn createDevice(
    instance: c.WGPUInstance,
    adapter: c.WGPUAdapter,
) WgpuError!c.WGPUDevice {
    var adapter_limits: c.WGPULimits = std.mem.zeroes(c.WGPULimits);
    if (c.wgpuAdapterGetLimits(adapter, &adapter_limits) != c.WGPUStatus_Success) {
        return WgpuError.DeviceLimitsFailed;
    }

    // WGPU sentinel: every limit field defaults to UINT32_MAX /
    // UINT64_MAX = "no requirement, use default". Zeroing would tell
    // the implementation we explicitly require 0, which rejects.
    const ONE_GIB: u64 = 1024 * 1024 * 1024;
    var required_limits: c.WGPULimits = .{
        .nextInChain = null,
        .maxTextureDimension1D = std.math.maxInt(u32),
        .maxTextureDimension2D = std.math.maxInt(u32),
        .maxTextureDimension3D = std.math.maxInt(u32),
        .maxTextureArrayLayers = std.math.maxInt(u32),
        .maxBindGroups = std.math.maxInt(u32),
        .maxBindGroupsPlusVertexBuffers = std.math.maxInt(u32),
        .maxBindingsPerBindGroup = std.math.maxInt(u32),
        .maxDynamicUniformBuffersPerPipelineLayout = std.math.maxInt(u32),
        .maxDynamicStorageBuffersPerPipelineLayout = std.math.maxInt(u32),
        .maxSampledTexturesPerShaderStage = std.math.maxInt(u32),
        .maxSamplersPerShaderStage = std.math.maxInt(u32),
        .maxStorageBuffersPerShaderStage = @min(adapter_limits.maxStorageBuffersPerShaderStage, 10),
        .maxStorageTexturesPerShaderStage = std.math.maxInt(u32),
        .maxUniformBuffersPerShaderStage = std.math.maxInt(u32),
        .maxUniformBufferBindingSize = std.math.maxInt(u64),
        .maxStorageBufferBindingSize = @min(adapter_limits.maxStorageBufferBindingSize, ONE_GIB),
        .minUniformBufferOffsetAlignment = std.math.maxInt(u32),
        .minStorageBufferOffsetAlignment = std.math.maxInt(u32),
        .maxVertexBuffers = std.math.maxInt(u32),
        .maxBufferSize = @min(adapter_limits.maxBufferSize, ONE_GIB),
        .maxVertexAttributes = std.math.maxInt(u32),
        .maxVertexBufferArrayStride = std.math.maxInt(u32),
        .maxInterStageShaderVariables = std.math.maxInt(u32),
        .maxColorAttachments = std.math.maxInt(u32),
        .maxColorAttachmentBytesPerSample = std.math.maxInt(u32),
        .maxComputeWorkgroupStorageSize = std.math.maxInt(u32),
        .maxComputeInvocationsPerWorkgroup = std.math.maxInt(u32),
        .maxComputeWorkgroupSizeX = std.math.maxInt(u32),
        .maxComputeWorkgroupSizeY = std.math.maxInt(u32),
        .maxComputeWorkgroupSizeZ = std.math.maxInt(u32),
        .maxComputeWorkgroupsPerDimension = std.math.maxInt(u32),
        .maxImmediateSize = std.math.maxInt(u32),
    };

    const required_features = [_]c.WGPUFeatureName{c.WGPUFeatureName_ShaderF16};

    var desc: c.WGPUDeviceDescriptor = std.mem.zeroes(c.WGPUDeviceDescriptor);
    desc.requiredFeatureCount = required_features.len;
    desc.requiredFeatures = &required_features;
    desc.requiredLimits = &required_limits;
    desc.uncapturedErrorCallbackInfo = .{
        .nextInChain = null,
        .callback = onUncapturedError,
        .userdata1 = null,
        .userdata2 = null,
    };

    var state = DeviceState{};
    const cb = c.WGPURequestDeviceCallbackInfo{
        .nextInChain = null,
        .mode = c.WGPUCallbackMode_AllowProcessEvents,
        .callback = onDevice,
        .userdata1 = &state,
        .userdata2 = null,
    };
    _ = c.wgpuAdapterRequestDevice(adapter, &desc, cb);

    // Device creation is heavier than adapter-request — Metal driver
    // brings up command queues, etc. Big spin budget; processEvents
    // is cheap when nothing's pending.
    var spins: u32 = 0;
    while (!state.fired and spins < 1_000_000) : (spins += 1) {
        c.wgpuInstanceProcessEvents(instance);
    }
    if (!state.fired or state.status != c.WGPURequestDeviceStatus_Success or state.device == null) {
        return WgpuError.DeviceUnavailable;
    }
    return state.device;
}

/// Read back the limits the device was actually granted.
pub fn getDeviceLimits(device: c.WGPUDevice) WgpuError!c.WGPULimits {
    var limits: c.WGPULimits = std.mem.zeroes(c.WGPULimits);
    if (c.wgpuDeviceGetLimits(device, &limits) != c.WGPUStatus_Success) {
        return WgpuError.DeviceLimitsFailed;
    }
    return limits;
}

/// Round-trip `input` through a wgpu buffer: write to GPU memory via
/// queue.writeBuffer, then map for read and copy back. Validates the
/// full buffer pipeline (create → write → submit → map → read) end-
/// to-end. `output` must have at least `input.len` capacity.
///
/// Allocates and releases its own instance + device — every call is
/// independent. The inference pipeline will hold device resources
/// long-term; this function exists for the conformance test only.
pub fn roundtripBuffer(input: []const u8, output: []u8) WgpuError!void {
    if (output.len < input.len) return WgpuError.BufferRangeFailed;

    const handles = try createInstanceAndAdapter();
    defer c.wgpuInstanceRelease(handles.instance);
    defer c.wgpuAdapterRelease(handles.adapter);

    try requireShaderF16(handles.adapter);
    const device = try createDevice(handles.instance, handles.adapter);
    defer c.wgpuDeviceRelease(device);

    const queue = c.wgpuDeviceGetQueue(device);
    defer c.wgpuQueueRelease(queue);

    // Staging buffer: created mapped, host writes the bytes via the
    // mapped range, then unmap. After unmap the bytes are owned by
    // the GPU and live in CopySrc-able storage. This is the canonical
    // host→GPU upload pattern; queue.writeBuffer with our combined
    // MapRead+CopyDst destination didn't surface the data on
    // wgpu-native v29.
    var staging_desc: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
    staging_desc.usage = c.WGPUBufferUsage_CopySrc;
    staging_desc.size = input.len;
    staging_desc.mappedAtCreation = 1;
    const staging = c.wgpuDeviceCreateBuffer(device, &staging_desc);
    if (staging == null) return WgpuError.BufferCreateFailed;
    defer c.wgpuBufferRelease(staging);

    const staging_mapped = c.wgpuBufferGetMappedRange(staging, 0, input.len);
    if (staging_mapped == null) return WgpuError.BufferRangeFailed;
    @memcpy(@as([*]u8, @ptrCast(staging_mapped))[0..input.len], input);
    c.wgpuBufferUnmap(staging);

    var read_desc: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
    read_desc.usage = c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst;
    read_desc.size = input.len;
    const readback = c.wgpuDeviceCreateBuffer(device, &read_desc);
    if (readback == null) return WgpuError.BufferCreateFailed;
    defer c.wgpuBufferRelease(readback);

    var encoder_desc: c.WGPUCommandEncoderDescriptor = std.mem.zeroes(c.WGPUCommandEncoderDescriptor);
    const encoder = c.wgpuDeviceCreateCommandEncoder(device, &encoder_desc);
    defer c.wgpuCommandEncoderRelease(encoder);
    c.wgpuCommandEncoderCopyBufferToBuffer(encoder, staging, 0, readback, 0, input.len);

    var cmd_desc: c.WGPUCommandBufferDescriptor = std.mem.zeroes(c.WGPUCommandBufferDescriptor);
    const cmd = c.wgpuCommandEncoderFinish(encoder, &cmd_desc);
    defer c.wgpuCommandBufferRelease(cmd);

    var cmds = [_]c.WGPUCommandBuffer{cmd};
    c.wgpuQueueSubmit(queue, cmds.len, &cmds);
    _ = c.wgpuDevicePoll(device, 1, null);

    var map_state = MapState{};
    const map_cb = c.WGPUBufferMapCallbackInfo{
        .nextInChain = null,
        .mode = c.WGPUCallbackMode_AllowProcessEvents,
        .callback = onBufferMap,
        .userdata1 = &map_state,
        .userdata2 = null,
    };
    _ = c.wgpuBufferMapAsync(readback, c.WGPUMapMode_Read, 0, input.len, map_cb);

    var spins: u32 = 0;
    while (!map_state.fired and spins < 1_000_000) : (spins += 1) {
        _ = c.wgpuDevicePoll(device, 0, null);
        c.wgpuInstanceProcessEvents(handles.instance);
    }
    if (!map_state.fired or map_state.status != c.WGPUMapAsyncStatus_Success) {
        return WgpuError.BufferMapFailed;
    }

    const mapped = c.wgpuBufferGetMappedRange(readback, 0, input.len);
    if (mapped == null) {
        c.wgpuBufferUnmap(readback);
        return WgpuError.BufferRangeFailed;
    }
    @memcpy(output[0..input.len], @as([*]const u8, @ptrCast(mapped))[0..input.len]);
    c.wgpuBufferUnmap(readback);
}

/// Run a "double everything" compute kernel against a Float32Array
/// input. Validates the full compute path: WGSL compile, pipeline
/// creation, bind-group binding, dispatch, output readback. The
/// kernel is the simplest possible storage-buffer compute — every
/// real shader port reuses the same surface.
pub fn dispatchDouble(input: []const f32, output: []f32) WgpuError!void {
    if (output.len < input.len) return WgpuError.BufferRangeFailed;
    const byte_len: u64 = input.len * @sizeOf(f32);

    const handles = try createInstanceAndAdapter();
    defer c.wgpuInstanceRelease(handles.instance);
    defer c.wgpuAdapterRelease(handles.adapter);

    try requireShaderF16(handles.adapter);
    const device = try createDevice(handles.instance, handles.adapter);
    defer c.wgpuDeviceRelease(device);
    const queue = c.wgpuDeviceGetQueue(device);
    defer c.wgpuQueueRelease(queue);

    // ── shader module ──
    const wgsl_src =
        \\@group(0) @binding(0) var<storage, read>       in_buf: array<f32>;
        \\@group(0) @binding(1) var<storage, read_write> out_buf: array<f32>;
        \\@compute @workgroup_size(64) fn main(
        \\  @builtin(global_invocation_id) gid: vec3<u32>
        \\) {
        \\  let i = gid.x;
        \\  if i < arrayLength(&in_buf) { out_buf[i] = in_buf[i] * 2.0; }
        \\}
    ;
    var wgsl_chain: c.WGPUShaderSourceWGSL = std.mem.zeroes(c.WGPUShaderSourceWGSL);
    wgsl_chain.chain.next = null;
    wgsl_chain.chain.sType = c.WGPUSType_ShaderSourceWGSL;
    wgsl_chain.code.data = wgsl_src.ptr;
    wgsl_chain.code.length = wgsl_src.len;

    var module_desc: c.WGPUShaderModuleDescriptor = std.mem.zeroes(c.WGPUShaderModuleDescriptor);
    module_desc.nextInChain = @ptrCast(&wgsl_chain.chain);
    module_desc.label = .{ .data = null, .length = std.math.maxInt(usize) };

    const module = c.wgpuDeviceCreateShaderModule(device, &module_desc);
    if (module == null) return WgpuError.ShaderModuleFailed;
    defer c.wgpuShaderModuleRelease(module);

    // ── pipeline (auto layout) ──
    var pipeline_desc: c.WGPUComputePipelineDescriptor = std.mem.zeroes(c.WGPUComputePipelineDescriptor);
    pipeline_desc.layout = null;
    pipeline_desc.compute.module = module;
    pipeline_desc.compute.entryPoint = .{ .data = "main", .length = 4 };
    pipeline_desc.label = .{ .data = null, .length = std.math.maxInt(usize) };

    const pipeline = c.wgpuDeviceCreateComputePipeline(device, &pipeline_desc);
    if (pipeline == null) return WgpuError.ComputePipelineFailed;
    defer c.wgpuComputePipelineRelease(pipeline);

    // ── input buffer (staged) ──
    var stage_desc: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
    stage_desc.usage = c.WGPUBufferUsage_CopySrc;
    stage_desc.size = byte_len;
    stage_desc.mappedAtCreation = 1;
    const stage = c.wgpuDeviceCreateBuffer(device, &stage_desc);
    if (stage == null) return WgpuError.BufferCreateFailed;
    defer c.wgpuBufferRelease(stage);
    const stage_ptr = c.wgpuBufferGetMappedRange(stage, 0, byte_len);
    if (stage_ptr == null) return WgpuError.BufferRangeFailed;
    @memcpy(@as([*]u8, @ptrCast(stage_ptr))[0..byte_len], std.mem.sliceAsBytes(input));
    c.wgpuBufferUnmap(stage);

    var in_desc: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
    in_desc.usage = c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst;
    in_desc.size = byte_len;
    const in_buf = c.wgpuDeviceCreateBuffer(device, &in_desc);
    if (in_buf == null) return WgpuError.BufferCreateFailed;
    defer c.wgpuBufferRelease(in_buf);

    var out_desc: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
    out_desc.usage = c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc;
    out_desc.size = byte_len;
    const out_buf = c.wgpuDeviceCreateBuffer(device, &out_desc);
    if (out_buf == null) return WgpuError.BufferCreateFailed;
    defer c.wgpuBufferRelease(out_buf);

    var read_desc: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
    read_desc.usage = c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst;
    read_desc.size = byte_len;
    const readback = c.wgpuDeviceCreateBuffer(device, &read_desc);
    if (readback == null) return WgpuError.BufferCreateFailed;
    defer c.wgpuBufferRelease(readback);

    // ── bind group ──
    const bgl = c.wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    defer c.wgpuBindGroupLayoutRelease(bgl);

    var entries = [_]c.WGPUBindGroupEntry{
        .{ .nextInChain = null, .binding = 0, .buffer = in_buf, .offset = 0, .size = byte_len, .sampler = null, .textureView = null },
        .{ .nextInChain = null, .binding = 1, .buffer = out_buf, .offset = 0, .size = byte_len, .sampler = null, .textureView = null },
    };
    var bg_desc: c.WGPUBindGroupDescriptor = std.mem.zeroes(c.WGPUBindGroupDescriptor);
    bg_desc.layout = bgl;
    bg_desc.entryCount = entries.len;
    bg_desc.entries = &entries;
    bg_desc.label = .{ .data = null, .length = std.math.maxInt(usize) };
    const bg = c.wgpuDeviceCreateBindGroup(device, &bg_desc);
    if (bg == null) return WgpuError.BindGroupFailed;
    defer c.wgpuBindGroupRelease(bg);

    // ── encoder: copy stage→in, dispatch, copy out→readback ──
    const encoder = c.wgpuDeviceCreateCommandEncoder(device, null);
    defer c.wgpuCommandEncoderRelease(encoder);
    c.wgpuCommandEncoderCopyBufferToBuffer(encoder, stage, 0, in_buf, 0, byte_len);

    const pass = c.wgpuCommandEncoderBeginComputePass(encoder, null);
    c.wgpuComputePassEncoderSetPipeline(pass, pipeline);
    c.wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, null);
    const wg_count: u32 = @intCast((input.len + 63) / 64);
    c.wgpuComputePassEncoderDispatchWorkgroups(pass, wg_count, 1, 1);
    c.wgpuComputePassEncoderEnd(pass);
    c.wgpuComputePassEncoderRelease(pass);

    c.wgpuCommandEncoderCopyBufferToBuffer(encoder, out_buf, 0, readback, 0, byte_len);
    const cmd = c.wgpuCommandEncoderFinish(encoder, null);
    defer c.wgpuCommandBufferRelease(cmd);

    var cmds = [_]c.WGPUCommandBuffer{cmd};
    c.wgpuQueueSubmit(queue, cmds.len, &cmds);
    _ = c.wgpuDevicePoll(device, 1, null);

    // ── readback ──
    var map_state = MapState{};
    const map_cb = c.WGPUBufferMapCallbackInfo{
        .nextInChain = null,
        .mode = c.WGPUCallbackMode_AllowProcessEvents,
        .callback = onBufferMap,
        .userdata1 = &map_state,
        .userdata2 = null,
    };
    _ = c.wgpuBufferMapAsync(readback, c.WGPUMapMode_Read, 0, byte_len, map_cb);
    var spins: u32 = 0;
    while (!map_state.fired and spins < 1_000_000) : (spins += 1) {
        _ = c.wgpuDevicePoll(device, 0, null);
        c.wgpuInstanceProcessEvents(handles.instance);
    }
    if (!map_state.fired or map_state.status != c.WGPUMapAsyncStatus_Success) {
        return WgpuError.BufferMapFailed;
    }
    const mapped = c.wgpuBufferGetMappedRange(readback, 0, byte_len);
    if (mapped == null) {
        c.wgpuBufferUnmap(readback);
        return WgpuError.BufferRangeFailed;
    }
    @memcpy(std.mem.sliceAsBytes(output[0..input.len]), @as([*]const u8, @ptrCast(mapped))[0..byte_len]);
    c.wgpuBufferUnmap(readback);
}

/// fp32 matmul: C[M,N] = A[M,K] · B[K,N]. Row-major. Validates the
/// multi-input compute path (two storage reads + one storage write
/// + uniform-style dimensions baked into a uniform buffer). The
/// shader is the "hello world" of GPU matmul — one workgroup per
/// output element. Real workhorse matmuls (int4-fp16) port next.
pub fn dispatchMatmulF32(
    a: []const f32,
    b: []const f32,
    out: []f32,
    m: u32,
    n: u32,
    k: u32,
) WgpuError!void {
    if (a.len < @as(usize, m) * @as(usize, k)) return WgpuError.BufferRangeFailed;
    if (b.len < @as(usize, k) * @as(usize, n)) return WgpuError.BufferRangeFailed;
    if (out.len < @as(usize, m) * @as(usize, n)) return WgpuError.BufferRangeFailed;

    const a_bytes: u64 = @as(u64, m) * @as(u64, k) * @sizeOf(f32);
    const b_bytes: u64 = @as(u64, k) * @as(u64, n) * @sizeOf(f32);
    const c_bytes: u64 = @as(u64, m) * @as(u64, n) * @sizeOf(f32);

    const handles = try createInstanceAndAdapter();
    defer c.wgpuInstanceRelease(handles.instance);
    defer c.wgpuAdapterRelease(handles.adapter);

    try requireShaderF16(handles.adapter);
    const device = try createDevice(handles.instance, handles.adapter);
    defer c.wgpuDeviceRelease(device);
    const queue = c.wgpuDeviceGetQueue(device);
    defer c.wgpuQueueRelease(queue);

    // ── shader ──
    const wgsl_src =
        \\struct Dims { m: u32, n: u32, k: u32, _pad: u32 };
        \\@group(0) @binding(0) var<uniform> dims: Dims;
        \\@group(0) @binding(1) var<storage, read>       a: array<f32>;
        \\@group(0) @binding(2) var<storage, read>       b: array<f32>;
        \\@group(0) @binding(3) var<storage, read_write> c: array<f32>;
        \\@compute @workgroup_size(8, 8) fn main(
        \\  @builtin(global_invocation_id) gid: vec3<u32>
        \\) {
        \\  let row = gid.x;
        \\  let col = gid.y;
        \\  if row >= dims.m || col >= dims.n { return; }
        \\  var acc: f32 = 0.0;
        \\  for (var i: u32 = 0u; i < dims.k; i = i + 1u) {
        \\    acc = acc + a[row * dims.k + i] * b[i * dims.n + col];
        \\  }
        \\  c[row * dims.n + col] = acc;
        \\}
    ;
    var wgsl_chain: c.WGPUShaderSourceWGSL = std.mem.zeroes(c.WGPUShaderSourceWGSL);
    wgsl_chain.chain.next = null;
    wgsl_chain.chain.sType = c.WGPUSType_ShaderSourceWGSL;
    wgsl_chain.code.data = wgsl_src.ptr;
    wgsl_chain.code.length = wgsl_src.len;
    var module_desc: c.WGPUShaderModuleDescriptor = std.mem.zeroes(c.WGPUShaderModuleDescriptor);
    module_desc.nextInChain = @ptrCast(&wgsl_chain.chain);
    module_desc.label = .{ .data = null, .length = std.math.maxInt(usize) };
    const module = c.wgpuDeviceCreateShaderModule(device, &module_desc);
    if (module == null) return WgpuError.ShaderModuleFailed;
    defer c.wgpuShaderModuleRelease(module);

    var pipeline_desc: c.WGPUComputePipelineDescriptor = std.mem.zeroes(c.WGPUComputePipelineDescriptor);
    pipeline_desc.layout = null;
    pipeline_desc.compute.module = module;
    pipeline_desc.compute.entryPoint = .{ .data = "main", .length = 4 };
    pipeline_desc.label = .{ .data = null, .length = std.math.maxInt(usize) };
    const pipeline = c.wgpuDeviceCreateComputePipeline(device, &pipeline_desc);
    if (pipeline == null) return WgpuError.ComputePipelineFailed;
    defer c.wgpuComputePipelineRelease(pipeline);

    // ── buffers ──
    const a_buf = try uploadStorageBuffer(device, std.mem.sliceAsBytes(a[0 .. m * k]));
    defer c.wgpuBufferRelease(a_buf);
    const b_buf = try uploadStorageBuffer(device, std.mem.sliceAsBytes(b[0 .. k * n]));
    defer c.wgpuBufferRelease(b_buf);

    var c_desc: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
    c_desc.usage = c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc;
    c_desc.size = c_bytes;
    const c_buf = c.wgpuDeviceCreateBuffer(device, &c_desc);
    if (c_buf == null) return WgpuError.BufferCreateFailed;
    defer c.wgpuBufferRelease(c_buf);

    // Uniform buffer: dims (m, n, k, pad)
    const dims = [_]u32{ m, n, k, 0 };
    var u_desc: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
    u_desc.usage = c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst;
    u_desc.size = @sizeOf(@TypeOf(dims));
    const u_buf = c.wgpuDeviceCreateBuffer(device, &u_desc);
    if (u_buf == null) return WgpuError.BufferCreateFailed;
    defer c.wgpuBufferRelease(u_buf);
    c.wgpuQueueWriteBuffer(queue, u_buf, 0, &dims, @sizeOf(@TypeOf(dims)));

    var read_desc: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
    read_desc.usage = c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst;
    read_desc.size = c_bytes;
    const readback = c.wgpuDeviceCreateBuffer(device, &read_desc);
    if (readback == null) return WgpuError.BufferCreateFailed;
    defer c.wgpuBufferRelease(readback);

    // ── bind group ──
    const bgl = c.wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    defer c.wgpuBindGroupLayoutRelease(bgl);
    var entries = [_]c.WGPUBindGroupEntry{
        .{ .nextInChain = null, .binding = 0, .buffer = u_buf, .offset = 0, .size = @sizeOf(@TypeOf(dims)), .sampler = null, .textureView = null },
        .{ .nextInChain = null, .binding = 1, .buffer = a_buf, .offset = 0, .size = a_bytes, .sampler = null, .textureView = null },
        .{ .nextInChain = null, .binding = 2, .buffer = b_buf, .offset = 0, .size = b_bytes, .sampler = null, .textureView = null },
        .{ .nextInChain = null, .binding = 3, .buffer = c_buf, .offset = 0, .size = c_bytes, .sampler = null, .textureView = null },
    };
    var bg_desc: c.WGPUBindGroupDescriptor = std.mem.zeroes(c.WGPUBindGroupDescriptor);
    bg_desc.layout = bgl;
    bg_desc.entryCount = entries.len;
    bg_desc.entries = &entries;
    const bg = c.wgpuDeviceCreateBindGroup(device, &bg_desc);
    if (bg == null) return WgpuError.BindGroupFailed;
    defer c.wgpuBindGroupRelease(bg);

    // ── dispatch ──
    const encoder = c.wgpuDeviceCreateCommandEncoder(device, null);
    defer c.wgpuCommandEncoderRelease(encoder);
    const pass = c.wgpuCommandEncoderBeginComputePass(encoder, null);
    c.wgpuComputePassEncoderSetPipeline(pass, pipeline);
    c.wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, null);
    const wg_x: u32 = (m + 7) / 8;
    const wg_y: u32 = (n + 7) / 8;
    c.wgpuComputePassEncoderDispatchWorkgroups(pass, wg_x, wg_y, 1);
    c.wgpuComputePassEncoderEnd(pass);
    c.wgpuComputePassEncoderRelease(pass);
    c.wgpuCommandEncoderCopyBufferToBuffer(encoder, c_buf, 0, readback, 0, c_bytes);
    const cmd = c.wgpuCommandEncoderFinish(encoder, null);
    defer c.wgpuCommandBufferRelease(cmd);
    var cmds = [_]c.WGPUCommandBuffer{cmd};
    c.wgpuQueueSubmit(queue, cmds.len, &cmds);
    _ = c.wgpuDevicePoll(device, 1, null);

    // ── readback ──
    var map_state = MapState{};
    const map_cb = c.WGPUBufferMapCallbackInfo{
        .nextInChain = null,
        .mode = c.WGPUCallbackMode_AllowProcessEvents,
        .callback = onBufferMap,
        .userdata1 = &map_state,
        .userdata2 = null,
    };
    _ = c.wgpuBufferMapAsync(readback, c.WGPUMapMode_Read, 0, c_bytes, map_cb);
    var spins: u32 = 0;
    while (!map_state.fired and spins < 1_000_000) : (spins += 1) {
        _ = c.wgpuDevicePoll(device, 0, null);
        c.wgpuInstanceProcessEvents(handles.instance);
    }
    if (!map_state.fired or map_state.status != c.WGPUMapAsyncStatus_Success) {
        return WgpuError.BufferMapFailed;
    }
    const mapped = c.wgpuBufferGetMappedRange(readback, 0, c_bytes);
    if (mapped == null) {
        c.wgpuBufferUnmap(readback);
        return WgpuError.BufferRangeFailed;
    }
    @memcpy(std.mem.sliceAsBytes(out[0 .. m * n]), @as([*]const u8, @ptrCast(mapped))[0..c_bytes]);
    c.wgpuBufferUnmap(readback);
}

pub const StorageInput = struct { binding: u32, bytes: []const u8 };
pub const KernelOutput = struct {
    binding: u32,
    bytes: []u8,
    /// How many dispatches to encode per submit/readback in
    /// `benchDispatch`. `1` (default) is per-call latency; larger
    /// values approximate a forward pass that amortizes encode +
    /// submit + map cost across many kernel invocations.
    chain_len: u32 = 1,
};

/// Generic kernel dispatcher. Every shader port becomes a thin
/// wrapper over this — pass the WGSL source, the uniform buffer
/// (or empty slice if the shader has none), the storage inputs by
/// binding index, the output binding+buffer, and the (x, y, z)
/// workgroup count. Handles instance/device/pipeline/bind-group/
/// dispatch/readback in one pass.
pub fn dispatchKernel(
    wgsl_src: []const u8,
    uniform_bytes: []const u8,
    uniform_binding: u32,
    inputs: []const StorageInput,
    output: KernelOutput,
    /// When non-null, these bytes are uploaded to the output buffer
    /// before dispatch — needed for kernels that read+write the same
    /// binding (e.g. rope_apply uses qk as both input and output).
    output_initial: ?[]const u8,
    /// Up to N additional uniform buffers beyond the first (used by
    /// shaders like cast_f32_to_fp16_scaled with two uniforms).
    extra_uniforms: []const StorageInput,
    dispatch: [3]u32,
) WgpuError!void {
    const handles = try createInstanceAndAdapter();
    defer c.wgpuInstanceRelease(handles.instance);
    defer c.wgpuAdapterRelease(handles.adapter);

    try requireShaderF16(handles.adapter);
    const device = try createDevice(handles.instance, handles.adapter);
    defer c.wgpuDeviceRelease(device);
    const queue = c.wgpuDeviceGetQueue(device);
    defer c.wgpuQueueRelease(queue);

    var wgsl_chain: c.WGPUShaderSourceWGSL = std.mem.zeroes(c.WGPUShaderSourceWGSL);
    wgsl_chain.chain.next = null;
    wgsl_chain.chain.sType = c.WGPUSType_ShaderSourceWGSL;
    wgsl_chain.code.data = wgsl_src.ptr;
    wgsl_chain.code.length = wgsl_src.len;
    var module_desc: c.WGPUShaderModuleDescriptor = std.mem.zeroes(c.WGPUShaderModuleDescriptor);
    module_desc.nextInChain = @ptrCast(&wgsl_chain.chain);
    module_desc.label = .{ .data = null, .length = std.math.maxInt(usize) };
    const module = c.wgpuDeviceCreateShaderModule(device, &module_desc);
    if (module == null) return WgpuError.ShaderModuleFailed;
    defer c.wgpuShaderModuleRelease(module);

    var pipeline_desc: c.WGPUComputePipelineDescriptor = std.mem.zeroes(c.WGPUComputePipelineDescriptor);
    pipeline_desc.layout = null;
    pipeline_desc.compute.module = module;
    pipeline_desc.compute.entryPoint = .{ .data = "main", .length = 4 };
    pipeline_desc.label = .{ .data = null, .length = std.math.maxInt(usize) };
    const pipeline = c.wgpuDeviceCreateComputePipeline(device, &pipeline_desc);
    if (pipeline == null) return WgpuError.ComputePipelineFailed;
    defer c.wgpuComputePipelineRelease(pipeline);

    // Uniform buffer (optional).
    var u_buf: c.WGPUBuffer = null;
    if (uniform_bytes.len > 0) {
        var u_desc: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
        u_desc.usage = c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst;
        u_desc.size = uniform_bytes.len;
        u_buf = c.wgpuDeviceCreateBuffer(device, &u_desc);
        if (u_buf == null) return WgpuError.BufferCreateFailed;
        c.wgpuQueueWriteBuffer(queue, u_buf, 0, uniform_bytes.ptr, uniform_bytes.len);
    }
    defer if (u_buf != null) c.wgpuBufferRelease(u_buf);

    // Storage input buffers (each uploaded via staging copy).
    var input_bufs: [16]c.WGPUBuffer = undefined;
    if (inputs.len > 16) return WgpuError.BufferRangeFailed;
    for (inputs, 0..) |input, i| {
        input_bufs[i] = try uploadStorageBuffer(device, input.bytes);
    }
    defer for (input_bufs[0..inputs.len]) |buf| c.wgpuBufferRelease(buf);

    // Output storage buffer. CopyDst added so we can pre-populate
    // for in-place kernels that read+write the same binding.
    var y_desc: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
    y_desc.usage = c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc | c.WGPUBufferUsage_CopyDst;
    y_desc.size = output.bytes.len;
    const y_buf = c.wgpuDeviceCreateBuffer(device, &y_desc);
    if (y_buf == null) return WgpuError.BufferCreateFailed;
    defer c.wgpuBufferRelease(y_buf);
    if (output_initial) |init_bytes| {
        if (init_bytes.len != output.bytes.len) return WgpuError.BufferRangeFailed;
        // Upload via staging — same pattern as the input buffers.
        var stage_desc: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
        stage_desc.usage = c.WGPUBufferUsage_CopySrc;
        stage_desc.size = init_bytes.len;
        stage_desc.mappedAtCreation = 1;
        const stage = c.wgpuDeviceCreateBuffer(device, &stage_desc);
        if (stage == null) return WgpuError.BufferCreateFailed;
        defer c.wgpuBufferRelease(stage);
        const stage_ptr = c.wgpuBufferGetMappedRange(stage, 0, init_bytes.len);
        if (stage_ptr == null) return WgpuError.BufferRangeFailed;
        @memcpy(@as([*]u8, @ptrCast(stage_ptr))[0..init_bytes.len], init_bytes);
        c.wgpuBufferUnmap(stage);
        const enc0 = c.wgpuDeviceCreateCommandEncoder(device, null);
        defer c.wgpuCommandEncoderRelease(enc0);
        c.wgpuCommandEncoderCopyBufferToBuffer(enc0, stage, 0, y_buf, 0, init_bytes.len);
        const cmd0 = c.wgpuCommandEncoderFinish(enc0, null);
        defer c.wgpuCommandBufferRelease(cmd0);
        var cmds0 = [_]c.WGPUCommandBuffer{cmd0};
        c.wgpuQueueSubmit(queue, cmds0.len, &cmds0);
    }

    // Extra uniforms (each created + written, like the primary uniform).
    var extra_uniform_bufs: [8]c.WGPUBuffer = undefined;
    if (extra_uniforms.len > 8) return WgpuError.BufferRangeFailed;
    for (extra_uniforms, 0..) |eu, i| {
        var ed: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
        ed.usage = c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst;
        ed.size = eu.bytes.len;
        const eb = c.wgpuDeviceCreateBuffer(device, &ed);
        if (eb == null) return WgpuError.BufferCreateFailed;
        c.wgpuQueueWriteBuffer(queue, eb, 0, eu.bytes.ptr, eu.bytes.len);
        extra_uniform_bufs[i] = eb;
    }
    defer for (extra_uniform_bufs[0..extra_uniforms.len]) |buf| c.wgpuBufferRelease(buf);

    // Readback buffer (host-visible).
    var read_desc: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
    read_desc.usage = c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst;
    read_desc.size = output.bytes.len;
    const readback = c.wgpuDeviceCreateBuffer(device, &read_desc);
    if (readback == null) return WgpuError.BufferCreateFailed;
    defer c.wgpuBufferRelease(readback);

    // Bind group: 1 (optional) uniform + N storage inputs + 1 storage output.
    var entries: [32]c.WGPUBindGroupEntry = undefined;
    var entry_count: usize = 0;
    if (uniform_bytes.len > 0) {
        entries[entry_count] = .{
            .nextInChain = null,
            .binding = uniform_binding,
            .buffer = u_buf,
            .offset = 0,
            .size = uniform_bytes.len,
            .sampler = null,
            .textureView = null,
        };
        entry_count += 1;
    }
    for (extra_uniforms, 0..) |eu, i| {
        entries[entry_count] = .{
            .nextInChain = null,
            .binding = eu.binding,
            .buffer = extra_uniform_bufs[i],
            .offset = 0,
            .size = eu.bytes.len,
            .sampler = null,
            .textureView = null,
        };
        entry_count += 1;
    }
    for (inputs, 0..) |input, i| {
        entries[entry_count] = .{
            .nextInChain = null,
            .binding = input.binding,
            .buffer = input_bufs[i],
            .offset = 0,
            .size = input.bytes.len,
            .sampler = null,
            .textureView = null,
        };
        entry_count += 1;
    }
    entries[entry_count] = .{
        .nextInChain = null,
        .binding = output.binding,
        .buffer = y_buf,
        .offset = 0,
        .size = output.bytes.len,
        .sampler = null,
        .textureView = null,
    };
    entry_count += 1;

    const bgl = c.wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    defer c.wgpuBindGroupLayoutRelease(bgl);
    var bg_desc: c.WGPUBindGroupDescriptor = std.mem.zeroes(c.WGPUBindGroupDescriptor);
    bg_desc.layout = bgl;
    bg_desc.entryCount = entry_count;
    bg_desc.entries = &entries;
    const bg = c.wgpuDeviceCreateBindGroup(device, &bg_desc);
    if (bg == null) return WgpuError.BindGroupFailed;
    defer c.wgpuBindGroupRelease(bg);

    // Encoder + dispatch + readback copy.
    const encoder = c.wgpuDeviceCreateCommandEncoder(device, null);
    defer c.wgpuCommandEncoderRelease(encoder);
    const pass = c.wgpuCommandEncoderBeginComputePass(encoder, null);
    c.wgpuComputePassEncoderSetPipeline(pass, pipeline);
    c.wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, null);
    c.wgpuComputePassEncoderDispatchWorkgroups(pass, dispatch[0], dispatch[1], dispatch[2]);
    c.wgpuComputePassEncoderEnd(pass);
    c.wgpuComputePassEncoderRelease(pass);
    c.wgpuCommandEncoderCopyBufferToBuffer(encoder, y_buf, 0, readback, 0, output.bytes.len);
    const cmd = c.wgpuCommandEncoderFinish(encoder, null);
    defer c.wgpuCommandBufferRelease(cmd);
    var cmds = [_]c.WGPUCommandBuffer{cmd};
    c.wgpuQueueSubmit(queue, cmds.len, &cmds);
    _ = c.wgpuDevicePoll(device, 1, null);

    // Map → memcpy → unmap.
    var map_state = MapState{};
    const map_cb = c.WGPUBufferMapCallbackInfo{
        .nextInChain = null,
        .mode = c.WGPUCallbackMode_AllowProcessEvents,
        .callback = onBufferMap,
        .userdata1 = &map_state,
        .userdata2 = null,
    };
    _ = c.wgpuBufferMapAsync(readback, c.WGPUMapMode_Read, 0, output.bytes.len, map_cb);
    var spins: u32 = 0;
    while (!map_state.fired and spins < 1_000_000) : (spins += 1) {
        _ = c.wgpuDevicePoll(device, 0, null);
        c.wgpuInstanceProcessEvents(handles.instance);
    }
    if (!map_state.fired or map_state.status != c.WGPUMapAsyncStatus_Success) {
        return WgpuError.BufferMapFailed;
    }
    const mapped = c.wgpuBufferGetMappedRange(readback, 0, output.bytes.len);
    if (mapped == null) {
        c.wgpuBufferUnmap(readback);
        return WgpuError.BufferRangeFailed;
    }
    @memcpy(output.bytes, @as([*]const u8, @ptrCast(mapped))[0..output.bytes.len]);
    c.wgpuBufferUnmap(readback);
}

pub fn dispatchRmsnorm(
    uniform_bytes: []const u8,
    x_bytes: []const u8,
    gamma_bytes: []const u8,
    output: []u8,
    dispatch_x: u32,
) WgpuError!void {
    return dispatchKernel(
        @embedFile("shaders/rms_norm.wgsl"),
        uniform_bytes,
        0,
        &[_]StorageInput{
            .{ .binding = 1, .bytes = x_bytes },
            .{ .binding = 2, .bytes = gamma_bytes },
        },
        .{ .binding = 3, .bytes = output },
        null,
        &[_]StorageInput{},
        .{ dispatch_x, 1, 1 },
    );
}

/// Persistent backend: owns instance + adapter + device + queue.
/// One-time setup; many dispatches per backend lifetime. This is
/// what the inference pipeline + benchmark scripts use — the per-call
/// dispatchByName entry point creates+destroys for one shot, so its
/// numbers include adapter init time.
pub const Backend = struct {
    instance: c.WGPUInstance,
    adapter: c.WGPUAdapter,
    device: c.WGPUDevice,
    queue: c.WGPUQueue,
};

pub fn createBackend() WgpuError!*Backend {
    const handles = try createInstanceAndAdapter();
    errdefer c.wgpuInstanceRelease(handles.instance);
    errdefer c.wgpuAdapterRelease(handles.adapter);

    try requireShaderF16(handles.adapter);
    const device = try createDevice(handles.instance, handles.adapter);
    errdefer c.wgpuDeviceRelease(device);
    const queue = c.wgpuDeviceGetQueue(device);
    errdefer c.wgpuQueueRelease(queue);

    const b = std.heap.c_allocator.create(Backend) catch return WgpuError.BufferCreateFailed;
    b.* = .{
        .instance = handles.instance,
        .adapter = handles.adapter,
        .device = device,
        .queue = queue,
    };
    return b;
}

pub fn destroyBackend(b: *Backend) void {
    c.wgpuQueueRelease(b.queue);
    c.wgpuDeviceRelease(b.device);
    c.wgpuAdapterRelease(b.adapter);
    c.wgpuInstanceRelease(b.instance);
    std.heap.c_allocator.destroy(b);
}

/// Same as dispatchKernel but uses an existing Backend (no per-call
/// instance/adapter/device init). Buffers + pipeline are still
/// per-call — the next milestone caches pipelines too.
pub fn dispatchOnBackend(
    b: *Backend,
    wgsl_src: []const u8,
    uniform_bytes: []const u8,
    uniform_binding: u32,
    inputs: []const StorageInput,
    output: KernelOutput,
    output_initial: ?[]const u8,
    extra_uniforms: []const StorageInput,
    dispatch: [3]u32,
) WgpuError!void {
    const device = b.device;
    const queue = b.queue;

    var wgsl_chain: c.WGPUShaderSourceWGSL = std.mem.zeroes(c.WGPUShaderSourceWGSL);
    wgsl_chain.chain.next = null;
    wgsl_chain.chain.sType = c.WGPUSType_ShaderSourceWGSL;
    wgsl_chain.code.data = wgsl_src.ptr;
    wgsl_chain.code.length = wgsl_src.len;
    var module_desc: c.WGPUShaderModuleDescriptor = std.mem.zeroes(c.WGPUShaderModuleDescriptor);
    module_desc.nextInChain = @ptrCast(&wgsl_chain.chain);
    module_desc.label = .{ .data = null, .length = std.math.maxInt(usize) };
    const module = c.wgpuDeviceCreateShaderModule(device, &module_desc);
    if (module == null) return WgpuError.ShaderModuleFailed;
    defer c.wgpuShaderModuleRelease(module);

    var pipeline_desc: c.WGPUComputePipelineDescriptor = std.mem.zeroes(c.WGPUComputePipelineDescriptor);
    pipeline_desc.layout = null;
    pipeline_desc.compute.module = module;
    pipeline_desc.compute.entryPoint = .{ .data = "main", .length = 4 };
    pipeline_desc.label = .{ .data = null, .length = std.math.maxInt(usize) };
    const pipeline = c.wgpuDeviceCreateComputePipeline(device, &pipeline_desc);
    if (pipeline == null) return WgpuError.ComputePipelineFailed;
    defer c.wgpuComputePipelineRelease(pipeline);

    var u_buf: c.WGPUBuffer = null;
    if (uniform_bytes.len > 0) {
        var u_desc: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
        u_desc.usage = c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst;
        u_desc.size = uniform_bytes.len;
        u_buf = c.wgpuDeviceCreateBuffer(device, &u_desc);
        if (u_buf == null) return WgpuError.BufferCreateFailed;
        c.wgpuQueueWriteBuffer(queue, u_buf, 0, uniform_bytes.ptr, uniform_bytes.len);
    }
    defer if (u_buf != null) c.wgpuBufferRelease(u_buf);

    var input_bufs: [16]c.WGPUBuffer = undefined;
    if (inputs.len > 16) return WgpuError.BufferRangeFailed;
    for (inputs, 0..) |input, i| {
        input_bufs[i] = try uploadStorageBuffer(device, input.bytes);
    }
    defer for (input_bufs[0..inputs.len]) |buf| c.wgpuBufferRelease(buf);

    var y_desc: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
    y_desc.usage = c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc | c.WGPUBufferUsage_CopyDst;
    y_desc.size = output.bytes.len;
    const y_buf = c.wgpuDeviceCreateBuffer(device, &y_desc);
    if (y_buf == null) return WgpuError.BufferCreateFailed;
    defer c.wgpuBufferRelease(y_buf);
    if (output_initial) |init_bytes| {
        if (init_bytes.len != output.bytes.len) return WgpuError.BufferRangeFailed;
        var stage_desc: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
        stage_desc.usage = c.WGPUBufferUsage_CopySrc;
        stage_desc.size = init_bytes.len;
        stage_desc.mappedAtCreation = 1;
        const stage = c.wgpuDeviceCreateBuffer(device, &stage_desc);
        if (stage == null) return WgpuError.BufferCreateFailed;
        defer c.wgpuBufferRelease(stage);
        const stage_ptr = c.wgpuBufferGetMappedRange(stage, 0, init_bytes.len);
        if (stage_ptr == null) return WgpuError.BufferRangeFailed;
        @memcpy(@as([*]u8, @ptrCast(stage_ptr))[0..init_bytes.len], init_bytes);
        c.wgpuBufferUnmap(stage);
        const enc0 = c.wgpuDeviceCreateCommandEncoder(device, null);
        defer c.wgpuCommandEncoderRelease(enc0);
        c.wgpuCommandEncoderCopyBufferToBuffer(enc0, stage, 0, y_buf, 0, init_bytes.len);
        const cmd0 = c.wgpuCommandEncoderFinish(enc0, null);
        defer c.wgpuCommandBufferRelease(cmd0);
        var cmds0 = [_]c.WGPUCommandBuffer{cmd0};
        c.wgpuQueueSubmit(queue, cmds0.len, &cmds0);
    }

    var extra_uniform_bufs: [8]c.WGPUBuffer = undefined;
    if (extra_uniforms.len > 8) return WgpuError.BufferRangeFailed;
    for (extra_uniforms, 0..) |eu, i| {
        var ed: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
        ed.usage = c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst;
        ed.size = eu.bytes.len;
        const eb = c.wgpuDeviceCreateBuffer(device, &ed);
        if (eb == null) return WgpuError.BufferCreateFailed;
        c.wgpuQueueWriteBuffer(queue, eb, 0, eu.bytes.ptr, eu.bytes.len);
        extra_uniform_bufs[i] = eb;
    }
    defer for (extra_uniform_bufs[0..extra_uniforms.len]) |buf| c.wgpuBufferRelease(buf);

    var read_desc: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
    read_desc.usage = c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst;
    read_desc.size = output.bytes.len;
    const readback = c.wgpuDeviceCreateBuffer(device, &read_desc);
    if (readback == null) return WgpuError.BufferCreateFailed;
    defer c.wgpuBufferRelease(readback);

    var entries: [32]c.WGPUBindGroupEntry = undefined;
    var entry_count: usize = 0;
    if (uniform_bytes.len > 0) {
        entries[entry_count] = .{ .nextInChain = null, .binding = uniform_binding, .buffer = u_buf, .offset = 0, .size = uniform_bytes.len, .sampler = null, .textureView = null };
        entry_count += 1;
    }
    for (extra_uniforms, 0..) |eu, i| {
        entries[entry_count] = .{ .nextInChain = null, .binding = eu.binding, .buffer = extra_uniform_bufs[i], .offset = 0, .size = eu.bytes.len, .sampler = null, .textureView = null };
        entry_count += 1;
    }
    for (inputs, 0..) |input, i| {
        entries[entry_count] = .{ .nextInChain = null, .binding = input.binding, .buffer = input_bufs[i], .offset = 0, .size = input.bytes.len, .sampler = null, .textureView = null };
        entry_count += 1;
    }
    entries[entry_count] = .{ .nextInChain = null, .binding = output.binding, .buffer = y_buf, .offset = 0, .size = output.bytes.len, .sampler = null, .textureView = null };
    entry_count += 1;

    const bgl = c.wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    defer c.wgpuBindGroupLayoutRelease(bgl);
    var bg_desc: c.WGPUBindGroupDescriptor = std.mem.zeroes(c.WGPUBindGroupDescriptor);
    bg_desc.layout = bgl;
    bg_desc.entryCount = entry_count;
    bg_desc.entries = &entries;
    const bg = c.wgpuDeviceCreateBindGroup(device, &bg_desc);
    if (bg == null) return WgpuError.BindGroupFailed;
    defer c.wgpuBindGroupRelease(bg);

    const encoder = c.wgpuDeviceCreateCommandEncoder(device, null);
    defer c.wgpuCommandEncoderRelease(encoder);
    const pass = c.wgpuCommandEncoderBeginComputePass(encoder, null);
    c.wgpuComputePassEncoderSetPipeline(pass, pipeline);
    c.wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, null);
    c.wgpuComputePassEncoderDispatchWorkgroups(pass, dispatch[0], dispatch[1], dispatch[2]);
    c.wgpuComputePassEncoderEnd(pass);
    c.wgpuComputePassEncoderRelease(pass);
    c.wgpuCommandEncoderCopyBufferToBuffer(encoder, y_buf, 0, readback, 0, output.bytes.len);
    const cmd = c.wgpuCommandEncoderFinish(encoder, null);
    defer c.wgpuCommandBufferRelease(cmd);
    var cmds = [_]c.WGPUCommandBuffer{cmd};
    c.wgpuQueueSubmit(queue, cmds.len, &cmds);
    _ = c.wgpuDevicePoll(device, 1, null);

    var map_state = MapState{};
    const map_cb = c.WGPUBufferMapCallbackInfo{
        .nextInChain = null,
        .mode = c.WGPUCallbackMode_AllowProcessEvents,
        .callback = onBufferMap,
        .userdata1 = &map_state,
        .userdata2 = null,
    };
    _ = c.wgpuBufferMapAsync(readback, c.WGPUMapMode_Read, 0, output.bytes.len, map_cb);
    var spins: u32 = 0;
    while (!map_state.fired and spins < 1_000_000) : (spins += 1) {
        _ = c.wgpuDevicePoll(device, 0, null);
        c.wgpuInstanceProcessEvents(b.instance);
    }
    if (!map_state.fired or map_state.status != c.WGPUMapAsyncStatus_Success) return WgpuError.BufferMapFailed;
    const mapped = c.wgpuBufferGetMappedRange(readback, 0, output.bytes.len);
    if (mapped == null) {
        c.wgpuBufferUnmap(readback);
        return WgpuError.BufferRangeFailed;
    }
    @memcpy(output.bytes, @as([*]const u8, @ptrCast(mapped))[0..output.bytes.len]);
    c.wgpuBufferUnmap(readback);
}

/// Setup-once / run-N kernel benchmark. Builds the shader module +
/// pipeline + uploads inputs + binds groups exactly once, then loops
/// the encode + submit + readback path N times. Returns per-iter
/// elapsed ms in `samples`.
///
/// This is what the bench script uses to compare native vs browser
/// — the browser path's microbench loop also amortizes setup, so
/// only the per-dispatch cost is being compared.
pub fn benchDispatch(
    b: *Backend,
    wgsl_src: []const u8,
    uniform_bytes: []const u8,
    uniform_binding: u32,
    inputs: []const StorageInput,
    output: KernelOutput,
    output_initial: ?[]const u8,
    extra_uniforms: []const StorageInput,
    dispatch: [3]u32,
    samples: []f64, // out: per-iter elapsed ms (length = N iterations)
) WgpuError!void {
    const device = b.device;
    const queue = b.queue;

    // ── ONE-TIME SETUP (excluded from timing) ──
    var wgsl_chain: c.WGPUShaderSourceWGSL = std.mem.zeroes(c.WGPUShaderSourceWGSL);
    wgsl_chain.chain.next = null;
    wgsl_chain.chain.sType = c.WGPUSType_ShaderSourceWGSL;
    wgsl_chain.code.data = wgsl_src.ptr;
    wgsl_chain.code.length = wgsl_src.len;
    var module_desc: c.WGPUShaderModuleDescriptor = std.mem.zeroes(c.WGPUShaderModuleDescriptor);
    module_desc.nextInChain = @ptrCast(&wgsl_chain.chain);
    module_desc.label = .{ .data = null, .length = std.math.maxInt(usize) };
    const module = c.wgpuDeviceCreateShaderModule(device, &module_desc);
    if (module == null) return WgpuError.ShaderModuleFailed;
    defer c.wgpuShaderModuleRelease(module);

    var pipeline_desc: c.WGPUComputePipelineDescriptor = std.mem.zeroes(c.WGPUComputePipelineDescriptor);
    pipeline_desc.layout = null;
    pipeline_desc.compute.module = module;
    pipeline_desc.compute.entryPoint = .{ .data = "main", .length = 4 };
    pipeline_desc.label = .{ .data = null, .length = std.math.maxInt(usize) };
    const pipeline = c.wgpuDeviceCreateComputePipeline(device, &pipeline_desc);
    if (pipeline == null) return WgpuError.ComputePipelineFailed;
    defer c.wgpuComputePipelineRelease(pipeline);

    var u_buf: c.WGPUBuffer = null;
    if (uniform_bytes.len > 0) {
        var u_desc: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
        u_desc.usage = c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst;
        u_desc.size = uniform_bytes.len;
        u_buf = c.wgpuDeviceCreateBuffer(device, &u_desc);
        if (u_buf == null) return WgpuError.BufferCreateFailed;
        c.wgpuQueueWriteBuffer(queue, u_buf, 0, uniform_bytes.ptr, uniform_bytes.len);
    }
    defer if (u_buf != null) c.wgpuBufferRelease(u_buf);

    var input_bufs: [16]c.WGPUBuffer = undefined;
    if (inputs.len > 16) return WgpuError.BufferRangeFailed;
    for (inputs, 0..) |input, i| {
        input_bufs[i] = try uploadStorageBuffer(device, input.bytes);
    }
    defer for (input_bufs[0..inputs.len]) |buf| c.wgpuBufferRelease(buf);

    var extra_uniform_bufs: [8]c.WGPUBuffer = undefined;
    if (extra_uniforms.len > 8) return WgpuError.BufferRangeFailed;
    for (extra_uniforms, 0..) |eu, i| {
        var ed: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
        ed.usage = c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst;
        ed.size = eu.bytes.len;
        const eb = c.wgpuDeviceCreateBuffer(device, &ed);
        if (eb == null) return WgpuError.BufferCreateFailed;
        c.wgpuQueueWriteBuffer(queue, eb, 0, eu.bytes.ptr, eu.bytes.len);
        extra_uniform_bufs[i] = eb;
    }
    defer for (extra_uniform_bufs[0..extra_uniforms.len]) |buf| c.wgpuBufferRelease(buf);

    var y_desc: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
    y_desc.usage = c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc | c.WGPUBufferUsage_CopyDst;
    y_desc.size = output.bytes.len;
    const y_buf = c.wgpuDeviceCreateBuffer(device, &y_desc);
    if (y_buf == null) return WgpuError.BufferCreateFailed;
    defer c.wgpuBufferRelease(y_buf);

    var read_desc: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
    read_desc.usage = c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst;
    read_desc.size = output.bytes.len;
    const readback = c.wgpuDeviceCreateBuffer(device, &read_desc);
    if (readback == null) return WgpuError.BufferCreateFailed;
    defer c.wgpuBufferRelease(readback);

    var entries: [32]c.WGPUBindGroupEntry = undefined;
    var entry_count: usize = 0;
    if (uniform_bytes.len > 0) {
        entries[entry_count] = .{ .nextInChain = null, .binding = uniform_binding, .buffer = u_buf, .offset = 0, .size = uniform_bytes.len, .sampler = null, .textureView = null };
        entry_count += 1;
    }
    for (extra_uniforms, 0..) |eu, i| {
        entries[entry_count] = .{ .nextInChain = null, .binding = eu.binding, .buffer = extra_uniform_bufs[i], .offset = 0, .size = eu.bytes.len, .sampler = null, .textureView = null };
        entry_count += 1;
    }
    for (inputs, 0..) |input, i| {
        entries[entry_count] = .{ .nextInChain = null, .binding = input.binding, .buffer = input_bufs[i], .offset = 0, .size = input.bytes.len, .sampler = null, .textureView = null };
        entry_count += 1;
    }
    entries[entry_count] = .{ .nextInChain = null, .binding = output.binding, .buffer = y_buf, .offset = 0, .size = output.bytes.len, .sampler = null, .textureView = null };
    entry_count += 1;

    const bgl = c.wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    defer c.wgpuBindGroupLayoutRelease(bgl);
    var bg_desc: c.WGPUBindGroupDescriptor = std.mem.zeroes(c.WGPUBindGroupDescriptor);
    bg_desc.layout = bgl;
    bg_desc.entryCount = entry_count;
    bg_desc.entries = &entries;
    const bg = c.wgpuDeviceCreateBindGroup(device, &bg_desc);
    if (bg == null) return WgpuError.BindGroupFailed;
    defer c.wgpuBindGroupRelease(bg);

    // ── TIMED LOOP ──
    // chain_len controls how many dispatches per submit. chain_len=1
    // is the per-call latency case; chain_len=N approximates a forward
    // pass (one submit + one readback amortized over N kernel calls).
    var i: usize = 0;
    const chain_len = output.chain_len;
    while (i < samples.len) : (i += 1) {
        if (output_initial) |init_bytes| {
            c.wgpuQueueWriteBuffer(queue, y_buf, 0, init_bytes.ptr, init_bytes.len);
        }
        const t0 = nanos();
        const encoder = c.wgpuDeviceCreateCommandEncoder(device, null);
        const pass = c.wgpuCommandEncoderBeginComputePass(encoder, null);
        c.wgpuComputePassEncoderSetPipeline(pass, pipeline);
        c.wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, null);
        var ci: u32 = 0;
        while (ci < chain_len) : (ci += 1) {
            c.wgpuComputePassEncoderDispatchWorkgroups(pass, dispatch[0], dispatch[1], dispatch[2]);
        }
        c.wgpuComputePassEncoderEnd(pass);
        c.wgpuComputePassEncoderRelease(pass);
        c.wgpuCommandEncoderCopyBufferToBuffer(encoder, y_buf, 0, readback, 0, output.bytes.len);
        const cmd = c.wgpuCommandEncoderFinish(encoder, null);
        c.wgpuCommandEncoderRelease(encoder);
        var cmds = [_]c.WGPUCommandBuffer{cmd};
        c.wgpuQueueSubmit(queue, cmds.len, &cmds);
        c.wgpuCommandBufferRelease(cmd);

        // Skip the explicit blocking devicePoll — mapAsync on a buffer
        // that's the dst of a CopyBufferToBuffer queued AFTER the
        // dispatch implicitly waits for both. We just need to drive
        // the event loop until the map callback fires.
        var map_state = MapState{};
        const map_cb = c.WGPUBufferMapCallbackInfo{
            .nextInChain = null,
            .mode = c.WGPUCallbackMode_AllowProcessEvents,
            .callback = onBufferMap,
            .userdata1 = &map_state,
            .userdata2 = null,
        };
        _ = c.wgpuBufferMapAsync(readback, c.WGPUMapMode_Read, 0, output.bytes.len, map_cb);
        var spins: u32 = 0;
        while (!map_state.fired and spins < 1_000_000) : (spins += 1) {
            _ = c.wgpuDevicePoll(device, 0, null);
            c.wgpuInstanceProcessEvents(b.instance);
        }
        if (!map_state.fired or map_state.status != c.WGPUMapAsyncStatus_Success) return WgpuError.BufferMapFailed;
        c.wgpuBufferUnmap(readback);
        const t1 = nanos();
        samples[i] = @as(f64, @floatFromInt(t1 - t0)) / 1_000_000.0;
    }
}

/// Generic shader-by-name dispatch driver. The conformance test
/// passes the WGSL filename + raw fixture bytes; we map name → embed
/// → call dispatchKernel. Each entry is one line — no per-shader
/// bespoke wrapper code.
pub fn dispatchByName(
    name: []const u8,
    uniform_bytes: []const u8,
    extra_uniforms: []const StorageInput,
    inputs: []const StorageInput,
    output: KernelOutput,
    output_initial: ?[]const u8,
    dispatch: [3]u32,
) WgpuError!void {
    const wgsl = shaderByName(name) orelse return WgpuError.ShaderModuleFailed;
    return dispatchKernel(wgsl, uniform_bytes, 0, inputs, output, output_initial, extra_uniforms, dispatch);
}

pub fn shaderByName_pub(name: []const u8) ?[]const u8 {
    return shaderByName(name);
}

/// Static lookup of every embedded shader. Adding a new shader =
/// one line below + writing the .wgsl file + adding it to the
/// conformance harness page.
fn shaderByName(name: []const u8) ?[]const u8 {
    const eql = std.mem.eql;
    if (eql(u8, name, "rms_norm")) return @embedFile("shaders/rms_norm.wgsl");
    if (eql(u8, name, "embed_lookup_int4")) return @embedFile("shaders/embed_lookup_int4.wgsl");
    if (eql(u8, name, "rope_apply")) return @embedFile("shaders/rope_apply.wgsl");
    if (eql(u8, name, "matmul_int4_fp16_f16")) return @embedFile("shaders/matmul_int4_fp16_f16.wgsl");
    if (eql(u8, name, "matmul_int4_f32_f32")) return @embedFile("shaders/matmul_int4_f32_f32.wgsl");
    if (eql(u8, name, "banded_attention")) return @embedFile("shaders/banded_attention.wgsl");
    if (eql(u8, name, "swiglu_clamp")) return @embedFile("shaders/swiglu_clamp.wgsl");
    if (eql(u8, name, "router_topk")) return @embedFile("shaders/router_topk.wgsl");
    if (eql(u8, name, "qmoe_gate_up")) return @embedFile("shaders/qmoe_gate_up.wgsl");
    if (eql(u8, name, "qmoe_down_scatter")) return @embedFile("shaders/qmoe_down_scatter.wgsl");
    if (eql(u8, name, "add_rmsnorm_fp16_to_f32")) return @embedFile("shaders/add_rmsnorm_fp16_to_f32.wgsl");
    if (eql(u8, name, "cast_fp16_to_f32")) return @embedFile("shaders/cast_fp16_to_f32.wgsl");
    if (eql(u8, name, "cast_f32_to_fp16_scaled")) return @embedFile("shaders/cast_f32_to_fp16_scaled.wgsl");
    if (eql(u8, name, "add_fp16")) return @embedFile("shaders/add_fp16.wgsl");
    if (eql(u8, name, "zero_f32")) return @embedFile("shaders/zero_f32.wgsl");
    return null;
}

/// Helper: create a Storage|CopyDst buffer, upload `bytes` via a
/// staged copy, return the buffer handle.
fn uploadStorageBuffer(device: c.WGPUDevice, bytes: []const u8) WgpuError!c.WGPUBuffer {
    var stage_desc: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
    stage_desc.usage = c.WGPUBufferUsage_CopySrc;
    stage_desc.size = bytes.len;
    stage_desc.mappedAtCreation = 1;
    const stage = c.wgpuDeviceCreateBuffer(device, &stage_desc);
    if (stage == null) return WgpuError.BufferCreateFailed;
    defer c.wgpuBufferRelease(stage);
    const stage_ptr = c.wgpuBufferGetMappedRange(stage, 0, bytes.len);
    if (stage_ptr == null) return WgpuError.BufferRangeFailed;
    @memcpy(@as([*]u8, @ptrCast(stage_ptr))[0..bytes.len], bytes);
    c.wgpuBufferUnmap(stage);

    var dst_desc: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
    dst_desc.usage = c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst;
    dst_desc.size = bytes.len;
    const dst = c.wgpuDeviceCreateBuffer(device, &dst_desc);
    if (dst == null) return WgpuError.BufferCreateFailed;

    const queue = c.wgpuDeviceGetQueue(device);
    defer c.wgpuQueueRelease(queue);
    const encoder = c.wgpuDeviceCreateCommandEncoder(device, null);
    defer c.wgpuCommandEncoderRelease(encoder);
    c.wgpuCommandEncoderCopyBufferToBuffer(encoder, stage, 0, dst, 0, bytes.len);
    const cmd = c.wgpuCommandEncoderFinish(encoder, null);
    defer c.wgpuCommandBufferRelease(cmd);
    var cmds = [_]c.WGPUCommandBuffer{cmd};
    c.wgpuQueueSubmit(queue, cmds.len, &cmds);
    return dst;
}

/// Bring up the full instance + adapter + device chain, return both
/// the adapter info and the limits the device was granted, then
/// release everything. Used by `getDeviceInfo()` to give callers a
/// one-shot capability probe.
pub fn probeDevice(allocator: std.mem.Allocator) WgpuError!DeviceInfo {
    const handles = try createInstanceAndAdapter();
    defer c.wgpuInstanceRelease(handles.instance);
    defer c.wgpuAdapterRelease(handles.adapter);

    try requireShaderF16(handles.adapter);
    const adapter_info = try getAdapterInfo(handles.adapter, allocator);
    const device = try createDevice(handles.instance, handles.adapter);
    defer c.wgpuDeviceRelease(device);

    const limits = try getDeviceLimits(device);
    return DeviceInfo{
        .adapter = adapter_info,
        .max_storage_buffer_binding_size = limits.maxStorageBufferBindingSize,
        .max_buffer_size = limits.maxBufferSize,
        .max_storage_buffers_per_shader_stage = limits.maxStorageBuffersPerShaderStage,
        .max_compute_workgroup_storage_size = limits.maxComputeWorkgroupStorageSize,
        .max_compute_invocations_per_workgroup = limits.maxComputeInvocationsPerWorkgroup,
        .max_compute_workgroup_size_x = limits.maxComputeWorkgroupSizeX,
    };
}
