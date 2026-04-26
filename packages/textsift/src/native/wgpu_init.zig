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
