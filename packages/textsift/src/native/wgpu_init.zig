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
