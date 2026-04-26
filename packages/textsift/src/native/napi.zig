// Node-API shim. Single entry point `napi_register_module_v1` that
// surfaces the native functions implemented in `wgpu_init.zig`
// (and future inference modules) as JS exports.
//
// Functions exported today:
//   getAdapterInfo() → { vendor, architecture, device, description,
//                        backendType, adapterType }
//     Throws if no GPU adapter is available — caller should catch
//     and route to textsift/browser per the contract in src/index.ts.

const std = @import("std");
const wgpu = @import("wgpu_init.zig");

const c = @cImport({
    @cInclude("node_api.h");
});

export fn napi_register_module_v1(env: c.napi_env, exports: c.napi_value) c.napi_value {
    register(env, exports, "getAdapterInfo", napiGetAdapterInfo) catch return null;
    register(env, exports, "getDeviceInfo", napiGetDeviceInfo) catch return null;
    register(env, exports, "roundtripBuffer", napiRoundtripBuffer) catch return null;
    return exports;
}

fn register(
    env: c.napi_env,
    exports: c.napi_value,
    name: [*:0]const u8,
    cb: c.napi_callback,
) !void {
    var fn_value: c.napi_value = undefined;
    if (c.napi_create_function(env, name, c.NAPI_AUTO_LENGTH, cb, null, &fn_value) != c.napi_ok) {
        return error.RegisterFailed;
    }
    if (c.napi_set_named_property(env, exports, name, fn_value) != c.napi_ok) {
        return error.RegisterFailed;
    }
}

fn napiThrow(env: c.napi_env, msg: [*:0]const u8) c.napi_value {
    _ = c.napi_throw_error(env, null, msg);
    return null;
}

fn napiSetString(env: c.napi_env, obj: c.napi_value, key: [*:0]const u8, value: []const u8) !void {
    var jstr: c.napi_value = undefined;
    if (c.napi_create_string_utf8(env, value.ptr, value.len, &jstr) != c.napi_ok) {
        return error.SetStringFailed;
    }
    if (c.napi_set_named_property(env, obj, key, jstr) != c.napi_ok) {
        return error.SetStringFailed;
    }
}

fn napiSetU32(env: c.napi_env, obj: c.napi_value, key: [*:0]const u8, value: u32) !void {
    var num: c.napi_value = undefined;
    if (c.napi_create_uint32(env, value, &num) != c.napi_ok) {
        return error.SetU32Failed;
    }
    if (c.napi_set_named_property(env, obj, key, num) != c.napi_ok) {
        return error.SetU32Failed;
    }
}

fn napiSetU64AsBigint(env: c.napi_env, obj: c.napi_value, key: [*:0]const u8, value: u64) !void {
    var bn: c.napi_value = undefined;
    if (c.napi_create_bigint_uint64(env, value, &bn) != c.napi_ok) {
        return error.SetU64Failed;
    }
    if (c.napi_set_named_property(env, obj, key, bn) != c.napi_ok) {
        return error.SetU64Failed;
    }
}

fn napiSetObject(env: c.napi_env, obj: c.napi_value, key: [*:0]const u8, value: c.napi_value) !void {
    if (c.napi_set_named_property(env, obj, key, value) != c.napi_ok) {
        return error.SetObjectFailed;
    }
}

fn napiGetAdapterInfo(env: c.napi_env, _: c.napi_callback_info) callconv(.c) c.napi_value {
    const handles = wgpu.createInstanceAndAdapter() catch |err| {
        return switch (err) {
            wgpu.WgpuError.InstanceCreateFailed =>
                napiThrow(env, "wgpu: failed to create instance"),
            wgpu.WgpuError.AdapterRequestFailed =>
                napiThrow(env, "wgpu: adapter request future failed"),
            wgpu.WgpuError.AdapterUnavailable =>
                napiThrow(env, "WebGPU adapter not available on this host. Use textsift/browser instead."),
            else => napiThrow(env, "wgpu: adapter init failed"),
        };
    };
    defer wgpu.c.wgpuAdapterRelease(handles.adapter);
    defer wgpu.c.wgpuInstanceRelease(handles.instance);

    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();
    const info = wgpu.getAdapterInfo(handles.adapter, arena.allocator()) catch {
        return napiThrow(env, "wgpu: adapter info query failed");
    };

    var obj: c.napi_value = undefined;
    if (c.napi_create_object(env, &obj) != c.napi_ok) {
        return napiThrow(env, "napi: failed to create object");
    }
    napiSetString(env, obj, "vendor", info.vendor) catch return napiThrow(env, "napi: failed setting vendor");
    napiSetString(env, obj, "architecture", info.architecture) catch return napiThrow(env, "napi: failed setting architecture");
    napiSetString(env, obj, "device", info.device) catch return napiThrow(env, "napi: failed setting device");
    napiSetString(env, obj, "description", info.description) catch return napiThrow(env, "napi: failed setting description");
    napiSetU32(env, obj, "backendType", info.backend_type) catch return napiThrow(env, "napi: failed setting backendType");
    napiSetU32(env, obj, "adapterType", info.adapter_type) catch return napiThrow(env, "napi: failed setting adapterType");
    return obj;
}

fn buildAdapterObject(env: c.napi_env, info: wgpu.AdapterInfo) !c.napi_value {
    var obj: c.napi_value = undefined;
    if (c.napi_create_object(env, &obj) != c.napi_ok) return error.CreateObjectFailed;
    try napiSetString(env, obj, "vendor", info.vendor);
    try napiSetString(env, obj, "architecture", info.architecture);
    try napiSetString(env, obj, "device", info.device);
    try napiSetString(env, obj, "description", info.description);
    try napiSetU32(env, obj, "backendType", info.backend_type);
    try napiSetU32(env, obj, "adapterType", info.adapter_type);
    return obj;
}

fn napiRoundtripBuffer(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 1;
    var argv: [1]c.napi_value = undefined;
    if (c.napi_get_cb_info(env, info, &argc, &argv, null, null) != c.napi_ok) {
        return napiThrow(env, "napi_get_cb_info failed");
    }
    if (argc < 1) {
        return napiThrow(env, "roundtripBuffer(input: Uint8Array) requires one argument");
    }

    var data_ptr: ?*anyopaque = null;
    var byte_len: usize = 0;
    if (c.napi_get_typedarray_info(env, argv[0], null, &byte_len, &data_ptr, null, null) != c.napi_ok) {
        return napiThrow(env, "argument must be a Uint8Array");
    }
    if (data_ptr == null or byte_len == 0) {
        return napiThrow(env, "input Uint8Array is empty");
    }

    const input = @as([*]const u8, @ptrCast(data_ptr.?))[0..byte_len];

    // Allocate the output Uint8Array first so wgpu writes into it
    // directly (avoids a stage-then-copy).
    var out_arraybuffer: c.napi_value = undefined;
    var out_data_ptr: ?*anyopaque = null;
    if (c.napi_create_arraybuffer(env, byte_len, &out_data_ptr, &out_arraybuffer) != c.napi_ok) {
        return napiThrow(env, "napi: failed to allocate output ArrayBuffer");
    }
    var out_typedarray: c.napi_value = undefined;
    if (c.napi_create_typedarray(env, c.napi_uint8_array, byte_len, out_arraybuffer, 0, &out_typedarray) != c.napi_ok) {
        return napiThrow(env, "napi: failed to allocate output Uint8Array");
    }
    const output = @as([*]u8, @ptrCast(out_data_ptr.?))[0..byte_len];

    wgpu.roundtripBuffer(input, output) catch |err| {
        return switch (err) {
            wgpu.WgpuError.BufferCreateFailed => napiThrow(env, "wgpu: device.createBuffer failed"),
            wgpu.WgpuError.BufferMapFailed => napiThrow(env, "wgpu: buffer.mapAsync failed"),
            wgpu.WgpuError.BufferRangeFailed => napiThrow(env, "wgpu: buffer.getMappedRange failed"),
            wgpu.WgpuError.AdapterUnavailable =>
                napiThrow(env, "WebGPU adapter not available on this host. Use textsift/browser instead."),
            wgpu.WgpuError.ShaderF16Unavailable =>
                napiThrow(env, "WebGPU adapter lacks shader-f16. Use textsift/browser instead."),
            else => napiThrow(env, "wgpu: roundtripBuffer failed"),
        };
    };

    return out_typedarray;
}

fn napiGetDeviceInfo(env: c.napi_env, _: c.napi_callback_info) callconv(.c) c.napi_value {
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();

    const info = wgpu.probeDevice(arena.allocator()) catch |err| {
        return switch (err) {
            wgpu.WgpuError.InstanceCreateFailed =>
                napiThrow(env, "wgpu: failed to create instance"),
            wgpu.WgpuError.AdapterRequestFailed =>
                napiThrow(env, "wgpu: adapter request future failed"),
            wgpu.WgpuError.AdapterUnavailable =>
                napiThrow(env, "WebGPU adapter not available on this host. Use textsift/browser instead."),
            wgpu.WgpuError.AdapterInfoFailed =>
                napiThrow(env, "wgpu: adapter info query failed"),
            wgpu.WgpuError.ShaderF16Unavailable =>
                napiThrow(env, "WebGPU adapter lacks shader-f16 feature; required for the int4-fp16 matmul kernels. Use textsift/browser instead."),
            wgpu.WgpuError.DeviceRequestFailed =>
                napiThrow(env, "wgpu: device request future failed"),
            wgpu.WgpuError.DeviceUnavailable =>
                napiThrow(env, "wgpu: requestDevice returned no device"),
            wgpu.WgpuError.DeviceLimitsFailed =>
                napiThrow(env, "wgpu: failed to query device limits"),
            else => napiThrow(env, "wgpu: device init failed"),
        };
    };

    var obj: c.napi_value = undefined;
    if (c.napi_create_object(env, &obj) != c.napi_ok) {
        return napiThrow(env, "napi: failed to create object");
    }
    const adapter_obj = buildAdapterObject(env, info.adapter) catch
        return napiThrow(env, "napi: failed to build adapter object");
    napiSetObject(env, obj, "adapter", adapter_obj) catch
        return napiThrow(env, "napi: failed setting adapter");
    napiSetU64AsBigint(env, obj, "maxStorageBufferBindingSize", info.max_storage_buffer_binding_size) catch
        return napiThrow(env, "napi: failed setting maxStorageBufferBindingSize");
    napiSetU64AsBigint(env, obj, "maxBufferSize", info.max_buffer_size) catch
        return napiThrow(env, "napi: failed setting maxBufferSize");
    napiSetU32(env, obj, "maxStorageBuffersPerShaderStage", info.max_storage_buffers_per_shader_stage) catch
        return napiThrow(env, "napi: failed setting maxStorageBuffersPerShaderStage");
    napiSetU32(env, obj, "maxComputeWorkgroupStorageSize", info.max_compute_workgroup_storage_size) catch
        return napiThrow(env, "napi: failed setting maxComputeWorkgroupStorageSize");
    napiSetU32(env, obj, "maxComputeInvocationsPerWorkgroup", info.max_compute_invocations_per_workgroup) catch
        return napiThrow(env, "napi: failed setting maxComputeInvocationsPerWorkgroup");
    napiSetU32(env, obj, "maxComputeWorkgroupSizeX", info.max_compute_workgroup_size_x) catch
        return napiThrow(env, "napi: failed setting maxComputeWorkgroupSizeX");
    return obj;
}
