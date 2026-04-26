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
    register(env, exports, "dispatchDouble", napiDispatchDouble) catch return null;
    register(env, exports, "matmulF32", napiMatmulF32) catch return null;
    register(env, exports, "dispatchRmsnorm", napiDispatchRmsnorm) catch return null;
    register(env, exports, "dispatchByName", napiDispatchByName) catch return null;
    register(env, exports, "createBackend", napiCreateBackend) catch return null;
    register(env, exports, "backendDispatch", napiBackendDispatch) catch return null;
    register(env, exports, "destroyBackend", napiDestroyBackend) catch return null;
    register(env, exports, "benchDispatch", napiBenchDispatch) catch return null;
    register(env, exports, "createBuffer", napiCreateBuffer) catch return null;
    register(env, exports, "releaseBuffer", napiReleaseBuffer) catch return null;
    register(env, exports, "dispatchByBuffers", napiDispatchByBuffers) catch return null;
    register(env, exports, "writeBuffer", napiWriteBuffer) catch return null;
    register(env, exports, "readBuffer", napiReadBuffer) catch return null;
    register(env, exports, "createEmptyBuffer", napiCreateEmptyBuffer) catch return null;
    register(env, exports, "beginEncoder", napiBeginEncoder) catch return null;
    register(env, exports, "enqueueDispatch", napiEnqueueDispatch) catch return null;
    register(env, exports, "submitAndReadback", napiSubmitAndReadback) catch return null;
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

fn napiDispatchDouble(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 1;
    var argv: [1]c.napi_value = undefined;
    if (c.napi_get_cb_info(env, info, &argc, &argv, null, null) != c.napi_ok) {
        return napiThrow(env, "napi_get_cb_info failed");
    }
    if (argc < 1) {
        return napiThrow(env, "dispatchDouble(input: Float32Array) requires one argument");
    }

    var arr_type: c.napi_typedarray_type = undefined;
    var data_ptr: ?*anyopaque = null;
    var elem_count: usize = 0;
    if (c.napi_get_typedarray_info(env, argv[0], &arr_type, &elem_count, &data_ptr, null, null) != c.napi_ok) {
        return napiThrow(env, "argument must be a Float32Array");
    }
    if (arr_type != c.napi_float32_array) {
        return napiThrow(env, "argument must be a Float32Array (got different element type)");
    }
    if (data_ptr == null or elem_count == 0) {
        return napiThrow(env, "input Float32Array is empty");
    }

    const input = @as([*]const f32, @ptrCast(@alignCast(data_ptr.?)))[0..elem_count];

    const byte_len = elem_count * @sizeOf(f32);
    var out_arraybuffer: c.napi_value = undefined;
    var out_data_ptr: ?*anyopaque = null;
    if (c.napi_create_arraybuffer(env, byte_len, &out_data_ptr, &out_arraybuffer) != c.napi_ok) {
        return napiThrow(env, "napi: failed to allocate output ArrayBuffer");
    }
    var out_typedarray: c.napi_value = undefined;
    if (c.napi_create_typedarray(env, c.napi_float32_array, elem_count, out_arraybuffer, 0, &out_typedarray) != c.napi_ok) {
        return napiThrow(env, "napi: failed to allocate output Float32Array");
    }
    const output = @as([*]f32, @ptrCast(@alignCast(out_data_ptr.?)))[0..elem_count];

    wgpu.dispatchDouble(input, output) catch |err| {
        return switch (err) {
            wgpu.WgpuError.BufferCreateFailed => napiThrow(env, "wgpu: device.createBuffer failed"),
            wgpu.WgpuError.BufferMapFailed => napiThrow(env, "wgpu: buffer.mapAsync failed"),
            wgpu.WgpuError.BufferRangeFailed => napiThrow(env, "wgpu: buffer.getMappedRange failed"),
            wgpu.WgpuError.ShaderModuleFailed => napiThrow(env, "wgpu: device.createShaderModule failed"),
            wgpu.WgpuError.ComputePipelineFailed => napiThrow(env, "wgpu: device.createComputePipeline failed"),
            wgpu.WgpuError.BindGroupFailed => napiThrow(env, "wgpu: device.createBindGroup failed"),
            wgpu.WgpuError.AdapterUnavailable =>
                napiThrow(env, "WebGPU adapter not available on this host. Use textsift/browser instead."),
            wgpu.WgpuError.ShaderF16Unavailable =>
                napiThrow(env, "WebGPU adapter lacks shader-f16. Use textsift/browser instead."),
            else => napiThrow(env, "wgpu: dispatchDouble failed"),
        };
    };

    return out_typedarray;
}

fn napiGetU32(env: c.napi_env, value: c.napi_value, name: [*:0]const u8) ?u32 {
    var out: u32 = 0;
    if (c.napi_get_value_uint32(env, value, &out) != c.napi_ok) {
        _ = c.napi_throw_error(env, null, name);
        return null;
    }
    return out;
}

fn napiGetFloat32Array(env: c.napi_env, value: c.napi_value, label: [*:0]const u8) ?[]const f32 {
    var arr_type: c.napi_typedarray_type = undefined;
    var data_ptr: ?*anyopaque = null;
    var elem_count: usize = 0;
    if (c.napi_get_typedarray_info(env, value, &arr_type, &elem_count, &data_ptr, null, null) != c.napi_ok) {
        _ = c.napi_throw_error(env, null, label);
        return null;
    }
    if (arr_type != c.napi_float32_array or data_ptr == null) {
        _ = c.napi_throw_error(env, null, label);
        return null;
    }
    return @as([*]const f32, @ptrCast(@alignCast(data_ptr.?)))[0..elem_count];
}

fn napiMatmulF32(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 5;
    var argv: [5]c.napi_value = undefined;
    if (c.napi_get_cb_info(env, info, &argc, &argv, null, null) != c.napi_ok) {
        return napiThrow(env, "napi_get_cb_info failed");
    }
    if (argc < 5) {
        return napiThrow(env, "matmulF32(a, b, m, n, k) requires five arguments");
    }
    const a = napiGetFloat32Array(env, argv[0], "argument 0 must be Float32Array (a)") orelse return null;
    const b = napiGetFloat32Array(env, argv[1], "argument 1 must be Float32Array (b)") orelse return null;
    const m = napiGetU32(env, argv[2], "argument 2 must be u32 (m)") orelse return null;
    const n = napiGetU32(env, argv[3], "argument 3 must be u32 (n)") orelse return null;
    const k = napiGetU32(env, argv[4], "argument 4 must be u32 (k)") orelse return null;

    if (a.len < @as(usize, m) * @as(usize, k)) return napiThrow(env, "a length < m*k");
    if (b.len < @as(usize, k) * @as(usize, n)) return napiThrow(env, "b length < k*n");

    const out_count: usize = @as(usize, m) * @as(usize, n);
    const byte_len = out_count * @sizeOf(f32);
    var out_arraybuffer: c.napi_value = undefined;
    var out_data_ptr: ?*anyopaque = null;
    if (c.napi_create_arraybuffer(env, byte_len, &out_data_ptr, &out_arraybuffer) != c.napi_ok) {
        return napiThrow(env, "napi: failed to allocate output ArrayBuffer");
    }
    var out_typedarray: c.napi_value = undefined;
    if (c.napi_create_typedarray(env, c.napi_float32_array, out_count, out_arraybuffer, 0, &out_typedarray) != c.napi_ok) {
        return napiThrow(env, "napi: failed to allocate output Float32Array");
    }
    const output = @as([*]f32, @ptrCast(@alignCast(out_data_ptr.?)))[0..out_count];

    wgpu.dispatchMatmulF32(a, b, output, m, n, k) catch |err| {
        return switch (err) {
            wgpu.WgpuError.BufferCreateFailed => napiThrow(env, "wgpu: createBuffer failed"),
            wgpu.WgpuError.BufferMapFailed => napiThrow(env, "wgpu: mapAsync failed"),
            wgpu.WgpuError.BufferRangeFailed => napiThrow(env, "wgpu: buffer range failed"),
            wgpu.WgpuError.ShaderModuleFailed => napiThrow(env, "wgpu: createShaderModule failed"),
            wgpu.WgpuError.ComputePipelineFailed => napiThrow(env, "wgpu: createComputePipeline failed"),
            wgpu.WgpuError.BindGroupFailed => napiThrow(env, "wgpu: createBindGroup failed"),
            wgpu.WgpuError.AdapterUnavailable =>
                napiThrow(env, "WebGPU adapter not available on this host. Use textsift/browser instead."),
            wgpu.WgpuError.ShaderF16Unavailable =>
                napiThrow(env, "WebGPU adapter lacks shader-f16. Use textsift/browser instead."),
            else => napiThrow(env, "wgpu: matmulF32 failed"),
        };
    };

    return out_typedarray;
}

fn napiGetUint8Array(env: c.napi_env, value: c.napi_value, label: [*:0]const u8) ?[]const u8 {
    var arr_type: c.napi_typedarray_type = undefined;
    var data_ptr: ?*anyopaque = null;
    var byte_len: usize = 0;
    if (c.napi_get_typedarray_info(env, value, &arr_type, &byte_len, &data_ptr, null, null) != c.napi_ok) {
        _ = c.napi_throw_error(env, null, label);
        return null;
    }
    if (arr_type != c.napi_uint8_array or data_ptr == null) {
        _ = c.napi_throw_error(env, null, label);
        return null;
    }
    return @as([*]const u8, @ptrCast(data_ptr.?))[0..byte_len];
}

fn napiDispatchRmsnorm(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 5;
    var argv: [5]c.napi_value = undefined;
    if (c.napi_get_cb_info(env, info, &argc, &argv, null, null) != c.napi_ok) {
        return napiThrow(env, "napi_get_cb_info failed");
    }
    if (argc < 5) {
        return napiThrow(env, "dispatchRmsnorm(uniform, x, gamma, dispatchX, outputSize) requires five arguments");
    }
    const uniform = napiGetUint8Array(env, argv[0], "argument 0 must be Uint8Array (uniform)") orelse return null;
    const x = napiGetUint8Array(env, argv[1], "argument 1 must be Uint8Array (x)") orelse return null;
    const gamma = napiGetUint8Array(env, argv[2], "argument 2 must be Uint8Array (gamma)") orelse return null;
    const dispatch_x = napiGetU32(env, argv[3], "argument 3 must be u32 (dispatchX)") orelse return null;
    const output_size = napiGetU32(env, argv[4], "argument 4 must be u32 (outputSize)") orelse return null;

    var out_ab: c.napi_value = undefined;
    var out_data_ptr: ?*anyopaque = null;
    if (c.napi_create_arraybuffer(env, output_size, &out_data_ptr, &out_ab) != c.napi_ok) {
        return napiThrow(env, "napi: failed to allocate output ArrayBuffer");
    }
    var out_typed: c.napi_value = undefined;
    if (c.napi_create_typedarray(env, c.napi_uint8_array, output_size, out_ab, 0, &out_typed) != c.napi_ok) {
        return napiThrow(env, "napi: failed to allocate output Uint8Array");
    }
    const output = @as([*]u8, @ptrCast(out_data_ptr.?))[0..output_size];

    wgpu.dispatchRmsnorm(uniform, x, gamma, output, dispatch_x) catch |err| {
        return switch (err) {
            wgpu.WgpuError.BufferCreateFailed => napiThrow(env, "wgpu: createBuffer failed"),
            wgpu.WgpuError.BufferMapFailed => napiThrow(env, "wgpu: mapAsync failed"),
            wgpu.WgpuError.BufferRangeFailed => napiThrow(env, "wgpu: buffer range failed"),
            wgpu.WgpuError.ShaderModuleFailed => napiThrow(env, "wgpu: createShaderModule failed"),
            wgpu.WgpuError.ComputePipelineFailed => napiThrow(env, "wgpu: createComputePipeline failed"),
            wgpu.WgpuError.BindGroupFailed => napiThrow(env, "wgpu: createBindGroup failed"),
            wgpu.WgpuError.AdapterUnavailable =>
                napiThrow(env, "WebGPU adapter not available on this host. Use textsift/browser instead."),
            wgpu.WgpuError.ShaderF16Unavailable =>
                napiThrow(env, "WebGPU adapter lacks shader-f16. Use textsift/browser instead."),
            else => napiThrow(env, "wgpu: dispatchRmsnorm failed"),
        };
    };
    return out_typed;
}

fn napiGetString(env: c.napi_env, value: c.napi_value, label: [*:0]const u8) ?[]u8 {
    var byte_len: usize = 0;
    if (c.napi_get_value_string_utf8(env, value, null, 0, &byte_len) != c.napi_ok) {
        _ = c.napi_throw_error(env, null, label);
        return null;
    }
    // Zig allocator-managed buffer for the duration of the call.
    const buf = std.heap.c_allocator.alloc(u8, byte_len + 1) catch {
        _ = c.napi_throw_error(env, null, "napi: oom reading string");
        return null;
    };
    var actual: usize = 0;
    if (c.napi_get_value_string_utf8(env, value, buf.ptr, buf.len, &actual) != c.napi_ok) {
        std.heap.c_allocator.free(buf);
        _ = c.napi_throw_error(env, null, label);
        return null;
    }
    return buf[0..actual];
}

fn napiArrayLen(env: c.napi_env, arr: c.napi_value) ?u32 {
    var n: u32 = 0;
    if (c.napi_get_array_length(env, arr, &n) != c.napi_ok) return null;
    return n;
}

fn napiArrayGet(env: c.napi_env, arr: c.napi_value, i: u32) ?c.napi_value {
    var v: c.napi_value = undefined;
    if (c.napi_get_element(env, arr, i, &v) != c.napi_ok) return null;
    return v;
}

fn napiNamedProperty(env: c.napi_env, obj: c.napi_value, name: [*:0]const u8) ?c.napi_value {
    var v: c.napi_value = undefined;
    if (c.napi_get_named_property(env, obj, name, &v) != c.napi_ok) return null;
    return v;
}

/// Generic shader dispatch driven by name + JS-provided bindings.
///
/// JS signature:
///   dispatchByName(
///     name: string,
///     uniformBytes: Uint8Array,                    // primary uniform (binding 0); empty if no uniform
///     extraUniforms: Array<{binding: number, bytes: Uint8Array}>,
///     inputs: Array<{binding: number, bytes: Uint8Array}>,
///     output: {binding: number, byteLength: number, initial?: Uint8Array},
///     dispatch: [number, number, number],
///   ) → Uint8Array
fn napiDispatchByName(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 6;
    var argv: [6]c.napi_value = undefined;
    if (c.napi_get_cb_info(env, info, &argc, &argv, null, null) != c.napi_ok) {
        return napiThrow(env, "napi_get_cb_info failed");
    }
    if (argc < 6) {
        return napiThrow(env, "dispatchByName requires six arguments");
    }
    const name_buf = napiGetString(env, argv[0], "argument 0 must be string (name)") orelse return null;
    defer std.heap.c_allocator.free(name_buf);
    const uniform_bytes = napiGetUint8Array(env, argv[1], "argument 1 must be Uint8Array (uniform)") orelse return null;

    // Parse extra uniforms array.
    const extra_n = napiArrayLen(env, argv[2]) orelse return napiThrow(env, "argument 2 must be array (extraUniforms)");
    if (extra_n > 8) return napiThrow(env, "too many extraUniforms (max 8)");
    var extra_storage: [8]wgpu.StorageInput = undefined;
    var extra_buf_storage: [8][]const u8 = undefined;
    _ = &extra_buf_storage;
    var ei: u32 = 0;
    while (ei < extra_n) : (ei += 1) {
        const item = napiArrayGet(env, argv[2], ei) orelse return napiThrow(env, "extraUniforms[i] read failed");
        const binding_v = napiNamedProperty(env, item, "binding") orelse return napiThrow(env, "extraUniforms[i].binding missing");
        const bytes_v = napiNamedProperty(env, item, "bytes") orelse return napiThrow(env, "extraUniforms[i].bytes missing");
        const binding = napiGetU32(env, binding_v, "extraUniforms[i].binding must be u32") orelse return null;
        const bytes = napiGetUint8Array(env, bytes_v, "extraUniforms[i].bytes must be Uint8Array") orelse return null;
        extra_storage[ei] = .{ .binding = binding, .bytes = bytes };
    }

    // Parse inputs array.
    const in_n = napiArrayLen(env, argv[3]) orelse return napiThrow(env, "argument 3 must be array (inputs)");
    if (in_n > 16) return napiThrow(env, "too many inputs (max 16)");
    var in_storage: [16]wgpu.StorageInput = undefined;
    var ii: u32 = 0;
    while (ii < in_n) : (ii += 1) {
        const item = napiArrayGet(env, argv[3], ii) orelse return napiThrow(env, "inputs[i] read failed");
        const binding_v = napiNamedProperty(env, item, "binding") orelse return napiThrow(env, "inputs[i].binding missing");
        const bytes_v = napiNamedProperty(env, item, "bytes") orelse return napiThrow(env, "inputs[i].bytes missing");
        const binding = napiGetU32(env, binding_v, "inputs[i].binding must be u32") orelse return null;
        const bytes = napiGetUint8Array(env, bytes_v, "inputs[i].bytes must be Uint8Array") orelse return null;
        in_storage[ii] = .{ .binding = binding, .bytes = bytes };
    }

    // Parse output object: {binding, byteLength, initial?}.
    const out_obj = argv[4];
    const out_binding_v = napiNamedProperty(env, out_obj, "binding") orelse return napiThrow(env, "output.binding missing");
    const out_blen_v = napiNamedProperty(env, out_obj, "byteLength") orelse return napiThrow(env, "output.byteLength missing");
    const out_binding = napiGetU32(env, out_binding_v, "output.binding must be u32") orelse return null;
    const out_blen = napiGetU32(env, out_blen_v, "output.byteLength must be u32") orelse return null;

    // Parse optional output.initial.
    var out_initial: ?[]const u8 = null;
    {
        var v: c.napi_value = undefined;
        if (c.napi_get_named_property(env, out_obj, "initial", &v) == c.napi_ok) {
            var t: c.napi_valuetype = undefined;
            if (c.napi_typeof(env, v, &t) == c.napi_ok and t != c.napi_undefined and t != c.napi_null) {
                const b = napiGetUint8Array(env, v, "output.initial must be Uint8Array") orelse return null;
                out_initial = b;
            }
        }
    }

    // Parse dispatch [x, y, z].
    const disp_arr = argv[5];
    const dx_v = napiArrayGet(env, disp_arr, 0) orelse return napiThrow(env, "dispatch[0] missing");
    const dy_v = napiArrayGet(env, disp_arr, 1) orelse return napiThrow(env, "dispatch[1] missing");
    const dz_v = napiArrayGet(env, disp_arr, 2) orelse return napiThrow(env, "dispatch[2] missing");
    const dx = napiGetU32(env, dx_v, "dispatch[0] must be u32") orelse return null;
    const dy = napiGetU32(env, dy_v, "dispatch[1] must be u32") orelse return null;
    const dz = napiGetU32(env, dz_v, "dispatch[2] must be u32") orelse return null;

    // Allocate output Uint8Array.
    var out_ab: c.napi_value = undefined;
    var out_data_ptr: ?*anyopaque = null;
    if (c.napi_create_arraybuffer(env, out_blen, &out_data_ptr, &out_ab) != c.napi_ok) {
        return napiThrow(env, "napi: failed to allocate output ArrayBuffer");
    }
    var out_typed: c.napi_value = undefined;
    if (c.napi_create_typedarray(env, c.napi_uint8_array, out_blen, out_ab, 0, &out_typed) != c.napi_ok) {
        return napiThrow(env, "napi: failed to allocate output Uint8Array");
    }
    const out_slice = @as([*]u8, @ptrCast(out_data_ptr.?))[0..out_blen];

    wgpu.dispatchByName(
        name_buf,
        uniform_bytes,
        extra_storage[0..extra_n],
        in_storage[0..in_n],
        .{ .binding = out_binding, .bytes = out_slice },
        out_initial,
        .{ dx, dy, dz },
    ) catch |err| {
        return switch (err) {
            wgpu.WgpuError.ShaderModuleFailed => napiThrow(env, "wgpu: createShaderModule failed (unknown shader name?)"),
            wgpu.WgpuError.ComputePipelineFailed => napiThrow(env, "wgpu: createComputePipeline failed"),
            wgpu.WgpuError.BindGroupFailed => napiThrow(env, "wgpu: createBindGroup failed"),
            wgpu.WgpuError.BufferCreateFailed => napiThrow(env, "wgpu: createBuffer failed"),
            wgpu.WgpuError.BufferMapFailed => napiThrow(env, "wgpu: buffer.mapAsync failed"),
            wgpu.WgpuError.BufferRangeFailed => napiThrow(env, "wgpu: buffer range failed"),
            wgpu.WgpuError.AdapterUnavailable =>
                napiThrow(env, "WebGPU adapter not available on this host. Use textsift/browser instead."),
            wgpu.WgpuError.ShaderF16Unavailable =>
                napiThrow(env, "WebGPU adapter lacks shader-f16. Use textsift/browser instead."),
            else => napiThrow(env, "wgpu: dispatchByName failed"),
        };
    };
    return out_typed;
}

// Backend lifetime: a JS BigInt holds the raw pointer; createBackend
// returns one, backendDispatch reads it, destroyBackend frees. Plain
// pointer arithmetic — no NAPI external/finalizer dance. The JS layer
// owns disposal.

fn napiCreateBackend(env: c.napi_env, _: c.napi_callback_info) callconv(.c) c.napi_value {
    const b = wgpu.createBackend() catch |err| {
        return switch (err) {
            wgpu.WgpuError.AdapterUnavailable =>
                napiThrow(env, "WebGPU adapter not available on this host. Use textsift/browser instead."),
            wgpu.WgpuError.ShaderF16Unavailable =>
                napiThrow(env, "WebGPU adapter lacks shader-f16. Use textsift/browser instead."),
            else => napiThrow(env, "wgpu: createBackend failed"),
        };
    };
    var bn: c.napi_value = undefined;
    if (c.napi_create_bigint_uint64(env, @intCast(@intFromPtr(b)), &bn) != c.napi_ok) {
        wgpu.destroyBackend(b);
        return napiThrow(env, "napi: failed to create handle bigint");
    }
    return bn;
}

fn napiDestroyBackend(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 1;
    var argv: [1]c.napi_value = undefined;
    if (c.napi_get_cb_info(env, info, &argc, &argv, null, null) != c.napi_ok) {
        return napiThrow(env, "napi_get_cb_info failed");
    }
    if (argc < 1) return napiThrow(env, "destroyBackend(handle) requires one argument");
    var raw: u64 = 0;
    var lossless: bool = false;
    if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok) {
        return napiThrow(env, "argument 0 must be a BigInt handle");
    }
    if (raw != 0) {
        const b: *wgpu.Backend = @ptrFromInt(@as(usize, @intCast(raw)));
        wgpu.destroyBackend(b);
    }
    var undef: c.napi_value = undefined;
    _ = c.napi_get_undefined(env, &undef);
    return undef;
}

fn napiBackendDispatch(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 7;
    var argv: [7]c.napi_value = undefined;
    if (c.napi_get_cb_info(env, info, &argc, &argv, null, null) != c.napi_ok) {
        return napiThrow(env, "napi_get_cb_info failed");
    }
    if (argc < 7) {
        return napiThrow(env, "backendDispatch(handle, name, uniform, extraUniforms, inputs, output, dispatch) requires 7 args");
    }

    var raw: u64 = 0;
    var lossless: bool = false;
    if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
        return napiThrow(env, "argument 0 must be a non-null BigInt handle");
    }
    const backend: *wgpu.Backend = @ptrFromInt(@as(usize, @intCast(raw)));

    const name_buf = napiGetString(env, argv[1], "argument 1 must be string (name)") orelse return null;
    defer std.heap.c_allocator.free(name_buf);
    const uniform_bytes = napiGetUint8Array(env, argv[2], "argument 2 must be Uint8Array (uniform)") orelse return null;

    const extra_n = napiArrayLen(env, argv[3]) orelse return napiThrow(env, "argument 3 must be array");
    if (extra_n > 8) return napiThrow(env, "too many extraUniforms");
    var extra_storage: [8]wgpu.StorageInput = undefined;
    var ei: u32 = 0;
    while (ei < extra_n) : (ei += 1) {
        const item = napiArrayGet(env, argv[3], ei) orelse return napiThrow(env, "extraUniforms[i] read failed");
        const binding_v = napiNamedProperty(env, item, "binding") orelse return napiThrow(env, "extraUniforms[i].binding missing");
        const bytes_v = napiNamedProperty(env, item, "bytes") orelse return napiThrow(env, "extraUniforms[i].bytes missing");
        const binding = napiGetU32(env, binding_v, "extraUniforms[i].binding must be u32") orelse return null;
        const bytes = napiGetUint8Array(env, bytes_v, "extraUniforms[i].bytes must be Uint8Array") orelse return null;
        extra_storage[ei] = .{ .binding = binding, .bytes = bytes };
    }

    const in_n = napiArrayLen(env, argv[4]) orelse return napiThrow(env, "argument 4 must be array");
    if (in_n > 16) return napiThrow(env, "too many inputs");
    var in_storage: [16]wgpu.StorageInput = undefined;
    var ii: u32 = 0;
    while (ii < in_n) : (ii += 1) {
        const item = napiArrayGet(env, argv[4], ii) orelse return napiThrow(env, "inputs[i] read failed");
        const binding_v = napiNamedProperty(env, item, "binding") orelse return napiThrow(env, "inputs[i].binding missing");
        const bytes_v = napiNamedProperty(env, item, "bytes") orelse return napiThrow(env, "inputs[i].bytes missing");
        const binding = napiGetU32(env, binding_v, "inputs[i].binding must be u32") orelse return null;
        const bytes = napiGetUint8Array(env, bytes_v, "inputs[i].bytes must be Uint8Array") orelse return null;
        in_storage[ii] = .{ .binding = binding, .bytes = bytes };
    }

    const out_obj = argv[5];
    const out_binding_v = napiNamedProperty(env, out_obj, "binding") orelse return napiThrow(env, "output.binding missing");
    const out_blen_v = napiNamedProperty(env, out_obj, "byteLength") orelse return napiThrow(env, "output.byteLength missing");
    const out_binding = napiGetU32(env, out_binding_v, "output.binding must be u32") orelse return null;
    const out_blen = napiGetU32(env, out_blen_v, "output.byteLength must be u32") orelse return null;

    var out_initial: ?[]const u8 = null;
    {
        var v: c.napi_value = undefined;
        if (c.napi_get_named_property(env, out_obj, "initial", &v) == c.napi_ok) {
            var t: c.napi_valuetype = undefined;
            if (c.napi_typeof(env, v, &t) == c.napi_ok and t != c.napi_undefined and t != c.napi_null) {
                const b = napiGetUint8Array(env, v, "output.initial must be Uint8Array") orelse return null;
                out_initial = b;
            }
        }
    }

    const dx_v = napiArrayGet(env, argv[6], 0) orelse return napiThrow(env, "dispatch[0] missing");
    const dy_v = napiArrayGet(env, argv[6], 1) orelse return napiThrow(env, "dispatch[1] missing");
    const dz_v = napiArrayGet(env, argv[6], 2) orelse return napiThrow(env, "dispatch[2] missing");
    const dx = napiGetU32(env, dx_v, "dispatch[0] must be u32") orelse return null;
    const dy = napiGetU32(env, dy_v, "dispatch[1] must be u32") orelse return null;
    const dz = napiGetU32(env, dz_v, "dispatch[2] must be u32") orelse return null;

    var out_ab: c.napi_value = undefined;
    var out_data_ptr: ?*anyopaque = null;
    if (c.napi_create_arraybuffer(env, out_blen, &out_data_ptr, &out_ab) != c.napi_ok) {
        return napiThrow(env, "napi: failed to allocate output ArrayBuffer");
    }
    var out_typed: c.napi_value = undefined;
    if (c.napi_create_typedarray(env, c.napi_uint8_array, out_blen, out_ab, 0, &out_typed) != c.napi_ok) {
        return napiThrow(env, "napi: failed to allocate output Uint8Array");
    }
    const out_slice = @as([*]u8, @ptrCast(out_data_ptr.?))[0..out_blen];

    const wgsl = wgpu.shaderByName_pub(name_buf) orelse return napiThrow(env, "unknown shader name");

    wgpu.dispatchOnBackend(
        backend,
        wgsl,
        uniform_bytes,
        0,
        in_storage[0..in_n],
        .{ .binding = out_binding, .bytes = out_slice },
        out_initial,
        extra_storage[0..extra_n],
        .{ dx, dy, dz },
    ) catch |err| {
        return switch (err) {
            wgpu.WgpuError.ShaderModuleFailed => napiThrow(env, "wgpu: createShaderModule failed"),
            wgpu.WgpuError.ComputePipelineFailed => napiThrow(env, "wgpu: createComputePipeline failed"),
            wgpu.WgpuError.BindGroupFailed => napiThrow(env, "wgpu: createBindGroup failed"),
            wgpu.WgpuError.BufferCreateFailed => napiThrow(env, "wgpu: createBuffer failed"),
            wgpu.WgpuError.BufferMapFailed => napiThrow(env, "wgpu: mapAsync failed"),
            wgpu.WgpuError.BufferRangeFailed => napiThrow(env, "wgpu: buffer range failed"),
            else => napiThrow(env, "wgpu: backendDispatch failed"),
        };
    };
    return out_typed;
}

// ── persistent buffer management ──

fn napiCreateBuffer(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 2;
    var argv: [2]c.napi_value = undefined;
    if (c.napi_get_cb_info(env, info, &argc, &argv, null, null) != c.napi_ok) {
        return napiThrow(env, "napi_get_cb_info failed");
    }
    if (argc < 2) return napiThrow(env, "createBuffer(handle, bytes) requires 2 args");

    var raw: u64 = 0;
    var lossless: bool = false;
    if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
        return napiThrow(env, "argument 0 must be a non-null backend BigInt handle");
    }
    const backend: *wgpu.Backend = @ptrFromInt(@as(usize, @intCast(raw)));
    const bytes = napiGetUint8Array(env, argv[1], "argument 1 must be Uint8Array (bytes)") orelse return null;

    const buf = wgpu.createPersistentBuffer(backend, bytes) catch
        return napiThrow(env, "wgpu: createPersistentBuffer failed");

    var bn: c.napi_value = undefined;
    if (c.napi_create_bigint_uint64(env, @intCast(@intFromPtr(buf)), &bn) != c.napi_ok) {
        wgpu.releasePersistentBuffer(buf);
        return napiThrow(env, "napi: failed to create buffer handle");
    }
    return bn;
}

fn napiReleaseBuffer(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 1;
    var argv: [1]c.napi_value = undefined;
    if (c.napi_get_cb_info(env, info, &argc, &argv, null, null) != c.napi_ok) {
        return napiThrow(env, "napi_get_cb_info failed");
    }
    if (argc < 1) return napiThrow(env, "releaseBuffer(bufPtr) requires 1 arg");

    var raw: u64 = 0;
    var lossless: bool = false;
    if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok) {
        return napiThrow(env, "argument 0 must be BigInt buffer pointer");
    }
    if (raw != 0) {
        const buf: wgpu.c.WGPUBuffer = @ptrFromInt(@as(usize, @intCast(raw)));
        wgpu.releasePersistentBuffer(buf);
    }
    var undef: c.napi_value = undefined;
    _ = c.napi_get_undefined(env, &undef);
    return undef;
}

fn napiWriteBuffer(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 4;
    var argv: [4]c.napi_value = undefined;
    if (c.napi_get_cb_info(env, info, &argc, &argv, null, null) != c.napi_ok) {
        return napiThrow(env, "napi_get_cb_info failed");
    }
    if (argc < 4) return napiThrow(env, "writeBuffer(handle, bufPtr, offset, bytes) requires 4 args");

    var raw: u64 = 0;
    var lossless: bool = false;
    if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
        return napiThrow(env, "argument 0 must be backend BigInt");
    }
    const backend: *wgpu.Backend = @ptrFromInt(@as(usize, @intCast(raw)));

    var bufRaw: u64 = 0;
    if (c.napi_get_value_bigint_uint64(env, argv[1], &bufRaw, &lossless) != c.napi_ok or bufRaw == 0) {
        return napiThrow(env, "argument 1 must be buffer BigInt");
    }
    const buf: wgpu.c.WGPUBuffer = @ptrFromInt(@as(usize, @intCast(bufRaw)));

    const offset = napiGetU32(env, argv[2], "argument 2 must be u32 (offset)") orelse return null;
    const bytes = napiGetUint8Array(env, argv[3], "argument 3 must be Uint8Array (bytes)") orelse return null;

    wgpu.c.wgpuQueueWriteBuffer(backend.queue, buf, offset, bytes.ptr, bytes.len);

    var undef: c.napi_value = undefined;
    _ = c.napi_get_undefined(env, &undef);
    return undef;
}

fn napiReadBuffer(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 4;
    var argv: [4]c.napi_value = undefined;
    if (c.napi_get_cb_info(env, info, &argc, &argv, null, null) != c.napi_ok) {
        return napiThrow(env, "napi_get_cb_info failed");
    }
    if (argc < 4) return napiThrow(env, "readBuffer(handle, bufPtr, offset, byteLen) requires 4 args");

    var raw: u64 = 0;
    var lossless: bool = false;
    if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
        return napiThrow(env, "argument 0 must be backend BigInt");
    }
    const backend: *wgpu.Backend = @ptrFromInt(@as(usize, @intCast(raw)));

    var bufRaw: u64 = 0;
    if (c.napi_get_value_bigint_uint64(env, argv[1], &bufRaw, &lossless) != c.napi_ok or bufRaw == 0) {
        return napiThrow(env, "argument 1 must be buffer BigInt");
    }
    const src: wgpu.c.WGPUBuffer = @ptrFromInt(@as(usize, @intCast(bufRaw)));

    const offset = napiGetU32(env, argv[2], "argument 2 must be u32 (offset)") orelse return null;
    const byte_len = napiGetU32(env, argv[3], "argument 3 must be u32 (byteLen)") orelse return null;

    var ab: c.napi_value = undefined;
    var data_ptr: ?*anyopaque = null;
    if (c.napi_create_arraybuffer(env, byte_len, &data_ptr, &ab) != c.napi_ok) {
        return napiThrow(env, "napi: alloc out ArrayBuffer failed");
    }
    var typed: c.napi_value = undefined;
    if (c.napi_create_typedarray(env, c.napi_uint8_array, byte_len, ab, 0, &typed) != c.napi_ok) {
        return napiThrow(env, "napi: alloc Uint8Array failed");
    }
    const out = @as([*]u8, @ptrCast(data_ptr.?))[0..byte_len];

    wgpu.readPersistentBuffer(backend, src, offset, out) catch
        return napiThrow(env, "wgpu: readPersistentBuffer failed");

    return typed;
}

fn parseStorageBindings(
    env: c.napi_env,
    arr: c.napi_value,
    out: []wgpu.StorageBinding,
    out_n: *usize,
) ?void {
    const n = napiArrayLen(env, arr) orelse {
        _ = c.napi_throw_error(env, null, "inputs must be array");
        return null;
    };
    if (n > out.len) {
        _ = c.napi_throw_error(env, null, "too many inputs");
        return null;
    }
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        const item = napiArrayGet(env, arr, i) orelse return null;
        const binding_v = napiNamedProperty(env, item, "binding") orelse {
            _ = c.napi_throw_error(env, null, "input[i].binding missing");
            return null;
        };
        const binding = napiGetU32(env, binding_v, "input[i].binding") orelse return null;

        // Either {bytes: Uint8Array} or {bufPtr: BigInt, byteLen: u32}.
        var bytes_v: c.napi_value = undefined;
        if (c.napi_get_named_property(env, item, "bytes", &bytes_v) == c.napi_ok) {
            var t: c.napi_valuetype = undefined;
            if (c.napi_typeof(env, bytes_v, &t) == c.napi_ok and t != c.napi_undefined and t != c.napi_null) {
                const bytes = napiGetUint8Array(env, bytes_v, "input[i].bytes must be Uint8Array") orelse return null;
                out[i] = .{ .bytes = .{ .binding = binding, .bytes = bytes } };
                continue;
            }
        }
        const ptr_v = napiNamedProperty(env, item, "bufPtr") orelse {
            _ = c.napi_throw_error(env, null, "input[i] needs bytes or bufPtr");
            return null;
        };
        const len_v = napiNamedProperty(env, item, "byteLen") orelse {
            _ = c.napi_throw_error(env, null, "input[i].byteLen missing");
            return null;
        };
        var raw: u64 = 0;
        var lossless: bool = false;
        if (c.napi_get_value_bigint_uint64(env, ptr_v, &raw, &lossless) != c.napi_ok or raw == 0) {
            _ = c.napi_throw_error(env, null, "input[i].bufPtr must be non-null BigInt");
            return null;
        }
        const buf: wgpu.c.WGPUBuffer = @ptrFromInt(@as(usize, @intCast(raw)));
        const byte_len = napiGetU32(env, len_v, "input[i].byteLen") orelse return null;
        out[i] = .{ .buffer = .{ .binding = binding, .buf = buf, .byte_len = byte_len } };
    }
    out_n.* = n;
}

fn napiDispatchByBuffers(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 7;
    var argv: [7]c.napi_value = undefined;
    if (c.napi_get_cb_info(env, info, &argc, &argv, null, null) != c.napi_ok) {
        return napiThrow(env, "napi_get_cb_info failed");
    }
    if (argc < 7) {
        return napiThrow(env, "dispatchByBuffers(handle, name, uniform, extras, inputs, output, dispatch) requires 7 args");
    }

    var raw: u64 = 0;
    var lossless: bool = false;
    if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
        return napiThrow(env, "argument 0 must be backend BigInt");
    }
    const backend: *wgpu.Backend = @ptrFromInt(@as(usize, @intCast(raw)));

    const name_buf = napiGetString(env, argv[1], "argument 1 must be string (name)") orelse return null;
    defer std.heap.c_allocator.free(name_buf);
    const uniform_bytes = napiGetUint8Array(env, argv[2], "argument 2 must be Uint8Array (uniform)") orelse return null;

    const extra_n = napiArrayLen(env, argv[3]) orelse return napiThrow(env, "argument 3 must be array");
    if (extra_n > 8) return napiThrow(env, "too many extraUniforms");
    var extra_storage: [8]wgpu.StorageInput = undefined;
    var ei: u32 = 0;
    while (ei < extra_n) : (ei += 1) {
        const item = napiArrayGet(env, argv[3], ei) orelse return napiThrow(env, "extraUniforms[i]");
        const binding = napiGetU32(env, napiNamedProperty(env, item, "binding") orelse return napiThrow(env, "binding"), "binding") orelse return null;
        const bytes = napiGetUint8Array(env, napiNamedProperty(env, item, "bytes") orelse return napiThrow(env, "bytes"), "bytes") orelse return null;
        extra_storage[ei] = .{ .binding = binding, .bytes = bytes };
    }

    var in_storage: [16]wgpu.StorageBinding = undefined;
    var in_n: usize = 0;
    parseStorageBindings(env, argv[4], in_storage[0..], &in_n) orelse return null;

    const out_obj = argv[5];
    const out_binding = napiGetU32(env, napiNamedProperty(env, out_obj, "binding") orelse return napiThrow(env, "output.binding"), "output.binding") orelse return null;
    const out_blen = napiGetU32(env, napiNamedProperty(env, out_obj, "byteLength") orelse return napiThrow(env, "output.byteLength"), "output.byteLength") orelse return null;
    var out_initial: ?[]const u8 = null;
    {
        var v: c.napi_value = undefined;
        if (c.napi_get_named_property(env, out_obj, "initial", &v) == c.napi_ok) {
            var t: c.napi_valuetype = undefined;
            if (c.napi_typeof(env, v, &t) == c.napi_ok and t != c.napi_undefined and t != c.napi_null) {
                const b = napiGetUint8Array(env, v, "output.initial must be Uint8Array") orelse return null;
                out_initial = b;
            }
        }
    }

    const dx = napiGetU32(env, napiArrayGet(env, argv[6], 0) orelse return napiThrow(env, "dispatch[0]"), "dispatch[0]") orelse return null;
    const dy = napiGetU32(env, napiArrayGet(env, argv[6], 1) orelse return napiThrow(env, "dispatch[1]"), "dispatch[1]") orelse return null;
    const dz = napiGetU32(env, napiArrayGet(env, argv[6], 2) orelse return napiThrow(env, "dispatch[2]"), "dispatch[2]") orelse return null;

    var out_ab: c.napi_value = undefined;
    var out_data_ptr: ?*anyopaque = null;
    if (c.napi_create_arraybuffer(env, out_blen, &out_data_ptr, &out_ab) != c.napi_ok) {
        return napiThrow(env, "napi: alloc output ArrayBuffer failed");
    }
    var out_typed: c.napi_value = undefined;
    if (c.napi_create_typedarray(env, c.napi_uint8_array, out_blen, out_ab, 0, &out_typed) != c.napi_ok) {
        return napiThrow(env, "napi: alloc output Uint8Array failed");
    }
    const out_slice = @as([*]u8, @ptrCast(out_data_ptr.?))[0..out_blen];

    const wgsl = wgpu.shaderByName_pub(name_buf) orelse return napiThrow(env, "unknown shader name");

    wgpu.dispatchOnBackendMixed(
        backend,
        wgsl,
        uniform_bytes,
        0,
        in_storage[0..in_n],
        .{ .binding = out_binding, .bytes = out_slice },
        out_initial,
        extra_storage[0..extra_n],
        .{ dx, dy, dz },
    ) catch |err| {
        return switch (err) {
            wgpu.WgpuError.ShaderModuleFailed => napiThrow(env, "wgpu: createShaderModule failed"),
            wgpu.WgpuError.ComputePipelineFailed => napiThrow(env, "wgpu: createComputePipeline failed"),
            wgpu.WgpuError.BindGroupFailed => napiThrow(env, "wgpu: createBindGroup failed"),
            wgpu.WgpuError.BufferCreateFailed => napiThrow(env, "wgpu: createBuffer failed"),
            wgpu.WgpuError.BufferMapFailed => napiThrow(env, "wgpu: mapAsync failed"),
            wgpu.WgpuError.BufferRangeFailed => napiThrow(env, "wgpu: buffer range failed"),
            else => napiThrow(env, "wgpu: dispatchByBuffers failed"),
        };
    };
    return out_typed;
}

// Convenience: allocate an uninitialized persistent buffer of `size`
// bytes (zero-filled). Use for transient activation scratch buffers
// that the forward overwrites every call.
fn napiCreateEmptyBuffer(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 2;
    var argv: [2]c.napi_value = undefined;
    if (c.napi_get_cb_info(env, info, &argc, &argv, null, null) != c.napi_ok) {
        return napiThrow(env, "napi_get_cb_info failed");
    }
    if (argc < 2) return napiThrow(env, "createEmptyBuffer(handle, byteLen) requires 2 args");

    var raw: u64 = 0;
    var lossless: bool = false;
    if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
        return napiThrow(env, "argument 0 must be backend BigInt");
    }
    const backend: *wgpu.Backend = @ptrFromInt(@as(usize, @intCast(raw)));
    const byte_len = napiGetU32(env, argv[1], "argument 1 must be u32 (byteLen)") orelse return null;

    const padded = std.mem.alignForward(usize, byte_len, 4);
    const zeros = std.heap.c_allocator.alloc(u8, padded) catch
        return napiThrow(env, "oom for empty buffer init");
    defer std.heap.c_allocator.free(zeros);
    @memset(zeros, 0);

    const buf = wgpu.createPersistentBuffer(backend, zeros) catch
        return napiThrow(env, "wgpu: createPersistentBuffer failed");

    var bn: c.napi_value = undefined;
    if (c.napi_create_bigint_uint64(env, @intCast(@intFromPtr(buf)), &bn) != c.napi_ok) {
        wgpu.releasePersistentBuffer(buf);
        return napiThrow(env, "napi: failed to create buffer handle");
    }
    return bn;
}

fn napiBeginEncoder(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 1;
    var argv: [1]c.napi_value = undefined;
    if (c.napi_get_cb_info(env, info, &argc, &argv, null, null) != c.napi_ok) {
        return napiThrow(env, "napi_get_cb_info failed");
    }
    if (argc < 1) return napiThrow(env, "beginEncoder(handle) requires 1 arg");

    var raw: u64 = 0;
    var lossless: bool = false;
    if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
        return napiThrow(env, "argument 0 must be backend BigInt");
    }
    const backend: *wgpu.Backend = @ptrFromInt(@as(usize, @intCast(raw)));

    const enc = wgpu.beginEncoder(backend) catch
        return napiThrow(env, "wgpu: beginEncoder failed");

    var bn: c.napi_value = undefined;
    if (c.napi_create_bigint_uint64(env, @intCast(@intFromPtr(enc)), &bn) != c.napi_ok) {
        return napiThrow(env, "napi: failed to create encoder handle");
    }
    return bn;
}

fn napiEnqueueDispatch(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 7;
    var argv: [7]c.napi_value = undefined;
    if (c.napi_get_cb_info(env, info, &argc, &argv, null, null) != c.napi_ok) {
        return napiThrow(env, "napi_get_cb_info failed");
    }
    if (argc < 7) {
        return napiThrow(env, "enqueueDispatch(encoder, name, uniform, extras, inputs, output, dispatch) requires 7 args");
    }

    var raw: u64 = 0;
    var lossless: bool = false;
    if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
        return napiThrow(env, "argument 0 must be encoder BigInt");
    }
    const enc: *wgpu.Encoder = @ptrFromInt(@as(usize, @intCast(raw)));

    const name_buf = napiGetString(env, argv[1], "argument 1 must be string (name)") orelse return null;
    defer std.heap.c_allocator.free(name_buf);
    const uniform_bytes = napiGetUint8Array(env, argv[2], "argument 2 must be Uint8Array (uniform)") orelse return null;

    const extra_n = napiArrayLen(env, argv[3]) orelse return napiThrow(env, "argument 3 must be array");
    if (extra_n > 8) return napiThrow(env, "too many extraUniforms");
    var extra_storage: [8]wgpu.StorageInput = undefined;
    var ei: u32 = 0;
    while (ei < extra_n) : (ei += 1) {
        const item = napiArrayGet(env, argv[3], ei) orelse return napiThrow(env, "extras[i]");
        const binding = napiGetU32(env, napiNamedProperty(env, item, "binding") orelse return napiThrow(env, "binding"), "binding") orelse return null;
        const bytes = napiGetUint8Array(env, napiNamedProperty(env, item, "bytes") orelse return napiThrow(env, "bytes"), "bytes") orelse return null;
        extra_storage[ei] = .{ .binding = binding, .bytes = bytes };
    }

    var in_storage: [16]wgpu.StorageBinding = undefined;
    var in_n: usize = 0;
    parseStorageBindings(env, argv[4], in_storage[0..], &in_n) orelse return null;

    const out_obj = argv[5];
    const out_binding = napiGetU32(env, napiNamedProperty(env, out_obj, "binding") orelse return napiThrow(env, "output.binding"), "output.binding") orelse return null;
    const out_byte_len_v = napiNamedProperty(env, out_obj, "byteLen") orelse return napiThrow(env, "output.byteLen");
    const out_byte_len = napiGetU32(env, out_byte_len_v, "output.byteLen") orelse return null;
    const out_buf_ptr_v = napiNamedProperty(env, out_obj, "bufPtr") orelse return napiThrow(env, "output.bufPtr");
    var out_raw: u64 = 0;
    if (c.napi_get_value_bigint_uint64(env, out_buf_ptr_v, &out_raw, &lossless) != c.napi_ok or out_raw == 0) {
        return napiThrow(env, "output.bufPtr must be non-null BigInt");
    }
    const out_buf: wgpu.c.WGPUBuffer = @ptrFromInt(@as(usize, @intCast(out_raw)));

    var out_initial: ?[]const u8 = null;
    {
        var v: c.napi_value = undefined;
        if (c.napi_get_named_property(env, out_obj, "initial", &v) == c.napi_ok) {
            var t: c.napi_valuetype = undefined;
            if (c.napi_typeof(env, v, &t) == c.napi_ok and t != c.napi_undefined and t != c.napi_null) {
                const b = napiGetUint8Array(env, v, "output.initial must be Uint8Array") orelse return null;
                out_initial = b;
            }
        }
    }

    const dx = napiGetU32(env, napiArrayGet(env, argv[6], 0) orelse return napiThrow(env, "dispatch[0]"), "dispatch[0]") orelse return null;
    const dy = napiGetU32(env, napiArrayGet(env, argv[6], 1) orelse return napiThrow(env, "dispatch[1]"), "dispatch[1]") orelse return null;
    const dz = napiGetU32(env, napiArrayGet(env, argv[6], 2) orelse return napiThrow(env, "dispatch[2]"), "dispatch[2]") orelse return null;

    wgpu.enqueueOnEncoder(
        enc,
        name_buf,
        uniform_bytes,
        0,
        in_storage[0..in_n],
        out_buf,
        out_binding,
        out_byte_len,
        out_initial,
        extra_storage[0..extra_n],
        .{ dx, dy, dz },
    ) catch |err| {
        return switch (err) {
            wgpu.WgpuError.ShaderModuleFailed => napiThrow(env, "wgpu: createShaderModule failed"),
            wgpu.WgpuError.ComputePipelineFailed => napiThrow(env, "wgpu: createComputePipeline failed"),
            wgpu.WgpuError.BindGroupFailed => napiThrow(env, "wgpu: createBindGroup failed"),
            wgpu.WgpuError.BufferCreateFailed => napiThrow(env, "wgpu: createBuffer failed"),
            wgpu.WgpuError.BufferRangeFailed => napiThrow(env, "wgpu: buffer range failed"),
            else => napiThrow(env, "wgpu: enqueueDispatch failed"),
        };
    };
    var undef: c.napi_value = undefined;
    _ = c.napi_get_undefined(env, &undef);
    return undef;
}

fn napiSubmitAndReadback(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 4;
    var argv: [4]c.napi_value = undefined;
    if (c.napi_get_cb_info(env, info, &argc, &argv, null, null) != c.napi_ok) {
        return napiThrow(env, "napi_get_cb_info failed");
    }
    if (argc < 4) return napiThrow(env, "submitAndReadback(encoder, srcBuf, offset, byteLen) requires 4 args");

    var raw: u64 = 0;
    var lossless: bool = false;
    if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
        return napiThrow(env, "argument 0 must be encoder BigInt");
    }
    const enc: *wgpu.Encoder = @ptrFromInt(@as(usize, @intCast(raw)));

    var srcRaw: u64 = 0;
    if (c.napi_get_value_bigint_uint64(env, argv[1], &srcRaw, &lossless) != c.napi_ok or srcRaw == 0) {
        return napiThrow(env, "argument 1 must be source buffer BigInt");
    }
    const src: wgpu.c.WGPUBuffer = @ptrFromInt(@as(usize, @intCast(srcRaw)));

    const offset = napiGetU32(env, argv[2], "argument 2 must be u32 (offset)") orelse return null;
    const byte_len = napiGetU32(env, argv[3], "argument 3 must be u32 (byteLen)") orelse return null;

    var ab: c.napi_value = undefined;
    var data_ptr: ?*anyopaque = null;
    if (c.napi_create_arraybuffer(env, byte_len, &data_ptr, &ab) != c.napi_ok) {
        return napiThrow(env, "napi: alloc out ArrayBuffer failed");
    }
    var typed: c.napi_value = undefined;
    if (c.napi_create_typedarray(env, c.napi_uint8_array, byte_len, ab, 0, &typed) != c.napi_ok) {
        return napiThrow(env, "napi: alloc Uint8Array failed");
    }
    const out = @as([*]u8, @ptrCast(data_ptr.?))[0..byte_len];

    wgpu.submitAndReadback(enc, src, offset, out) catch
        return napiThrow(env, "wgpu: submitAndReadback failed");

    return typed;
}

fn napiBenchDispatch(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 8;
    var argv: [8]c.napi_value = undefined;
    if (c.napi_get_cb_info(env, info, &argc, &argv, null, null) != c.napi_ok) {
        return napiThrow(env, "napi_get_cb_info failed");
    }
    if (argc < 8) {
        return napiThrow(env, "benchDispatch(handle, name, uniform, extraUniforms, inputs, output, dispatch, iters) requires 8 args");
    }

    var raw: u64 = 0;
    var lossless: bool = false;
    if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
        return napiThrow(env, "argument 0 must be a non-null BigInt handle");
    }
    const backend: *wgpu.Backend = @ptrFromInt(@as(usize, @intCast(raw)));

    const name_buf = napiGetString(env, argv[1], "argument 1 must be string (name)") orelse return null;
    defer std.heap.c_allocator.free(name_buf);
    const uniform_bytes = napiGetUint8Array(env, argv[2], "argument 2 must be Uint8Array (uniform)") orelse return null;

    const extra_n = napiArrayLen(env, argv[3]) orelse return napiThrow(env, "argument 3 must be array");
    if (extra_n > 8) return napiThrow(env, "too many extraUniforms");
    var extra_storage: [8]wgpu.StorageInput = undefined;
    var ei: u32 = 0;
    while (ei < extra_n) : (ei += 1) {
        const item = napiArrayGet(env, argv[3], ei) orelse return napiThrow(env, "extraUniforms[i] read failed");
        const binding = napiGetU32(env, napiNamedProperty(env, item, "binding") orelse return napiThrow(env, "binding"), "binding") orelse return null;
        const bytes = napiGetUint8Array(env, napiNamedProperty(env, item, "bytes") orelse return napiThrow(env, "bytes"), "bytes") orelse return null;
        extra_storage[ei] = .{ .binding = binding, .bytes = bytes };
    }

    const in_n = napiArrayLen(env, argv[4]) orelse return napiThrow(env, "argument 4 must be array");
    if (in_n > 16) return napiThrow(env, "too many inputs");
    var in_storage: [16]wgpu.StorageInput = undefined;
    var ii: u32 = 0;
    while (ii < in_n) : (ii += 1) {
        const item = napiArrayGet(env, argv[4], ii) orelse return napiThrow(env, "inputs[i] read failed");
        const binding = napiGetU32(env, napiNamedProperty(env, item, "binding") orelse return napiThrow(env, "binding"), "binding") orelse return null;
        const bytes = napiGetUint8Array(env, napiNamedProperty(env, item, "bytes") orelse return napiThrow(env, "bytes"), "bytes") orelse return null;
        in_storage[ii] = .{ .binding = binding, .bytes = bytes };
    }

    const out_obj = argv[5];
    const out_binding = napiGetU32(env, napiNamedProperty(env, out_obj, "binding") orelse return napiThrow(env, "output.binding"), "output.binding") orelse return null;
    const out_blen = napiGetU32(env, napiNamedProperty(env, out_obj, "byteLength") orelse return napiThrow(env, "output.byteLength"), "output.byteLength") orelse return null;
    var out_initial: ?[]const u8 = null;
    {
        var v: c.napi_value = undefined;
        if (c.napi_get_named_property(env, out_obj, "initial", &v) == c.napi_ok) {
            var t: c.napi_valuetype = undefined;
            if (c.napi_typeof(env, v, &t) == c.napi_ok and t != c.napi_undefined and t != c.napi_null) {
                const b = napiGetUint8Array(env, v, "output.initial must be Uint8Array") orelse return null;
                out_initial = b;
            }
        }
    }

    // Optional output.chainLen (defaults to 1).
    var chain_len: u32 = 1;
    {
        var v: c.napi_value = undefined;
        if (c.napi_get_named_property(env, out_obj, "chainLen", &v) == c.napi_ok) {
            var t: c.napi_valuetype = undefined;
            if (c.napi_typeof(env, v, &t) == c.napi_ok and t == c.napi_number) {
                chain_len = napiGetU32(env, v, "output.chainLen must be u32") orelse return null;
                if (chain_len == 0) chain_len = 1;
            }
        }
    }

    const dx = napiGetU32(env, napiArrayGet(env, argv[6], 0) orelse return napiThrow(env, "dispatch[0]"), "dispatch[0]") orelse return null;
    const dy = napiGetU32(env, napiArrayGet(env, argv[6], 1) orelse return napiThrow(env, "dispatch[1]"), "dispatch[1]") orelse return null;
    const dz = napiGetU32(env, napiArrayGet(env, argv[6], 2) orelse return napiThrow(env, "dispatch[2]"), "dispatch[2]") orelse return null;

    const iters = napiGetU32(env, argv[7], "argument 7 must be u32 (iters)") orelse return null;
    if (iters == 0 or iters > 100000) return napiThrow(env, "iters must be in 1..100000");

    // Allocate result Float64Array (one ms sample per iter).
    var ab: c.napi_value = undefined;
    var data_ptr: ?*anyopaque = null;
    if (c.napi_create_arraybuffer(env, iters * @sizeOf(f64), &data_ptr, &ab) != c.napi_ok) {
        return napiThrow(env, "napi: failed to allocate samples ArrayBuffer");
    }
    var typed: c.napi_value = undefined;
    if (c.napi_create_typedarray(env, c.napi_float64_array, iters, ab, 0, &typed) != c.napi_ok) {
        return napiThrow(env, "napi: failed to allocate Float64Array");
    }
    const samples = @as([*]f64, @ptrCast(@alignCast(data_ptr.?)))[0..iters];

    // Output buffer is throwaway during bench (we only care about timing,
    // not value), but the bench function still needs a target slice.
    const out_dummy = std.heap.c_allocator.alloc(u8, out_blen) catch
        return napiThrow(env, "oom for bench output buffer");
    defer std.heap.c_allocator.free(out_dummy);

    const wgsl = wgpu.shaderByName_pub(name_buf) orelse return napiThrow(env, "unknown shader name");

    wgpu.benchDispatch(
        backend,
        wgsl,
        uniform_bytes,
        0,
        in_storage[0..in_n],
        .{ .binding = out_binding, .bytes = out_dummy, .chain_len = chain_len },
        out_initial,
        extra_storage[0..extra_n],
        .{ dx, dy, dz },
        samples,
    ) catch |err| {
        return switch (err) {
            wgpu.WgpuError.ShaderModuleFailed => napiThrow(env, "wgpu: createShaderModule failed"),
            wgpu.WgpuError.ComputePipelineFailed => napiThrow(env, "wgpu: createComputePipeline failed"),
            wgpu.WgpuError.BindGroupFailed => napiThrow(env, "wgpu: createBindGroup failed"),
            wgpu.WgpuError.BufferCreateFailed => napiThrow(env, "wgpu: createBuffer failed"),
            wgpu.WgpuError.BufferMapFailed => napiThrow(env, "wgpu: mapAsync failed"),
            wgpu.WgpuError.BufferRangeFailed => napiThrow(env, "wgpu: buffer range failed"),
            else => napiThrow(env, "wgpu: benchDispatch failed"),
        };
    };
    return typed;
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
