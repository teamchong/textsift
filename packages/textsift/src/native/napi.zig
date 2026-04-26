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

fn napiGetAdapterInfo(env: c.napi_env, _: c.napi_callback_info) callconv(.c) c.napi_value {
    const handles = wgpu.createInstanceAndAdapter() catch |err| {
        return switch (err) {
            wgpu.WgpuError.InstanceCreateFailed =>
                napiThrow(env, "wgpu: failed to create instance"),
            wgpu.WgpuError.AdapterRequestFailed =>
                napiThrow(env, "wgpu: adapter request future failed"),
            wgpu.WgpuError.AdapterUnavailable =>
                napiThrow(env, "WebGPU adapter not available on this host. Use textsift/browser instead."),
            wgpu.WgpuError.AdapterInfoFailed =>
                napiThrow(env, "wgpu: adapter info query failed"),
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
