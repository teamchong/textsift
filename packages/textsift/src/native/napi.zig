// Node-API shim. Single entry point `napi_register_module_v1` that
// exposes the Metal-direct (macOS) and Vulkan-direct (non-macOS) NAPI
// surfaces. Comptime-gated: each platform builds only the surface it
// actually has — see the Metal/Vulkan struct branches below.

const std = @import("std");
const builtin = @import("builtin");

// Per-platform backend routing, comptime-gated so each .node binary
// only compiles + links the surfaces relevant to its target:
//
//   macOS   → Metal-direct (hand-written MSL via Obj-C bridge)
//   Linux   → Vulkan-direct (hand-written GLSL → SPIR-V via glslangValidator)
//             + Dawn-direct (Tint codegen, kept available for measurement)
//   Windows → Dawn-direct (Tint → D3D12 via Dawn's Vulkan/D3D12 backend selection)
//
// Each backend's struct is `if (gate) struct { ... } else struct { empty registerAll }`,
// so the un-taken branch is never analyzed by Zig — that's how we keep
// metal_backend.zig (Obj-C) out of Linux/Windows builds and vulkan_backend.zig
// (libvulkan-dev) out of macOS/Windows builds.
const is_macos = builtin.os.tag == .macos;
const is_linux = builtin.os.tag == .linux;

const c = @cImport({
    @cInclude("node_api.h");
});

export fn napi_register_module_v1(env: c.napi_env, exports: c.napi_value) c.napi_value {
    Metal.registerAll(env, exports) catch return null;
    Vulkan.registerAll(env, exports) catch return null;
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
fn napiGetU32(env: c.napi_env, value: c.napi_value, name: [*:0]const u8) ?u32 {
    var out: u32 = 0;
    if (c.napi_get_value_uint32(env, value, &out) != c.napi_ok) {
        _ = c.napi_throw_error(env, null, name);
        return null;
    }
    return out;
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

// ── Metal-direct backend (macOS only) ──
//
// Comptime-gated: on non-macOS the if-branch struct body is not analyzed,
// so metal_backend.zig and the Mac Obj-C bridge headers never compile.
// The else-branch exposes a no-op registerAll so the call site stays platform-agnostic.
const Metal = if (is_macos) struct {
    const metal = @import("metal_backend.zig");

    pub fn registerAll(env: c.napi_env, exports: c.napi_value) !void {
        try register(env, exports, "metalCreateBackend", napiMetalCreateBackend);
        try register(env, exports, "metalDestroyBackend", napiMetalDestroyBackend);
        try register(env, exports, "metalDeviceName", napiMetalDeviceName);
        try register(env, exports, "metalCreateBuffer", napiMetalCreateBuffer);
        try register(env, exports, "metalReleaseBuffer", napiMetalReleaseBuffer);
        try register(env, exports, "metalReadBuffer", napiMetalReadBuffer);
        try register(env, exports, "metalDispatchOneShot", napiMetalDispatchOneShot);
        try register(env, exports, "metalBeginEncoder", napiMetalBeginEncoder);
        try register(env, exports, "metalEnqueueDispatch", napiMetalEnqueueDispatch);
        try register(env, exports, "metalSubmitAndReadback", napiMetalSubmitAndReadback);
        try register(env, exports, "metalWriteBuffer", napiMetalWriteBuffer);
        try register(env, exports, "metalCreateEmptyBuffer", napiMetalCreateEmptyBuffer);
    }

    fn napiMetalCreateBackend(env: c.napi_env, _: c.napi_callback_info) callconv(.c) c.napi_value {
    const b = metal.createBackend() catch |err| {
        return switch (err) {
            metal.MetalError.DeviceCreateFailed => napiThrow(env, "metal: no MTLDevice (Apple Silicon required)"),
            metal.MetalError.LibraryCompileFailed => napiThrow(env, "metal: MSL library compile failed (see stderr)"),
            else => napiThrow(env, "metal: createBackend failed"),
        };
    };
    var bn: c.napi_value = undefined;
    if (c.napi_create_bigint_uint64(env, @intCast(@intFromPtr(b)), &bn) != c.napi_ok) {
        metal.destroyBackend(b);
        return napiThrow(env, "napi: failed to create handle bigint");
    }
    return bn;
}

fn napiMetalDestroyBackend(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 1;
    var argv: [1]c.napi_value = undefined;
    _ = c.napi_get_cb_info(env, info, &argc, &argv, null, null);
    if (argc < 1) return napiThrow(env, "metalDestroyBackend(handle) requires 1 arg");
    var raw: u64 = 0; var lossless: bool = false;
    if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok) {
        return napiThrow(env, "argument 0 must be BigInt handle");
    }
    if (raw != 0) {
        const b: *metal.Backend = @ptrFromInt(@as(usize, @intCast(raw)));
        metal.destroyBackend(b);
    }
    var u: c.napi_value = undefined;
    _ = c.napi_get_undefined(env, &u);
    return u;
}

fn napiMetalDeviceName(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 1; var argv: [1]c.napi_value = undefined;
    _ = c.napi_get_cb_info(env, info, &argc, &argv, null, null);
    if (argc < 1) return napiThrow(env, "metalDeviceName(handle) requires 1 arg");
    var raw: u64 = 0; var lossless: bool = false;
    if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
        return napiThrow(env, "argument 0 must be non-null BigInt");
    }
    const b: *metal.Backend = @ptrFromInt(@as(usize, @intCast(raw)));
    const name = metal.deviceName(b);
    var s: c.napi_value = undefined;
    if (c.napi_create_string_utf8(env, name, c.NAPI_AUTO_LENGTH, &s) != c.napi_ok) {
        return napiThrow(env, "napi: failed to create name string");
    }
    return s;
}

fn napiMetalCreateBuffer(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 2; var argv: [2]c.napi_value = undefined;
    _ = c.napi_get_cb_info(env, info, &argc, &argv, null, null);
    if (argc < 2) return napiThrow(env, "metalCreateBuffer(handle, bytes) requires 2 args");
    var raw: u64 = 0; var lossless: bool = false;
    if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
        return napiThrow(env, "argument 0 must be backend BigInt");
    }
    const b: *metal.Backend = @ptrFromInt(@as(usize, @intCast(raw)));
    const bytes = napiGetUint8Array(env, argv[1], "argument 1 must be Uint8Array") orelse return null;
    const buf = metal.createBuffer(b, bytes) catch return napiThrow(env, "metal: createBuffer failed");
    var bn: c.napi_value = undefined;
    if (c.napi_create_bigint_uint64(env, @intCast(@intFromPtr(buf)), &bn) != c.napi_ok) {
        metal.releaseBuffer(buf);
        return napiThrow(env, "napi: failed to create buffer handle");
    }
    return bn;
}

fn napiMetalReleaseBuffer(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 1; var argv: [1]c.napi_value = undefined;
    _ = c.napi_get_cb_info(env, info, &argc, &argv, null, null);
    if (argc < 1) return napiThrow(env, "metalReleaseBuffer(bufPtr) requires 1 arg");
    var raw: u64 = 0; var lossless: bool = false;
    _ = c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless);
    if (raw != 0) {
        const buf: metal.cb.TsMetalBuffer = @ptrFromInt(@as(usize, @intCast(raw)));
        metal.releaseBuffer(buf);
    }
    var u: c.napi_value = undefined;
    _ = c.napi_get_undefined(env, &u);
    return u;
}

fn napiMetalReadBuffer(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 3; var argv: [3]c.napi_value = undefined;
    _ = c.napi_get_cb_info(env, info, &argc, &argv, null, null);
    if (argc < 3) return napiThrow(env, "metalReadBuffer(bufPtr, offset, byteLen) requires 3 args");
    var raw: u64 = 0; var lossless: bool = false;
    if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
        return napiThrow(env, "argument 0 must be non-null buffer BigInt");
    }
    const buf: metal.cb.TsMetalBuffer = @ptrFromInt(@as(usize, @intCast(raw)));
    const offset = napiGetU32(env, argv[1], "argument 1 must be u32 (offset)") orelse return null;
    const byte_len = napiGetU32(env, argv[2], "argument 2 must be u32 (byteLen)") orelse return null;

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
    metal.readBuffer(buf, offset, out);
    return typed;
}

fn napiMetalDispatchOneShot(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 5; var argv: [5]c.napi_value = undefined;
    _ = c.napi_get_cb_info(env, info, &argc, &argv, null, null);
    if (argc < 5) return napiThrow(env, "metalDispatchOneShot(handle, name, bindings, grid, threadgroup) requires 5 args");

    var raw: u64 = 0; var lossless: bool = false;
    if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
        return napiThrow(env, "argument 0 must be backend BigInt");
    }
    const b: *metal.Backend = @ptrFromInt(@as(usize, @intCast(raw)));

    const name_buf = napiGetString(env, argv[1], "argument 1 must be string (name)") orelse return null;
    defer std.heap.c_allocator.free(name_buf);

    const n = napiArrayLen(env, argv[2]) orelse return napiThrow(env, "argument 2 must be array (bindings)");
    if (n > 16) return napiThrow(env, "too many bindings");
    var bindings: [16]metal.Binding = undefined;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        const item = napiArrayGet(env, argv[2], i) orelse return napiThrow(env, "bindings[i]");
        const idx_v = napiNamedProperty(env, item, "index") orelse return napiThrow(env, "bindings[i].index");
        const idx = napiGetU32(env, idx_v, "bindings[i].index") orelse return null;
        // Either {bytes: Uint8Array} or {bufPtr: BigInt, offset?: u32}.
        var bytes_v: c.napi_value = undefined;
        if (c.napi_get_named_property(env, item, "bytes", &bytes_v) == c.napi_ok) {
            var t: c.napi_valuetype = undefined;
            if (c.napi_typeof(env, bytes_v, &t) == c.napi_ok and t != c.napi_undefined and t != c.napi_null) {
                const bytes = napiGetUint8Array(env, bytes_v, "bindings[i].bytes") orelse return null;
                bindings[i] = .{ .bytes = .{ .index = idx, .bytes = bytes } };
                continue;
            }
        }
        const ptr_v = napiNamedProperty(env, item, "bufPtr") orelse return napiThrow(env, "bindings[i] needs bytes or bufPtr");
        var bufRaw: u64 = 0;
        if (c.napi_get_value_bigint_uint64(env, ptr_v, &bufRaw, &lossless) != c.napi_ok or bufRaw == 0) {
            return napiThrow(env, "bindings[i].bufPtr must be non-null BigInt");
        }
        const buf: metal.cb.TsMetalBuffer = @ptrFromInt(@as(usize, @intCast(bufRaw)));
        var off: u32 = 0;
        var off_v: c.napi_value = undefined;
        if (c.napi_get_named_property(env, item, "offset", &off_v) == c.napi_ok) {
            var t: c.napi_valuetype = undefined;
            if (c.napi_typeof(env, off_v, &t) == c.napi_ok and t == c.napi_number) {
                off = napiGetU32(env, off_v, "bindings[i].offset") orelse return null;
            }
        }
        bindings[i] = .{ .buffer = .{ .index = idx, .buf = buf, .offset = off } };
    }

    const grid_arr = argv[3];
    const tg_arr = argv[4];
    const gx = napiGetU32(env, napiArrayGet(env, grid_arr, 0) orelse return napiThrow(env, "grid[0]"), "grid[0]") orelse return null;
    const gy = napiGetU32(env, napiArrayGet(env, grid_arr, 1) orelse return napiThrow(env, "grid[1]"), "grid[1]") orelse return null;
    const gz = napiGetU32(env, napiArrayGet(env, grid_arr, 2) orelse return napiThrow(env, "grid[2]"), "grid[2]") orelse return null;
    const tx = napiGetU32(env, napiArrayGet(env, tg_arr, 0) orelse return napiThrow(env, "tg[0]"), "tg[0]") orelse return null;
    const ty = napiGetU32(env, napiArrayGet(env, tg_arr, 1) orelse return napiThrow(env, "tg[1]"), "tg[1]") orelse return null;
    const tz = napiGetU32(env, napiArrayGet(env, tg_arr, 2) orelse return napiThrow(env, "tg[2]"), "tg[2]") orelse return null;

    metal.dispatchOneShot(b, name_buf, bindings[0..n], .{ gx, gy, gz }, .{ tx, ty, tz }) catch |err| {
        return switch (err) {
            metal.MetalError.UnknownKernel => napiThrow(env, "metal: unknown kernel name (not in shaders.metal)"),
            metal.MetalError.PipelineCreateFailed => napiThrow(env, "metal: pipeline creation failed (see stderr)"),
            else => napiThrow(env, "metal: dispatchOneShot failed"),
        };
    };
    var u: c.napi_value = undefined;
    _ = c.napi_get_undefined(env, &u);
    return u;
}

fn napiMetalBeginEncoder(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 1; var argv: [1]c.napi_value = undefined;
    _ = c.napi_get_cb_info(env, info, &argc, &argv, null, null);
    if (argc < 1) return napiThrow(env, "metalBeginEncoder(handle) requires 1 arg");
    var raw: u64 = 0; var lossless: bool = false;
    if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
        return napiThrow(env, "argument 0 must be backend BigInt");
    }
    const b: *metal.Backend = @ptrFromInt(@as(usize, @intCast(raw)));
    const e = metal.beginEncoder(b) catch return napiThrow(env, "metal: beginEncoder failed");
    var bn: c.napi_value = undefined;
    if (c.napi_create_bigint_uint64(env, @intCast(@intFromPtr(e)), &bn) != c.napi_ok) {
        return napiThrow(env, "napi: failed to create encoder handle");
    }
    return bn;
}

fn napiMetalEnqueueDispatch(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 5; var argv: [5]c.napi_value = undefined;
    _ = c.napi_get_cb_info(env, info, &argc, &argv, null, null);
    if (argc < 5) return napiThrow(env, "metalEnqueueDispatch(enc, name, bindings, grid, threadgroup) requires 5 args");

    var raw: u64 = 0; var lossless: bool = false;
    if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
        return napiThrow(env, "argument 0 must be encoder BigInt");
    }
    const e: *metal.Encoder = @ptrFromInt(@as(usize, @intCast(raw)));

    const name_buf = napiGetString(env, argv[1], "argument 1 must be string (name)") orelse return null;
    defer std.heap.c_allocator.free(name_buf);

    const n = napiArrayLen(env, argv[2]) orelse return napiThrow(env, "argument 2 must be array (bindings)");
    if (n > 16) return napiThrow(env, "too many bindings");
    var bindings: [16]metal.Binding = undefined;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        const item = napiArrayGet(env, argv[2], i) orelse return napiThrow(env, "bindings[i]");
        const idx_v = napiNamedProperty(env, item, "index") orelse return napiThrow(env, "bindings[i].index");
        const idx = napiGetU32(env, idx_v, "bindings[i].index") orelse return null;
        var bytes_v: c.napi_value = undefined;
        if (c.napi_get_named_property(env, item, "bytes", &bytes_v) == c.napi_ok) {
            var t: c.napi_valuetype = undefined;
            if (c.napi_typeof(env, bytes_v, &t) == c.napi_ok and t != c.napi_undefined and t != c.napi_null) {
                const bytes = napiGetUint8Array(env, bytes_v, "bindings[i].bytes") orelse return null;
                bindings[i] = .{ .bytes = .{ .index = idx, .bytes = bytes } };
                continue;
            }
        }
        const ptr_v = napiNamedProperty(env, item, "bufPtr") orelse return napiThrow(env, "bindings[i] needs bytes or bufPtr");
        var bufRaw: u64 = 0;
        if (c.napi_get_value_bigint_uint64(env, ptr_v, &bufRaw, &lossless) != c.napi_ok or bufRaw == 0) {
            return napiThrow(env, "bindings[i].bufPtr must be non-null BigInt");
        }
        const buf: metal.cb.TsMetalBuffer = @ptrFromInt(@as(usize, @intCast(bufRaw)));
        var off: u32 = 0;
        var off_v: c.napi_value = undefined;
        if (c.napi_get_named_property(env, item, "offset", &off_v) == c.napi_ok) {
            var t: c.napi_valuetype = undefined;
            if (c.napi_typeof(env, off_v, &t) == c.napi_ok and t == c.napi_number) {
                off = napiGetU32(env, off_v, "bindings[i].offset") orelse return null;
            }
        }
        bindings[i] = .{ .buffer = .{ .index = idx, .buf = buf, .offset = off } };
    }

    const grid_arr = argv[3]; const tg_arr = argv[4];
    const gx = napiGetU32(env, napiArrayGet(env, grid_arr, 0) orelse return napiThrow(env, "grid[0]"), "grid[0]") orelse return null;
    const gy = napiGetU32(env, napiArrayGet(env, grid_arr, 1) orelse return napiThrow(env, "grid[1]"), "grid[1]") orelse return null;
    const gz = napiGetU32(env, napiArrayGet(env, grid_arr, 2) orelse return napiThrow(env, "grid[2]"), "grid[2]") orelse return null;
    const tx = napiGetU32(env, napiArrayGet(env, tg_arr, 0) orelse return napiThrow(env, "tg[0]"), "tg[0]") orelse return null;
    const ty = napiGetU32(env, napiArrayGet(env, tg_arr, 1) orelse return napiThrow(env, "tg[1]"), "tg[1]") orelse return null;
    const tz = napiGetU32(env, napiArrayGet(env, tg_arr, 2) orelse return napiThrow(env, "tg[2]"), "tg[2]") orelse return null;

    metal.enqueueOnEncoder(e, name_buf, bindings[0..n], .{ gx, gy, gz }, .{ tx, ty, tz }) catch |err| {
        return switch (err) {
            metal.MetalError.UnknownKernel => napiThrow(env, "metal: unknown kernel name"),
            metal.MetalError.PipelineCreateFailed => napiThrow(env, "metal: pipeline creation failed"),
            else => napiThrow(env, "metal: enqueueDispatch failed"),
        };
    };
    var u: c.napi_value = undefined;
    _ = c.napi_get_undefined(env, &u);
    return u;
}

fn napiMetalSubmitAndReadback(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 4; var argv: [4]c.napi_value = undefined;
    _ = c.napi_get_cb_info(env, info, &argc, &argv, null, null);
    if (argc < 4) return napiThrow(env, "metalSubmitAndReadback(enc, outBufPtr, offset, byteLen) requires 4 args");

    var raw: u64 = 0; var lossless: bool = false;
    if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
        return napiThrow(env, "argument 0 must be encoder BigInt");
    }
    const e: *metal.Encoder = @ptrFromInt(@as(usize, @intCast(raw)));

    var bufRaw: u64 = 0;
    if (c.napi_get_value_bigint_uint64(env, argv[1], &bufRaw, &lossless) != c.napi_ok or bufRaw == 0) {
        return napiThrow(env, "argument 1 must be buffer BigInt");
    }
    const buf: metal.cb.TsMetalBuffer = @ptrFromInt(@as(usize, @intCast(bufRaw)));
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
    metal.submitAndReadback(e, buf, offset, out) catch return napiThrow(env, "metal: submitAndReadback failed");
    return typed;
}

fn napiMetalWriteBuffer(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 3; var argv: [3]c.napi_value = undefined;
    _ = c.napi_get_cb_info(env, info, &argc, &argv, null, null);
    if (argc < 3) return napiThrow(env, "metalWriteBuffer(bufPtr, offset, bytes) requires 3 args");
    var raw: u64 = 0; var lossless: bool = false;
    if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
        return napiThrow(env, "argument 0 must be buffer BigInt");
    }
    const buf: metal.cb.TsMetalBuffer = @ptrFromInt(@as(usize, @intCast(raw)));
    const offset = napiGetU32(env, argv[1], "argument 1 must be u32 (offset)") orelse return null;
    const bytes = napiGetUint8Array(env, argv[2], "argument 2 must be Uint8Array") orelse return null;
    metal.writeBuffer(buf, offset, bytes);
    var u: c.napi_value = undefined;
    _ = c.napi_get_undefined(env, &u);
    return u;
}

fn napiMetalCreateEmptyBuffer(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
    var argc: usize = 2; var argv: [2]c.napi_value = undefined;
    _ = c.napi_get_cb_info(env, info, &argc, &argv, null, null);
    if (argc < 2) return napiThrow(env, "metalCreateEmptyBuffer(handle, byteLen) requires 2 args");
    var raw: u64 = 0; var lossless: bool = false;
    if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
        return napiThrow(env, "argument 0 must be backend BigInt");
    }
    const b: *metal.Backend = @ptrFromInt(@as(usize, @intCast(raw)));
    const len = napiGetU32(env, argv[1], "argument 1 must be u32 (byteLen)") orelse return null;
    const buf = metal.createEmptyBuffer(b, len) catch return napiThrow(env, "metal: createEmptyBuffer failed");
    var bn: c.napi_value = undefined;
    if (c.napi_create_bigint_uint64(env, @intCast(@intFromPtr(buf)), &bn) != c.napi_ok) {
        metal.releaseBuffer(buf);
        return napiThrow(env, "napi: failed to create buffer handle");
    }
    return bn;
}
} else struct {
    pub fn registerAll(env: c.napi_env, exports: c.napi_value) !void {
        _ = env;
        _ = exports;
    }
};

// ── Vulkan-direct backend (Linux only) ──
//
// Hand-tuned GLSL ports compiled to SPIR-V at build time via glslangValidator.
// Gated to Linux only — Windows would need its own build-script work (glslang
// isn't standard on Windows) and we ship Dawn-direct there instead.
//
// JS calling convention for vulkan* dispatches:
//   - bindings: Array<BigInt> — buf handles in storage-binding-slot order
//                (slot 0 = first SSBO in the GLSL, etc.)
//   - pushData: Uint8Array — push-constant block bytes (length must match
//                pipeline's push_constant_size; empty if 0)
//   - grid:     Array<u32>[3] — workgroup count (local_size_* is baked
//                into the GLSL, no threadgroup passed at dispatch time)
const Vulkan = if (is_linux) struct {
    const vk = @import("vulkan_backend.zig");

    pub fn registerAll(env: c.napi_env, exports: c.napi_value) !void {
        try register(env, exports, "vulkanCreateBackend", napiVkCreateBackend);
        try register(env, exports, "vulkanDestroyBackend", napiVkDestroyBackend);
        try register(env, exports, "vulkanDeviceName", napiVkDeviceName);
        try register(env, exports, "vulkanCreateBuffer", napiVkCreateBuffer);
        try register(env, exports, "vulkanCreateEmptyBuffer", napiVkCreateEmptyBuffer);
        try register(env, exports, "vulkanReleaseBuffer", napiVkReleaseBuffer);
        try register(env, exports, "vulkanReadBuffer", napiVkReadBuffer);
        try register(env, exports, "vulkanWriteBuffer", napiVkWriteBuffer);
        try register(env, exports, "vulkanDispatchOneShot", napiVkDispatchOneShot);
        try register(env, exports, "vulkanBeginEncoder", napiVkBeginEncoder);
        try register(env, exports, "vulkanEnqueueDispatch", napiVkEnqueueDispatch);
        try register(env, exports, "vulkanSubmitAndReadback", napiVkSubmitAndReadback);
    }

    fn napiVkCreateBackend(env: c.napi_env, _: c.napi_callback_info) callconv(.c) c.napi_value {
        const b = vk.createBackend() catch |err| {
            return switch (err) {
                vk.VulkanError.BackendCreateFailed => napiThrow(env, "vulkan: createBackend failed (no Vulkan device, or required features missing — see stderr)"),
                else => napiThrow(env, "vulkan: createBackend failed"),
            };
        };
        var bn: c.napi_value = undefined;
        if (c.napi_create_bigint_uint64(env, @intCast(@intFromPtr(b)), &bn) != c.napi_ok) {
            vk.destroyBackend(b);
            return napiThrow(env, "napi: failed to create handle bigint");
        }
        return bn;
    }

    fn napiVkDestroyBackend(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
        var argc: usize = 1; var argv: [1]c.napi_value = undefined;
        _ = c.napi_get_cb_info(env, info, &argc, &argv, null, null);
        if (argc < 1) return napiThrow(env, "vulkanDestroyBackend(handle) requires 1 arg");
        var raw: u64 = 0; var lossless: bool = false;
        if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok) {
            return napiThrow(env, "argument 0 must be BigInt handle");
        }
        if (raw != 0) {
            const b: *vk.Backend = @ptrFromInt(@as(usize, @intCast(raw)));
            vk.destroyBackend(b);
        }
        var u: c.napi_value = undefined;
        _ = c.napi_get_undefined(env, &u);
        return u;
    }

    fn napiVkDeviceName(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
        var argc: usize = 1; var argv: [1]c.napi_value = undefined;
        _ = c.napi_get_cb_info(env, info, &argc, &argv, null, null);
        if (argc < 1) return napiThrow(env, "vulkanDeviceName(handle) requires 1 arg");
        var raw: u64 = 0; var lossless: bool = false;
        if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
            return napiThrow(env, "argument 0 must be non-null BigInt handle");
        }
        const b: *vk.Backend = @ptrFromInt(@as(usize, @intCast(raw)));
        const name = vk.deviceName(b);
        var s: c.napi_value = undefined;
        if (c.napi_create_string_utf8(env, name, c.NAPI_AUTO_LENGTH, &s) != c.napi_ok) {
            return napiThrow(env, "napi: failed to create string");
        }
        return s;
    }

    fn napiVkCreateBuffer(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
        var argc: usize = 2; var argv: [2]c.napi_value = undefined;
        _ = c.napi_get_cb_info(env, info, &argc, &argv, null, null);
        if (argc < 2) return napiThrow(env, "vulkanCreateBuffer(handle, bytes) requires 2 args");
        var raw: u64 = 0; var lossless: bool = false;
        if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
            return napiThrow(env, "argument 0 must be non-null BigInt handle");
        }
        const b: *vk.Backend = @ptrFromInt(@as(usize, @intCast(raw)));
        const bytes = napiGetUint8Array(env, argv[1], "argument 1 must be Uint8Array (bytes)") orelse return null;
        const buf = vk.createBuffer(b, bytes) catch return napiThrow(env, "vulkan: createBuffer failed");
        var bn: c.napi_value = undefined;
        if (c.napi_create_bigint_uint64(env, @intCast(@intFromPtr(buf)), &bn) != c.napi_ok) {
            vk.releaseBuffer(b, buf);
            return napiThrow(env, "napi: failed to create buffer handle");
        }
        return bn;
    }

    fn napiVkCreateEmptyBuffer(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
        var argc: usize = 2; var argv: [2]c.napi_value = undefined;
        _ = c.napi_get_cb_info(env, info, &argc, &argv, null, null);
        if (argc < 2) return napiThrow(env, "vulkanCreateEmptyBuffer(handle, byteLen) requires 2 args");
        var raw: u64 = 0; var lossless: bool = false;
        if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
            return napiThrow(env, "argument 0 must be non-null BigInt handle");
        }
        const b: *vk.Backend = @ptrFromInt(@as(usize, @intCast(raw)));
        const len = napiGetU32(env, argv[1], "argument 1 must be u32 (byteLen)") orelse return null;
        const buf = vk.createEmptyBuffer(b, len) catch return napiThrow(env, "vulkan: createEmptyBuffer failed");
        var bn: c.napi_value = undefined;
        if (c.napi_create_bigint_uint64(env, @intCast(@intFromPtr(buf)), &bn) != c.napi_ok) {
            vk.releaseBuffer(b, buf);
            return napiThrow(env, "napi: failed to create buffer handle");
        }
        return bn;
    }

    fn napiVkReleaseBuffer(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
        var argc: usize = 2; var argv: [2]c.napi_value = undefined;
        _ = c.napi_get_cb_info(env, info, &argc, &argv, null, null);
        if (argc < 2) return napiThrow(env, "vulkanReleaseBuffer(handle, buf) requires 2 args");
        var raw: u64 = 0; var lossless: bool = false;
        if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
            return napiThrow(env, "argument 0 must be non-null BigInt handle");
        }
        const b: *vk.Backend = @ptrFromInt(@as(usize, @intCast(raw)));
        var bufRaw: u64 = 0;
        if (c.napi_get_value_bigint_uint64(env, argv[1], &bufRaw, &lossless) != c.napi_ok) {
            return napiThrow(env, "argument 1 must be BigInt buffer handle");
        }
        if (bufRaw != 0) {
            const buf: vk.cb.TsVkBuffer = @ptrFromInt(@as(usize, @intCast(bufRaw)));
            vk.releaseBuffer(b, buf);
        }
        var u: c.napi_value = undefined;
        _ = c.napi_get_undefined(env, &u);
        return u;
    }

    fn napiVkWriteBuffer(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
        var argc: usize = 4; var argv: [4]c.napi_value = undefined;
        _ = c.napi_get_cb_info(env, info, &argc, &argv, null, null);
        if (argc < 4) return napiThrow(env, "vulkanWriteBuffer(handle, buf, offset, bytes) requires 4 args");
        var raw: u64 = 0; var lossless: bool = false;
        if (c.napi_get_value_bigint_uint64(env, argv[1], &raw, &lossless) != c.napi_ok or raw == 0) {
            return napiThrow(env, "argument 1 must be non-null buffer BigInt");
        }
        const buf: vk.cb.TsVkBuffer = @ptrFromInt(@as(usize, @intCast(raw)));
        const offset = napiGetU32(env, argv[2], "argument 2 must be u32 (offset)") orelse return null;
        const bytes = napiGetUint8Array(env, argv[3], "argument 3 must be Uint8Array (bytes)") orelse return null;
        vk.writeBuffer(buf, offset, bytes);
        var u: c.napi_value = undefined;
        _ = c.napi_get_undefined(env, &u);
        return u;
    }

    fn napiVkReadBuffer(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
        var argc: usize = 4; var argv: [4]c.napi_value = undefined;
        _ = c.napi_get_cb_info(env, info, &argc, &argv, null, null);
        if (argc < 4) return napiThrow(env, "vulkanReadBuffer(handle, buf, offset, byteLen) requires 4 args");
        var raw: u64 = 0; var lossless: bool = false;
        if (c.napi_get_value_bigint_uint64(env, argv[1], &raw, &lossless) != c.napi_ok or raw == 0) {
            return napiThrow(env, "argument 1 must be non-null buffer BigInt");
        }
        const buf: vk.cb.TsVkBuffer = @ptrFromInt(@as(usize, @intCast(raw)));
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
        vk.readBuffer(buf, offset, out);
        return typed;
    }

    // Helper: extract bindings array (Array<BigInt>) and grid array
    // for both dispatchOneShot and enqueueDispatch.
    const ParseError = error{ ParseFailed };
    fn parseBindings(env: c.napi_env, arr: c.napi_value, out: *[16]vk.cb.TsVkBuffer) ParseError!u32 {
        const n = napiArrayLen(env, arr) orelse {
            _ = napiThrow(env, "bindings must be Array");
            return ParseError.ParseFailed;
        };
        if (n > 16) {
            _ = napiThrow(env, "too many bindings (max 16)");
            return ParseError.ParseFailed;
        }
        var i: u32 = 0;
        while (i < n) : (i += 1) {
            const item = napiArrayGet(env, arr, i) orelse {
                _ = napiThrow(env, "bindings[i] missing");
                return ParseError.ParseFailed;
            };
            var raw: u64 = 0; var lossless: bool = false;
            if (c.napi_get_value_bigint_uint64(env, item, &raw, &lossless) != c.napi_ok or raw == 0) {
                _ = napiThrow(env, "bindings[i] must be non-null BigInt buffer handle");
                return ParseError.ParseFailed;
            }
            out[i] = @ptrFromInt(@as(usize, @intCast(raw)));
        }
        return n;
    }

    fn parseGrid(env: c.napi_env, arr: c.napi_value) ?[3]u32 {
        const gx = napiGetU32(env, napiArrayGet(env, arr, 0) orelse return null, "grid[0]") orelse return null;
        const gy = napiGetU32(env, napiArrayGet(env, arr, 1) orelse return null, "grid[1]") orelse return null;
        const gz = napiGetU32(env, napiArrayGet(env, arr, 2) orelse return null, "grid[2]") orelse return null;
        return .{ gx, gy, gz };
    }

    fn napiVkDispatchOneShot(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
        var argc: usize = 5; var argv: [5]c.napi_value = undefined;
        _ = c.napi_get_cb_info(env, info, &argc, &argv, null, null);
        if (argc < 5) return napiThrow(env, "vulkanDispatchOneShot(handle, name, bindings, pushData, grid) requires 5 args");

        var raw: u64 = 0; var lossless: bool = false;
        if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
            return napiThrow(env, "argument 0 must be non-null backend BigInt");
        }
        const b: *vk.Backend = @ptrFromInt(@as(usize, @intCast(raw)));

        const name_buf = napiGetString(env, argv[1], "argument 1 must be string (name)") orelse return null;
        defer std.heap.c_allocator.free(name_buf);

        var bindings: [16]vk.cb.TsVkBuffer = undefined;
        const n = parseBindings(env, argv[2], &bindings) catch return null;

        const push = napiGetUint8Array(env, argv[3], "argument 3 must be Uint8Array (pushData)") orelse return null;

        const grid = parseGrid(env, argv[4]) orelse return null;

        vk.dispatchOneShot(b, name_buf, bindings[0..n], push, grid) catch |err| {
            return switch (err) {
                vk.VulkanError.UnknownKernel => napiThrow(env, "vulkan: unknown kernel name (not in SHADERS table)"),
                vk.VulkanError.PipelineCreateFailed => napiThrow(env, "vulkan: pipeline creation failed (see stderr)"),
                vk.VulkanError.BindingMismatch => napiThrow(env, "vulkan: binding count or pushData size mismatch vs pipeline"),
                else => napiThrow(env, "vulkan: dispatchOneShot failed"),
            };
        };
        var u: c.napi_value = undefined;
        _ = c.napi_get_undefined(env, &u);
        return u;
    }

    fn napiVkBeginEncoder(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
        var argc: usize = 1; var argv: [1]c.napi_value = undefined;
        _ = c.napi_get_cb_info(env, info, &argc, &argv, null, null);
        if (argc < 1) return napiThrow(env, "vulkanBeginEncoder(handle) requires 1 arg");
        var raw: u64 = 0; var lossless: bool = false;
        if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
            return napiThrow(env, "argument 0 must be non-null backend BigInt");
        }
        const b: *vk.Backend = @ptrFromInt(@as(usize, @intCast(raw)));
        const e = vk.beginEncoder(b) catch return napiThrow(env, "vulkan: beginEncoder failed");
        var bn: c.napi_value = undefined;
        if (c.napi_create_bigint_uint64(env, @intCast(@intFromPtr(e)), &bn) != c.napi_ok) {
            return napiThrow(env, "napi: failed to create encoder handle");
        }
        return bn;
    }

    fn napiVkEnqueueDispatch(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
        var argc: usize = 5; var argv: [5]c.napi_value = undefined;
        _ = c.napi_get_cb_info(env, info, &argc, &argv, null, null);
        if (argc < 5) return napiThrow(env, "vulkanEnqueueDispatch(encoder, name, bindings, pushData, grid) requires 5 args");

        var raw: u64 = 0; var lossless: bool = false;
        if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
            return napiThrow(env, "argument 0 must be non-null encoder BigInt");
        }
        const e: *vk.Encoder = @ptrFromInt(@as(usize, @intCast(raw)));

        const name_buf = napiGetString(env, argv[1], "argument 1 must be string (name)") orelse return null;
        defer std.heap.c_allocator.free(name_buf);

        var bindings: [16]vk.cb.TsVkBuffer = undefined;
        const n = parseBindings(env, argv[2], &bindings) catch return null;

        const push = napiGetUint8Array(env, argv[3], "argument 3 must be Uint8Array (pushData)") orelse return null;

        const grid = parseGrid(env, argv[4]) orelse return null;

        vk.enqueueOnEncoder(e, name_buf, bindings[0..n], push, grid) catch |err| {
            return switch (err) {
                vk.VulkanError.UnknownKernel => napiThrow(env, "vulkan: unknown kernel name (not in SHADERS table)"),
                vk.VulkanError.PipelineCreateFailed => napiThrow(env, "vulkan: pipeline creation failed (see stderr)"),
                vk.VulkanError.BindingMismatch => napiThrow(env, "vulkan: binding count or pushData size mismatch vs pipeline"),
                else => napiThrow(env, "vulkan: enqueueDispatch failed"),
            };
        };
        var u: c.napi_value = undefined;
        _ = c.napi_get_undefined(env, &u);
        return u;
    }

    fn napiVkSubmitAndReadback(env: c.napi_env, info: c.napi_callback_info) callconv(.c) c.napi_value {
        var argc: usize = 4; var argv: [4]c.napi_value = undefined;
        _ = c.napi_get_cb_info(env, info, &argc, &argv, null, null);
        if (argc < 4) return napiThrow(env, "vulkanSubmitAndReadback(encoder, outBuf, offset, byteLen) requires 4 args");

        var raw: u64 = 0; var lossless: bool = false;
        if (c.napi_get_value_bigint_uint64(env, argv[0], &raw, &lossless) != c.napi_ok or raw == 0) {
            return napiThrow(env, "argument 0 must be non-null encoder BigInt");
        }
        const e: *vk.Encoder = @ptrFromInt(@as(usize, @intCast(raw)));

        var bufRaw: u64 = 0;
        if (c.napi_get_value_bigint_uint64(env, argv[1], &bufRaw, &lossless) != c.napi_ok or bufRaw == 0) {
            return napiThrow(env, "argument 1 must be non-null buffer BigInt");
        }
        const out_buf: vk.cb.TsVkBuffer = @ptrFromInt(@as(usize, @intCast(bufRaw)));

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
        vk.submitAndReadback(e, out_buf, offset, out);
        return typed;
    }
} else struct {
    pub fn registerAll(env: c.napi_env, exports: c.napi_value) !void {
        _ = env;
        _ = exports;
    }
};
