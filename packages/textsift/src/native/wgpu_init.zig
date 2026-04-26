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
};

const RequestState = struct {
    // 0 isn't a valid WGPURequestAdapterStatus value (the enum starts
    // at _Success = 0x1), so it serves as a "callback hasn't run yet"
    // sentinel without name-collision with the upstream header.
    status: c.WGPURequestAdapterStatus = 0,
    adapter: c.WGPUAdapter = null,
    fired: bool = false,
};

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
