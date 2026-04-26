// Dawn-direct bridge implementation. Mirrors vulkan/bridge.c's shape.
//
// Pattern:
//   ts_dawn_create_backend() — Instance + RequestAdapter + RequestDevice +
//                              GetQueue. Async ops use wgpuInstanceWaitAny.
//   ts_dawn_pipeline_create() — Tint compiles WGSL → SPIR-V/MSL/HLSL at
//                               this point; expensive once, then cached.
//   ts_dawn_buffer_create() — host-visible storage buffer (UMA on Iris Xe).
//                             Initial data goes in via mappedAtCreation=true.
//   ts_dawn_encoder_begin/dispatch/submit_and_wait — one command buffer
//                              per encoder; per-dispatch uniform buffer
//                              + bind group are tracked and released
//                              in submit_and_wait.
//
// Synchronization: every async function returns WGPUFuture; we call
// wgpuInstanceWaitAny(instance, 1, &wait_info, UINT64_MAX) to block.
// This is the C-equivalent of `await` and sidesteps the setImmediate
// event-loop hang that broke the npm `webgpu` package on heavy loads.

#include "bridge.h"

#include <webgpu/webgpu.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ── helpers ────────────────────────────────────────────────────────────

static void format_err(char* err_out, size_t err_cap, const char* fmt, ...) {
    if (!err_out || err_cap == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err_out, err_cap, fmt, ap);
    va_end(ap);
}

static WGPUStringView make_sv(const char* s, size_t len) {
    WGPUStringView sv;
    sv.data = s;
    sv.length = len;
    return sv;
}

static WGPUStringView cstr_sv(const char* s) {
    return make_sv(s, s ? strlen(s) : 0);
}

// ── opaque struct definitions ──────────────────────────────────────────

struct TsDawnBackendImpl {
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
    WGPUQueue queue;
    char device_name[256];
};

struct TsDawnBufferImpl {
    WGPUBuffer buffer;
    size_t length;
    // Companion readback buffer (lazy-created on first read).
    WGPUBuffer readback;
};

#define MAX_UNIFORMS_PER_PIPELINE 4

struct TsDawnPipelineImpl {
    WGPUShaderModule shader;
    WGPUBindGroupLayout bgl;
    WGPUPipelineLayout layout;
    WGPUComputePipeline pipeline;
    uint32_t num_storage_buffers;
    uint32_t num_uniforms;
    uint32_t uniform_sizes[MAX_UNIFORMS_PER_PIPELINE];
    uint32_t uniform_total_size;  // sum of uniform_sizes
};

#define MAX_DISPATCHES_PER_ENCODER 256

struct TsDawnEncoderImpl {
    TsDawnBackend backend;
    WGPUCommandEncoder cmd_encoder;
    WGPUComputePassEncoder pass;
    // Per-dispatch resources tracked here so we can release them after
    // the queue drains. Each dispatch may use 1..MAX_UNIFORMS uniform
    // buffers; the flat array is indexed by total count.
    WGPUBuffer uniform_buffers[MAX_DISPATCHES_PER_ENCODER * MAX_UNIFORMS_PER_PIPELINE];
    size_t uniform_buffer_count;
    WGPUBindGroup bind_groups[MAX_DISPATCHES_PER_ENCODER];
    size_t dispatch_count;
};

// ── async-to-sync helpers using wgpuInstanceWaitAny ─────────────────

typedef struct {
    bool done;
    WGPURequestAdapterStatus status;
    WGPUAdapter adapter;
    char message[512];
} AdapterRequest;

static void on_adapter_request(
    WGPURequestAdapterStatus status,
    WGPUAdapter adapter,
    WGPUStringView message,
    void* userdata1,
    void* userdata2)
{
    (void)userdata2;
    AdapterRequest* r = (AdapterRequest*)userdata1;
    r->status = status;
    r->adapter = adapter;
    r->done = true;
    if (message.data && message.length > 0) {
        size_t n = message.length < sizeof(r->message) - 1
                   ? message.length : sizeof(r->message) - 1;
        memcpy(r->message, message.data, n);
        r->message[n] = '\0';
    }
}

typedef struct {
    bool done;
    WGPURequestDeviceStatus status;
    WGPUDevice device;
    char message[512];
} DeviceRequest;

static void on_device_request(
    WGPURequestDeviceStatus status,
    WGPUDevice device,
    WGPUStringView message,
    void* userdata1,
    void* userdata2)
{
    (void)userdata2;
    DeviceRequest* r = (DeviceRequest*)userdata1;
    r->status = status;
    r->device = device;
    r->done = true;
    if (message.data && message.length > 0) {
        size_t n = message.length < sizeof(r->message) - 1
                   ? message.length : sizeof(r->message) - 1;
        memcpy(r->message, message.data, n);
        r->message[n] = '\0';
    }
}

typedef struct {
    bool done;
    WGPUMapAsyncStatus status;
    char message[256];
} MapRequest;

static void on_map(
    WGPUMapAsyncStatus status,
    WGPUStringView message,
    void* userdata1,
    void* userdata2)
{
    (void)userdata2;
    MapRequest* r = (MapRequest*)userdata1;
    r->status = status;
    r->done = true;
    if (message.data && message.length > 0) {
        size_t n = message.length < sizeof(r->message) - 1
                   ? message.length : sizeof(r->message) - 1;
        memcpy(r->message, message.data, n);
        r->message[n] = '\0';
    }
}

typedef struct {
    bool done;
    WGPUQueueWorkDoneStatus status;
} WorkDoneRequest;

static void on_work_done(
    WGPUQueueWorkDoneStatus status,
    WGPUStringView message,
    void* userdata1,
    void* userdata2)
{
    (void)message; (void)userdata2;
    WorkDoneRequest* r = (WorkDoneRequest*)userdata1;
    r->status = status;
    r->done = true;
}

static void on_uncaptured_error(
    WGPUDevice const * device,
    WGPUErrorType type,
    WGPUStringView message,
    void* userdata1,
    void* userdata2)
{
    (void)device; (void)userdata1; (void)userdata2;
    fprintf(stderr, "dawn: uncaptured error type=%d: %.*s\n",
            (int)type, (int)message.length, message.data);
}

static void on_device_lost(
    WGPUDevice const * device,
    WGPUDeviceLostReason reason,
    WGPUStringView message,
    void* userdata1,
    void* userdata2)
{
    (void)device; (void)userdata1; (void)userdata2;
    fprintf(stderr, "dawn: device lost reason=%d: %.*s\n",
            (int)reason, (int)message.length, message.data);
}

// ── backend lifecycle ──────────────────────────────────────────────────

TsDawnBackend ts_dawn_create_backend(char* err_out, size_t err_cap) {
    TsDawnBackend b = (TsDawnBackend)calloc(1, sizeof(*b));
    if (!b) { format_err(err_out, err_cap, "calloc backend"); return NULL; }

    // Instance with TimedWaitAny enabled so we can use wgpuInstanceWaitAny.
    WGPUInstanceFeatureName instance_features[] = { WGPUInstanceFeatureName_TimedWaitAny };
    WGPUInstanceDescriptor idesc = {0};
    idesc.requiredFeatureCount = 1;
    idesc.requiredFeatures = instance_features;
    b->instance = wgpuCreateInstance(&idesc);
    if (!b->instance) { format_err(err_out, err_cap, "wgpuCreateInstance failed"); free(b); return NULL; }

    // Request adapter (async → sync via WaitAny).
    WGPURequestAdapterOptions opts = {0};
    opts.powerPreference = WGPUPowerPreference_HighPerformance;

    AdapterRequest areq = {0};
    WGPURequestAdapterCallbackInfo acb = {0};
    acb.mode = WGPUCallbackMode_WaitAnyOnly;
    acb.callback = on_adapter_request;
    acb.userdata1 = &areq;

    WGPUFuture afuture = wgpuInstanceRequestAdapter(b->instance, &opts, acb);
    WGPUFutureWaitInfo await = {0};
    await.future = afuture;
    WGPUWaitStatus astatus = wgpuInstanceWaitAny(b->instance, 1, &await, UINT64_MAX);
    if (astatus != WGPUWaitStatus_Success || areq.status != WGPURequestAdapterStatus_Success || !areq.adapter) {
        format_err(err_out, err_cap, "requestAdapter failed: %s", areq.message);
        wgpuInstanceRelease(b->instance);
        free(b);
        return NULL;
    }
    b->adapter = areq.adapter;

    // Read device name from adapter info.
    WGPUAdapterInfo info = {0};
    wgpuAdapterGetInfo(b->adapter, &info);
    if (info.device.data && info.device.length > 0) {
        size_t n = info.device.length < sizeof(b->device_name) - 1
                   ? info.device.length : sizeof(b->device_name) - 1;
        memcpy(b->device_name, info.device.data, n);
        b->device_name[n] = '\0';
    } else {
        snprintf(b->device_name, sizeof(b->device_name), "(unknown Dawn device)");
    }
    wgpuAdapterInfoFreeMembers(info);

    // Request device with shader-f16 + 16-bit storage features.
    WGPUFeatureName features[] = {
        WGPUFeatureName_ShaderF16,
    };
    WGPUDeviceDescriptor ddesc = {0};
    ddesc.requiredFeatureCount = sizeof(features) / sizeof(features[0]);
    ddesc.requiredFeatures = features;
    ddesc.uncapturedErrorCallbackInfo.callback = on_uncaptured_error;
    ddesc.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
    ddesc.deviceLostCallbackInfo.callback = on_device_lost;

    DeviceRequest dreq = {0};
    WGPURequestDeviceCallbackInfo dcb = {0};
    dcb.mode = WGPUCallbackMode_WaitAnyOnly;
    dcb.callback = on_device_request;
    dcb.userdata1 = &dreq;

    WGPUFuture dfuture = wgpuAdapterRequestDevice(b->adapter, &ddesc, dcb);
    WGPUFutureWaitInfo dwait = {0};
    dwait.future = dfuture;
    WGPUWaitStatus dstatus = wgpuInstanceWaitAny(b->instance, 1, &dwait, UINT64_MAX);
    if (dstatus != WGPUWaitStatus_Success || dreq.status != WGPURequestDeviceStatus_Success || !dreq.device) {
        format_err(err_out, err_cap, "requestDevice failed: %s", dreq.message);
        wgpuAdapterRelease(b->adapter);
        wgpuInstanceRelease(b->instance);
        free(b);
        return NULL;
    }
    b->device = dreq.device;
    b->queue = wgpuDeviceGetQueue(b->device);

    return b;
}

void ts_dawn_destroy_backend(TsDawnBackend b) {
    if (!b) return;
    if (b->queue) wgpuQueueRelease(b->queue);
    if (b->device) wgpuDeviceRelease(b->device);
    if (b->adapter) wgpuAdapterRelease(b->adapter);
    if (b->instance) wgpuInstanceRelease(b->instance);
    free(b);
}

const char* ts_dawn_device_name(TsDawnBackend b) {
    return b ? b->device_name : "";
}

// ── pipelines ──────────────────────────────────────────────────────────

TsDawnPipeline ts_dawn_pipeline_create(
    TsDawnBackend b,
    const char* wgsl_source, size_t wgsl_len,
    const char* entry_point,
    uint32_t num_storage_buffers,
    uint32_t num_uniforms,
    const uint32_t* uniform_sizes,
    char* err_out, size_t err_cap)
{
    if (!b || !wgsl_source || wgsl_len == 0) {
        format_err(err_out, err_cap, "ts_dawn_pipeline_create: null arg");
        return NULL;
    }
    if (num_uniforms > MAX_UNIFORMS_PER_PIPELINE) {
        format_err(err_out, err_cap, "ts_dawn_pipeline_create: too many uniforms (%u > %d)",
                   num_uniforms, MAX_UNIFORMS_PER_PIPELINE);
        return NULL;
    }
    TsDawnPipeline p = (TsDawnPipeline)calloc(1, sizeof(*p));
    if (!p) { format_err(err_out, err_cap, "calloc pipeline"); return NULL; }
    p->num_storage_buffers = num_storage_buffers;
    p->num_uniforms = num_uniforms;
    p->uniform_total_size = 0;
    for (uint32_t i = 0; i < num_uniforms; i++) {
        p->uniform_sizes[i] = uniform_sizes[i];
        p->uniform_total_size += uniform_sizes[i];
    }

    // Shader module from WGSL.
    WGPUShaderSourceWGSL wgsl_src = {0};
    wgsl_src.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl_src.code = make_sv(wgsl_source, wgsl_len);

    WGPUShaderModuleDescriptor smd = {0};
    smd.nextInChain = (WGPUChainedStruct*)&wgsl_src;

    p->shader = wgpuDeviceCreateShaderModule(b->device, &smd);
    if (!p->shader) {
        format_err(err_out, err_cap, "createShaderModule returned NULL");
        free(p); return NULL;
    }

    // Auto-layout: pass NULL for `layout` and let Dawn derive the
    // bind-group layout from the WGSL source. This handles read-only
    // vs read-write storage automatically (the WGSL declares
    // `var<storage, read>` vs `var<storage, read_write>` per binding,
    // and Dawn's auto-layout matches that — building it manually
    // would mean propagating the access mode through our API).
    WGPUComputePipelineDescriptor cpd = {0};
    cpd.layout = NULL;
    cpd.compute.module = p->shader;
    cpd.compute.entryPoint = cstr_sv(entry_point ? entry_point : "main");

    p->pipeline = wgpuDeviceCreateComputePipeline(b->device, &cpd);
    if (!p->pipeline) {
        format_err(err_out, err_cap, "createComputePipeline failed");
        wgpuShaderModuleRelease(p->shader);
        free(p); return NULL;
    }

    // Cache the auto-derived BGL so the dispatch path can build bind
    // groups against it without re-fetching every dispatch.
    p->bgl = wgpuComputePipelineGetBindGroupLayout(p->pipeline, 0);
    if (!p->bgl) {
        format_err(err_out, err_cap, "getBindGroupLayout(0) returned NULL");
        wgpuComputePipelineRelease(p->pipeline);
        wgpuShaderModuleRelease(p->shader);
        free(p); return NULL;
    }
    p->layout = NULL;  // auto layout: no explicit pipeline layout to release

    return p;
}

void ts_dawn_pipeline_release(TsDawnBackend b, TsDawnPipeline p) {
    (void)b;
    if (!p) return;
    if (p->pipeline) wgpuComputePipelineRelease(p->pipeline);
    if (p->layout) wgpuPipelineLayoutRelease(p->layout);
    if (p->bgl) wgpuBindGroupLayoutRelease(p->bgl);
    if (p->shader) wgpuShaderModuleRelease(p->shader);
    free(p);
}

// ── buffers ────────────────────────────────────────────────────────────

TsDawnBuffer ts_dawn_buffer_create(TsDawnBackend b, size_t length, const void* init_data) {
    if (!b || length == 0) return NULL;
    TsDawnBuffer buf = (TsDawnBuffer)calloc(1, sizeof(*buf));
    if (!buf) return NULL;
    buf->length = length;

    // Round size up to 4-byte multiple (WebGPU buffer-size requirement).
    size_t padded = (length + 3) & ~(size_t)3;

    WGPUBufferDescriptor bd = {0};
    bd.size = padded;
    bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
    bd.mappedAtCreation = init_data != NULL;

    buf->buffer = wgpuDeviceCreateBuffer(b->device, &bd);
    if (!buf->buffer) { free(buf); return NULL; }

    if (init_data) {
        void* mapped = wgpuBufferGetMappedRange(buf->buffer, 0, padded);
        if (mapped) {
            memcpy(mapped, init_data, length);
            if (padded > length) memset((char*)mapped + length, 0, padded - length);
        }
        wgpuBufferUnmap(buf->buffer);
    }
    return buf;
}

void ts_dawn_buffer_release(TsDawnBackend b, TsDawnBuffer buf) {
    (void)b;
    if (!buf) return;
    if (buf->buffer) wgpuBufferRelease(buf->buffer);
    if (buf->readback) wgpuBufferRelease(buf->readback);
    free(buf);
}

void ts_dawn_buffer_write(TsDawnBackend b, TsDawnBuffer buf, size_t offset, const void* data, size_t len) {
    if (!buf || !data || !b) return;
    if (offset + len > buf->length) return;
    // wgpuQueueWriteBuffer stages internally; safe between encoder calls.
    wgpuQueueWriteBuffer(b->queue, buf->buffer, offset, data, (len + 3) & ~(size_t)3);
}

void ts_dawn_buffer_read(TsDawnBackend b, TsDawnBuffer buf, size_t offset, void* dst, size_t len) {
    if (!b || !buf || !dst || len == 0) return;
    if (offset + len > buf->length) return;

    size_t padded = (len + 3) & ~(size_t)3;

    // Lazy-create the companion readback buffer (or grow it if needed).
    if (!buf->readback || wgpuBufferGetSize(buf->readback) < padded) {
        if (buf->readback) {
            wgpuBufferRelease(buf->readback);
            buf->readback = NULL;
        }
        WGPUBufferDescriptor rd = {0};
        rd.size = padded;
        rd.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
        buf->readback = wgpuDeviceCreateBuffer(b->device, &rd);
        if (!buf->readback) return;
    }

    // Copy storage → readback via a one-shot encoder.
    WGPUCommandEncoderDescriptor ced = {0};
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(b->device, &ced);
    wgpuCommandEncoderCopyBufferToBuffer(enc, buf->buffer, offset, buf->readback, 0, padded);
    WGPUCommandBufferDescriptor cbd = {0};
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, &cbd);
    wgpuQueueSubmit(b->queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);

    // Wait for submit to drain, then map read.
    WorkDoneRequest wd = {0};
    WGPUQueueWorkDoneCallbackInfo wdcb = {0};
    wdcb.mode = WGPUCallbackMode_WaitAnyOnly;
    wdcb.callback = on_work_done;
    wdcb.userdata1 = &wd;
    WGPUFuture wf = wgpuQueueOnSubmittedWorkDone(b->queue, wdcb);
    WGPUFutureWaitInfo wwait = { wf };
    wgpuInstanceWaitAny(b->instance, 1, &wwait, UINT64_MAX);

    MapRequest mr = {0};
    WGPUBufferMapCallbackInfo mcb = {0};
    mcb.mode = WGPUCallbackMode_WaitAnyOnly;
    mcb.callback = on_map;
    mcb.userdata1 = &mr;
    WGPUFuture mf = wgpuBufferMapAsync(buf->readback, WGPUMapMode_Read, 0, padded, mcb);
    WGPUFutureWaitInfo mwait = { mf };
    wgpuInstanceWaitAny(b->instance, 1, &mwait, UINT64_MAX);
    if (mr.status != WGPUMapAsyncStatus_Success) {
        fprintf(stderr, "dawn: mapAsync failed: %s\n", mr.message);
        return;
    }

    const void* mapped = wgpuBufferGetConstMappedRange(buf->readback, 0, padded);
    if (mapped) memcpy(dst, mapped, len);
    wgpuBufferUnmap(buf->readback);
}

// ── encoder ────────────────────────────────────────────────────────────

TsDawnEncoder ts_dawn_encoder_begin(TsDawnBackend b) {
    if (!b) return NULL;
    TsDawnEncoder e = (TsDawnEncoder)calloc(1, sizeof(*e));
    if (!e) return NULL;
    e->backend = b;

    WGPUCommandEncoderDescriptor ced = {0};
    e->cmd_encoder = wgpuDeviceCreateCommandEncoder(b->device, &ced);
    if (!e->cmd_encoder) { free(e); return NULL; }

    WGPUComputePassDescriptor cpd = {0};
    e->pass = wgpuCommandEncoderBeginComputePass(e->cmd_encoder, &cpd);
    if (!e->pass) {
        wgpuCommandEncoderRelease(e->cmd_encoder);
        free(e); return NULL;
    }
    return e;
}

void ts_dawn_encoder_release(TsDawnEncoder e) {
    if (!e) return;
    if (e->pass) {
        wgpuComputePassEncoderEnd(e->pass);
        wgpuComputePassEncoderRelease(e->pass);
    }
    if (e->cmd_encoder) wgpuCommandEncoderRelease(e->cmd_encoder);
    for (size_t i = 0; i < e->dispatch_count; i++) {
        if (e->bind_groups[i]) wgpuBindGroupRelease(e->bind_groups[i]);
    }
    for (size_t i = 0; i < e->uniform_buffer_count; i++) {
        if (e->uniform_buffers[i]) wgpuBufferRelease(e->uniform_buffers[i]);
    }
    free(e);
}

void ts_dawn_encoder_dispatch(
    TsDawnEncoder e,
    TsDawnPipeline p,
    const TsDawnBuffer* bindings, uint32_t num_bindings,
    const void* uniform_data, uint32_t uniform_size,
    uint32_t gx, uint32_t gy, uint32_t gz)
{
    if (!e || !p) return;
    if (e->dispatch_count >= MAX_DISPATCHES_PER_ENCODER) {
        fprintf(stderr, "dawn: too many dispatches per encoder (>%d)\n",
                MAX_DISPATCHES_PER_ENCODER);
        return;
    }
    if (num_bindings != p->num_storage_buffers) {
        fprintf(stderr, "dawn: binding count mismatch (got %u, expected %u)\n",
                num_bindings, p->num_storage_buffers);
        return;
    }
    if (uniform_size != p->uniform_total_size) {
        fprintf(stderr, "dawn: uniform total size mismatch (got %u, expected %u)\n",
                uniform_size, p->uniform_total_size);
        return;
    }

    TsDawnBackend b = e->backend;

    // Build bind-group entries. WGSL bindings 0..num_uniforms-1 are
    // uniform buffers, then num_uniforms..num_uniforms+num_storage-1
    // are the user-supplied storage buffers.
    WGPUBindGroupEntry entries[17];
    uint32_t entry_count = 0;

    // Allocate one uniform buffer per uniform binding, sliced from
    // the concatenated `uniform_data` payload.
    size_t uoff = 0;
    for (uint32_t u = 0; u < p->num_uniforms; u++) {
        uint32_t sz = p->uniform_sizes[u];
        size_t padded = (sz + 15) & ~(size_t)15;  // 16-byte align (WebGPU min)
        WGPUBufferDescriptor ubd = {0};
        ubd.size = padded;
        ubd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        ubd.mappedAtCreation = 1;
        WGPUBuffer ubuf = wgpuDeviceCreateBuffer(b->device, &ubd);
        if (!ubuf) return;
        void* mapped = wgpuBufferGetMappedRange(ubuf, 0, padded);
        if (mapped) {
            memcpy(mapped, (const char*)uniform_data + uoff, sz);
            if (padded > sz) memset((char*)mapped + sz, 0, padded - sz);
        }
        wgpuBufferUnmap(ubuf);
        if (e->uniform_buffer_count >= MAX_DISPATCHES_PER_ENCODER * MAX_UNIFORMS_PER_PIPELINE) {
            fprintf(stderr, "dawn: uniform buffer pool exhausted\n");
            wgpuBufferRelease(ubuf);
            return;
        }
        e->uniform_buffers[e->uniform_buffer_count++] = ubuf;

        entries[entry_count] = (WGPUBindGroupEntry){0};
        entries[entry_count].binding = u;
        entries[entry_count].buffer = ubuf;
        entries[entry_count].size = sz;
        entry_count++;
        uoff += sz;
    }

    for (uint32_t i = 0; i < num_bindings; i++) {
        if (!bindings[i] || !bindings[i]->buffer) {
            fprintf(stderr, "dawn: null binding at slot %u\n", i);
            return;
        }
        entries[entry_count] = (WGPUBindGroupEntry){0};
        entries[entry_count].binding = p->num_uniforms + i;
        entries[entry_count].buffer = bindings[i]->buffer;
        entries[entry_count].size = bindings[i]->length;
        entry_count++;
    }

    WGPUBindGroupDescriptor bgd = {0};
    bgd.layout = p->bgl;
    bgd.entryCount = entry_count;
    bgd.entries = entries;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(b->device, &bgd);
    if (!bg) return;
    e->bind_groups[e->dispatch_count] = bg;

    wgpuComputePassEncoderSetPipeline(e->pass, p->pipeline);
    wgpuComputePassEncoderSetBindGroup(e->pass, 0, bg, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(e->pass, gx, gy, gz);

    e->dispatch_count++;
}

void ts_dawn_encoder_submit_and_wait(TsDawnEncoder e) {
    if (!e) return;
    TsDawnBackend b = e->backend;

    if (e->pass) {
        wgpuComputePassEncoderEnd(e->pass);
        wgpuComputePassEncoderRelease(e->pass);
        e->pass = NULL;
    }
    WGPUCommandBufferDescriptor cbd = {0};
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(e->cmd_encoder, &cbd);
    wgpuCommandEncoderRelease(e->cmd_encoder);
    e->cmd_encoder = NULL;
    if (!cb) {
        ts_dawn_encoder_release(e);
        return;
    }
    wgpuQueueSubmit(b->queue, 1, &cb);
    wgpuCommandBufferRelease(cb);

    WorkDoneRequest wd = {0};
    WGPUQueueWorkDoneCallbackInfo wdcb = {0};
    wdcb.mode = WGPUCallbackMode_WaitAnyOnly;
    wdcb.callback = on_work_done;
    wdcb.userdata1 = &wd;
    WGPUFuture wf = wgpuQueueOnSubmittedWorkDone(b->queue, wdcb);
    WGPUFutureWaitInfo wwait = { wf };
    wgpuInstanceWaitAny(b->instance, 1, &wwait, UINT64_MAX);

    // Release per-dispatch resources.
    for (size_t i = 0; i < e->dispatch_count; i++) {
        if (e->bind_groups[i]) wgpuBindGroupRelease(e->bind_groups[i]);
    }
    for (size_t i = 0; i < e->uniform_buffer_count; i++) {
        if (e->uniform_buffers[i]) wgpuBufferRelease(e->uniform_buffers[i]);
    }
    free(e);
}
