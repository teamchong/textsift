// Objective-C implementation of the Metal C bridge. Each function
// takes/returns plain-C types so Zig can call them through the
// header without needing Obj-C runtime knowledge.
//
// All Metal Obj-C objects are reference-counted via ARC. We bridge-
// retain on creation (to hand out an opaque void* that owns a +1
// retain), and bridge-release on the matching *_release.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <stdio.h>
#import <string.h>

#include "bridge.h"

// ── helpers ──
static void copy_err(NSError* err, char* buf, size_t cap) {
  if (!buf || cap == 0) return;
  if (!err) { buf[0] = 0; return; }
  const char* msg = [[err localizedDescription] UTF8String];
  if (!msg) msg = "(no error message)";
  strncpy(buf, msg, cap - 1);
  buf[cap - 1] = 0;
}

// ── device + queue ──
TsMetalDevice ts_metal_create_device(void) {
  id<MTLDevice> d = MTLCreateSystemDefaultDevice();
  if (!d) return NULL;
  return (__bridge_retained void*)d;
}

void ts_metal_device_release(TsMetalDevice d) {
  if (!d) return;
  CFRelease(d);
}

const char* ts_metal_device_name(TsMetalDevice d) {
  if (!d) return "(null)";
  id<MTLDevice> dev = (__bridge id<MTLDevice>)d;
  return [[dev name] UTF8String];
}

TsMetalQueue ts_metal_device_queue(TsMetalDevice d) {
  if (!d) return NULL;
  id<MTLDevice> dev = (__bridge id<MTLDevice>)d;
  id<MTLCommandQueue> q = [dev newCommandQueue];
  if (!q) return NULL;
  return (__bridge_retained void*)q;
}

void ts_metal_queue_release(TsMetalQueue q) {
  if (!q) return;
  CFRelease(q);
}

// ── library ──
TsMetalLibrary ts_metal_library_from_source(
    TsMetalDevice d, const char* msl_source, size_t source_len,
    char* err_out, size_t err_cap) {
  if (!d || !msl_source) return NULL;
  id<MTLDevice> dev = (__bridge id<MTLDevice>)d;
  NSString* src = [[NSString alloc]
      initWithBytes:msl_source length:source_len encoding:NSUTF8StringEncoding];
  if (!src) {
    copy_err(nil, err_out, err_cap);
    if (err_out && err_cap > 0) {
      const char* m = "msl source not valid UTF-8";
      strncpy(err_out, m, err_cap - 1);
      err_out[err_cap - 1] = 0;
    }
    return NULL;
  }
  MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
  opts.languageVersion = MTLLanguageVersion3_0; // M3+ supports simdgroup_matrix
  NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
  if (!lib) {
    copy_err(err, err_out, err_cap);
    return NULL;
  }
  return (__bridge_retained void*)lib;
}

void ts_metal_library_release(TsMetalLibrary lib) {
  if (!lib) return;
  CFRelease(lib);
}

TsMetalPipeline ts_metal_pipeline_for(
    TsMetalDevice d, TsMetalLibrary lib, const char* entry_name,
    char* err_out, size_t err_cap) {
  if (!d || !lib || !entry_name) return NULL;
  id<MTLDevice> dev = (__bridge id<MTLDevice>)d;
  id<MTLLibrary> mlib = (__bridge id<MTLLibrary>)lib;
  NSString* nm = [NSString stringWithUTF8String:entry_name];
  id<MTLFunction> fn = [mlib newFunctionWithName:nm];
  if (!fn) {
    if (err_out && err_cap > 0) {
      snprintf(err_out, err_cap, "function '%s' not found in MSL library", entry_name);
    }
    return NULL;
  }
  NSError* err = nil;
  id<MTLComputePipelineState> p =
      [dev newComputePipelineStateWithFunction:fn error:&err];
  if (!p) {
    copy_err(err, err_out, err_cap);
    return NULL;
  }
  return (__bridge_retained void*)p;
}

void ts_metal_pipeline_release(TsMetalPipeline p) {
  if (!p) return;
  CFRelease(p);
}

size_t ts_metal_pipeline_threadgroup_size(TsMetalPipeline p) {
  if (!p) return 0;
  id<MTLComputePipelineState> ps = (__bridge id<MTLComputePipelineState>)p;
  return [ps maxTotalThreadsPerThreadgroup];
}

// ── buffers ──
TsMetalBuffer ts_metal_buffer_create(
    TsMetalDevice d, size_t length, const void* init_data) {
  if (!d || length == 0) return NULL;
  id<MTLDevice> dev = (__bridge id<MTLDevice>)d;
  id<MTLBuffer> b;
  if (init_data) {
    b = [dev newBufferWithBytes:init_data length:length options:MTLResourceStorageModeShared];
  } else {
    b = [dev newBufferWithLength:length options:MTLResourceStorageModeShared];
  }
  if (!b) return NULL;
  return (__bridge_retained void*)b;
}

void ts_metal_buffer_release(TsMetalBuffer b) {
  if (!b) return;
  CFRelease(b);
}

void ts_metal_buffer_write(TsMetalBuffer b, size_t offset, const void* data, size_t len) {
  if (!b || !data || len == 0) return;
  id<MTLBuffer> mb = (__bridge id<MTLBuffer>)b;
  void* dst = (char*)[mb contents] + offset;
  memcpy(dst, data, len);
}

void ts_metal_buffer_read(TsMetalBuffer b, size_t offset, void* dst, size_t len) {
  if (!b || !dst || len == 0) return;
  id<MTLBuffer> mb = (__bridge id<MTLBuffer>)b;
  const void* src = (const char*)[mb contents] + offset;
  memcpy(dst, src, len);
}

size_t ts_metal_buffer_length(TsMetalBuffer b) {
  if (!b) return 0;
  id<MTLBuffer> mb = (__bridge id<MTLBuffer>)b;
  return [mb length];
}

// ── command encoding ──
TsMetalCommandBuffer ts_metal_queue_command_buffer(TsMetalQueue q) {
  if (!q) return NULL;
  id<MTLCommandQueue> mq = (__bridge id<MTLCommandQueue>)q;
  id<MTLCommandBuffer> cb = [mq commandBuffer];
  if (!cb) return NULL;
  return (__bridge_retained void*)cb;
}

void ts_metal_command_buffer_release(TsMetalCommandBuffer cb) {
  if (!cb) return;
  CFRelease(cb);
}

TsMetalEncoder ts_metal_command_buffer_compute_encoder(TsMetalCommandBuffer cb) {
  if (!cb) return NULL;
  id<MTLCommandBuffer> mcb = (__bridge id<MTLCommandBuffer>)cb;
  id<MTLComputeCommandEncoder> e = [mcb computeCommandEncoder];
  if (!e) return NULL;
  return (__bridge_retained void*)e;
}

void ts_metal_encoder_set_pipeline(TsMetalEncoder enc, TsMetalPipeline p) {
  if (!enc || !p) return;
  id<MTLComputeCommandEncoder> e = (__bridge id<MTLComputeCommandEncoder>)enc;
  id<MTLComputePipelineState> ps = (__bridge id<MTLComputePipelineState>)p;
  [e setComputePipelineState:ps];
}

void ts_metal_encoder_set_buffer(TsMetalEncoder enc, TsMetalBuffer b, size_t offset, uint32_t index) {
  if (!enc || !b) return;
  id<MTLComputeCommandEncoder> e = (__bridge id<MTLComputeCommandEncoder>)enc;
  id<MTLBuffer> mb = (__bridge id<MTLBuffer>)b;
  [e setBuffer:mb offset:offset atIndex:index];
}

void ts_metal_encoder_set_bytes(TsMetalEncoder enc, const void* data, size_t len, uint32_t index) {
  if (!enc || !data) return;
  id<MTLComputeCommandEncoder> e = (__bridge id<MTLComputeCommandEncoder>)enc;
  [e setBytes:data length:len atIndex:index];
}

void ts_metal_encoder_dispatch(
    TsMetalEncoder enc,
    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    uint32_t tg_x, uint32_t tg_y, uint32_t tg_z) {
  if (!enc) return;
  id<MTLComputeCommandEncoder> e = (__bridge id<MTLComputeCommandEncoder>)enc;
  // Metal uses threadgroups-per-grid (not threads-per-grid like WGSL),
  // matching the WGSL @workgroup_size convention. The dispatch params
  // here are workgroup counts in each dimension.
  MTLSize tg = MTLSizeMake(tg_x, tg_y, tg_z);
  MTLSize tpg = MTLSizeMake(grid_x, grid_y, grid_z);
  [e dispatchThreadgroups:tpg threadsPerThreadgroup:tg];
}

void ts_metal_encoder_end(TsMetalEncoder enc) {
  if (!enc) return;
  id<MTLComputeCommandEncoder> e = (__bridge id<MTLComputeCommandEncoder>)enc;
  [e endEncoding];
  CFRelease(enc);
}

void ts_metal_command_buffer_commit(TsMetalCommandBuffer cb) {
  if (!cb) return;
  id<MTLCommandBuffer> mcb = (__bridge id<MTLCommandBuffer>)cb;
  [mcb commit];
}

void ts_metal_command_buffer_wait(TsMetalCommandBuffer cb) {
  if (!cb) return;
  id<MTLCommandBuffer> mcb = (__bridge id<MTLCommandBuffer>)cb;
  [mcb waitUntilCompleted];
}

void ts_metal_encoder_blit_copy(
    TsMetalCommandBuffer cb,
    TsMetalBuffer src, size_t src_off,
    TsMetalBuffer dst, size_t dst_off,
    size_t length) {
  if (!cb || !src || !dst) return;
  id<MTLCommandBuffer> mcb = (__bridge id<MTLCommandBuffer>)cb;
  id<MTLBuffer> ms = (__bridge id<MTLBuffer>)src;
  id<MTLBuffer> md = (__bridge id<MTLBuffer>)dst;
  id<MTLBlitCommandEncoder> blit = [mcb blitCommandEncoder];
  [blit copyFromBuffer:ms sourceOffset:src_off
              toBuffer:md destinationOffset:dst_off
                  size:length];
  [blit endEncoding];
}
