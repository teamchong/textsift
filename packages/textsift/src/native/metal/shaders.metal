// Hand-tuned Metal Shading Language kernels. One file, one library —
// all entry points compile together so we share a single MTLLibrary
// per backend instance.
//
// Conformance: each kernel must produce byte-equal output (within
// 1 fp16 ULP) to the reference WGSL kernel in
// packages/textsift/src/native/shaders/<name>.wgsl when given the
// same input. Tested by tests/native/metal/conformance.test.js.

#include <metal_stdlib>
using namespace metal;

// ────────────────────────────────────────────────────────────────
// rms_norm  ←  reference: src/native/shaders/rms_norm.wgsl
//
// y[t, d] = x[t, d] * gamma[d] / sqrt(mean_d(x[t, :]^2) + eps)
//
// One threadgroup per row of T. 64 threads per workgroup, each walks
// a strided slice of D for the sumsq reduction, then again for the
// per-element write. f32 accumulator (640-wide fp16 sum loses bits).
// ────────────────────────────────────────────────────────────────

struct RmsNormDims { uint T; uint D; float eps; uint _pad; };

kernel void rms_norm(
    device const RmsNormDims&  dims    [[buffer(0)]],
    device const half*         x       [[buffer(1)]],
    device const half*         gamma   [[buffer(2)]],
    device       half*         y       [[buffer(3)]],
    uint3                      lid_v   [[thread_position_in_threadgroup]],
    uint3                      gid     [[threadgroup_position_in_grid]])
{
    threadgroup float wg_sum[64];
    const uint tid = lid_v.x;
    const uint t = gid.x;
    if (t >= dims.T) return;
    const uint D = dims.D;
    const uint row = t * D;

    // Pass 1: per-thread strided sum-of-squares.
    float ssq = 0.0f;
    for (uint d = tid; d < D; d += 64u) {
        const float v = float(x[row + d]);
        ssq = fma(v, v, ssq);
    }
    wg_sum[tid] = ssq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction: 32 → 16 → 8 → 4 → 2 → 1.
    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (tid < stride) wg_sum[tid] += wg_sum[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_rms = rsqrt(wg_sum[0] / float(D) + dims.eps);

    // Pass 2: per-element scale.
    for (uint d = tid; d < D; d += 64u) {
        const float g = float(gamma[d]);
        const float xv = float(x[row + d]);
        y[row + d] = half(xv * inv_rms * g);
    }
}
