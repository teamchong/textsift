// rms_norm — per-token RMS normalization with f16 storage and f32 reduction.
//
// Mirrors src/native/shaders/rms_norm.wgsl byte-for-byte semantically:
//   - workgroup size 64
//   - parallel reduction strides 32→16→8→4→2→1
//   - f16 storage / f32 reduction (no precision loss in mean-square)
//   - inverseSqrt of (mean(x^2) + eps), then multiply by gamma
//
// Differences from the WGSL source:
//   - `var<uniform> dims` → push_constant block (≤128 B, faster than UBO).
//   - Storage buffer bindings start at 0 (since uniform moved to push constants),
//     vs binding(1..3) in WGSL.

#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(push_constant) uniform Dims {
    uint T;
    uint D;
    float eps;
    uint _pad;
} dims;

layout(set = 0, binding = 0, std430) readonly buffer X { float16_t x[]; };
layout(set = 0, binding = 1, std430) readonly buffer Gamma { float16_t gamma[]; };
layout(set = 0, binding = 2, std430) buffer Y { float16_t y[]; };

shared float wg_sum[64];

layout(local_size_x = 64) in;

void main() {
    uint t = gl_WorkGroupID.x;
    if (t >= dims.T) return;
    uint D = dims.D;
    uint row = t * D;
    uint tid = gl_LocalInvocationID.x;

    float ssq = 0.0;
    for (uint d = tid; d < D; d += 64u) {
        float v = float(x[row + d]);
        ssq += v * v;
    }
    wg_sum[tid] = ssq;
    barrier();

    // Strides: 32, 16, 8, 4, 2, 1. Matches the WGSL loop exactly.
    for (uint stride = 32u; stride >= 1u; stride /= 2u) {
        if (tid < stride) {
            wg_sum[tid] += wg_sum[tid + stride];
        }
        barrier();
    }

    float inv_rms = inversesqrt(wg_sum[0] / float(D) + dims.eps);

    for (uint d = tid; d < D; d += 64u) {
        float g = float(gamma[d]);
        float xv = float(x[row + d]);
        y[row + d] = float16_t(xv * inv_rms * g);
    }
}
