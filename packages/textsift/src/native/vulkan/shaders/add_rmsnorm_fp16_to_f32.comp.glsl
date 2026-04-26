// add_rmsnorm_fp16_to_f32 — fused residual-add + RMS norm.
//   sum_out = a + b      (fp16 storage; written so pass 2 can re-read)
//   norm_out = sum_out * rsqrt(mean(sum^2) + eps) * gamma  (f32 storage)
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(push_constant) uniform Dims { uint T; uint D; float eps; uint _pad; } dims;

layout(set = 0, binding = 0, std430) readonly buffer A     { float16_t a[]; };
layout(set = 0, binding = 1, std430) readonly buffer B     { float16_t b[]; };
layout(set = 0, binding = 2, std430) readonly buffer Gamma { float16_t gamma[]; };
layout(set = 0, binding = 3, std430) buffer SumOut         { float16_t sum_out[]; };
layout(set = 0, binding = 4, std430) buffer NormOut        { float norm_out[]; };

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
        float av = float(a[row + d]);
        float bv = float(b[row + d]);
        float sv = av + bv;
        sum_out[row + d] = float16_t(sv);
        ssq += sv * sv;
    }
    wg_sum[tid] = ssq;
    barrier();

    for (uint stride = 32u; stride >= 1u; stride /= 2u) {
        if (tid < stride) {
            wg_sum[tid] += wg_sum[tid + stride];
        }
        barrier();
    }

    float inv_rms = inversesqrt(wg_sum[0] / float(D) + dims.eps);

    // Re-read fp16 sum_out we just wrote: each thread reads only its
    // own writes (strided by 64) so write→read of the same fp16 is
    // round-trip identical with no cross-thread dependency.
    for (uint d = tid; d < D; d += 64u) {
        float g  = float(gamma[d]);
        float sv = float(sum_out[row + d]);
        norm_out[row + d] = sv * inv_rms * g;
    }
}
