// matmul_int4_fp16_f16 — int4-quantized matmul, fp16 input/output.
// The workhorse kernel for Q/K/V/O projections in the attention block.
//
// 2D dispatch: workgroup_id.x indexes 64-wide N tiles, .y indexes
// 4-wide T tiles. Each thread owns one N column and four T rows; the
// int4 weight decode for that (n, k_block) is reused across the four
// T accumulators, and the X reads are amortized across the workgroup
// via on-chip x_tile.
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(push_constant) uniform Dims { uint T; uint N; uint K; uint _pad; } dims;

layout(set = 0, binding = 0, std430) readonly buffer X       { float16_t x[]; };
layout(set = 0, binding = 1, std430) readonly buffer Wint4   { uint w_int4[]; };
layout(set = 0, binding = 2, std430) readonly buffer Wscales { float16_t w_scales[]; };
layout(set = 0, binding = 3, std430) readonly buffer Wzp     { uint w_zp[]; };
layout(set = 0, binding = 4, std430) readonly buffer Bias    { float16_t bias[]; };
layout(set = 0, binding = 5, std430) buffer YBuf             { float16_t y[]; };

const uint BLOCK = 32u;
const uint TR = 4u;
const uint WG = 64u;
const uint X_TILE_SIZE = TR * BLOCK;  // 128

shared float x_tile[X_TILE_SIZE];

layout(local_size_x = 64) in;

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint n   = gl_WorkGroupID.x * WG + tid;
    uint t_base = gl_WorkGroupID.y * TR;

    uint K = dims.K;
    uint n_blocks = K / BLOCK;
    uint zp_per_row = (n_blocks + 1u) >> 1u;

    float acc0 = 0.0;
    float acc1 = 0.0;
    float acc2 = 0.0;
    float acc3 = 0.0;

    bool n_active = n < dims.N;
    uint nibble_row_base = n_active ? (n * K) : 0u;
    uint scale_row = n_active ? (n * n_blocks) : 0u;

    for (uint b = 0u; b < n_blocks; b = b + 1u) {
        // Cooperative X load: 128 elements / 64 threads = 2 per thread.
        uint i0 = tid * 2u;
        uint i1 = i0 + 1u;
        uint t0_row = i0 / BLOCK;
        uint k0     = i0 % BLOCK;
        uint t1_row = i1 / BLOCK;
        uint k1     = i1 % BLOCK;
        uint g0 = t_base + t0_row;
        uint g1 = t_base + t1_row;
        x_tile[i0] = (g0 < dims.T) ? float(x[g0 * K + b * BLOCK + k0]) : 0.0;
        x_tile[i1] = (g1 < dims.T) ? float(x[g1 * K + b * BLOCK + k1]) : 0.0;
        barrier();

        if (n_active) {
            float scale  = float(w_scales[scale_row + b]);
            uint zp_byte_idx = n * zp_per_row + (b >> 1u);
            uint zp_byte = (w_zp[zp_byte_idx >> 2u] >> ((zp_byte_idx & 3u) * 8u)) & 0xFFu;
            uint zp_nib  = ((b & 1u) == 1u) ? ((zp_byte >> 4u) & 0xFu) : (zp_byte & 0xFu);
            float zp_f   = float(zp_nib);
            uint word_base = (nibble_row_base + b * BLOCK) >> 3u;

            float blk0 = 0.0;
            float blk1 = 0.0;
            float blk2 = 0.0;
            float blk3 = 0.0;

            for (uint w = 0u; w < 4u; w = w + 1u) {
                uint word = w_int4[word_base + w];
                uint kb = w * 8u;
                vec4 q_lo = vec4(
                    float( word        & 0xFu) - zp_f,
                    float((word >>  4u) & 0xFu) - zp_f,
                    float((word >>  8u) & 0xFu) - zp_f,
                    float((word >> 12u) & 0xFu) - zp_f
                );
                vec4 q_hi = vec4(
                    float((word >> 16u) & 0xFu) - zp_f,
                    float((word >> 20u) & 0xFu) - zp_f,
                    float((word >> 24u) & 0xFu) - zp_f,
                    float((word >> 28u) & 0xFu) - zp_f
                );

                uint xb0 = 0u * BLOCK + kb;
                vec4 x0_lo = vec4(x_tile[xb0+0u], x_tile[xb0+1u], x_tile[xb0+2u], x_tile[xb0+3u]);
                vec4 x0_hi = vec4(x_tile[xb0+4u], x_tile[xb0+5u], x_tile[xb0+6u], x_tile[xb0+7u]);
                blk0 = blk0 + dot(q_lo, x0_lo) + dot(q_hi, x0_hi);

                uint xb1 = 1u * BLOCK + kb;
                vec4 x1_lo = vec4(x_tile[xb1+0u], x_tile[xb1+1u], x_tile[xb1+2u], x_tile[xb1+3u]);
                vec4 x1_hi = vec4(x_tile[xb1+4u], x_tile[xb1+5u], x_tile[xb1+6u], x_tile[xb1+7u]);
                blk1 = blk1 + dot(q_lo, x1_lo) + dot(q_hi, x1_hi);

                uint xb2 = 2u * BLOCK + kb;
                vec4 x2_lo = vec4(x_tile[xb2+0u], x_tile[xb2+1u], x_tile[xb2+2u], x_tile[xb2+3u]);
                vec4 x2_hi = vec4(x_tile[xb2+4u], x_tile[xb2+5u], x_tile[xb2+6u], x_tile[xb2+7u]);
                blk2 = blk2 + dot(q_lo, x2_lo) + dot(q_hi, x2_hi);

                uint xb3 = 3u * BLOCK + kb;
                vec4 x3_lo = vec4(x_tile[xb3+0u], x_tile[xb3+1u], x_tile[xb3+2u], x_tile[xb3+3u]);
                vec4 x3_hi = vec4(x_tile[xb3+4u], x_tile[xb3+5u], x_tile[xb3+6u], x_tile[xb3+7u]);
                blk3 = blk3 + dot(q_lo, x3_lo) + dot(q_hi, x3_hi);
            }
            acc0 = fma(blk0, scale, acc0);
            acc1 = fma(blk1, scale, acc1);
            acc2 = fma(blk2, scale, acc2);
            acc3 = fma(blk3, scale, acc3);
        }
        barrier();
    }

    if (n_active) {
        float bias_f = float(bias[n]);
        uint t0 = t_base + 0u;
        uint t1 = t_base + 1u;
        uint t2 = t_base + 2u;
        uint t3 = t_base + 3u;
        if (t0 < dims.T) y[t0 * dims.N + n] = float16_t(acc0 + bias_f);
        if (t1 < dims.T) y[t1 * dims.N + n] = float16_t(acc1 + bias_f);
        if (t2 < dims.T) y[t2 * dims.N + n] = float16_t(acc2 + bias_f);
        if (t3 < dims.T) y[t3 * dims.N + n] = float16_t(acc3 + bias_f);
    }
}
