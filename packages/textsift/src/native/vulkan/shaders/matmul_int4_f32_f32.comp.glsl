// matmul_int4_f32_f32 — int4-quantized matmul, f32 input/output.
// Used for the classifier head (small enough that the workgroup-tile
// optimization isn't worth the complexity).
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(push_constant) uniform Dims { uint T; uint N; uint K; uint _pad; } dims;

layout(set = 0, binding = 0, std430) readonly buffer X       { float x[]; };
layout(set = 0, binding = 1, std430) readonly buffer Wint4   { uint w_int4[]; };
layout(set = 0, binding = 2, std430) readonly buffer Wscales { float16_t w_scales[]; };
layout(set = 0, binding = 3, std430) readonly buffer Wzp     { uint w_zp[]; };
layout(set = 0, binding = 4, std430) readonly buffer Bias    { float16_t bias[]; };
layout(set = 0, binding = 5, std430) buffer YBuf             { float y[]; };

const uint BLOCK = 32u;

layout(local_size_x = 64) in;

void main() {
    uint tn = gl_GlobalInvocationID.x;
    uint total = dims.T * dims.N;
    if (tn >= total) return;
    uint t = tn / dims.N;
    uint n = tn % dims.N;
    uint K = dims.K;
    uint n_blocks = K / BLOCK;
    uint zp_per_row = (n_blocks + 1u) >> 1u;

    uint nibble_row_base = n * K;
    uint scale_row = n * n_blocks;
    uint x_row = t * K;

    float acc = 0.0;
    for (uint b = 0u; b < n_blocks; b = b + 1u) {
        float scale  = float(w_scales[scale_row + b]);
        uint zp_byte_idx = n * zp_per_row + (b >> 1u);
        uint zp_byte = (w_zp[zp_byte_idx >> 2u] >> ((zp_byte_idx & 3u) * 8u)) & 0xFFu;
        uint zp_nib  = ((b & 1u) == 1u) ? ((zp_byte >> 4u) & 0xFu) : (zp_byte & 0xFu);
        float zp_f   = float(zp_nib);

        uint word_base = (nibble_row_base + b * BLOCK) >> 3u;
        uint xb = x_row + b * BLOCK;

        float block_sum = 0.0;
        for (uint w = 0u; w < 4u; w = w + 1u) {
            uint word = w_int4[word_base + w];
            uint kb = w * 8u;
            float q0 = float( word        & 0xFu) - zp_f;
            float q1 = float((word >>  4u) & 0xFu) - zp_f;
            float q2 = float((word >>  8u) & 0xFu) - zp_f;
            float q3 = float((word >> 12u) & 0xFu) - zp_f;
            float q4 = float((word >> 16u) & 0xFu) - zp_f;
            float q5 = float((word >> 20u) & 0xFu) - zp_f;
            float q6 = float((word >> 24u) & 0xFu) - zp_f;
            float q7 = float((word >> 28u) & 0xFu) - zp_f;
            block_sum = fma(q0, x[xb + kb + 0u], block_sum);
            block_sum = fma(q1, x[xb + kb + 1u], block_sum);
            block_sum = fma(q2, x[xb + kb + 2u], block_sum);
            block_sum = fma(q3, x[xb + kb + 3u], block_sum);
            block_sum = fma(q4, x[xb + kb + 4u], block_sum);
            block_sum = fma(q5, x[xb + kb + 5u], block_sum);
            block_sum = fma(q6, x[xb + kb + 6u], block_sum);
            block_sum = fma(q7, x[xb + kb + 7u], block_sum);
        }
        acc = fma(block_sum, scale, acc);
    }
    acc = acc + float(bias[n]);
    y[t * dims.N + n] = acc;
}
