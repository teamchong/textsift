// qmoe_gate_up — MoE gate+up projection: int4 matmul, expert-routed.
// 2D dispatch: x is the (token, k_pick) pair, y is the N-tile index.
// All 64 threads in a workgroup share the same (t, k_pick) so they
// cooperatively load the X row into shared memory once per workgroup.
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(push_constant) uniform Dims {
    uint T; uint K; uint N; uint D;
    uint _pad0; uint _pad1; uint _pad2; uint _pad3;
} dims;

layout(set = 0, binding = 0, std430) readonly buffer X       { float x[]; };
layout(set = 0, binding = 1, std430) readonly buffer RIdx    { int routing_idx[]; };
layout(set = 0, binding = 2, std430) readonly buffer Wint4   { uint w_int4[]; };
layout(set = 0, binding = 3, std430) readonly buffer Wscales { float16_t w_scales[]; };
layout(set = 0, binding = 4, std430) readonly buffer Wzp     { uint w_zp[]; };
layout(set = 0, binding = 5, std430) readonly buffer Bias    { float16_t bias[]; };
layout(set = 0, binding = 6, std430) buffer OutBuf           { float outBuf[]; };

const uint BLOCK = 32u;
const uint D_MAX = 640u;
const uint WG = 64u;

shared float x_tile[D_MAX];

layout(local_size_x = 64) in;

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint tk = gl_WorkGroupID.x;
    if (tk >= dims.T * dims.K) return;
    uint n_base = gl_WorkGroupID.y * WG;
    uint n = n_base + tid;

    uint t = tk / dims.K;
    uint e = uint(routing_idx[tk]);

    uint D = dims.D;
    uint N = dims.N;
    uint n_blocks = D / BLOCK;
    uint zp_per_row = (n_blocks + 1u) >> 1u;

    // Cooperative X load: D=640, 64 threads, 10 elements each.
    for (uint i = 0u; i < 10u; i = i + 1u) {
        uint k_idx = i * WG + tid;
        if (k_idx < D) {
            x_tile[k_idx] = x[t * D + k_idx];
        }
    }
    barrier();

    if (n >= N) return;

    uint expert_nibble_base = e * N * D + n * D;
    uint expert_scale_base  = e * N * n_blocks + n * n_blocks;
    uint expert_zp_base     = e * N * zp_per_row + n * zp_per_row;
    uint expert_bias_base   = e * N + n;

    float acc = 0.0;
    for (uint b = 0u; b < n_blocks; b = b + 1u) {
        float scale  = float(w_scales[expert_scale_base + b]);
        uint zp_byte_idx = expert_zp_base + (b >> 1u);
        uint zp_byte = (w_zp[zp_byte_idx >> 2u] >> ((zp_byte_idx & 3u) * 8u)) & 0xFFu;
        uint zp_nib  = ((b & 1u) == 1u) ? ((zp_byte >> 4u) & 0xFu) : (zp_byte & 0xFu);
        float zp_f   = float(zp_nib);

        uint word_base = (expert_nibble_base + b * BLOCK) >> 3u;
        uint xb = b * BLOCK;

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
            block_sum = fma(q0, x_tile[xb + kb + 0u], block_sum);
            block_sum = fma(q1, x_tile[xb + kb + 1u], block_sum);
            block_sum = fma(q2, x_tile[xb + kb + 2u], block_sum);
            block_sum = fma(q3, x_tile[xb + kb + 3u], block_sum);
            block_sum = fma(q4, x_tile[xb + kb + 4u], block_sum);
            block_sum = fma(q5, x_tile[xb + kb + 5u], block_sum);
            block_sum = fma(q6, x_tile[xb + kb + 6u], block_sum);
            block_sum = fma(q7, x_tile[xb + kb + 7u], block_sum);
        }
        acc = fma(block_sum, scale, acc);
    }
    acc = acc + float(bias[expert_bias_base]);
    outBuf[tk * N + n] = acc;
}
