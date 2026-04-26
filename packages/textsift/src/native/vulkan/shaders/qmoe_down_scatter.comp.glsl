// qmoe_down_scatter — MoE down projection with weighted atomic scatter.
//
// Each thread computes one (n) column for a (token, k_pick) pair via
// int4-quantized matmul, then scatters w * acc into the output buffer
// using a CAS-loop f32 atomic add (Vulkan core lacks native f32 atomics
// without VK_EXT_shader_atomic_float; the CAS pattern is portable).
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(push_constant) uniform Dims {
    uint T; uint K; uint N; uint D;
    uint _pad0; uint _pad1; uint _pad2; uint _pad3;
} dims;

layout(set = 0, binding = 0, std430) readonly buffer Glu       { float glu[]; };
layout(set = 0, binding = 1, std430) readonly buffer RIdx      { int routing_idx[]; };
layout(set = 0, binding = 2, std430) readonly buffer RScores   { float routing_scores[]; };
layout(set = 0, binding = 3, std430) readonly buffer Wint4     { uint w_int4[]; };
layout(set = 0, binding = 4, std430) readonly buffer Wscales   { float16_t w_scales[]; };
layout(set = 0, binding = 5, std430) readonly buffer Wzp       { uint w_zp[]; };
layout(set = 0, binding = 6, std430) readonly buffer Bias      { float16_t bias[]; };
layout(set = 0, binding = 7, std430) coherent buffer Acc       { uint acc[]; };

const uint BLOCK = 32u;
const uint D_MAX = 640u;
const uint WG = 64u;

shared float glu_tile[D_MAX];

layout(local_size_x = 64) in;

void atomic_add_f32(uint slot, float val) {
    // Non-atomic initial read is safe: u32 reads on aligned memory are
    // tear-free on every Vulkan-supported arch, and the CAS retry
    // verifies the prior value anyway.
    uint old_u = acc[slot];
    while (true) {
        float new_f = uintBitsToFloat(old_u) + val;
        uint  new_u = floatBitsToUint(new_f);
        uint  prev  = atomicCompSwap(acc[slot], old_u, new_u);
        if (prev == old_u) break;
        old_u = prev;
    }
}

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint tk = gl_WorkGroupID.x;
    if (tk >= dims.T * dims.K) return;
    uint n_base = gl_WorkGroupID.y * WG;
    uint n = n_base + tid;

    uint t = tk / dims.K;
    uint e = uint(routing_idx[tk]);
    float w = routing_scores[tk];

    uint D = dims.D;
    uint N = dims.N;
    uint n_blocks = D / BLOCK;
    uint zp_per_row = (n_blocks + 1u) >> 1u;

    // Cooperative GLU load: D=640, 64 threads, 10 elements each.
    for (uint i = 0u; i < 10u; i = i + 1u) {
        uint k_idx = i * WG + tid;
        if (k_idx < D) {
            glu_tile[k_idx] = glu[tk * D + k_idx];
        }
    }
    barrier();

    if (n >= N) return;

    uint expert_nibble_base = e * N * D + n * D;
    uint expert_scale_base  = e * N * n_blocks + n * n_blocks;
    uint expert_zp_base     = e * N * zp_per_row + n * zp_per_row;
    uint expert_bias_base   = e * N + n;

    float acc_local = 0.0;
    for (uint b = 0u; b < n_blocks; b = b + 1u) {
        float scale  = float(w_scales[expert_scale_base + b]);
        uint zp_byte_idx = expert_zp_base + (b >> 1u);
        uint zp_byte = (w_zp[zp_byte_idx >> 2u] >> ((zp_byte_idx & 3u) * 8u)) & 0xFFu;
        uint zp_nib  = ((b & 1u) == 1u) ? ((zp_byte >> 4u) & 0xFu) : (zp_byte & 0xFu);
        float zp_f   = float(zp_nib);

        uint word_base = (expert_nibble_base + b * BLOCK) >> 3u;
        uint xb = b * BLOCK;

        float block_sum = 0.0;
        for (uint wi = 0u; wi < 4u; wi = wi + 1u) {
            uint word = w_int4[word_base + wi];
            uint kb = wi * 8u;
            float q0 = float( word        & 0xFu) - zp_f;
            float q1 = float((word >>  4u) & 0xFu) - zp_f;
            float q2 = float((word >>  8u) & 0xFu) - zp_f;
            float q3 = float((word >> 12u) & 0xFu) - zp_f;
            float q4 = float((word >> 16u) & 0xFu) - zp_f;
            float q5 = float((word >> 20u) & 0xFu) - zp_f;
            float q6 = float((word >> 24u) & 0xFu) - zp_f;
            float q7 = float((word >> 28u) & 0xFu) - zp_f;
            block_sum = fma(q0, glu_tile[xb + kb + 0u], block_sum);
            block_sum = fma(q1, glu_tile[xb + kb + 1u], block_sum);
            block_sum = fma(q2, glu_tile[xb + kb + 2u], block_sum);
            block_sum = fma(q3, glu_tile[xb + kb + 3u], block_sum);
            block_sum = fma(q4, glu_tile[xb + kb + 4u], block_sum);
            block_sum = fma(q5, glu_tile[xb + kb + 5u], block_sum);
            block_sum = fma(q6, glu_tile[xb + kb + 6u], block_sum);
            block_sum = fma(q7, glu_tile[xb + kb + 7u], block_sum);
        }
        acc_local = fma(block_sum, scale, acc_local);
    }
    acc_local = acc_local + float(bias[expert_bias_base]);

    atomic_add_f32(t * N + n, w * acc_local);
}
