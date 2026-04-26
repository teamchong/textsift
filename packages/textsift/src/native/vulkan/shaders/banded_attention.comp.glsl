// banded_attention — masked sliding-window attention with sink term.
// 4-pass design within a single workgroup:
//   1. score & per-thread max
//   2. exp(score - max) & per-thread sum
//   3. normalize (in-place)
//   4. value combine, one thread per head_dim lane
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(push_constant) uniform Dims {
    uint T; uint H_q; uint H_kv; uint head_dim;
    uint window; uint use_mask; uint _pad0; uint _pad1;
} dims;

layout(set = 0, binding = 0, std430) readonly buffer Q     { float16_t q[]; };
layout(set = 0, binding = 1, std430) readonly buffer K     { float16_t k[]; };
layout(set = 0, binding = 2, std430) readonly buffer V     { float16_t v[]; };
layout(set = 0, binding = 3, std430) readonly buffer Sinks { float sinks[]; };
layout(set = 0, binding = 4, std430) readonly buffer Mask  { uint mask[]; };
layout(set = 0, binding = 5, std430) buffer Out            { float16_t outBuf[]; };

const uint MAX_WINDOW_TOTAL = 257u;
const float NEG_INF = -1e30;

shared float wg_scores[MAX_WINDOW_TOTAL];
shared float wg_tmp[64];
shared float wg_broadcast[2];  // [0]=row_max, [1]=inv_sum

uint read_mask_byte(uint i) {
    uint word = mask[i >> 2u];
    return (word >> ((i & 3u) * 8u)) & 0xFFu;
}

layout(local_size_x = 64) in;

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint t_q_h_q = gl_WorkGroupID.x;
    uint H_q = dims.H_q;
    if (t_q_h_q >= dims.T * H_q) return;
    uint t_q = t_q_h_q / H_q;
    uint h_q = t_q_h_q % H_q;
    uint head_dim = dims.head_dim;
    uint kv_group = H_q / dims.H_kv;
    uint h_kv = h_q / kv_group;

    uint q_stride_t = H_q * head_dim;
    uint k_stride_t = dims.H_kv * head_dim;
    uint q_base = t_q * q_stride_t + h_q * head_dim;

    uint window = dims.window;
    uint ws = (t_q > window) ? (t_q - window) : 0u;
    uint we = (t_q + window + 1u < dims.T) ? (t_q + window + 1u) : dims.T;
    uint n_keys = we - ws;

    // Pass 1: scores + per-thread max.
    float thread_max = NEG_INF;
    for (uint idx = tid; idx < n_keys; idx += 64u) {
        uint abs_k = ws + idx;
        bool is_valid = true;
        if (dims.use_mask == 1u) {
            is_valid = read_mask_byte(abs_k) != 0u;
        }
        float s = NEG_INF;
        if (is_valid) {
            uint k_base = abs_k * k_stride_t + h_kv * head_dim;
            float dotv = 0.0;
            for (uint d = 0u; d < head_dim; d = d + 1u) {
                dotv = fma(float(q[q_base + d]), float(k[k_base + d]), dotv);
            }
            s = dotv;
        }
        wg_scores[idx] = s;
        if (s > thread_max) thread_max = s;
    }
    if (tid == 0u) {
        float sink_s = sinks[h_q];
        if (sink_s > thread_max) thread_max = sink_s;
    }
    wg_tmp[tid] = thread_max;
    barrier();

    // Reduce max.
    for (uint stride = 32u; stride >= 1u; stride /= 2u) {
        if (tid < stride) {
            wg_tmp[tid] = max(wg_tmp[tid], wg_tmp[tid + stride]);
        }
        barrier();
    }
    if (tid == 0u) wg_broadcast[0] = wg_tmp[0];
    barrier();
    float row_max = wg_broadcast[0];

    // Pass 2: exp(s - max), per-thread sum.
    float thread_sum = 0.0;
    for (uint idx = tid; idx < n_keys; idx += 64u) {
        float e = exp(wg_scores[idx] - row_max);
        wg_scores[idx] = e;
        thread_sum += e;
    }
    if (tid == 0u) {
        thread_sum += exp(sinks[h_q] - row_max);
    }
    wg_tmp[tid] = thread_sum;
    barrier();

    for (uint stride = 32u; stride >= 1u; stride /= 2u) {
        if (tid < stride) {
            wg_tmp[tid] = wg_tmp[tid] + wg_tmp[tid + stride];
        }
        barrier();
    }
    if (tid == 0u) {
        float s = wg_tmp[0];
        wg_broadcast[1] = (s > 0.0) ? (1.0 / s) : 0.0;
    }
    barrier();
    float inv_sum = wg_broadcast[1];

    // Pass 3: normalize in-place.
    for (uint idx = tid; idx < n_keys; idx += 64u) {
        wg_scores[idx] *= inv_sum;
    }
    barrier();

    // Pass 4: value combine, one thread per head_dim lane.
    if (tid < head_dim) {
        float acc = 0.0;
        for (uint i = 0u; i < n_keys; i = i + 1u) {
            uint abs_k = ws + i;
            uint v_base = abs_k * k_stride_t + h_kv * head_dim;
            acc = fma(wg_scores[i], float(v[v_base + tid]), acc);
        }
        uint out_base = t_q * q_stride_t + h_q * head_dim;
        outBuf[out_base + tid] = float16_t(acc);
    }
}
