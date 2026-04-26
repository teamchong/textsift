// router_topk — MoE routing: pick top-K experts per token, softmax their
// scores, scale by 1/K. Writes int indices and f32 scores.
#version 450

layout(push_constant) uniform Dims { uint T; uint E; uint K; uint _pad; } dims;

layout(set = 0, binding = 0, std430) readonly buffer Logits { float logits[]; };
layout(set = 0, binding = 1, std430) buffer OutIdx          { int out_idx[]; };
layout(set = 0, binding = 2, std430) buffer OutScores       { float out_scores[]; };

const uint MAX_K = 8u;
const float NEG_INF = -1e30;

layout(local_size_x = 64) in;

void main() {
    uint t = gl_GlobalInvocationID.x;
    if (t >= dims.T) return;
    uint K = dims.K;
    uint row = t * dims.E;

    float top_val[MAX_K];
    uint  top_idx[MAX_K];
    for (uint i = 0u; i < MAX_K; i = i + 1u) {
        top_val[i] = NEG_INF;
        top_idx[i] = 0u;
    }

    for (uint e = 0u; e < dims.E; e = e + 1u) {
        float v = logits[row + e];
        // Locate slot with the smallest current value.
        uint min_k = 0u;
        float min_v = top_val[0];
        for (uint k = 1u; k < K; k = k + 1u) {
            if (top_val[k] < min_v) {
                min_v = top_val[k];
                min_k = k;
            }
        }
        if (v > min_v) {
            top_val[min_k] = v;
            top_idx[min_k] = e;
        }
    }

    // Softmax over top K.
    float max_v = top_val[0];
    for (uint k = 1u; k < K; k = k + 1u) {
        max_v = max(max_v, top_val[k]);
    }
    float sum = 0.0;
    for (uint k = 0u; k < K; k = k + 1u) {
        float e = exp(top_val[k] - max_v);
        top_val[k] = e;
        sum = sum + e;
    }
    float inv_sum = (sum > 0.0) ? (1.0 / sum) : 0.0;
    float inv_K   = 1.0 / float(K);

    uint out_base = t * K;
    for (uint k = 0u; k < K; k = k + 1u) {
        out_idx[out_base + k] = int(top_idx[k]);
        out_scores[out_base + k] = top_val[k] * inv_sum * inv_K;
    }
}
