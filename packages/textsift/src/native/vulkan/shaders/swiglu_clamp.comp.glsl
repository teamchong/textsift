// swiglu_clamp — SwiGLU activation with gate clamp at LIMIT.
//   gate = min(gate_raw, LIMIT)            // one-sided clamp
//   up   = clamp(up_raw, -LIMIT, LIMIT)    // two-sided
//   glu  = gate * sigmoid(gate * ALPHA)
//   out  = (up + 1) * glu
//
// Numerically-stable sigmoid: keeps the exp argument ≤ 0 to avoid
// intermediate overflow on large positive inputs.
#version 450

layout(push_constant) uniform Dims { uint rows; uint dff; uint _pad0; uint _pad1; } dims;

layout(set = 0, binding = 0, std430) readonly buffer GateUp { float gate_up[]; };
layout(set = 0, binding = 1, std430) buffer Out { float outBuf[]; };

const float LIMIT = 7.0;
const float ALPHA = 1.702;

float sigmoid_f32(float x) {
    if (x >= 0.0) {
        return 1.0 / (1.0 + exp(-x));
    }
    float e = exp(x);
    return e / (1.0 + e);
}

layout(local_size_x = 64) in;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = dims.rows * dims.dff;
    if (idx >= total) return;
    uint row = idx / dims.dff;
    uint col = idx % dims.dff;

    uint stride = 2u * dims.dff;
    float gate_raw = gate_up[row * stride + 2u * col];
    float up_raw   = gate_up[row * stride + 2u * col + 1u];
    float gate = min(gate_raw, LIMIT);
    float upv  = clamp(up_raw, -LIMIT, LIMIT);
    float glu  = gate * sigmoid_f32(gate * ALPHA);
    outBuf[idx] = (upv + 1.0) * glu;
}
