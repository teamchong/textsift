// add_fp16 — element-wise add of two fp16 buffers, widen-to-f32 and
// round once on output (matches the WASM reference semantics).
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(push_constant) uniform Dims { uint n; uint _pad0; uint _pad1; uint _pad2; } dims;

layout(set = 0, binding = 0, std430) readonly buffer A { float16_t a[]; };
layout(set = 0, binding = 1, std430) readonly buffer B { float16_t b[]; };
layout(set = 0, binding = 2, std430) buffer Out { float16_t outBuf[]; };

layout(local_size_x = 64) in;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= dims.n) return;
    outBuf[i] = float16_t(float(a[i]) + float(b[i]));
}
