// cast_fp16_to_f32 — widen f16 buffer to f32 buffer.
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(push_constant) uniform Dims { uint n; uint _pad0; uint _pad1; uint _pad2; } dims;

layout(set = 0, binding = 0, std430) readonly buffer Src { float16_t src[]; };
layout(set = 0, binding = 1, std430) buffer Dst { float dst[]; };

layout(local_size_x = 64) in;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= dims.n) return;
    dst[i] = float(src[i]);
}
