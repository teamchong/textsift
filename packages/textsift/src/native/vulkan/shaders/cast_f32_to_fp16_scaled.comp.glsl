// cast_f32_to_fp16_scaled — multiply f32 by scale, narrow to f16.
//
// WGSL uses two separate uniform buffers (Dims at binding 0, Scale at
// binding 1), each padded to 16 bytes. We collapse them into one 32-byte
// push-constant block. Layout matches the concatenation
// `uniform.bin || scale.bin` so the existing fixture data drops in
// without re-encoding.
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(push_constant) uniform PC {
    uint  n;        uint _pad0; uint _pad1; uint _pad2;     // offset 0..15  (Dims)
    float scale_v;  uint _pad3; uint _pad4; uint _pad5;     // offset 16..31 (Scale)
} pc;

layout(set = 0, binding = 0, std430) readonly buffer Src { float src[]; };
layout(set = 0, binding = 1, std430) buffer Dst { float16_t dst[]; };

layout(local_size_x = 64) in;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;
    dst[i] = float16_t(src[i] * pc.scale_v);
}
