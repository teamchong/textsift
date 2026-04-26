// zero_f32 — fill a u32 buffer with zeros (used to clear scratch
// activations before kernel ops that expect zero-initialized output).
#version 450

layout(push_constant) uniform Dims { uint n; uint _pad0; uint _pad1; uint _pad2; } dims;

layout(set = 0, binding = 0, std430) buffer Buf { uint buf[]; };

layout(local_size_x = 64) in;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= dims.n) return;
    buf[i] = 0u;
}
