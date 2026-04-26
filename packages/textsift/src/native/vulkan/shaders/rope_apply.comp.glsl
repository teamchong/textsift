// rope_apply — apply rotary position embedding to qk in-place.
// qk is read+written; cos_tab and sin_tab are read-only lookup tables
// indexed by (token, half_dim).
//
// Binding order in this Vulkan port:
//   0: qk      (read+write SSBO)
//   1: cos_tab (read-only)
//   2: sin_tab (read-only)
// (WGSL has qk at binding 1 and cos/sin at 2/3 — order is just a label,
// what matters is the slot order matching how the JS test passes them.)
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(push_constant) uniform Dims {
    uint T; uint H; uint head_dim; uint _pad;
} dims;

layout(set = 0, binding = 0, std430) buffer QK { float16_t qk[]; };
layout(set = 0, binding = 1, std430) readonly buffer Cos { float16_t cos_tab[]; };
layout(set = 0, binding = 2, std430) readonly buffer Sin { float16_t sin_tab[]; };

layout(local_size_x = 64) in;

void main() {
    uint half_dim = dims.head_dim / 2u;
    uint idx = gl_GlobalInvocationID.x;
    uint tp_total = dims.T * dims.H * half_dim;
    if (idx >= tp_total) return;

    uint p  = idx % half_dim;
    uint th = idx / half_dim;
    uint h  = th % dims.H;
    uint t  = th / dims.H;

    float16_t c = cos_tab[t * half_dim + p];
    float16_t s = sin_tab[t * half_dim + p];

    uint head_base = (t * dims.H + h) * dims.head_dim;
    float16_t a = qk[head_base + 2u * p];
    float16_t b = qk[head_base + 2u * p + 1u];

    float16_t ac = a * c;
    float16_t bs = b * s;
    float16_t bc = b * c;
    float16_t as_ = a * s;

    qk[head_base + 2u * p]      = ac - bs;
    qk[head_base + 2u * p + 1u] = bc + as_;
}
