// embed_lookup_int4 — int4-quantized embedding lookup.
// Each thread emits one (token, dim) element of the embedding output.
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require

layout(push_constant) uniform Dims { uint T; uint V; uint D; uint _pad; } dims;

layout(set = 0, binding = 0, std430) readonly buffer Eint4   { uint embed_int4[]; };
layout(set = 0, binding = 1, std430) readonly buffer Escales { float16_t embed_scales[]; };
layout(set = 0, binding = 2, std430) readonly buffer Ezp     { uint embed_zp[]; };
layout(set = 0, binding = 3, std430) readonly buffer Ids     { int ids[]; };
layout(set = 0, binding = 4, std430) buffer OutBuf          { float16_t outBuf[]; };

const uint EMBED_BLOCK = 32u;

layout(local_size_x = 64) in;

void main() {
    uint td = gl_GlobalInvocationID.x;
    uint D = dims.D;
    uint total = dims.T * D;
    if (td >= total) return;
    uint t = td / D;
    uint d = td % D;

    int id = ids[t];
    if (id < 0 || uint(id) >= dims.V) {
        outBuf[td] = float16_t(0.0);
        return;
    }
    uint row = uint(id);
    uint n_blocks = D / EMBED_BLOCK;
    uint zp_per_row = (n_blocks + 1u) >> 1u;

    uint b = d / EMBED_BLOCK;
    float scale = float(embed_scales[row * n_blocks + b]);
    uint zp_byte_idx = row * zp_per_row + (b >> 1u);
    uint zp_byte = (embed_zp[zp_byte_idx >> 2u] >> ((zp_byte_idx & 3u) * 8u)) & 0xFFu;
    uint zp_nib  = ((b & 1u) == 1u) ? ((zp_byte >> 4u) & 0xFu) : (zp_byte & 0xFu);
    uint nib_idx = row * D + d;
    uint nib = (embed_int4[nib_idx >> 3u] >> ((nib_idx & 7u) * 4u)) & 0xFu;
    float q = float(nib) - float(zp_nib);
    outBuf[td] = float16_t(q * scale);
}
