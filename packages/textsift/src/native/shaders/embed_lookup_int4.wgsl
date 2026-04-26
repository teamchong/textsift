enable f16;

struct Dims { T: u32, V: u32, D: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> embed_int4: array<u32>;
@group(0) @binding(2) var<storage, read> embed_scales: array<f16>;
@group(0) @binding(3) var<storage, read> embed_zp: array<u32>;
@group(0) @binding(4) var<storage, read> ids: array<i32>;
@group(0) @binding(5) var<storage, read_write> out: array<f16>;

const EMBED_BLOCK: u32 = 32u;


// Inlined int4-access primitives: Naga (wgpu-native) rejects
// ptr<storage, ...> as a function argument, so the helpers from
// int4_access.wgsl are inlined here as direct array reads.
//   nibble at index n: (arr[n >> 3] >> ((n & 7) * 4)) & 0xF
//   byte at index   b: (arr[b >> 2] >> ((b & 3) * 8)) & 0xFF


@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let td = gid.x;
    let D = dims.D;
    let total = dims.T * D;
    if (td >= total) { return; }
    let t = td / D;
    let d = td % D;

    let id = ids[t];
    if (id < 0 || u32(id) >= dims.V) {
        out[td] = f16(0.0);
        return;
    }
    let row = u32(id);
    let n_blocks = D / EMBED_BLOCK;
    let zp_per_row = (n_blocks + 1u) >> 1u;

    let b = d / EMBED_BLOCK;
    let scale = f32(embed_scales[row * n_blocks + b]);
    let zp_byte_idx = row * zp_per_row + (b >> 1u);
    let zp_byte = (embed_zp[zp_byte_idx >> 2u] >> ((zp_byte_idx & 3u) * 8u)) & 0xFFu;
    let zp_nib: u32 = select(zp_byte & 0xFu, (zp_byte >> 4u) & 0xFu, (b & 1u) == 1u);
    let nib_idx = row * D + d;
    let nib = (embed_int4[nib_idx >> 3u] >> ((nib_idx & 7u) * 4u)) & 0xFu;
    let q = f32(nib) - f32(zp_nib);
    out[td] = f16(q * scale);
}
