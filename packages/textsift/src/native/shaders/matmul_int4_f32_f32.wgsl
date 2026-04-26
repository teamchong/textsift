enable f16;

struct Dims { T: u32, N: u32, K: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w_int4: array<u32>;
@group(0) @binding(3) var<storage, read> w_scales: array<f16>;
@group(0) @binding(4) var<storage, read> w_zp: array<u32>;
@group(0) @binding(5) var<storage, read> bias: array<f16>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;

const BLOCK: u32 = 32u;


// Inlined int4-access (Naga rejects ptr<storage, ...> as fn args).


@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tn = gid.x;
    let total = dims.T * dims.N;
    if (tn >= total) { return; }
    let t = tn / dims.N;
    let n = tn % dims.N;
    let K = dims.K;
    let n_blocks = K / BLOCK;
    let zp_per_row = (n_blocks + 1u) >> 1u;

    let nibble_row_base = n * K;
    let scale_row = n * n_blocks;
    let x_row = t * K;

    var acc: f32 = 0.0;
    for (var b: u32 = 0u; b < n_blocks; b = b + 1u) {
        let scale: f32 = f32(w_scales[scale_row + b]);
        let zp_byte_idx = n * zp_per_row + (b >> 1u);
        let zp_byte = (w_zp[zp_byte_idx >> 2u] >> ((zp_byte_idx & 3u) * 8u)) & 0xFFu;
        let zp_nib = select(zp_byte & 0xFu, (zp_byte >> 4u) & 0xFu, (b & 1u) == 1u);
        let zp_f: f32 = f32(zp_nib);

        let word_base = (nibble_row_base + b * BLOCK) >> 3u;
        let xb = x_row + b * BLOCK;

        var block_sum: f32 = 0.0;
        for (var w: u32 = 0u; w < 4u; w = w + 1u) {
            let word = w_int4[word_base + w];
            let kb = w * 8u;
            let q0 = f32( word        & 0xFu) - zp_f;
            let q1 = f32((word >>  4u) & 0xFu) - zp_f;
            let q2 = f32((word >>  8u) & 0xFu) - zp_f;
            let q3 = f32((word >> 12u) & 0xFu) - zp_f;
            let q4 = f32((word >> 16u) & 0xFu) - zp_f;
            let q5 = f32((word >> 20u) & 0xFu) - zp_f;
            let q6 = f32((word >> 24u) & 0xFu) - zp_f;
            let q7 = f32((word >> 28u) & 0xFu) - zp_f;
            block_sum = fma(q0, x[xb + kb + 0u], block_sum);
            block_sum = fma(q1, x[xb + kb + 1u], block_sum);
            block_sum = fma(q2, x[xb + kb + 2u], block_sum);
            block_sum = fma(q3, x[xb + kb + 3u], block_sum);
            block_sum = fma(q4, x[xb + kb + 4u], block_sum);
            block_sum = fma(q5, x[xb + kb + 5u], block_sum);
            block_sum = fma(q6, x[xb + kb + 6u], block_sum);
            block_sum = fma(q7, x[xb + kb + 7u], block_sum);
        }
        acc = fma(block_sum, scale, acc);
    }
    acc = acc + f32(bias[n]);
    y[t * dims.N + n] = acc;
}
