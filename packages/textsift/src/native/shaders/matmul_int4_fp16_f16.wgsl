enable f16;

struct Dims { T: u32, N: u32, K: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f16>;
@group(0) @binding(2) var<storage, read> w_int4: array<u32>;
@group(0) @binding(3) var<storage, read> w_scales: array<f16>;
@group(0) @binding(4) var<storage, read> w_zp: array<u32>;
@group(0) @binding(5) var<storage, read> bias: array<f16>;
@group(0) @binding(6) var<storage, read_write> y: array<f16>;

const BLOCK: u32 = 32u;
const TR: u32 = 4u;
const WG: u32 = 64u;
const X_TILE_SIZE: u32 = TR * BLOCK;  // 4 * 32 = 128 floats per tile

// Workgroup-shared X tile: each iteration of the K loop loads
// (TR rows × BLOCK cols) of X into on-chip memory once, and all 64
// threads read from this tile rather than reissuing 64 × TR global
// loads per K block.
var<workgroup> x_tile: array<f32, X_TILE_SIZE>;


fn load_byte(arr: ptr<storage, array<u32>, read>, byte_idx: u32) -> u32 {
    let word = (*arr)[byte_idx >> 2u];
    let shift = (byte_idx & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

fn load_nibble(arr: ptr<storage, array<u32>, read>, nibble_idx: u32) -> u32 {
    let byte_val = load_byte(arr, nibble_idx >> 1u);
    let hi = (nibble_idx & 1u) == 1u;
    return select(byte_val & 0xFu, (byte_val >> 4u) & 0xFu, hi);
}


// 2D dispatch: workgroup_id.x indexes 64-wide N tiles, .y indexes
// 4-wide T tiles. Each thread owns one N column and four T rows; the
// int4 weight decode for that (n, k_block) is reused across the four
// T accumulators, and the X reads are amortized across the workgroup
// via on-chip x_tile. TR=8 was tried and produced a wash (register
// pressure on Apple Silicon GPRs cancelled the W-decode amortization
// gain).
@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let n = wg_id.x * WG + lid.x;
    let t_base = wg_id.y * TR;

    let K = dims.K;
    let n_blocks = K / BLOCK;
    let zp_per_row = (n_blocks + 1u) >> 1u;

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;

    let n_active = n < dims.N;
    let nibble_row_base = select(0u, n * K, n_active);
    let scale_row = select(0u, n * n_blocks, n_active);

    for (var b: u32 = 0u; b < n_blocks; b = b + 1u) {
        // Cooperative X load: 128 elements across 64 threads = 2 per
        // thread, pulling block-b's slice of all four T rows into x_tile.
        let i0 = tid * 2u;
        let i1 = i0 + 1u;
        let t0_row = i0 / BLOCK;
        let k0 = i0 % BLOCK;
        let t1_row = i1 / BLOCK;
        let k1 = i1 % BLOCK;
        let g0 = t_base + t0_row;
        let g1 = t_base + t1_row;
        x_tile[i0] = select(0.0, f32(x[g0 * K + b * BLOCK + k0]), g0 < dims.T);
        x_tile[i1] = select(0.0, f32(x[g1 * K + b * BLOCK + k1]), g1 < dims.T);
        workgroupBarrier();

        if (n_active) {
            let scale: f32 = f32(w_scales[scale_row + b]);
            let zp_byte = load_byte(&w_zp, n * zp_per_row + (b >> 1u));
            let zp_nib = select(zp_byte & 0xFu, (zp_byte >> 4u) & 0xFu, (b & 1u) == 1u);
            let zp_f: f32 = f32(zp_nib);
            let word_base = (nibble_row_base + b * BLOCK) >> 3u;

            var blk0: f32 = 0.0;
            var blk1: f32 = 0.0;
            var blk2: f32 = 0.0;
            var blk3: f32 = 0.0;

            for (var w: u32 = 0u; w < 4u; w = w + 1u) {
                let word = w_int4[word_base + w];
                let kb = w * 8u;
                let q_lo = vec4<f32>(
                    f32( word        & 0xFu) - zp_f,
                    f32((word >>  4u) & 0xFu) - zp_f,
                    f32((word >>  8u) & 0xFu) - zp_f,
                    f32((word >> 12u) & 0xFu) - zp_f,
                );
                let q_hi = vec4<f32>(
                    f32((word >> 16u) & 0xFu) - zp_f,
                    f32((word >> 20u) & 0xFu) - zp_f,
                    f32((word >> 24u) & 0xFu) - zp_f,
                    f32((word >> 28u) & 0xFu) - zp_f,
                );

                let xb0 = 0u * BLOCK + kb;
                let x0_lo = vec4<f32>(x_tile[xb0 + 0u], x_tile[xb0 + 1u], x_tile[xb0 + 2u], x_tile[xb0 + 3u]);
                let x0_hi = vec4<f32>(x_tile[xb0 + 4u], x_tile[xb0 + 5u], x_tile[xb0 + 6u], x_tile[xb0 + 7u]);
                blk0 = blk0 + dot(q_lo, x0_lo) + dot(q_hi, x0_hi);

                let xb1 = 1u * BLOCK + kb;
                let x1_lo = vec4<f32>(x_tile[xb1 + 0u], x_tile[xb1 + 1u], x_tile[xb1 + 2u], x_tile[xb1 + 3u]);
                let x1_hi = vec4<f32>(x_tile[xb1 + 4u], x_tile[xb1 + 5u], x_tile[xb1 + 6u], x_tile[xb1 + 7u]);
                blk1 = blk1 + dot(q_lo, x1_lo) + dot(q_hi, x1_hi);

                let xb2 = 2u * BLOCK + kb;
                let x2_lo = vec4<f32>(x_tile[xb2 + 0u], x_tile[xb2 + 1u], x_tile[xb2 + 2u], x_tile[xb2 + 3u]);
                let x2_hi = vec4<f32>(x_tile[xb2 + 4u], x_tile[xb2 + 5u], x_tile[xb2 + 6u], x_tile[xb2 + 7u]);
                blk2 = blk2 + dot(q_lo, x2_lo) + dot(q_hi, x2_hi);

                let xb3 = 3u * BLOCK + kb;
                let x3_lo = vec4<f32>(x_tile[xb3 + 0u], x_tile[xb3 + 1u], x_tile[xb3 + 2u], x_tile[xb3 + 3u]);
                let x3_hi = vec4<f32>(x_tile[xb3 + 4u], x_tile[xb3 + 5u], x_tile[xb3 + 6u], x_tile[xb3 + 7u]);
                blk3 = blk3 + dot(q_lo, x3_lo) + dot(q_hi, x3_hi);
            }
            acc0 = fma(blk0, scale, acc0);
            acc1 = fma(blk1, scale, acc1);
            acc2 = fma(blk2, scale, acc2);
            acc3 = fma(blk3, scale, acc3);
        }
        workgroupBarrier();
    }

    if (n_active) {
        let bias_f = f32(bias[n]);
        let t0 = t_base + 0u;
        let t1 = t_base + 1u;
        let t2 = t_base + 2u;
        let t3 = t_base + 3u;
        if (t0 < dims.T) { y[t0 * dims.N + n] = f16(acc0 + bias_f); }
        if (t1 < dims.T) { y[t1 * dims.N + n] = f16(acc1 + bias_f); }
        if (t2 < dims.T) { y[t2 * dims.N + n] = f16(acc2 + bias_f); }
        if (t3 < dims.T) { y[t3 * dims.N + n] = f16(acc3 + bias_f); }
    }
}
