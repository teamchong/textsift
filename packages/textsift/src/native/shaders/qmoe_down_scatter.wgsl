enable f16;

struct Dims {
    T: u32, K: u32, N: u32, D: u32,
    _pad0: u32, _pad1: u32, _pad2: u32, _pad3: u32,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> glu: array<f32>;
@group(0) @binding(2) var<storage, read> routing_idx: array<i32>;
@group(0) @binding(3) var<storage, read> routing_scores: array<f32>;
@group(0) @binding(4) var<storage, read> w_int4: array<u32>;
@group(0) @binding(5) var<storage, read> w_scales: array<f16>;
@group(0) @binding(6) var<storage, read> w_zp: array<u32>;
@group(0) @binding(7) var<storage, read> bias: array<f16>;
@group(0) @binding(8) var<storage, read_write> acc: array<atomic<u32>>;

const BLOCK: u32 = 32u;
const D_MAX: u32 = 640u;  // dff for openai/privacy-filter
const WG: u32 = 64u;

// Workgroup-shared input tile. All 64 threads in a workgroup share the
// same (token, k_pick) so they read the same glu row — load it once
// cooperatively and reuse from on-chip memory rather than reissuing
// 64 redundant global loads per K block.
var<workgroup> glu_tile: array<f32, D_MAX>;


// Inlined int4-access (Naga rejects ptr<storage, ...> as fn args).


fn atomic_add_f32(slot: u32, val: f32) {
    var old_u = atomicLoad(&acc[slot]);
    loop {
        let new_f = bitcast<f32>(old_u) + val;
        let new_u = bitcast<u32>(new_f);
        let r = atomicCompareExchangeWeak(&acc[slot], old_u, new_u);
        if (r.exchanged) { break; }
        old_u = r.old_value;
    }
}

// 2D dispatch: workgroup_id.x is the (token, k_pick) pair, .y is the
// N-tile index (0..N/64). Each thread writes one N column for that
// (tk, n_tile) tile via atomic scatter-add.
@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let tk = wg_id.x;
    if (tk >= dims.T * dims.K) { return; }
    let n_base = wg_id.y * WG;
    let n = n_base + tid;

    let t = tk / dims.K;
    let e = u32(routing_idx[tk]);
    let w = routing_scores[tk];

    let D = dims.D;  // inner dim = dff
    let N = dims.N;  // output dim = d_model
    let n_blocks = D / BLOCK;
    let zp_per_row = (n_blocks + 1u) >> 1u;

    // Cooperatively load the glu row (D = dff = 640 elements) once.
    // 64 threads, 10 elements each.
    for (var i: u32 = 0u; i < 10u; i = i + 1u) {
        let k_idx = i * WG + tid;
        if (k_idx < D) {
            glu_tile[k_idx] = glu[tk * D + k_idx];
        }
    }
    workgroupBarrier();

    if (n >= N) { return; }

    let expert_nibble_base = e * N * D + n * D;
    let expert_scale_base  = e * N * n_blocks + n * n_blocks;
    let expert_zp_base     = e * N * zp_per_row + n * zp_per_row;
    let expert_bias_base   = e * N + n;

    var acc_local: f32 = 0.0;
    for (var b: u32 = 0u; b < n_blocks; b = b + 1u) {
        let scale: f32 = f32(w_scales[expert_scale_base + b]);
        let zp_byte_idx = expert_zp_base + (b >> 1u);
        let zp_byte = (w_zp[zp_byte_idx >> 2u] >> ((zp_byte_idx & 3u) * 8u)) & 0xFFu;
        let zp_nib = select(zp_byte & 0xFu, (zp_byte >> 4u) & 0xFu, (b & 1u) == 1u);
        let zp_f: f32 = f32(zp_nib);

        let word_base = (expert_nibble_base + b * BLOCK) >> 3u;
        let xb = b * BLOCK;

        var block_sum: f32 = 0.0;
        for (var wi: u32 = 0u; wi < 4u; wi = wi + 1u) {
            let word = w_int4[word_base + wi];
            let kb = wi * 8u;
            let q0 = f32( word        & 0xFu) - zp_f;
            let q1 = f32((word >>  4u) & 0xFu) - zp_f;
            let q2 = f32((word >>  8u) & 0xFu) - zp_f;
            let q3 = f32((word >> 12u) & 0xFu) - zp_f;
            let q4 = f32((word >> 16u) & 0xFu) - zp_f;
            let q5 = f32((word >> 20u) & 0xFu) - zp_f;
            let q6 = f32((word >> 24u) & 0xFu) - zp_f;
            let q7 = f32((word >> 28u) & 0xFu) - zp_f;
            block_sum = fma(q0, glu_tile[xb + kb + 0u], block_sum);
            block_sum = fma(q1, glu_tile[xb + kb + 1u], block_sum);
            block_sum = fma(q2, glu_tile[xb + kb + 2u], block_sum);
            block_sum = fma(q3, glu_tile[xb + kb + 3u], block_sum);
            block_sum = fma(q4, glu_tile[xb + kb + 4u], block_sum);
            block_sum = fma(q5, glu_tile[xb + kb + 5u], block_sum);
            block_sum = fma(q6, glu_tile[xb + kb + 6u], block_sum);
            block_sum = fma(q7, glu_tile[xb + kb + 7u], block_sum);
        }
        acc_local = fma(block_sum, scale, acc_local);
    }
    acc_local = acc_local + f32(bias[expert_bias_base]);

    atomic_add_f32(t * N + n, w * acc_local);
}
