struct Dims { T: u32, E: u32, K: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_idx: array<i32>;
@group(0) @binding(3) var<storage, read_write> out_scores: array<f32>;

const MAX_K: u32 = 8u;
const NEG_INF: f32 = -1e30;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= dims.T) { return; }
    let K = dims.K;
    let row = t * dims.E;

    var top_val: array<f32, MAX_K>;
    var top_idx: array<u32, MAX_K>;
    for (var i: u32 = 0u; i < MAX_K; i = i + 1u) {
        top_val[i] = NEG_INF;
        top_idx[i] = 0u;
    }

    for (var e: u32 = 0u; e < dims.E; e = e + 1u) {
        let v = logits[row + e];
        // Locate slot with the smallest current value.
        var min_k: u32 = 0u;
        var min_v: f32 = top_val[0];
        for (var k: u32 = 1u; k < K; k = k + 1u) {
            if (top_val[k] < min_v) {
                min_v = top_val[k];
                min_k = k;
            }
        }
        if (v > min_v) {
            top_val[min_k] = v;
            top_idx[min_k] = e;
        }
    }

    // Softmax over top K.
    var max_v: f32 = top_val[0];
    for (var k: u32 = 1u; k < K; k = k + 1u) {
        max_v = max(max_v, top_val[k]);
    }
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < K; k = k + 1u) {
        let e = exp(top_val[k] - max_v);
        top_val[k] = e;
        sum = sum + e;
    }
    let inv_sum = select(0.0, 1.0 / sum, sum > 0.0);
    let inv_K = 1.0 / f32(K);

    let out_base = t * K;
    for (var k: u32 = 0u; k < K; k = k + 1u) {
        out_idx[out_base + k] = i32(top_idx[k]);
        out_scores[out_base + k] = top_val[k] * inv_sum * inv_K;
    }
}
