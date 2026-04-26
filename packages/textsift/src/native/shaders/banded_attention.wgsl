enable f16;

struct Dims {
    T: u32, H_q: u32, H_kv: u32, head_dim: u32,
    window: u32, use_mask: u32, _pad0: u32, _pad1: u32,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> q: array<f16>;
@group(0) @binding(2) var<storage, read> k: array<f16>;
@group(0) @binding(3) var<storage, read> v: array<f16>;
@group(0) @binding(4) var<storage, read> sinks: array<f32>;
@group(0) @binding(5) var<storage, read> mask: array<u32>;
@group(0) @binding(6) var<storage, read_write> out: array<f16>;

const MAX_WINDOW_TOTAL: u32 = 257u;
const NEG_INF: f32 = -1e30;

var<workgroup> wg_scores: array<f32, MAX_WINDOW_TOTAL>;
var<workgroup> wg_tmp: array<f32, 64>;
var<workgroup> wg_broadcast: array<f32, 2>;  // [0] = row_max, [1] = inv_sum

fn read_mask_byte(i: u32) -> u32 {
    let word = mask[i >> 2u];
    return (word >> ((i & 3u) * 8u)) & 0xFFu;
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let t_q_h_q = wg_id.x;
    let H_q = dims.H_q;
    if (t_q_h_q >= dims.T * H_q) { return; }
    let t_q = t_q_h_q / H_q;
    let h_q = t_q_h_q % H_q;
    let head_dim = dims.head_dim;
    let kv_group = H_q / dims.H_kv;
    let h_kv = h_q / kv_group;

    let q_stride_t = H_q * head_dim;
    let k_stride_t = dims.H_kv * head_dim;
    let q_base = t_q * q_stride_t + h_q * head_dim;

    let window = dims.window;
    var ws: u32 = 0u;
    if (t_q > window) { ws = t_q - window; }
    var we: u32 = dims.T;
    if (t_q + window + 1u < dims.T) { we = t_q + window + 1u; }
    let n_keys = we - ws;

    // Pass 1: each thread computes scores for its strided keys and
    // writes them into workgroup memory.
    var thread_max: f32 = NEG_INF;
    var idx = tid;
    loop {
        if (idx >= n_keys) { break; }
        let abs_k = ws + idx;
        var is_valid: bool = true;
        if (dims.use_mask == 1u) {
            is_valid = read_mask_byte(abs_k) != 0u;
        }
        var s: f32 = NEG_INF;
        if (is_valid) {
            let k_base = abs_k * k_stride_t + h_kv * head_dim;
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                dot = fma(f32(q[q_base + d]), f32(k[k_base + d]), dot);
            }
            s = dot;
        }
        wg_scores[idx] = s;
        if (s > thread_max) { thread_max = s; }
        idx = idx + 64u;
    }
    // Include sink in max comparison (thread 0 seeds with sink).
    if (tid == 0u) {
        let sink_s = sinks[h_q];
        if (sink_s > thread_max) { thread_max = sink_s; }
    }
    wg_tmp[tid] = thread_max;
    workgroupBarrier();

    // Tree reduce max across 64 threads.
    var stride: u32 = 32u;
    loop {
        if (tid < stride) {
            wg_tmp[tid] = max(wg_tmp[tid], wg_tmp[tid + stride]);
        }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride = stride / 2u;
    }
    if (tid == 0u) { wg_broadcast[0] = wg_tmp[0]; }
    workgroupBarrier();
    let row_max = wg_broadcast[0];

    // Pass 2: exp(score - max), accumulate sum.
    var thread_sum: f32 = 0.0;
    idx = tid;
    loop {
        if (idx >= n_keys) { break; }
        let e = exp(wg_scores[idx] - row_max);
        wg_scores[idx] = e;
        thread_sum = thread_sum + e;
        idx = idx + 64u;
    }
    // Sink contributes to denominator.
    if (tid == 0u) {
        thread_sum = thread_sum + exp(sinks[h_q] - row_max);
    }
    wg_tmp[tid] = thread_sum;
    workgroupBarrier();

    stride = 32u;
    loop {
        if (tid < stride) {
            wg_tmp[tid] = wg_tmp[tid] + wg_tmp[tid + stride];
        }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride = stride / 2u;
    }
    if (tid == 0u) {
        let s = wg_tmp[0];
        wg_broadcast[1] = select(0.0, 1.0 / s, s > 0.0);
    }
    workgroupBarrier();
    let inv_sum = wg_broadcast[1];

    // Pass 3: normalize softmax in-place so pass 4 reads a single value.
    idx = tid;
    loop {
        if (idx >= n_keys) { break; }
        wg_scores[idx] = wg_scores[idx] * inv_sum;
        idx = idx + 64u;
    }
    workgroupBarrier();

    // Pass 4: each thread owns one head_dim lane and combines V across
    // all keys in the window. Requires workgroup_size >= head_dim; our
    // head_dim is 64 so each of 64 threads handles exactly one lane.
    if (tid < head_dim) {
        var acc: f32 = 0.0;
        for (var i: u32 = 0u; i < n_keys; i = i + 1u) {
            let abs_k = ws + i;
            let v_base = abs_k * k_stride_t + h_kv * head_dim;
            acc = fma(wg_scores[i], f32(v[v_base + tid]), acc);
        }
        let out_base = t_q * q_stride_t + h_q * head_dim;
        out[out_base + tid] = f16(acc);
    }
}
