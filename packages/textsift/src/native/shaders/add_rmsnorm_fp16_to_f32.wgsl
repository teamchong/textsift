enable f16;

struct Dims { T: u32, D: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f16>;
@group(0) @binding(2) var<storage, read> b: array<f16>;
@group(0) @binding(3) var<storage, read> gamma: array<f16>;
@group(0) @binding(4) var<storage, read_write> sum_out: array<f16>;
@group(0) @binding(5) var<storage, read_write> norm_out: array<f32>;

var<workgroup> wg_sum: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let t = wg.x;
    if (t >= dims.T) { return; }
    let D = dims.D;
    let row = t * D;
    let tid = lid.x;

    var ssq: f32 = 0.0;
    var d = tid;
    loop {
        if (d >= D) { break; }
        let av = f32(a[row + d]);
        let bv = f32(b[row + d]);
        let sv = av + bv;
        sum_out[row + d] = f16(sv);
        ssq = ssq + sv * sv;
        d = d + 64u;
    }
    wg_sum[tid] = ssq;
    workgroupBarrier();

    var stride: u32 = 32u;
    loop {
        if (tid < stride) {
            wg_sum[tid] = wg_sum[tid] + wg_sum[tid + stride];
        }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride = stride / 2u;
    }

    let inv_rms = inverseSqrt(wg_sum[0] / f32(D) + dims.eps);

    // Re-read the fp16 sum we just wrote. Each thread reads only its
    // own writes (strided by 64) so there's no cross-thread dep, and
    // the write→read of the same fp16 is round-trip identical.
    d = tid;
    loop {
        if (d >= D) { break; }
        let g = f32(gamma[d]);
        let sv = f32(sum_out[row + d]);
        norm_out[row + d] = sv * inv_rms * g;
        d = d + 64u;
    }
}
