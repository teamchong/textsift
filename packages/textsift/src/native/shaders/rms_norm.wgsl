enable f16;

struct Dims { T: u32, D: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f16>;
@group(0) @binding(2) var<storage, read> gamma: array<f16>;
@group(0) @binding(3) var<storage, read_write> y: array<f16>;

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
        let v = f32(x[row + d]);
        ssq = ssq + v * v;
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

    d = tid;
    loop {
        if (d >= D) { break; }
        let g = f32(gamma[d]);
        let xv = f32(x[row + d]);
        y[row + d] = f16(xv * inv_rms * g);
        d = d + 64u;
    }
}
