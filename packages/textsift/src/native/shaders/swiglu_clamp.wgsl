struct Dims { rows: u32, dff: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> gate_up: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

const LIMIT: f32 = 7.0;
const ALPHA: f32 = 1.702;

fn sigmoid_f32(x: f32) -> f32 {
    // Numerically stable: σ(x) = 1/(1+exp(-|x|)) on both sides, flipped for
    // negative x. Keeps the exp argument ≤ 0 so no intermediate overflow.
    if (x >= 0.0) {
        return 1.0 / (1.0 + exp(-x));
    }
    let e = exp(x);
    return e / (1.0 + e);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = dims.rows * dims.dff;
    if (idx >= total) { return; }
    let row = idx / dims.dff;
    let col = idx % dims.dff;

    let stride = 2u * dims.dff;
    let gate_raw = gate_up[row * stride + 2u * col];
    let up_raw   = gate_up[row * stride + 2u * col + 1u];
    let gate = min(gate_raw, LIMIT);
    let up   = clamp(up_raw, -LIMIT, LIMIT);
    let glu  = gate * sigmoid_f32(gate * ALPHA);
    out[idx] = (up + 1.0) * glu;
}
