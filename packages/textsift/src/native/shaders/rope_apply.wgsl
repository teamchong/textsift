enable f16;

struct Dims { T: u32, H: u32, head_dim: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read_write> qk: array<f16>;
@group(0) @binding(2) var<storage, read> cos_tab: array<f16>;
@group(0) @binding(3) var<storage, read> sin_tab: array<f16>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_dim = dims.head_dim / 2u;
    let idx = gid.x;
    let tp_total = dims.T * dims.H * half_dim;
    if (idx >= tp_total) { return; }

    let p = idx % half_dim;
    let th = idx / half_dim;
    let h = th % dims.H;
    let t = th / dims.H;

    let c: f16 = cos_tab[t * half_dim + p];
    let s: f16 = sin_tab[t * half_dim + p];

    let head_base = (t * dims.H + h) * dims.head_dim;
    let a: f16 = qk[head_base + 2u * p];
    let b: f16 = qk[head_base + 2u * p + 1u];

    let ac: f16 = a * c;
    let bs: f16 = b * s;
    let bc: f16 = b * c;
    let as_: f16 = a * s;

    qk[head_base + 2u * p]      = ac - bs;
    qk[head_base + 2u * p + 1u] = bc + as_;
}
