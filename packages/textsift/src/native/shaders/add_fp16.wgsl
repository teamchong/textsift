enable f16;

struct Dims { n: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f16>;
@group(0) @binding(2) var<storage, read> b: array<f16>;
@group(0) @binding(3) var<storage, read_write> out: array<f16>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.n) { return; }
    // Widen to f32 for the add, then round once — matches WASM add_fp16
    // semantics: f32ToFp16(fp16ToF32(a) + fp16ToF32(b)).
    out[i] = f16(f32(a[i]) + f32(b[i]));
}
