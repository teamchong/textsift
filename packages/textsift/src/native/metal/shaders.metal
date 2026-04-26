// Hand-tuned Metal Shading Language kernels. One file, one library —
// all entry points compile together so we share a single MTLLibrary
// per backend instance.
//
// Conformance: each kernel must produce byte-equal output (within
// 1 fp16 ULP) to the reference WGSL kernel in
// packages/textsift/src/native/shaders/<name>.wgsl when given the
// same input. Tested by tests/native/metal/conformance.test.js.

#include <metal_stdlib>
using namespace metal;

// ────────────────────────────────────────────────────────────────
// rms_norm  ←  reference: src/native/shaders/rms_norm.wgsl
//
// y[t, d] = x[t, d] * gamma[d] / sqrt(mean_d(x[t, :]^2) + eps)
//
// One threadgroup per row of T. 64 threads per workgroup, each walks
// a strided slice of D for the sumsq reduction, then again for the
// per-element write. f32 accumulator (640-wide fp16 sum loses bits).
// ────────────────────────────────────────────────────────────────

struct RmsNormDims { uint T; uint D; float eps; uint _pad; };

kernel void rms_norm(
    device const RmsNormDims&  dims    [[buffer(0)]],
    device const half*         x       [[buffer(1)]],
    device const half*         gamma   [[buffer(2)]],
    device       half*         y       [[buffer(3)]],
    uint3                      lid_v   [[thread_position_in_threadgroup]],
    uint3                      gid     [[threadgroup_position_in_grid]])
{
    threadgroup float wg_sum[64];
    const uint tid = lid_v.x;
    const uint t = gid.x;
    if (t >= dims.T) return;
    const uint D = dims.D;
    const uint row = t * D;

    // Pass 1: per-thread strided sum-of-squares.
    float ssq = 0.0f;
    for (uint d = tid; d < D; d += 64u) {
        const float v = float(x[row + d]);
        ssq = fma(v, v, ssq);
    }
    wg_sum[tid] = ssq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction: 32 → 16 → 8 → 4 → 2 → 1.
    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (tid < stride) wg_sum[tid] += wg_sum[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_rms = rsqrt(wg_sum[0] / float(D) + dims.eps);

    // Pass 2: per-element scale.
    for (uint d = tid; d < D; d += 64u) {
        const float g = float(gamma[d]);
        const float xv = float(x[row + d]);
        y[row + d] = half(xv * inv_rms * g);
    }
}

// ────────────────────────────────────────────────────────────────
// matmul_int4_fp16_f16  ←  reference: src/native/shaders/matmul_int4_fp16_f16.wgsl
//
// y[T, N] = dequant(W[N, K]) @ x[T, K]^T + bias[N], all fp16.
// Dequant: int4 nibble × fp16 scale, blocked at K=32.
//
// Tile: 64 threads × 4-T-rows per workgroup. Cooperative X load
// shares one (TR=4)×BLOCK tile across all 64 threads. Each thread
// owns one N column and accumulates four T-row dot products.
// ────────────────────────────────────────────────────────────────

struct MatmulDims { uint T; uint N; uint K; uint _pad; };

constant constexpr uint BLOCK = 32u;
constant constexpr uint TR    = 4u;
constant constexpr uint WG    = 64u;
constant constexpr uint X_TILE_SIZE = TR * BLOCK;

kernel void matmul_int4_fp16_f16(
    device const MatmulDims&   dims      [[buffer(0)]],
    device const half*         x         [[buffer(1)]],
    device const uint*         w_int4    [[buffer(2)]],
    device const half*         w_scales  [[buffer(3)]],
    device const uint*         w_zp      [[buffer(4)]],
    device const half*         bias      [[buffer(5)]],
    device       half*         y         [[buffer(6)]],
    uint3                      lid_v     [[thread_position_in_threadgroup]],
    uint3                      gid       [[threadgroup_position_in_grid]])
{
    threadgroup float x_tile[X_TILE_SIZE];

    const uint tid = lid_v.x;
    const uint n = gid.x * WG + tid;
    const uint t_base = gid.y * TR;
    const uint K = dims.K;
    const uint n_blocks = K / BLOCK;
    const uint zp_per_row = (n_blocks + 1u) >> 1u;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    const bool n_active = n < dims.N;
    const uint nibble_row_base = n_active ? n * K : 0u;
    const uint scale_row = n_active ? n * n_blocks : 0u;

    for (uint b = 0u; b < n_blocks; ++b) {
        // Cooperative X load: 128 elements / 64 threads = 2/thread.
        const uint i0 = tid * 2u;
        const uint i1 = i0 + 1u;
        const uint t0_row = i0 / BLOCK, k0 = i0 % BLOCK;
        const uint t1_row = i1 / BLOCK, k1 = i1 % BLOCK;
        const uint g0 = t_base + t0_row;
        const uint g1 = t_base + t1_row;
        x_tile[i0] = (g0 < dims.T) ? float(x[g0 * K + b * BLOCK + k0]) : 0.0f;
        x_tile[i1] = (g1 < dims.T) ? float(x[g1 * K + b * BLOCK + k1]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (n_active) {
            const float scale = float(w_scales[scale_row + b]);
            const uint zp_byte_idx = n * zp_per_row + (b >> 1u);
            const uint zp_byte = (w_zp[zp_byte_idx >> 2u] >> ((zp_byte_idx & 3u) * 8u)) & 0xFFu;
            const uint zp_nib = (b & 1u) ? ((zp_byte >> 4u) & 0xFu) : (zp_byte & 0xFu);
            const float zp_f = float(zp_nib);
            const uint word_base = (nibble_row_base + b * BLOCK) >> 3u;

            float blk0 = 0.0f, blk1 = 0.0f, blk2 = 0.0f, blk3 = 0.0f;
            for (uint w = 0u; w < 4u; ++w) {
                const uint word = w_int4[word_base + w];
                const uint kb = w * 8u;
                const float4 q_lo = float4(
                    float( word        & 0xFu) - zp_f,
                    float((word >>  4u) & 0xFu) - zp_f,
                    float((word >>  8u) & 0xFu) - zp_f,
                    float((word >> 12u) & 0xFu) - zp_f);
                const float4 q_hi = float4(
                    float((word >> 16u) & 0xFu) - zp_f,
                    float((word >> 20u) & 0xFu) - zp_f,
                    float((word >> 24u) & 0xFu) - zp_f,
                    float((word >> 28u) & 0xFu) - zp_f);

                const uint xb0 = 0u * BLOCK + kb;
                blk0 += dot(q_lo, float4(x_tile[xb0+0u], x_tile[xb0+1u], x_tile[xb0+2u], x_tile[xb0+3u]))
                      + dot(q_hi, float4(x_tile[xb0+4u], x_tile[xb0+5u], x_tile[xb0+6u], x_tile[xb0+7u]));
                const uint xb1 = 1u * BLOCK + kb;
                blk1 += dot(q_lo, float4(x_tile[xb1+0u], x_tile[xb1+1u], x_tile[xb1+2u], x_tile[xb1+3u]))
                      + dot(q_hi, float4(x_tile[xb1+4u], x_tile[xb1+5u], x_tile[xb1+6u], x_tile[xb1+7u]));
                const uint xb2 = 2u * BLOCK + kb;
                blk2 += dot(q_lo, float4(x_tile[xb2+0u], x_tile[xb2+1u], x_tile[xb2+2u], x_tile[xb2+3u]))
                      + dot(q_hi, float4(x_tile[xb2+4u], x_tile[xb2+5u], x_tile[xb2+6u], x_tile[xb2+7u]));
                const uint xb3 = 3u * BLOCK + kb;
                blk3 += dot(q_lo, float4(x_tile[xb3+0u], x_tile[xb3+1u], x_tile[xb3+2u], x_tile[xb3+3u]))
                      + dot(q_hi, float4(x_tile[xb3+4u], x_tile[xb3+5u], x_tile[xb3+6u], x_tile[xb3+7u]));
            }
            acc0 = fma(blk0, scale, acc0);
            acc1 = fma(blk1, scale, acc1);
            acc2 = fma(blk2, scale, acc2);
            acc3 = fma(blk3, scale, acc3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (n_active) {
        const float bias_f = float(bias[n]);
        if (t_base + 0u < dims.T) y[(t_base + 0u) * dims.N + n] = half(acc0 + bias_f);
        if (t_base + 1u < dims.T) y[(t_base + 1u) * dims.N + n] = half(acc1 + bias_f);
        if (t_base + 2u < dims.T) y[(t_base + 2u) * dims.N + n] = half(acc2 + bias_f);
        if (t_base + 3u < dims.T) y[(t_base + 3u) * dims.N + n] = half(acc3 + bias_f);
    }
}

// ────────────────────────────────────────────────────────────────
// matmul_int4_f32_f32 — f32 x [T,K] × int4 W [N,K] → f32 y [T,N]
// One thread per output element. Used for router (T×E) and final
// classifier head (T×numClasses).
// ────────────────────────────────────────────────────────────────

kernel void matmul_int4_f32_f32(
    device const MatmulDims&  dims     [[buffer(0)]],
    device const float*       x        [[buffer(1)]],
    device const uint*        w_int4   [[buffer(2)]],
    device const half*        w_scales [[buffer(3)]],
    device const uint*        w_zp     [[buffer(4)]],
    device const half*        bias     [[buffer(5)]],
    device       float*       y        [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint tn = gid.x;
    const uint total = dims.T * dims.N;
    if (tn >= total) return;
    const uint t = tn / dims.N;
    const uint n = tn % dims.N;
    const uint K = dims.K;
    const uint n_blocks = K / BLOCK;
    const uint zp_per_row = (n_blocks + 1u) >> 1u;

    const uint nibble_row_base = n * K;
    const uint scale_row = n * n_blocks;
    const uint x_row = t * K;

    float acc = 0.0f;
    for (uint b = 0u; b < n_blocks; ++b) {
        const float scale = float(w_scales[scale_row + b]);
        const uint zp_byte_idx = n * zp_per_row + (b >> 1u);
        const uint zp_byte = (w_zp[zp_byte_idx >> 2u] >> ((zp_byte_idx & 3u) * 8u)) & 0xFFu;
        const uint zp_nib = (b & 1u) ? ((zp_byte >> 4u) & 0xFu) : (zp_byte & 0xFu);
        const float zp_f = float(zp_nib);
        const uint word_base = (nibble_row_base + b * BLOCK) >> 3u;
        const uint xb = x_row + b * BLOCK;

        float blk = 0.0f;
        for (uint w = 0u; w < 4u; ++w) {
            const uint word = w_int4[word_base + w];
            const uint kb = w * 8u;
            blk = fma(float( word        & 0xFu) - zp_f, x[xb + kb + 0u], blk);
            blk = fma(float((word >>  4u) & 0xFu) - zp_f, x[xb + kb + 1u], blk);
            blk = fma(float((word >>  8u) & 0xFu) - zp_f, x[xb + kb + 2u], blk);
            blk = fma(float((word >> 12u) & 0xFu) - zp_f, x[xb + kb + 3u], blk);
            blk = fma(float((word >> 16u) & 0xFu) - zp_f, x[xb + kb + 4u], blk);
            blk = fma(float((word >> 20u) & 0xFu) - zp_f, x[xb + kb + 5u], blk);
            blk = fma(float((word >> 24u) & 0xFu) - zp_f, x[xb + kb + 6u], blk);
            blk = fma(float((word >> 28u) & 0xFu) - zp_f, x[xb + kb + 7u], blk);
        }
        acc = fma(blk, scale, acc);
    }
    y[t * dims.N + n] = acc + float(bias[n]);
}

// ────────────────────────────────────────────────────────────────
// embed_lookup_int4 — ids[T] → int4-quantized rows of shape [V, D]
// → out[T, D] fp16
// ────────────────────────────────────────────────────────────────

struct EmbedDims { uint T; uint V; uint D; uint _pad; };

kernel void embed_lookup_int4(
    device const EmbedDims&    dims         [[buffer(0)]],
    device const uint*         embed_int4   [[buffer(1)]],
    device const half*         embed_scales [[buffer(2)]],
    device const uint*         embed_zp     [[buffer(3)]],
    device const int*          ids          [[buffer(4)]],
    device       half*         out          [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint EMBED_BLOCK = 32u;
    const uint td = gid.x;
    const uint D = dims.D;
    const uint total = dims.T * D;
    if (td >= total) return;
    const uint t = td / D;
    const uint d = td % D;

    const int id = ids[t];
    if (id < 0 || (uint)id >= dims.V) {
        out[td] = half(0.0f);
        return;
    }
    const uint row = (uint)id;
    const uint n_blocks = D / EMBED_BLOCK;
    const uint zp_per_row = (n_blocks + 1u) >> 1u;
    const uint b = d / EMBED_BLOCK;
    const float scale = float(embed_scales[row * n_blocks + b]);

    const uint zp_byte_idx = row * zp_per_row + (b >> 1u);
    const uint zp_byte = (embed_zp[zp_byte_idx >> 2u] >> ((zp_byte_idx & 3u) * 8u)) & 0xFFu;
    const uint zp_nib = (b & 1u) ? ((zp_byte >> 4u) & 0xFu) : (zp_byte & 0xFu);

    const uint nib_idx = row * D + d;
    const uint nib = (embed_int4[nib_idx >> 3u] >> ((nib_idx & 7u) * 4u)) & 0xFu;
    const float q = float(nib) - float(zp_nib);
    out[td] = half(q * scale);
}

// ────────────────────────────────────────────────────────────────
// add_rmsnorm_fp16_to_f32 — fused (a+b → fp16) + rmsnorm + widen→f32
// One workgroup per row.
// ────────────────────────────────────────────────────────────────

kernel void add_rmsnorm_fp16_to_f32(
    device const RmsNormDims&  dims      [[buffer(0)]],
    device const half*         a         [[buffer(1)]],
    device const half*         b         [[buffer(2)]],
    device const half*         gamma     [[buffer(3)]],
    device       half*         sum_out   [[buffer(4)]],
    device       float*        norm_out  [[buffer(5)]],
    uint3 lid_v [[thread_position_in_threadgroup]],
    uint3 gid   [[threadgroup_position_in_grid]])
{
    threadgroup float wg_sum[64];
    const uint tid = lid_v.x;
    const uint t = gid.x;
    if (t >= dims.T) return;
    const uint D = dims.D;
    const uint row = t * D;

    float ssq = 0.0f;
    for (uint d = tid; d < D; d += 64u) {
        const float av = float(a[row + d]);
        const float bv = float(b[row + d]);
        const half sum = half(av + bv);
        sum_out[row + d] = sum;
        const float sf = float(sum);
        ssq = fma(sf, sf, ssq);
    }
    wg_sum[tid] = ssq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (tid < stride) wg_sum[tid] += wg_sum[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_rms = rsqrt(wg_sum[0] / float(D) + dims.eps);
    for (uint d = tid; d < D; d += 64u) {
        const float g = float(gamma[d]);
        const float sv = float(sum_out[row + d]);
        norm_out[row + d] = sv * inv_rms * g;
    }
}

// ────────────────────────────────────────────────────────────────
// element-wise: cast/add/zero kernels
// ────────────────────────────────────────────────────────────────

struct ElemDims { uint n; uint _pad0; uint _pad1; uint _pad2; };

kernel void cast_fp16_to_f32(
    device const ElemDims& dims [[buffer(0)]],
    device const half*     src  [[buffer(1)]],
    device       float*    dst  [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint i = gid.x;
    if (i >= dims.n) return;
    dst[i] = float(src[i]);
}

struct ScaleU { float v; };

kernel void cast_f32_to_fp16_scaled(
    device const ElemDims& dims  [[buffer(0)]],
    device const ScaleU&   scale [[buffer(1)]],
    device const float*    src   [[buffer(2)]],
    device       half*     dst   [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint i = gid.x;
    if (i >= dims.n) return;
    dst[i] = half(src[i] * scale.v);
}

kernel void add_fp16(
    device const ElemDims& dims [[buffer(0)]],
    device const half*     a    [[buffer(1)]],
    device const half*     b    [[buffer(2)]],
    device       half*     out  [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint i = gid.x;
    if (i >= dims.n) return;
    // Match WGSL semantics: widen to f32 for the add, then round once.
    out[i] = half(float(a[i]) + float(b[i]));
}

kernel void zero_f32(
    device const ElemDims& dims [[buffer(0)]],
    device       uint*     buf  [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint i = gid.x;
    if (i >= dims.n) return;
    buf[i] = 0u;
}

// ────────────────────────────────────────────────────────────────
// rope_apply — interleaved-pair rotary positional encoding, in-place
// ────────────────────────────────────────────────────────────────

struct RopeDims { uint T; uint H; uint head_dim; uint _pad; };

kernel void rope_apply(
    device const RopeDims& dims    [[buffer(0)]],
    device       half*     qk      [[buffer(1)]],
    device const half*     cos_tab [[buffer(2)]],
    device const half*     sin_tab [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint half_dim = dims.head_dim / 2u;
    const uint idx = gid.x;
    const uint tp_total = dims.T * dims.H * half_dim;
    if (idx >= tp_total) return;

    const uint p = idx % half_dim;
    const uint th = idx / half_dim;
    const uint h = th % dims.H;
    const uint t = th / dims.H;

    const half c = cos_tab[t * half_dim + p];
    const half s = sin_tab[t * half_dim + p];
    const uint head_base = (t * dims.H + h) * dims.head_dim;
    const half a = qk[head_base + 2u * p];
    const half b = qk[head_base + 2u * p + 1u];

    qk[head_base + 2u * p]      = a * c - b * s;
    qk[head_base + 2u * p + 1u] = b * c + a * s;
}

// ────────────────────────────────────────────────────────────────
// swiglu_clamp — gated activation for MoE FFN
// ────────────────────────────────────────────────────────────────

struct SwigluDims { uint rows; uint dff; uint _pad0; uint _pad1; };

static inline float sigmoid_f32(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + exp(-x));
    const float e = exp(x);
    return e / (1.0f + e);
}

kernel void swiglu_clamp(
    device const SwigluDims& dims    [[buffer(0)]],
    device const float*      gate_up [[buffer(1)]],
    device       float*      out     [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint idx = gid.x;
    const uint total = dims.rows * dims.dff;
    if (idx >= total) return;
    const uint row = idx / dims.dff;
    const uint col = idx % dims.dff;
    const uint stride = 2u * dims.dff;
    const float gate_raw = gate_up[row * stride + 2u * col];
    const float up_raw   = gate_up[row * stride + 2u * col + 1u];
    const float LIMIT = 7.0f, ALPHA = 1.702f;
    const float gate = min(gate_raw, LIMIT);
    const float up   = clamp(up_raw, -LIMIT, LIMIT);
    const float glu  = gate * sigmoid_f32(gate * ALPHA);
    out[idx] = (up + 1.0f) * glu;
}

// ────────────────────────────────────────────────────────────────
// router_topk — per-token softmax + top-K expert selection
// ────────────────────────────────────────────────────────────────

struct RouterDims { uint T; uint E; uint K; uint _pad; };

kernel void router_topk(
    device const RouterDims& dims          [[buffer(0)]],
    device const float*      logits        [[buffer(1)]],
    device       int*        out_idx       [[buffer(2)]],
    device       float*      out_scores    [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint t = gid.x;
    if (t >= dims.T) return;
    const uint K = dims.K;
    const uint MAX_K = 8u;
    const uint row = t * dims.E;

    // Match the WGSL reference exactly: min-heap of K. Replace
    // the slot with the smallest current value when a new candidate
    // beats it. Output order = whatever the heap holds at the end
    // (NOT sorted), so byte-equal comparison vs the WGSL fixture
    // requires identical iteration semantics.
    float top_val[8];
    uint  top_idx[8];
    for (uint i = 0u; i < MAX_K; ++i) { top_val[i] = -1e30f; top_idx[i] = 0u; }

    for (uint e = 0u; e < dims.E; ++e) {
        const float v = logits[row + e];
        uint min_k = 0u;
        float min_v = top_val[0];
        for (uint k = 1u; k < K; ++k) {
            if (top_val[k] < min_v) { min_v = top_val[k]; min_k = k; }
        }
        if (v > min_v) {
            top_val[min_k] = v;
            top_idx[min_k] = e;
        }
    }

    float max_v = top_val[0];
    for (uint k = 1u; k < K; ++k) max_v = max(max_v, top_val[k]);
    float sum = 0.0f;
    for (uint k = 0u; k < K; ++k) {
        const float e = exp(top_val[k] - max_v);
        top_val[k] = e;
        sum += e;
    }
    const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    const float inv_K = 1.0f / float(K);

    const uint out_base = t * K;
    for (uint k = 0u; k < K; ++k) {
        out_idx[out_base + k] = (int)top_idx[k];
        out_scores[out_base + k] = top_val[k] * inv_sum * inv_K;
    }
}

// ────────────────────────────────────────────────────────────────
// banded_attention  ←  reference: src/native/shaders/banded_attention.wgsl
//
// Per-(t_q, h_q) row of attention restricted to a +/- window of keys.
// One threadgroup per query position-and-head (T*H_q in grid.x).
// Pass 1: per-key dot-product in workgroup memory, find row max.
// Pass 2: exp+sum (softmax denominator).
// Pass 3: normalize scores.
// Pass 4: each thread accumulates one head_dim lane via V.
// ────────────────────────────────────────────────────────────────

struct BandedDims {
    uint T; uint H_q; uint H_kv; uint head_dim;
    uint window; uint use_mask; uint _p0; uint _p1;
};

kernel void banded_attention(
    device const BandedDims&  dims    [[buffer(0)]],
    device const half*        q       [[buffer(1)]],
    device const half*        k       [[buffer(2)]],
    device const half*        v       [[buffer(3)]],
    device const float*       sinks   [[buffer(4)]],
    device const uint*        mask    [[buffer(5)]],
    device       half*        outv    [[buffer(6)]],
    uint3 lid_v [[thread_position_in_threadgroup]],
    uint3 gid   [[threadgroup_position_in_grid]])
{
    constexpr uint MAX_WINDOW_TOTAL = 257u;
    constexpr float NEG_INF = -1e30f;

    threadgroup float wg_scores[MAX_WINDOW_TOTAL];
    threadgroup float wg_tmp[64];
    threadgroup float wg_broadcast[2];

    const uint tid = lid_v.x;
    const uint t_q_h_q = gid.x;
    const uint H_q = dims.H_q;
    if (t_q_h_q >= dims.T * H_q) return;
    const uint t_q = t_q_h_q / H_q;
    const uint h_q = t_q_h_q % H_q;
    const uint head_dim = dims.head_dim;
    const uint kv_group = H_q / dims.H_kv;
    const uint h_kv = h_q / kv_group;

    const uint q_stride_t = H_q * head_dim;
    const uint k_stride_t = dims.H_kv * head_dim;
    const uint q_base = t_q * q_stride_t + h_q * head_dim;

    const uint window = dims.window;
    uint ws = 0u;
    if (t_q > window) ws = t_q - window;
    uint we = dims.T;
    if (t_q + window + 1u < dims.T) we = t_q + window + 1u;
    const uint n_keys = we - ws;

    // Pass 1: scores into workgroup memory + per-thread max.
    float thread_max = NEG_INF;
    for (uint idx = tid; idx < n_keys; idx += 64u) {
        const uint abs_k = ws + idx;
        bool is_valid = true;
        if (dims.use_mask == 1u) {
            const uint word = mask[abs_k >> 2u];
            const uint mb = (word >> ((abs_k & 3u) * 8u)) & 0xFFu;
            is_valid = (mb != 0u);
        }
        float s = NEG_INF;
        if (is_valid) {
            const uint k_base = abs_k * k_stride_t + h_kv * head_dim;
            float dot = 0.0f;
            for (uint d = 0u; d < head_dim; ++d) {
                dot = fma(float(q[q_base + d]), float(k[k_base + d]), dot);
            }
            s = dot;
        }
        wg_scores[idx] = s;
        if (s > thread_max) thread_max = s;
    }
    if (tid == 0u) {
        const float sink_s = sinks[h_q];
        if (sink_s > thread_max) thread_max = sink_s;
    }
    wg_tmp[tid] = thread_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            wg_tmp[tid] = max(wg_tmp[tid], wg_tmp[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) wg_broadcast[0] = wg_tmp[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float row_max = wg_broadcast[0];

    // Pass 2: exp(score - max), accumulate sum.
    float thread_sum = 0.0f;
    for (uint idx = tid; idx < n_keys; idx += 64u) {
        const float e = exp(wg_scores[idx] - row_max);
        wg_scores[idx] = e;
        thread_sum += e;
    }
    if (tid == 0u) {
        thread_sum += exp(sinks[h_q] - row_max);
    }
    wg_tmp[tid] = thread_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            wg_tmp[tid] = wg_tmp[tid] + wg_tmp[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        const float s = wg_tmp[0];
        wg_broadcast[1] = (s > 0.0f) ? (1.0f / s) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_sum = wg_broadcast[1];

    // Pass 3: normalize softmax in-place.
    for (uint idx = tid; idx < n_keys; idx += 64u) {
        wg_scores[idx] = wg_scores[idx] * inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 4: each thread owns one head_dim lane.
    if (tid < head_dim) {
        float acc = 0.0f;
        for (uint i = 0u; i < n_keys; ++i) {
            const uint abs_k = ws + i;
            const uint v_base = abs_k * k_stride_t + h_kv * head_dim;
            acc = fma(wg_scores[i], float(v[v_base + tid]), acc);
        }
        const uint out_base = t_q * q_stride_t + h_q * head_dim;
        outv[out_base + tid] = half(acc);
    }
}

// ────────────────────────────────────────────────────────────────
// qmoe_gate_up  ←  reference: src/native/shaders/qmoe_gate_up.wgsl
//
// Per-(token, k_pick) gated-up int4 matmul: out[tk, n] =
//   bias[expert, n] + sum_d x[t, d] * dequant(W[expert, n, d])
// 2D dispatch: gid.x = tk (T*K), gid.y = N tile (N/64). 64 threads
// per tg, each thread writes one N column. X row loaded cooperatively.
// ────────────────────────────────────────────────────────────────

struct QmoeDims { uint T; uint K; uint N; uint D; uint _p0; uint _p1; uint _p2; uint _p3; };

kernel void qmoe_gate_up(
    device const QmoeDims&    dims         [[buffer(0)]],
    device const float*       x            [[buffer(1)]],
    device const int*         routing_idx  [[buffer(2)]],
    device const uint*        w_int4       [[buffer(3)]],
    device const half*        w_scales     [[buffer(4)]],
    device const uint*        w_zp         [[buffer(5)]],
    device const half*        bias         [[buffer(6)]],
    device       float*       outv         [[buffer(7)]],
    uint3 lid_v [[thread_position_in_threadgroup]],
    uint3 gid   [[threadgroup_position_in_grid]])
{
    constexpr uint BLOCK = 32u;
    constexpr uint D_MAX = 640u;
    constexpr uint WG = 64u;
    threadgroup float x_tile[D_MAX];

    const uint tid = lid_v.x;
    const uint tk = gid.x;
    if (tk >= dims.T * dims.K) return;
    const uint n_base = gid.y * WG;
    const uint n = n_base + tid;

    const uint t = tk / dims.K;
    const uint e = (uint)routing_idx[tk];

    const uint D = dims.D;
    const uint N = dims.N;
    const uint n_blocks = D / BLOCK;
    const uint zp_per_row = (n_blocks + 1u) >> 1u;

    // Cooperative x-row load: D=640, WG=64 → 10 elements per thread.
    for (uint i = 0u; i < 10u; ++i) {
        const uint k_idx = i * WG + tid;
        if (k_idx < D) x_tile[k_idx] = x[t * D + k_idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (n >= N) return;

    const uint expert_nibble_base = e * N * D + n * D;
    const uint expert_scale_base  = e * N * n_blocks + n * n_blocks;
    const uint expert_zp_base     = e * N * zp_per_row + n * zp_per_row;
    const uint expert_bias_base   = e * N + n;

    float acc = 0.0f;
    for (uint b = 0u; b < n_blocks; ++b) {
        const float scale = float(w_scales[expert_scale_base + b]);
        const uint zp_byte_idx = expert_zp_base + (b >> 1u);
        const uint zp_byte = (w_zp[zp_byte_idx >> 2u] >> ((zp_byte_idx & 3u) * 8u)) & 0xFFu;
        const uint zp_nib = (b & 1u) ? ((zp_byte >> 4u) & 0xFu) : (zp_byte & 0xFu);
        const float zp_f = float(zp_nib);

        const uint word_base = (expert_nibble_base + b * BLOCK) >> 3u;
        const uint xb = b * BLOCK;

        float blk = 0.0f;
        for (uint w = 0u; w < 4u; ++w) {
            const uint word = w_int4[word_base + w];
            const uint kb = w * 8u;
            blk = fma(float( word        & 0xFu) - zp_f, x_tile[xb + kb + 0u], blk);
            blk = fma(float((word >>  4u) & 0xFu) - zp_f, x_tile[xb + kb + 1u], blk);
            blk = fma(float((word >>  8u) & 0xFu) - zp_f, x_tile[xb + kb + 2u], blk);
            blk = fma(float((word >> 12u) & 0xFu) - zp_f, x_tile[xb + kb + 3u], blk);
            blk = fma(float((word >> 16u) & 0xFu) - zp_f, x_tile[xb + kb + 4u], blk);
            blk = fma(float((word >> 20u) & 0xFu) - zp_f, x_tile[xb + kb + 5u], blk);
            blk = fma(float((word >> 24u) & 0xFu) - zp_f, x_tile[xb + kb + 6u], blk);
            blk = fma(float((word >> 28u) & 0xFu) - zp_f, x_tile[xb + kb + 7u], blk);
        }
        acc = fma(blk, scale, acc);
    }
    outv[tk * N + n] = acc + float(bias[expert_bias_base]);
}

// ────────────────────────────────────────────────────────────────
// qmoe_down_scatter  ←  reference: src/native/shaders/qmoe_down_scatter.wgsl
//
// Per-(token, k_pick) int4 down-projection scatter-add:
//   out[t, n] += routing_score[tk] * (bias + sum_d glu[tk, d] * dequant(W[e, n, d]))
// Atomic float-add via CAS on a u32-aliased buffer.
// ────────────────────────────────────────────────────────────────

kernel void qmoe_down_scatter(
    device const QmoeDims&        dims            [[buffer(0)]],
    device const float*           glu             [[buffer(1)]],
    device const int*             routing_idx     [[buffer(2)]],
    device const float*           routing_scores  [[buffer(3)]],
    device const uint*            w_int4          [[buffer(4)]],
    device const half*            w_scales        [[buffer(5)]],
    device const uint*            w_zp            [[buffer(6)]],
    device const half*            bias            [[buffer(7)]],
    device       atomic_uint*     acc_out         [[buffer(8)]],
    uint3 lid_v [[thread_position_in_threadgroup]],
    uint3 gid   [[threadgroup_position_in_grid]])
{
    constexpr uint BLOCK = 32u;
    constexpr uint D_MAX = 640u;
    constexpr uint WG = 64u;
    threadgroup float glu_tile[D_MAX];

    const uint tid = lid_v.x;
    const uint tk = gid.x;
    if (tk >= dims.T * dims.K) return;
    const uint n_base = gid.y * WG;
    const uint n = n_base + tid;

    const uint t = tk / dims.K;
    const uint e = (uint)routing_idx[tk];
    const float w_route = routing_scores[tk];

    const uint D = dims.D;
    const uint N = dims.N;
    const uint n_blocks = D / BLOCK;
    const uint zp_per_row = (n_blocks + 1u) >> 1u;

    for (uint i = 0u; i < 10u; ++i) {
        const uint k_idx = i * WG + tid;
        if (k_idx < D) glu_tile[k_idx] = glu[tk * D + k_idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (n >= N) return;

    const uint expert_nibble_base = e * N * D + n * D;
    const uint expert_scale_base  = e * N * n_blocks + n * n_blocks;
    const uint expert_zp_base     = e * N * zp_per_row + n * zp_per_row;
    const uint expert_bias_base   = e * N + n;

    float acc_local = 0.0f;
    for (uint b = 0u; b < n_blocks; ++b) {
        const float scale = float(w_scales[expert_scale_base + b]);
        const uint zp_byte_idx = expert_zp_base + (b >> 1u);
        const uint zp_byte = (w_zp[zp_byte_idx >> 2u] >> ((zp_byte_idx & 3u) * 8u)) & 0xFFu;
        const uint zp_nib = (b & 1u) ? ((zp_byte >> 4u) & 0xFu) : (zp_byte & 0xFu);
        const float zp_f = float(zp_nib);

        const uint word_base = (expert_nibble_base + b * BLOCK) >> 3u;
        const uint xb = b * BLOCK;

        float blk = 0.0f;
        for (uint wi = 0u; wi < 4u; ++wi) {
            const uint word = w_int4[word_base + wi];
            const uint kb = wi * 8u;
            blk = fma(float( word        & 0xFu) - zp_f, glu_tile[xb + kb + 0u], blk);
            blk = fma(float((word >>  4u) & 0xFu) - zp_f, glu_tile[xb + kb + 1u], blk);
            blk = fma(float((word >>  8u) & 0xFu) - zp_f, glu_tile[xb + kb + 2u], blk);
            blk = fma(float((word >> 12u) & 0xFu) - zp_f, glu_tile[xb + kb + 3u], blk);
            blk = fma(float((word >> 16u) & 0xFu) - zp_f, glu_tile[xb + kb + 4u], blk);
            blk = fma(float((word >> 20u) & 0xFu) - zp_f, glu_tile[xb + kb + 5u], blk);
            blk = fma(float((word >> 24u) & 0xFu) - zp_f, glu_tile[xb + kb + 6u], blk);
            blk = fma(float((word >> 28u) & 0xFu) - zp_f, glu_tile[xb + kb + 7u], blk);
        }
        acc_local = fma(blk, scale, acc_local);
    }
    acc_local = acc_local + float(bias[expert_bias_base]);
    const float contrib = w_route * acc_local;

    // Atomic float-add via CAS on u32-aliased atomic.
    device atomic_uint* slot = &acc_out[t * N + n];
    uint old_u = atomic_load_explicit(slot, memory_order_relaxed);
    while (true) {
        const float new_f = as_type<float>(old_u) + contrib;
        const uint  new_u = as_type<uint>(new_f);
        if (atomic_compare_exchange_weak_explicit(slot, &old_u, new_u,
                memory_order_relaxed, memory_order_relaxed)) break;
    }
}
