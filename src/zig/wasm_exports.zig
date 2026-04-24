// WASM32 exports for the pii-wasm Stage 1 inference engine.
//
// This module is the sole compilation unit. We target
// `wasm32-freestanding` with `simd128` + `relaxed_simd` CPU features —
// see `package.json` → `build:zig`. The JS bridge lives at
// `src/js/backends/wasm.ts`.
//
// Current contents: Phase A scaffolding only — one real allocator, one
// echo function, one bump-allocator-like reset. Real kernels land in
// later phases (see `docs/roadmap.md`).
//
// Memory model:
// - Linear memory is WASM's default single page + grow-on-demand.
// - A single 16-byte-aligned bump allocator hands out buffers to JS
//   for ONNX input/output staging. `reset()` rewinds the bump cursor.
// - No free-list and no heap. Every `redact()` call begins by calling
//   `reset()` — model weights live above the reset boundary.
//
// The bump allocator is enough for Phase A+B+C (fixed-size kernel
// scratch buffers). Phase D will revisit if attention scratch grows
// non-trivially with sequence length.

const std = @import("std");
const math = @import("math.zig");
// Activations + biases + norms are all fp16 throughout. matches what
// ORT Web runs for `model_q4f16.onnx` — same 10-bit mantissa, same
// decision boundaries.
const fp16ToF32 = math.fp16ToF32;
const f32ToFp16 = math.f32ToFp16;
const fp16x4ToF32x4 = math.fp16x4ToF32x4;
const alignUp = math.alignUp;
const fma4 = math.relaxed_madd_f32x4;

// --------------------------------------------------------------
// Bump allocator
// --------------------------------------------------------------
//
// Two-zone bump layout: a permanent region at the bottom holds model
// weights (loaded once per session), a scratch region above it holds
// per-forward-pass buffers. `reset()` rewinds only the scratch region,
// preserving the weights.
//
// Invariant:  heap_base <= heap_mark <= heap_end
//   heap_base  bottom of heap — set from wasm-ld's `__heap_base`
//              symbol, which marks the end of the statics/stack region.
//              Hard-coding to a constant (e.g. 64 KiB) is wrong because
//              Zig's linker places globals + shadow stack anywhere up
//              to ~1 MiB.
//   heap_mark  boundary between permanent region (below) and scratch (above).
//              JS calls `heap_mark_now()` after loading weights so subsequent
//              `reset()` calls keep the blob intact.
//   heap_end   current top of bump.

const WASM_PAGE = 64 * 1024;
const HEAP_ALIGN = 16;

// Provided by wasm-ld. Its *address* (not its value) is the first byte
// of heap memory safe to hand out.
extern const __heap_base: u8;

var heap_base: usize = 0;
var heap_mark: usize = 0;
var heap_end: usize = 0;

fn wasmPages() usize {
    return @wasmMemorySize(0);
}

fn growBy(pages: usize) bool {
    const prev = @wasmMemoryGrow(0, pages);
    return prev != std.math.maxInt(usize);
}

/// Initialize the heap. Idempotent — safe to call any number of times.
/// Explicit call is no longer required; `alloc`/`reset`/`heap_mark_now`
/// all lazy-init on first use. Kept as an export so JS can force init
/// early (e.g. for tests that instantiate the module directly).
/// Returns the byte offset the bump allocator starts handing out from.
export fn heap_init() usize {
    ensureHeapInit();
    return heap_base;
}

inline fn ensureHeapInit() void {
    if (heap_base == 0) {
        // Round the linker-provided `__heap_base` up to our alignment.
        heap_base = alignUp(@intFromPtr(&__heap_base), HEAP_ALIGN);
        heap_mark = heap_base;
        heap_end = heap_base;
    }
}

/// Allocate `n` bytes, 16-byte aligned. Returns a pointer (offset) into
/// linear memory. Returns 0 on OOM (memory growth failure).
export fn alloc(n: usize) usize {
    ensureHeapInit();
    const start = alignUp(heap_end, HEAP_ALIGN);
    const new_end = start + n;

    const needed_bytes = new_end;
    const have_bytes = wasmPages() * WASM_PAGE;
    if (needed_bytes > have_bytes) {
        const extra_bytes = needed_bytes - have_bytes;
        const extra_pages = (extra_bytes + WASM_PAGE - 1) / WASM_PAGE;
        if (!growBy(extra_pages)) return 0;
    }

    heap_end = new_end;
    return start;
}

/// Pin the current heap top as the new baseline for `reset()`. Called
/// once after the weight blob is loaded; every subsequent `reset()`
/// will free only the scratch above it. Safe to call repeatedly — the
/// mark only moves up.
export fn heap_mark_now() void {
    ensureHeapInit();
    if (heap_end > heap_mark) heap_mark = heap_end;
}

/// Rewind the scratch region. Weights loaded below `heap_mark` survive.
export fn reset() void {
    ensureHeapInit();
    heap_end = heap_mark;
}

/// Bytes of permanent (below-mark) memory — useful for verifying that
/// weights loaded successfully.
export fn heap_permanent() usize {
    return heap_mark - heap_base;
}

/// Bytes of scratch in use right now (above the mark).
export fn heap_used() usize {
    return heap_end - heap_mark;
}

/// DEBUG — address of `heap_end` in linear memory. If any alloc
/// returns a pointer below this, the write WILL trash the allocator
/// state. Expose so JS can sanity-check heap layout before trusting
/// kernel output.
export fn debug_heap_end_addr() usize {
    return @intFromPtr(&heap_end);
}

export fn debug_heap_end_value() usize {
    return heap_end;
}

// --------------------------------------------------------------
// Pipeline plumbing smoke test
// --------------------------------------------------------------

/// Identity function — JS side uses this to verify the WASM module
/// loaded and the ABI works before exercising real kernels.
export fn echo(x: i32) i32 {
    return x;
}

/// Sum of an int32 slice at the given offset. Two purposes: smoke-test
/// that we can pass a buffer from JS into WASM memory, and sanity-check
/// that SIMD codegen compiles at all (Zig auto-vectorizes this with
/// simd128 enabled).
export fn sum_i32(ptr: [*]const i32, len: usize) i32 {
    var acc: i32 = 0;
    var i: usize = 0;
    // Wrapping add: smoke-test input is arbitrary, overflow is not an error.
    while (i < len) : (i += 1) acc +%= ptr[i];
    return acc;
}

// --------------------------------------------------------------
// Kernel: RMSNorm
// --------------------------------------------------------------
//
// y[t, d] = x[t, d] * gamma[d] / sqrt(mean_d(x[t, :]^2) + eps)
//
// x:     fp16 [T, D]      row-major
// gamma: fp16 [D]
// out:   fp16 [T, D]      row-major
//
// Sum-of-squares accumulates in f32 for numerical stability (640-wide
// fp16 rows would lose precision if accumulated in fp16).
// The upstream model uses `rms_norm_eps = 1e-5` everywhere.

export fn rms_norm(
    x_ptr: [*]const u16,
    gamma_ptr: [*]const u16,
    out_ptr: [*]u16,
    T: u32,
    D: u32,
    eps: f32,
) void {
    const Tz: usize = T;
    const Dz: usize = D;
    const d_inv = 1.0 / @as(f32, @floatFromInt(Dz));

    var t: usize = 0;
    while (t < Tz) : (t += 1) {
        const row_base = t * Dz;

        var sumsq: f32 = 0.0;
        var d: usize = 0;
        while (d < Dz) : (d += 1) {
            const v = fp16ToF32(x_ptr[row_base + d]);
            sumsq += v * v;
        }

        const inv_rms: f32 = 1.0 / @sqrt(sumsq * d_inv + eps);

        d = 0;
        while (d < Dz) : (d += 1) {
            const xv = fp16ToF32(x_ptr[row_base + d]);
            const gv = fp16ToF32(gamma_ptr[d]);
            out_ptr[row_base + d] = f32ToFp16(xv * inv_rms * gv);
        }
    }
}


// --------------------------------------------------------------
// Kernel: row-wise top-k (partial selection)
// --------------------------------------------------------------
//
// For each row of `x: [rows, cols]`, produce the indices and values of
// the `k` largest entries, written into `out_idx` and `out_val`.
// Ordering within the k outputs is "largest first" — convenient for
// softmax-then-divide in the router where order doesn't matter but
// determinism does.
//
// Uses a fixed-size incremental insertion sort of k elements. For
// k=4 and n=128 (router), this is O(n*k) = 512 compares per row,
// which is faster than a heap for small k.

export fn topk_partial_f32(
    x_ptr: [*]const f32,
    out_idx: [*]i32,
    out_val: [*]f32,
    rows: u32,
    cols: u32,
    k: u32,
) void {
    const Rz: usize = rows;
    const Cz: usize = cols;
    const Kz: usize = k;

    var r: usize = 0;
    while (r < Rz) : (r += 1) {
        const row_base = r * Cz;
        const out_base = r * Kz;

        // Seed with the first k entries in input order, then sort
        // descending.
        var i: usize = 0;
        while (i < Kz) : (i += 1) {
            out_idx[out_base + i] = @intCast(i);
            out_val[out_base + i] = x_ptr[row_base + i];
        }
        // Descending insertion sort on the seed k elements.
        var s: usize = 1;
        while (s < Kz) : (s += 1) {
            var j: usize = s;
            while (j > 0 and out_val[out_base + j] > out_val[out_base + j - 1]) : (j -= 1) {
                const tv = out_val[out_base + j];
                out_val[out_base + j] = out_val[out_base + j - 1];
                out_val[out_base + j - 1] = tv;
                const ti = out_idx[out_base + j];
                out_idx[out_base + j] = out_idx[out_base + j - 1];
                out_idx[out_base + j - 1] = ti;
            }
        }

        // Scan the rest. If new value beats the smallest (last) slot,
        // insert and bubble up.
        var c: usize = Kz;
        while (c < Cz) : (c += 1) {
            const v = x_ptr[row_base + c];
            if (v > out_val[out_base + Kz - 1]) {
                // Replace last, bubble up.
                out_val[out_base + Kz - 1] = v;
                out_idx[out_base + Kz - 1] = @intCast(c);
                var j: usize = Kz - 1;
                while (j > 0 and out_val[out_base + j] > out_val[out_base + j - 1]) : (j -= 1) {
                    const tv = out_val[out_base + j];
                    out_val[out_base + j] = out_val[out_base + j - 1];
                    out_val[out_base + j - 1] = tv;
                    const ti = out_idx[out_base + j];
                    out_idx[out_base + j] = out_idx[out_base + j - 1];
                    out_idx[out_base + j - 1] = ti;
                }
            }
        }
    }
}

// --------------------------------------------------------------
// Kernel: fp16 × int4-blockwise matmul with bias
// --------------------------------------------------------------
//
// out = x @ dequant(W).T + bias     // PyTorch F.linear convention
//   x:    fp16                [T, D]
//   W:    int4 blockwise sym  [N, D]      (logical shape)
//   bias: fp16                [N]
//   out:  fp16                [T, N]
//
// Weight encoding (matches the "int4_block32_sym" dtype from
// scripts/convert-weights.py):
//   - Block size 32 along D.
//   - Signed int4 values, packed two per byte (low nibble = even d,
//     high nibble = odd d), layout [N, D/2].
//   - Per-block fp16 scale, layout [N, D/32], appended immediately
//     after the packed data in the same tensor blob.
//   - Dequant: x_fp = int4_sign_extend(q) * fp16_to_f32(scale).
//     No zero-point (scale set so the most-negative int4 value -8
//     is not required — max_abs maps to +7).
//
// Accumulate semantics:
//   per block: 4-wide SIMD load of 4 x-fp16 lanes + extract 4 int4
//     nibbles from 2 bytes, upcast both to f32, multiply. Drain lanes
//     into a scalar `block_sum` so the per-block sum order matches a
//     reference implementation. Then `acc += block_sum * scale_f32`.
//   across blocks: scalar accumulate into `acc` (each block has its
//     own scale; can't hoist).
// Final `acc + bias` rounded to fp16 via RNE.

inline fn unpackInt4Lo(b: u8) i8 {
    const n: u4 = @truncate(b);
    return @as(i8, @bitCast(@as(i8, @as(i4, @bitCast(n)))));
}

inline fn unpackInt4Hi(b: u8) i8 {
    const n: u4 = @truncate(b >> 4);
    return @as(i8, @bitCast(@as(i8, @as(i4, @bitCast(n)))));
}

/// Load 4 consecutive int4 values (2 packed bytes) as a 4-lane f32 vector.
/// Used inside the D-axis inner loop; each call advances by 4 elements.
inline fn int4x4LoadF32(w_int4: [*]const u8, nibble_index: usize) @Vector(4, f32) {
    // nibble_index is the index of the FIRST int4 element we want.
    // It must be even (4-element chunks start on byte boundaries).
    const byte_index = nibble_index >> 1;
    const b0 = w_int4[byte_index];
    const b1 = w_int4[byte_index + 1];
    const q: @Vector(4, i32) = .{
        @as(i32, unpackInt4Lo(b0)),
        @as(i32, unpackInt4Hi(b0)),
        @as(i32, unpackInt4Lo(b1)),
        @as(i32, unpackInt4Hi(b1)),
    };
    // i32 → f32 elementwise. `@floatFromInt` on vectors works.
    return @floatFromInt(q);
}

/// Load 4 elements of an X-row as an f32 vector. `XType = u16` treats
/// the source as fp16 and widens; `XType = f32` is a direct load.
inline fn loadX4(comptime XType: type, base: [*]const XType, offset: usize) @Vector(4, f32) {
    if (XType == u16) {
        const xu: @Vector(4, u16) = base[offset..][0..4].*;
        return fp16x4ToF32x4(xu);
    } else if (XType == f32) {
        return base[offset..][0..4].*;
    } else @compileError("loadX4: unsupported XType");
}

/// Store an f32 accumulator lane to out. `OutType = u16` rounds to fp16;
/// `OutType = f32` is a direct store.
inline fn storeOut(comptime OutType: type, out: [*]OutType, idx: usize, val: f32) void {
    if (OutType == u16) {
        out[idx] = f32ToFp16(val);
    } else if (OutType == f32) {
        out[idx] = val;
    } else @compileError("storeOut: unsupported OutType");
}

/// Decode 32 int4 values at `block_d` of row `w_int4_row` into eight
/// `@Vector(4, f32)` chunks covering the block's 32 lanes.
/// ONNX MatMulNBits stores weights as UINT4 (range 0..15); dequant
/// requires `(q - zp) * scale`. `zp` is the block's zero-point (also
/// UINT4, range 0..15); passing 0 yields a symmetric decode.
inline fn int4BlockDecode(
    w_int4_row: [*]const u8,
    block_d: usize,
    zp: u8,
) [8]@Vector(4, f32) {
    const packed_u: @Vector(16, u8) = w_int4_row[block_d / 2 ..][0..16].*;
    const lo_u: @Vector(16, u8) = packed_u & @as(@Vector(16, u8), @splat(0x0F));
    const hi_u: @Vector(16, u8) = (packed_u >> @splat(4)) & @as(@Vector(16, u8), @splat(0x0F));
    const zp_v: @Vector(16, u8) = @splat(zp);
    // Subtract zp in i16 space to allow negative results (range -15..15).
    const lo: @Vector(16, i16) = @as(@Vector(16, i16), lo_u) - @as(@Vector(16, i16), zp_v);
    const hi: @Vector(16, i16) = @as(@Vector(16, i16), hi_u) - @as(@Vector(16, i16), zp_v);
    var out: [8]@Vector(4, f32) = undefined;
    inline for (0..8) |chunk| {
        // Lane `chunk * 4 + c` in block corresponds to nibble `c` of
        // byte `chunk * 2 + c/2` if c is even, or the hi nibble if odd.
        const base_byte = chunk * 2;
        const slice: @Vector(4, i16) = @shuffle(
            i16, lo, hi,
            @Vector(4, i32){
                base_byte,          ~@as(i32, base_byte),
                base_byte + 1,      ~@as(i32, base_byte + 1),
            },
        );
        const widened: @Vector(4, i32) = slice;
        out[chunk] = @floatFromInt(widened);
    }
    return out;
}

/// Extract a 4-bit zero-point value for block `b` from a packed
/// `[N, ceil(nblocks / 2)]` byte buffer. Even blocks go in the low
/// nibble, odd blocks in the high nibble.
inline fn extractZp(zp_row: [*]const u8, b: usize) u8 {
    const byte = zp_row[b >> 1];
    return if ((b & 1) == 0) byte & 0x0F else (byte >> 4) & 0x0F;
}

/// Comptime-parameterized int4-block matmul body. Instantiated by the
/// three public exports (fp16→fp16, fp16→f32, f32→f32). Three variants
/// × ~100 lines each are worth a controlled monomorphization; we pay
/// the specialization cost exactly three times.
///
/// ONNX MatMulNBits layout:
///   w_int4_ptr:  uint4 weights packed 2-per-byte, shape [N, K/2]
///   w_scales_ptr: fp16 scales,                    shape [N, K/block]
///   w_zp_ptr:    uint4 zero-points packed 2-per-byte, shape [N, ceil(K/block/2)]
///                Pass null/0 for symmetric-decode (treats zp = 0).
inline fn matmulInt4BlockImpl(
    comptime XType: type,
    comptime OutType: type,
    x_ptr: [*]const XType,
    w_int4_ptr: [*]const u8,
    w_scales_ptr: [*]const u16,
    w_zp_ptr: ?[*]const u8,
    bias_ptr: [*]const u16,
    out_ptr: [*]OutType,
    T: u32,
    N: u32,
    D: u32,
) void {
    const Tz: usize = T;
    const Nz: usize = N;
    const Dz: usize = D;
    const BLOCK: usize = 32;
    const TR: usize = 8;

    const w_int4_stride: usize = Dz / 2;
    const w_scale_stride: usize = Dz / BLOCK;
    const w_zp_stride: usize = (w_scale_stride + 1) / 2;
    const w_ptr = w_int4_ptr;
    const scales_base: [*]const u16 = w_scales_ptr;

    // TR-tiled T loop: amortize one int4 decode across 8 rows of X
    // sharing the same W row. 8 × 32 = 256 MACs per decode, up from
    // 128 with TR=4. Single accumulator per T row (no ILP×2) keeps
    // the SIMD register set within wasm's 16-register budget.
    var tt: usize = 0;
    while (tt + TR <= Tz) : (tt += TR) {
        var n: usize = 0;
        while (n < Nz) : (n += 1) {
            const w_int4_row = w_ptr + n * w_int4_stride;
            const w_scale_row = scales_base + n * w_scale_stride;
            const w_zp_row: ?[*]const u8 = if (w_zp_ptr) |p| p + n * w_zp_stride else null;

            var sum: @Vector(8, f32) = @splat(0);
            var b: usize = 0;
            while (b < w_scale_stride) : (b += 1) {
                const scale = fp16ToF32(w_scale_row[b]);
                const block_d = b * BLOCK;
                const zp: u8 = if (w_zp_row) |p| extractZp(p, b) else 0;

                const q_chunks: [8]@Vector(4, f32) = int4BlockDecode(w_int4_row, block_d, zp);

                var v0: @Vector(4, f32) = @splat(0);
                var v1: @Vector(4, f32) = @splat(0);
                var v2: @Vector(4, f32) = @splat(0);
                var v3: @Vector(4, f32) = @splat(0);
                var v4: @Vector(4, f32) = @splat(0);
                var v5: @Vector(4, f32) = @splat(0);
                var v6: @Vector(4, f32) = @splat(0);
                var v7: @Vector(4, f32) = @splat(0);

                inline for (0..4) |k| {
                    const j = k * 8;
                    const qf0 = q_chunks[k * 2];
                    const qf1 = q_chunks[k * 2 + 1];
                    v0 = fma4(loadX4(XType, x_ptr, (tt + 0) * Dz + block_d + j),     qf0, v0);
                    v0 = fma4(loadX4(XType, x_ptr, (tt + 0) * Dz + block_d + j + 4), qf1, v0);
                    v1 = fma4(loadX4(XType, x_ptr, (tt + 1) * Dz + block_d + j),     qf0, v1);
                    v1 = fma4(loadX4(XType, x_ptr, (tt + 1) * Dz + block_d + j + 4), qf1, v1);
                    v2 = fma4(loadX4(XType, x_ptr, (tt + 2) * Dz + block_d + j),     qf0, v2);
                    v2 = fma4(loadX4(XType, x_ptr, (tt + 2) * Dz + block_d + j + 4), qf1, v2);
                    v3 = fma4(loadX4(XType, x_ptr, (tt + 3) * Dz + block_d + j),     qf0, v3);
                    v3 = fma4(loadX4(XType, x_ptr, (tt + 3) * Dz + block_d + j + 4), qf1, v3);
                    v4 = fma4(loadX4(XType, x_ptr, (tt + 4) * Dz + block_d + j),     qf0, v4);
                    v4 = fma4(loadX4(XType, x_ptr, (tt + 4) * Dz + block_d + j + 4), qf1, v4);
                    v5 = fma4(loadX4(XType, x_ptr, (tt + 5) * Dz + block_d + j),     qf0, v5);
                    v5 = fma4(loadX4(XType, x_ptr, (tt + 5) * Dz + block_d + j + 4), qf1, v5);
                    v6 = fma4(loadX4(XType, x_ptr, (tt + 6) * Dz + block_d + j),     qf0, v6);
                    v6 = fma4(loadX4(XType, x_ptr, (tt + 6) * Dz + block_d + j + 4), qf1, v6);
                    v7 = fma4(loadX4(XType, x_ptr, (tt + 7) * Dz + block_d + j),     qf0, v7);
                    v7 = fma4(loadX4(XType, x_ptr, (tt + 7) * Dz + block_d + j + 4), qf1, v7);
                }

                const block_contribs: @Vector(8, f32) = .{
                    (v0[0] + v0[1]) + (v0[2] + v0[3]),
                    (v1[0] + v1[1]) + (v1[2] + v1[3]),
                    (v2[0] + v2[1]) + (v2[2] + v2[3]),
                    (v3[0] + v3[1]) + (v3[2] + v3[3]),
                    (v4[0] + v4[1]) + (v4[2] + v4[3]),
                    (v5[0] + v5[1]) + (v5[2] + v5[3]),
                    (v6[0] + v6[1]) + (v6[2] + v6[3]),
                    (v7[0] + v7[1]) + (v7[2] + v7[3]),
                };
                sum += block_contribs * @as(@Vector(8, f32), @splat(scale));
            }

            const b_val = fp16ToF32(bias_ptr[n]);
            storeOut(OutType, out_ptr, (tt + 0) * Nz + n, sum[0] + b_val);
            storeOut(OutType, out_ptr, (tt + 1) * Nz + n, sum[1] + b_val);
            storeOut(OutType, out_ptr, (tt + 2) * Nz + n, sum[2] + b_val);
            storeOut(OutType, out_ptr, (tt + 3) * Nz + n, sum[3] + b_val);
            storeOut(OutType, out_ptr, (tt + 4) * Nz + n, sum[4] + b_val);
            storeOut(OutType, out_ptr, (tt + 5) * Nz + n, sum[5] + b_val);
            storeOut(OutType, out_ptr, (tt + 6) * Nz + n, sum[6] + b_val);
            storeOut(OutType, out_ptr, (tt + 7) * Nz + n, sum[7] + b_val);
        }
    }

    // T tail for T not divisible by TR=8 (hot for expert dispatch — each
    // expert sees m ∈ {1..K×T/E} tokens, often 1-6). Walk blocks in the
    // outer loop so one decode serves all m_tail tokens instead of
    // re-decoding per token. Accumulators are a small stack array
    // (m_tail ≤ TR-1 = 7).
    const m_tail: usize = Tz - tt;
    if (m_tail > 0) {
        var n: usize = 0;
        while (n < Nz) : (n += 1) {
            const w_int4_row = w_ptr + n * w_int4_stride;
            const w_scale_row = scales_base + n * w_scale_stride;
            const w_zp_row: ?[*]const u8 = if (w_zp_ptr) |p| p + n * w_zp_stride else null;

            var accs: [TR - 1]f32 = .{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            var b: usize = 0;
            while (b < w_scale_stride) : (b += 1) {
                const scale = fp16ToF32(w_scale_row[b]);
                const block_d = b * BLOCK;
                const zp: u8 = if (w_zp_row) |p| extractZp(p, b) else 0;
                const q_chunks = int4BlockDecode(w_int4_row, block_d, zp);

                var ti: usize = 0;
                while (ti < m_tail) : (ti += 1) {
                    var a0: @Vector(4, f32) = @splat(0);
                    var a1: @Vector(4, f32) = @splat(0);
                    inline for (0..4) |k| {
                        const j = k * 8;
                        const xu0 = loadX4(XType, x_ptr, (tt + ti) * Dz + block_d + j);
                        const xu1 = loadX4(XType, x_ptr, (tt + ti) * Dz + block_d + j + 4);
                        a0 = fma4(xu0, q_chunks[k * 2],     a0);
                        a1 = fma4(xu1, q_chunks[k * 2 + 1], a1);
                    }
                    const combined: @Vector(4, f32) = a0 + a1;
                    accs[ti] += ((combined[0] + combined[1]) + (combined[2] + combined[3])) * scale;
                }
            }

            const b_val = fp16ToF32(bias_ptr[n]);
            var ti: usize = 0;
            while (ti < m_tail) : (ti += 1) {
                storeOut(OutType, out_ptr, (tt + ti) * Nz + n, accs[ti] + b_val);
            }
        }
    }
}

/// fp16 x × int4-blockwise W + bias → fp16 out. ONNX MatMulNBits
/// semantics: uint4 weights, (q - zp) * scale dequant.
/// Pass `w_zp_ptr = 0` for symmetric-decode (treats zp = 0 for all blocks).
export fn matmul_fp16_x_int4block(
    x_ptr: [*]const u16,
    w_int4_ptr: [*]const u8,
    w_scales_ptr: [*]const u16,
    w_zp_ptr: ?[*]const u8,
    bias_ptr: [*]const u16,
    out_ptr: [*]u16,
    T: u32, N: u32, D: u32,
) void {
    matmulInt4BlockImpl(u16, u16, x_ptr, w_int4_ptr, w_scales_ptr, w_zp_ptr, bias_ptr, out_ptr, T, N, D);
}

/// fp16 x × int4-blockwise W + bias → f32 out. Used for the MoE gate_up
/// matmul — upstream expert forward keeps the chain in fp32.
export fn matmul_fp16_x_int4block_out_f32(
    x_ptr: [*]const u16,
    w_int4_ptr: [*]const u8,
    w_scales_ptr: [*]const u16,
    w_zp_ptr: ?[*]const u8,
    bias_ptr: [*]const u16,
    out_ptr: [*]f32,
    T: u32, N: u32, D: u32,
) void {
    matmulInt4BlockImpl(u16, f32, x_ptr, w_int4_ptr, w_scales_ptr, w_zp_ptr, bias_ptr, out_ptr, T, N, D);
}

/// f32 x × int4-blockwise W + bias → f32 out. Used for the MoE down_proj
/// matmul — the SwiGLU output feeding it is fp32.
export fn matmul_f32_x_int4block_out_f32(
    x_ptr: [*]const f32,
    w_int4_ptr: [*]const u8,
    w_scales_ptr: [*]const u16,
    w_zp_ptr: ?[*]const u8,
    bias_ptr: [*]const u16,
    out_ptr: [*]f32,
    T: u32, N: u32, D: u32,
) void {
    matmulInt4BlockImpl(f32, f32, x_ptr, w_int4_ptr, w_scales_ptr, w_zp_ptr, bias_ptr, out_ptr, T, N, D);
}

/// f32 x × int4-blockwise W + bias → fp16 out. Used by attention Q/K/V/O
/// and the classifier head, where callers pre-widen x once and use this
/// kernel to skip the per-load fp16 → f32 conversion in the inner loop.
export fn matmul_f32_x_int4block(
    x_ptr: [*]const f32,
    w_int4_ptr: [*]const u8,
    w_scales_ptr: [*]const u16,
    w_zp_ptr: ?[*]const u8,
    bias_ptr: [*]const u16,
    out_ptr: [*]u16,
    T: u32, N: u32, D: u32,
) void {
    matmulInt4BlockImpl(f32, u16, x_ptr, w_int4_ptr, w_scales_ptr, w_zp_ptr, bias_ptr, out_ptr, T, N, D);
}

// --------------------------------------------------------------
// Kernel: SwiGLU with clamp (privacy-filter / ORT QMoE variant)
// --------------------------------------------------------------
//
// Matches ORT's QMoE `swiglu_fusion=1` and upstream `swiglu(packed=True)`:
//   gate = gate_up[..., 0::2]       # even indices
//   up   = gate_up[..., 1::2]       # odd indices
//   gate = gate.clamp(max=7.0)
//   up   = up.clamp(-7.0, 7.0)
//   glu  = gate * sigmoid(gate * 1.702)
//   out  = (up + 1.0) * glu
//
// All compute in f32 — the upstream expert forward runs in fp32
// throughout (`.float()` on every op). Input and output are f32.
//
// gate_up: [T, 2*D]   f32       (packed: (gate[0], up[0], gate[1], up[1], …))
// out:     [T, D]     f32
//
// Note on the 1.702 constant: this is the GELU-approximation scale
// `α = 1.702` that makes `x * sigmoid(α * x)` ≈ gelu(x). Combined with
// the `+1` on the up branch and the clamps, this is the privacy-filter
// specific gate (differs from vanilla SwiGLU's `silu(gate) * up`).

const SWIGLU_LIMIT: f32 = 7.0;
const SWIGLU_ALPHA: f32 = 1.702;

inline fn sigmoidF32(x: f32) f32 {
    // Numerically stable: for positive x, σ(x) = 1/(1+exp(-x)); for
    // negative x, σ(x) = exp(x)/(1+exp(x)). Keeps the exp argument ≤ 0
    // so intermediate values can't overflow.
    if (x >= 0.0) {
        const e = @exp(-x);
        return 1.0 / (1.0 + e);
    }
    const e = @exp(x);
    return e / (1.0 + e);
}

/// 4-lane approximate exp for non-positive arguments (softmax /
/// sigmoid inputs after the row-max subtraction / sign flip). Uses a
/// range reduction `exp(x) = 2^(x * log2(e))` plus a 5-term
/// polynomial for `2^f, f ∈ [0, 1)`. Max relative error ≈ 3×10⁻⁷ on
/// x ∈ [-40, 0]; errors are symmetric around lanes so the softmax
/// weights stay a normalisation apart.
inline fn expApproxNonPositiveF32x4(x: @Vector(4, f32)) @Vector(4, f32) {
    const LOG2_E: @Vector(4, f32) = @splat(1.44269504088896340736);
    // Clamp to a safe range: exp(-90) ≈ 0 in f32.
    const LO: @Vector(4, f32) = @splat(-88.0);
    const clamped = @max(x, LO);
    const y = clamped * LOG2_E; // y ≤ 0
    // int = floor(y); frac = y - int, frac ∈ [0, 1)
    const int_v: @Vector(4, i32) = @intFromFloat(@floor(y));
    const frac: @Vector(4, f32) = y - @as(@Vector(4, f32), @floatFromInt(int_v));
    // Horner polynomial for 2^frac: coefficients from a minimax fit.
    const c0: @Vector(4, f32) = @splat(1.000000012);
    const c1: @Vector(4, f32) = @splat(0.693154774);
    const c2: @Vector(4, f32) = @splat(0.240226522);
    const c3: @Vector(4, f32) = @splat(0.055053319);
    const c4: @Vector(4, f32) = @splat(0.009679494);
    const c5: @Vector(4, f32) = @splat(0.001333355);
    var poly = fma4(c5, frac, c4);
    poly = fma4(poly, frac, c3);
    poly = fma4(poly, frac, c2);
    poly = fma4(poly, frac, c1);
    poly = fma4(poly, frac, c0);
    // Scale by 2^int via bit twiddling: build f32 with exponent bias 127+int.
    const bias: @Vector(4, i32) = @splat(127);
    const exp_bits: @Vector(4, i32) = (int_v + bias) << @splat(23);
    const scale: @Vector(4, f32) = @bitCast(exp_bits);
    return poly * scale;
}

/// 4-lane sigmoid using the approximate exp above. Takes any real x;
/// produces values in [0, 1]. Worst-case error ≈ 10⁻⁶, plenty for
/// SwiGLU's gate which is itself an activation not a softmax
/// normaliser.
inline fn sigmoidF32x4(x: @Vector(4, f32)) @Vector(4, f32) {
    const ZERO: @Vector(4, f32) = @splat(0.0);
    const ONE: @Vector(4, f32) = @splat(1.0);
    const neg_abs: @Vector(4, f32) = @min(x, ZERO) - @max(x, ZERO); // = -|x|
    const e = expApproxNonPositiveF32x4(neg_abs);
    const sig_neg = e / (ONE + e); // σ(-|x|)
    // σ(x) = sig_neg when x < 0, else 1 - sig_neg.
    const pos_mask = x >= ZERO;
    return @select(f32, pos_mask, ONE - sig_neg, sig_neg);
}

export fn swiglu_clamp_f32(
    gate_up_ptr: [*]const f32,
    out_ptr: [*]f32,
    T: u32,
    D: u32,
) void {
    const Tz: usize = T;
    const Dz: usize = D;
    const stride: usize = 2 * Dz;
    const LIMIT_VEC: @Vector(4, f32) = @splat(SWIGLU_LIMIT);
    const NEG_LIMIT_VEC: @Vector(4, f32) = @splat(-SWIGLU_LIMIT);
    const ALPHA_VEC: @Vector(4, f32) = @splat(SWIGLU_ALPHA);
    const ONE_VEC: @Vector(4, f32) = @splat(1.0);

    var t: usize = 0;
    while (t < Tz) : (t += 1) {
        const row_base = t * stride;
        const out_base = t * Dz;

        // 4-at-a-time SIMD loop: load two v128s of interleaved (gate, up)
        // pairs, shuffle-deinterleave, clamp in SIMD; sigmoid stays
        // scalar-per-lane (wasm has no SIMD exp).
        var d: usize = 0;
        while (d + 4 <= Dz) : (d += 4) {
            const v0: @Vector(4, f32) = gate_up_ptr[row_base + 2 * d ..][0..4].*;
            const v1: @Vector(4, f32) = gate_up_ptr[row_base + 2 * d + 4 ..][0..4].*;
            const gates_raw = @shuffle(f32, v0, v1, @Vector(4, i32){ 0, 2, -1, -3 });
            const ups_raw   = @shuffle(f32, v0, v1, @Vector(4, i32){ 1, 3, -2, -4 });
            const gates = @min(gates_raw, LIMIT_VEC);
            const ups = @max(@min(ups_raw, LIMIT_VEC), NEG_LIMIT_VEC);
            const scaled = gates * ALPHA_VEC;
            const sigs = sigmoidF32x4(scaled);
            const glu = gates * sigs;
            const result = (ups + ONE_VEC) * glu;
            out_ptr[out_base + d ..][0..4].* = result;
        }

        // Scalar tail for D not divisible by 4.
        while (d < Dz) : (d += 1) {
            var gate = gate_up_ptr[row_base + 2 * d];
            var up = gate_up_ptr[row_base + 2 * d + 1];
            if (gate > SWIGLU_LIMIT) gate = SWIGLU_LIMIT;
            if (up > SWIGLU_LIMIT) up = SWIGLU_LIMIT;
            if (up < -SWIGLU_LIMIT) up = -SWIGLU_LIMIT;
            const glu = gate * sigmoidF32(gate * SWIGLU_ALPHA);
            out_ptr[out_base + d] = (up + 1.0) * glu;
        }
    }
}

// --------------------------------------------------------------
// Kernel: row-wise softmax, f32 in / f32 out, numerically stable
// --------------------------------------------------------------
//
// out[r, c] = exp(x[r, c] - max_c(x[r, :])) / sum_c(exp(...))
//
// Used by attention (scores + sink, softmax over the last axis in f32
// per the upstream `dtype=torch.float32` argument) and by the router
// (top-4 scores).
//
// Three passes over each row: max, exp+sum, normalize. Exp is scalar
// per lane — WASM has no vector exp; Zig's `@exp` compiles to its
// freestanding software implementation.

export fn softmax_f32(
    x_ptr: [*]const f32,
    out_ptr: [*]f32,
    rows: u32,
    cols: u32,
) void {
    const Rz: usize = rows;
    const Cz: usize = cols;

    var r: usize = 0;
    while (r < Rz) : (r += 1) {
        const row_base = r * Cz;

        // Pass 1: max over the row.
        var row_max: f32 = x_ptr[row_base];
        var c: usize = 1;
        while (c < Cz) : (c += 1) {
            const v = x_ptr[row_base + c];
            if (v > row_max) row_max = v;
        }

        // Pass 2: exp(x - max), accumulate sum.
        var sum: f32 = 0.0;
        c = 0;
        while (c < Cz) : (c += 1) {
            const e = @exp(x_ptr[row_base + c] - row_max);
            out_ptr[row_base + c] = e;
            sum += e;
        }

        // Pass 3: divide by sum. If sum is zero (shouldn't happen with
        // max-subtract, but defensive), emit 1/C as a uniform distribution
        // so downstream ops don't NaN.
        const inv_sum: f32 = if (sum > 0.0) 1.0 / sum else 1.0 / @as(f32, @floatFromInt(Cz));
        c = 0;
        while (c < Cz) : (c += 1) {
            out_ptr[row_base + c] *= inv_sum;
        }
    }
}

// --------------------------------------------------------------
// Kernel: elementwise fp16 add
// --------------------------------------------------------------
//
// out[i] = round_fp16(a[i] + b[i])
// Used for residual connections in the block forward pass. Upstream's
// `residual + hidden_states` is a fp16 + fp16 → fp16 with an implicit
// f32 widen inside — matched here.

export fn add_fp16(
    a_ptr: [*]const u16,
    b_ptr: [*]const u16,
    out_ptr: [*]u16,
    n: u32,
) void {
    const nz: usize = n;
    const LANES: usize = 4;
    var i: usize = 0;
    while (i + LANES <= nz) : (i += LANES) {
        const av: @Vector(4, u16) = a_ptr[i ..][0..LANES].*;
        const bv: @Vector(4, u16) = b_ptr[i ..][0..LANES].*;
        const sum: @Vector(4, f32) = fp16x4ToF32x4(av) + fp16x4ToF32x4(bv);
        out_ptr[i + 0] = f32ToFp16(sum[0]);
        out_ptr[i + 1] = f32ToFp16(sum[1]);
        out_ptr[i + 2] = f32ToFp16(sum[2]);
        out_ptr[i + 3] = f32ToFp16(sum[3]);
    }
    while (i < nz) : (i += 1) {
        out_ptr[i] = f32ToFp16(fp16ToF32(a_ptr[i]) + fp16ToF32(b_ptr[i]));
    }
}

// --------------------------------------------------------------
// Kernel: gather fp16 rows by int32 index
// --------------------------------------------------------------
//
// dst[i, :] = src[indices[i], :]   for i in 0..m
// Used by expert dispatch to pull the token rows assigned to one expert
// into a contiguous scratch before the matmul chain.

export fn gather_fp16(
    src_ptr: [*]const u16,
    indices_ptr: [*]const i32,
    dst_ptr: [*]u16,
    m: u32,
    D: u32,
) void {
    const mz: usize = m;
    const Dz: usize = D;
    var i: usize = 0;
    while (i < mz) : (i += 1) {
        const t: usize = @intCast(indices_ptr[i]);
        const src_row = src_ptr + t * Dz;
        const dst_row = dst_ptr + i * Dz;
        var d: usize = 0;
        while (d < Dz) : (d += 1) dst_row[d] = src_row[d];
    }
}

// --------------------------------------------------------------
// Kernel: weighted scatter-add into f32 accumulator
// --------------------------------------------------------------
//
// target[indices[i], :] += weights[i] * values[i, :]   for i in 0..m
// Expert dispatch scatters per-expert outputs back into the per-token
// accumulator with the routing score as weight.

export fn scatter_add_weighted_f32(
    target_ptr: [*]f32,
    values_ptr: [*]const f32,
    indices_ptr: [*]const i32,
    weights_ptr: [*]const f32,
    m: u32,
    D: u32,
) void {
    const mz: usize = m;
    const Dz: usize = D;
    const LANES: usize = 4;

    var i: usize = 0;
    while (i < mz) : (i += 1) {
        const t: usize = @intCast(indices_ptr[i]);
        const w: f32 = weights_ptr[i];
        const w_vec: @Vector(4, f32) = @splat(w);
        const val_row = values_ptr + i * Dz;
        const tgt_row = target_ptr + t * Dz;

        var d: usize = 0;
        while (d + LANES <= Dz) : (d += LANES) {
            const tgt: @Vector(4, f32) = tgt_row[d ..][0..LANES].*;
            const val: @Vector(4, f32) = val_row[d ..][0..LANES].*;
            tgt_row[d ..][0..LANES].* = fma4(w_vec, val, tgt);
        }
        while (d < Dz) : (d += 1) tgt_row[d] += w * val_row[d];
    }
}

// --------------------------------------------------------------
// Kernel: zero-fill f32 buffer (bump-alloc'd scratch comes uninit)
// --------------------------------------------------------------

export fn zero_f32(ptr: [*]f32, n: u32) void {
    const nz: usize = n;
    const LANES: usize = 4;
    const zero_vec: @Vector(4, f32) = @splat(0.0);
    var i: usize = 0;
    while (i + LANES <= nz) : (i += LANES) {
        ptr[i ..][0..LANES].* = zero_vec;
    }
    while (i < nz) : (i += 1) ptr[i] = 0.0;
}

// --------------------------------------------------------------
// Kernel: fp16 → f32 widen (bulk)
// --------------------------------------------------------------
//
// Pre-converting X from fp16 to f32 once before a matmul lets the
// inner MAC loop skip the per-load widening (was ~4 SIMD ops per
// 4-lane x load). Saves real time in the MoE gate_up path where the
// fp16 int4 matmul reads each x column ~N times per call.

export fn convert_fp16_to_f32(src_ptr: [*]const u16, dst_ptr: [*]f32, n: u32) void {
    const nz: usize = n;
    const LANES: usize = 4;
    var i: usize = 0;
    while (i + LANES <= nz) : (i += LANES) {
        const u: @Vector(4, u16) = src_ptr[i..][0..LANES].*;
        const f: @Vector(4, f32) = fp16x4ToF32x4(u);
        dst_ptr[i..][0..LANES].* = f;
    }
    while (i < nz) : (i += 1) dst_ptr[i] = fp16ToF32(src_ptr[i]);
}

// --------------------------------------------------------------
// Kernel: gather f32 rows
// --------------------------------------------------------------
//
// Copies `m` rows of width `D` from `src` (indexed by `indices[0..m]`)
// into a dense `dst [m, D]`. f32 analogue of `gather_fp16`, used when
// the caller has already widened hidden activations to f32 once per
// block.

export fn gather_f32(src_ptr: [*]const f32, indices_ptr: [*]const i32, dst_ptr: [*]f32, m: u32, D: u32) void {
    const mz: usize = m;
    const Dz: usize = D;
    const LANES: usize = 4;
    var i: usize = 0;
    while (i < mz) : (i += 1) {
        const idx: usize = @intCast(indices_ptr[i]);
        const src_row = src_ptr + idx * Dz;
        const dst_row = dst_ptr + i * Dz;
        var d: usize = 0;
        while (d + LANES <= Dz) : (d += LANES) {
            const v: @Vector(4, f32) = src_row[d..][0..LANES].*;
            dst_row[d..][0..LANES].* = v;
        }
        while (d < Dz) : (d += 1) dst_row[d] = src_row[d];
    }
}

// --------------------------------------------------------------
// Kernel: f32 → fp16 with optional scalar multiply
// --------------------------------------------------------------
//
// dst[i] = round_fp16(src[i] * scale)
// Used to finish the MoE accumulator: multiply by `num_experts_per_tok`
// and downcast to fp16 in one pass (saves a buffer).

export fn cast_f32_to_fp16_scaled(
    src_ptr: [*]const f32,
    dst_ptr: [*]u16,
    n: u32,
    scale: f32,
) void {
    const nz: usize = n;
    var i: usize = 0;
    while (i < nz) : (i += 1) {
        dst_ptr[i] = f32ToFp16(src_ptr[i] * scale);
    }
}

// --------------------------------------------------------------
// Kernel: in-place fp16 scale
// --------------------------------------------------------------
//
// `x[i] = fp16(fp16(x[i]) * scale)` for i in 0..n.
// Used by the attention composition to multiply Q and K by
// `head_dim^-0.25` after RoPE. Upstream applies this as an explicit
// Python multiply, one rounding per element — we match.

export fn scale_fp16_inplace(
    x_ptr: [*]u16,
    scale: f32,
    n: u32,
) void {
    const Nz: usize = n;
    const scale_vec: @Vector(4, f32) = @splat(scale);
    var i: usize = 0;
    while (i + 4 <= Nz) : (i += 4) {
        const xu: @Vector(4, u16) = x_ptr[i ..][0..4].*;
        const scaled: @Vector(4, f32) = fp16x4ToF32x4(xu) * scale_vec;
        x_ptr[i + 0] = f32ToFp16(scaled[0]);
        x_ptr[i + 1] = f32ToFp16(scaled[1]);
        x_ptr[i + 2] = f32ToFp16(scaled[2]);
        x_ptr[i + 3] = f32ToFp16(scaled[3]);
    }
    while (i < Nz) : (i += 1) {
        x_ptr[i] = f32ToFp16(fp16ToF32(x_ptr[i]) * scale);
    }
}

// --------------------------------------------------------------
// Kernel: banded GQA attention with sinks
// --------------------------------------------------------------
//
// Implements the upstream `eager_attention_forward` pipeline for a
// single batch, restricted to a bidirectional sliding window.
// Expects Q and K already scaled by `head_dim^-0.25` (upstream applies
// this to Q and K individually before attention — see
// `OpenAIPrivacyFilterAttention.forward`). Our attention call uses
// `scaling = 1.0` to avoid a second multiply.
//
//   for each (t_q, h_q):
//     h_kv = h_q / num_kv_groups
//     ws = max(0, t_q - window)
//     we = min(T, t_q + window + 1)
//     scores[i] = dot(Q[t_q, h_q, :], K[ws+i, h_kv, :])   for i in 0..we-ws
//     combined = cat(scores, [sinks[h_q]])                 # append sink col
//     m = max(combined)
//     probs = exp(combined - m) / sum(exp(combined - m))
//     scores_drop_sink = probs[..., :-1]                    # drop sink column
//     out[t_q, h_q, :] = sum_i scores_drop_sink[i] * V[ws+i, h_kv, :]
//
// Layouts (all [T, H * head_dim] row-major, NOT transposed to [H, T, ...]):
//   q:     fp16 [T, H_q * head_dim]
//   k:     fp16 [T, H_kv * head_dim]
//   v:     fp16 [T, H_kv * head_dim]
//   sinks: f32  [H_q]                 (upstream keeps sinks in fp32)
//   mask:  u8   [T] or NULL (= 0 ptr) (1 = valid key, 0 = padding — skip)
//   out:   fp16 [T, H_q * head_dim]
//
// Constraint: `H_q % H_kv == 0` (GQA). Window is one-sided — `window=128`
// means total attended keys ≤ `2*window + 1 = 257`.

const MAX_WINDOW_TOTAL: usize = 512; // must be ≥ 2*window + 1
const MAX_HEAD_DIM: usize = 256;      // stack acc cap

inline fn dotBf16F32(a: [*]const u16, b: [*]const u16, n: usize) f32 {
    const LANES: usize = 4;
    const UNROLL: usize = 4;
    var a0: @Vector(4, f32) = @splat(0);
    var a1: @Vector(4, f32) = @splat(0);
    var a2: @Vector(4, f32) = @splat(0);
    var a3: @Vector(4, f32) = @splat(0);

    const step: usize = LANES * UNROLL;
    var i: usize = 0;
    while (i + step <= n) : (i += step) {
        const x0: @Vector(4, u16) = a[i ..][0..LANES].*;
        const x1: @Vector(4, u16) = a[i + 4 ..][0..LANES].*;
        const x2: @Vector(4, u16) = a[i + 8 ..][0..LANES].*;
        const x3: @Vector(4, u16) = a[i + 12 ..][0..LANES].*;
        const y0: @Vector(4, u16) = b[i ..][0..LANES].*;
        const y1: @Vector(4, u16) = b[i + 4 ..][0..LANES].*;
        const y2: @Vector(4, u16) = b[i + 8 ..][0..LANES].*;
        const y3: @Vector(4, u16) = b[i + 12 ..][0..LANES].*;
        a0 = fma4(fp16x4ToF32x4(x0), fp16x4ToF32x4(y0), a0);
        a1 = fma4(fp16x4ToF32x4(x1), fp16x4ToF32x4(y1), a1);
        a2 = fma4(fp16x4ToF32x4(x2), fp16x4ToF32x4(y2), a2);
        a3 = fma4(fp16x4ToF32x4(x3), fp16x4ToF32x4(y3), a3);
    }
    while (i + LANES <= n) : (i += LANES) {
        const xu: @Vector(4, u16) = a[i ..][0..LANES].*;
        const yu: @Vector(4, u16) = b[i ..][0..LANES].*;
        a0 = fma4(fp16x4ToF32x4(xu), fp16x4ToF32x4(yu), a0);
    }
    const combined: @Vector(4, f32) = (a0 + a1) + (a2 + a3);
    var sum: f32 = combined[0] + combined[1] + combined[2] + combined[3];
    while (i < n) : (i += 1) {
        sum += fp16ToF32(a[i]) * fp16ToF32(b[i]);
    }
    return sum;
}

inline fn saxpyF32Bf16(p: f32, v_ptr: [*]const u16, acc: [*]f32, n: usize) void {
    const LANES: usize = 4;
    const p_vec: @Vector(4, f32) = @splat(p);
    var i: usize = 0;
    while (i + LANES <= n) : (i += LANES) {
        const vv: @Vector(4, u16) = v_ptr[i ..][0..LANES].*;
        const acc_vec: @Vector(4, f32) = acc[i ..][0..LANES].*;
        acc[i ..][0..LANES].* = fma4(p_vec, fp16x4ToF32x4(vv), acc_vec);
    }
    while (i < n) : (i += 1) {
        acc[i] += p * fp16ToF32(v_ptr[i]);
    }
}

export fn banded_attention(
    q_ptr: [*]const u16,
    k_ptr: [*]const u16,
    v_ptr: [*]const u16,
    sinks_ptr: [*]const f32,
    mask_ptr: ?[*]const u8,    // NULL = all keys valid
    out_ptr: [*]u16,
    T: u32,
    H_q: u32,
    H_kv: u32,
    head_dim: u32,
    window: u32,
) void {
    const Tz: usize = T;
    const Hqz: usize = H_q;
    const Hkvz: usize = H_kv;
    const Dz: usize = head_dim;
    const Wz: usize = window;
    const kv_group: usize = Hqz / Hkvz;

    const q_stride_t: usize = Hqz * Dz;
    const k_stride_t: usize = Hkvz * Dz;

    // Per-query scratch: one entry per key in the window, plus the sink.
    var scores: [MAX_WINDOW_TOTAL + 1]f32 = undefined;
    var acc: [MAX_HEAD_DIM]f32 = undefined;

    var t_q: usize = 0;
    while (t_q < Tz) : (t_q += 1) {
        var h_q: usize = 0;
        while (h_q < Hqz) : (h_q += 1) {
            const h_kv = h_q / kv_group;
            const q_base: usize = t_q * q_stride_t + h_q * Dz;
            const out_base: usize = t_q * q_stride_t + h_q * Dz;

            const ws: usize = if (t_q > Wz) t_q - Wz else 0;
            const we: usize = if (t_q + Wz + 1 < Tz) t_q + Wz + 1 else Tz;
            const n_keys: usize = we - ws;

            const sink_logit: f32 = sinks_ptr[h_q];
            var row_max: f32 = sink_logit;

            var t_k: usize = 0;
            while (t_k < n_keys) : (t_k += 1) {
                const abs_k = ws + t_k;
                const is_valid: bool = if (mask_ptr) |m| (m[abs_k] != 0) else true;
                const k_base = abs_k * k_stride_t + h_kv * Dz;
                // Padding keys get -∞ logit so exp(·) rounds to 0 and
                // they contribute nothing to the softmax or AV combine.
                const s: f32 = if (is_valid)
                    dotBf16F32(q_ptr + q_base, k_ptr + k_base, Dz)
                else
                    -std.math.inf(f32);
                scores[t_k] = s;
                if (s > row_max) row_max = s;
            }
            scores[n_keys] = sink_logit;

            // exp(s - max) and running sum (including sink). Scores
            // are already ≤ row_max so (s - row_max) ≤ 0 — use the
            // approximate-exp path that assumes non-positive inputs.
            var sum: f32 = 0.0;
            const total = n_keys + 1;
            const max_vec: @Vector(4, f32) = @splat(row_max);
            var sum_vec: @Vector(4, f32) = @splat(0.0);
            var i: usize = 0;
            while (i + 4 <= total) : (i += 4) {
                const s_vec: @Vector(4, f32) = scores[i..][0..4].*;
                const e_vec = expApproxNonPositiveF32x4(s_vec - max_vec);
                scores[i..][0..4].* = e_vec;
                sum_vec += e_vec;
            }
            sum = sum_vec[0] + sum_vec[1] + sum_vec[2] + sum_vec[3];
            while (i < total) : (i += 1) {
                const e = @exp(scores[i] - row_max);
                scores[i] = e;
                sum += e;
            }
            const inv_sum: f32 = if (sum > 0.0) 1.0 / sum else 0.0;

            // Zero accumulator for this query head.
            var d: usize = 0;
            while (d < Dz) : (d += 1) acc[d] = 0.0;

            // Weighted sum of V over window keys (sink does not contribute).
            t_k = 0;
            while (t_k < n_keys) : (t_k += 1) {
                const p: f32 = scores[t_k] * inv_sum;
                const v_base = (ws + t_k) * k_stride_t + h_kv * Dz;
                saxpyF32Bf16(p, v_ptr + v_base, &acc, Dz);
            }

            d = 0;
            while (d < Dz) : (d += 1) {
                out_ptr[out_base + d] = f32ToFp16(acc[d]);
            }
        }
    }
}

// --------------------------------------------------------------
// Kernel: RoPE apply (interleaved layout)
// --------------------------------------------------------------
//
// Applies rotary position embedding in-place to an already-projected
// Q or K tensor. The privacy-filter layout is "interleaved pairs":
// for each head's head_dim-wide vector `x`, pair (x[2p], x[2p+1]) is
// rotated by angle `theta_{t,p}`, with cos/sin tables precomputed
// by the JS caller (yarn scaling + mscale folded into cos/sin).
//
//   for each (t, h, p):
//     c = cos[t, p]
//     s = sin[t, p]
//     a = x[t, h, 2p]
//     b = x[t, h, 2p+1]
//     x[t, h, 2p]   = a*c - b*s
//     x[t, h, 2p+1] = b*c + a*s
//
// qk:   fp16 [T, H, head_dim]          row-major, in-place
// cos:  fp16 [T, head_dim/2]            yarn's mscale already baked in
// sin:  fp16 [T, head_dim/2]
//
// cos/sin are fp16 because that's what the upstream layer returns (it
// downcasts its f32 compute to the caller's dtype, which is fp16 in
// Phase D). Widened to f32 inside the kernel for the rotation math.

export fn rope_apply(
    qk_ptr: [*]u16,
    cos_ptr: [*]const u16,
    sin_ptr: [*]const u16,
    T: u32,
    H: u32,
    head_dim: u32,
) void {
    const Tz: usize = T;
    const Hz: usize = H;
    const Dz: usize = head_dim;
    const HalfDz: usize = Dz / 2;

    var t: usize = 0;
    while (t < Tz) : (t += 1) {
        const cos_row: [*]const u16 = cos_ptr + t * HalfDz;
        const sin_row: [*]const u16 = sin_ptr + t * HalfDz;

        var h: usize = 0;
        while (h < Hz) : (h += 1) {
            const head_base: usize = (t * Hz + h) * Dz;

            var p: usize = 0;
            while (p < HalfDz) : (p += 1) {
                const c = fp16ToF32(cos_row[p]);
                const s = fp16ToF32(sin_row[p]);
                const a = fp16ToF32(qk_ptr[head_base + 2 * p]);
                const b = fp16ToF32(qk_ptr[head_base + 2 * p + 1]);
                // PyTorch eager fp16 arithmetic rounds between every op:
                // upcast → mul → round to fp16 → upcast → combine → round.
                // Three roundings total per lane, which produces bit-identical
                // output vs the reference but is slightly noisier than a
                // single-rounding f32 chain.
                const ac = fp16ToF32(f32ToFp16(a * c));
                const bs = fp16ToF32(f32ToFp16(b * s));
                const bc = fp16ToF32(f32ToFp16(b * c));
                const as_ = fp16ToF32(f32ToFp16(a * s));
                qk_ptr[head_base + 2 * p] = f32ToFp16(ac - bs);
                qk_ptr[head_base + 2 * p + 1] = f32ToFp16(bc + as_);
            }
        }
    }
}

// --------------------------------------------------------------
// Kernel: ONNX GatherBlockQuantized embedding lookup
// --------------------------------------------------------------
//
// out[t, d] = (uint4(embed[ids[t], d]) - zp[ids[t], block(d)])
//             * scale[ids[t], block(d)]
//
// Matches ONNX GatherBlockQuantized semantics (block_size=32, uint4
// weights, uint4 zero-points, fp16 scales). Layout of the ONNX embed
// table as unpacked by the reader:
//   embed_int4:    uint4 packed, [V, D/2] bytes
//   embed_scales:  fp16,         [V, D/32]
//   embed_zp:      uint4 packed, [V, ceil(D/32/2)] bytes
//
// Output is fp16 so the downstream kernels (rms_norm, q/k/v proj) read
// their usual input dtype.
//
// OOB ids zero-fill, same safety policy as the fp16 variant.

export fn embed_lookup_int4(
    embed_int4: [*]const u8,
    embed_scales: [*]const u16,
    embed_zp: [*]const u8,
    ids_ptr: [*]const i32,
    out_ptr: [*]u16,
    T: u32,
    V: u32,
    D: u32,
) void {
    const Tz: usize = T;
    const Dz: usize = D;
    const BLOCK: usize = 32;
    const n_blocks: usize = Dz / BLOCK;
    const int4_stride: usize = Dz / 2;
    const scale_stride: usize = n_blocks;
    const zp_stride: usize = (n_blocks + 1) / 2;

    var t: usize = 0;
    while (t < Tz) : (t += 1) {
        const id = ids_ptr[t];
        const out_base = t * Dz;
        if (id < 0 or @as(u32, @intCast(id)) >= V) {
            var d: usize = 0;
            while (d < Dz) : (d += 1) out_ptr[out_base + d] = 0;
            continue;
        }
        const row: usize = @intCast(id);
        const w_int4_row = embed_int4 + row * int4_stride;
        const w_scale_row = embed_scales + row * scale_stride;
        const w_zp_row = embed_zp + row * zp_stride;

        var b: usize = 0;
        while (b < n_blocks) : (b += 1) {
            const scale = fp16ToF32(w_scale_row[b]);
            const zp = extractZp(w_zp_row, b);
            const block_d = b * BLOCK;

            const q_chunks = int4BlockDecode(w_int4_row, block_d, zp);
            // 8 chunks of 4 lanes each = 32 output lanes per block.
            inline for (0..8) |chunk| {
                const base = out_base + block_d + chunk * 4;
                const qf = q_chunks[chunk];
                out_ptr[base + 0] = f32ToFp16(qf[0] * scale);
                out_ptr[base + 1] = f32ToFp16(qf[1] * scale);
                out_ptr[base + 2] = f32ToFp16(qf[2] * scale);
                out_ptr[base + 3] = f32ToFp16(qf[3] * scale);
            }
        }
    }
}

