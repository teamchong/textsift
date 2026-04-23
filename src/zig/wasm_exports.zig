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
const bf16ToF32 = math.bf16ToF32;
const f32ToBf16 = math.f32ToBf16;
const bf16x4ToF32x4 = math.bf16x4ToF32x4;
const alignUp = math.alignUp;
const readU32LE = math.readU32LE;
const readU64LE = math.readU64LE;

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
// Weight blob parser (pii-weights.bin v1)
// --------------------------------------------------------------
//
// Format produced by `scripts/convert-weights.py`. See that script for
// the full layout. Summary:
//   Header (16  B): magic "PIIW" | version u32 | num_tensors u32 | data_offset u32
//   Entry  (104 B): name[64] | dtype u32 | ndim u32 | shape[4] u32 | data_off u64 | data_size u64
//   Data          : each tensor 64-byte aligned
//
// The parser validates magic + version, walks the table, and records
// each tensor's `(data_ptr, data_size, shape, dtype)`. Kernels read
// the stored pointers directly — no copying, no dynamic allocation.

const WEIGHTS_MAGIC: u32 = 0x5749_4950; // "PIIW" little-endian
const WEIGHTS_VERSION: u32 = 1;
const MAX_TENSORS: usize = 256;
const NAME_FIELD: usize = 64;
const HEADER_SIZE: usize = 16;
const ENTRY_SIZE: usize = 104;

const TensorRecord = struct {
    data_ptr: [*]const u8,
    data_size: usize,
    name: [NAME_FIELD]u8,
    dtype: u32,
    ndim: u32,
    shape: [4]u32,
};

var g_tensors: [MAX_TENSORS]TensorRecord = undefined;
var g_num_tensors: u32 = 0;

const WEIGHTS_ERR_MAGIC: i32 = -1;
const WEIGHTS_ERR_VERSION: i32 = -2;
const WEIGHTS_ERR_TOO_MANY: i32 = -3;
const WEIGHTS_ERR_BAD_OFFSET: i32 = -4;

/// Parse a weight blob already copied into WASM linear memory.
/// Returns 0 on success, negative error code otherwise. Safe to call
/// multiple times — replaces any prior state.
export fn weights_load(ptr: [*]const u8, size: usize) i32 {
    g_num_tensors = 0;
    if (size < HEADER_SIZE) return WEIGHTS_ERR_BAD_OFFSET;

    const magic = readU32LE(ptr, 0);
    if (magic != WEIGHTS_MAGIC) return WEIGHTS_ERR_MAGIC;
    const version = readU32LE(ptr, 4);
    if (version != WEIGHTS_VERSION) return WEIGHTS_ERR_VERSION;

    const n = readU32LE(ptr, 8);
    if (n > MAX_TENSORS) return WEIGHTS_ERR_TOO_MANY;

    const data_offset = readU32LE(ptr, 12);
    const table_end = HEADER_SIZE + ENTRY_SIZE * @as(usize, n);
    if (data_offset < table_end) return WEIGHTS_ERR_BAD_OFFSET;
    if (data_offset > size) return WEIGHTS_ERR_BAD_OFFSET;

    var i: u32 = 0;
    while (i < n) : (i += 1) {
        const base: usize = HEADER_SIZE + ENTRY_SIZE * @as(usize, i);
        var rec: TensorRecord = undefined;

        // Copy 64-byte name.
        var j: usize = 0;
        while (j < NAME_FIELD) : (j += 1) {
            rec.name[j] = ptr[base + j];
        }

        rec.dtype = readU32LE(ptr, base + 64);
        rec.ndim = readU32LE(ptr, base + 68);
        rec.shape[0] = readU32LE(ptr, base + 72);
        rec.shape[1] = readU32LE(ptr, base + 76);
        rec.shape[2] = readU32LE(ptr, base + 80);
        rec.shape[3] = readU32LE(ptr, base + 84);

        // Layout past the shape fields:
        //   +88..+95  data_offset u64
        //   +96..+103 data_size  u64
        const data_off = readU64LE(ptr, base + 88);
        const data_size = readU64LE(ptr, base + 96);
        rec.data_ptr = ptr + @as(usize, @intCast(data_off));
        rec.data_size = @intCast(data_size);

        if (data_off + rec.data_size > size) return WEIGHTS_ERR_BAD_OFFSET;

        g_tensors[i] = rec;
    }

    g_num_tensors = n;
    return 0;
}

export fn weights_count() u32 {
    return g_num_tensors;
}

export fn weights_dtype(idx: u32) u32 {
    if (idx >= g_num_tensors) return 0xFFFF_FFFF;
    return g_tensors[idx].dtype;
}

export fn weights_ndim(idx: u32) u32 {
    if (idx >= g_num_tensors) return 0xFFFF_FFFF;
    return g_tensors[idx].ndim;
}

/// Write shape into `out[0..4]`. Returns 0 on success, 1 on bad index.
export fn weights_shape(idx: u32, out: [*]u32) u32 {
    if (idx >= g_num_tensors) return 1;
    const s = g_tensors[idx].shape;
    out[0] = s[0];
    out[1] = s[1];
    out[2] = s[2];
    out[3] = s[3];
    return 0;
}

/// Return byte offset (into linear memory) of the tensor's data, or 0 on bad idx.
export fn weights_data_ptr(idx: u32) u32 {
    if (idx >= g_num_tensors) return 0;
    const p: [*]const u8 = g_tensors[idx].data_ptr;
    return @intCast(@intFromPtr(p));
}

export fn weights_data_size(idx: u32) u32 {
    if (idx >= g_num_tensors) return 0;
    return @intCast(g_tensors[idx].data_size);
}

/// Write the tensor's name into out[0..NAME_FIELD]. Returns string length
/// (excluding null padding) on success, or 0xFFFFFFFF on bad idx.
export fn weights_name(idx: u32, out: [*]u8) u32 {
    if (idx >= g_num_tensors) return 0xFFFF_FFFF;
    const name = g_tensors[idx].name;
    var len: u32 = 0;
    while (len < NAME_FIELD and name[len] != 0) : (len += 1) {
        out[len] = name[len];
    }
    // Zero-pad the rest so JS can treat the buffer as a fixed-size field.
    var i: usize = len;
    while (i < NAME_FIELD) : (i += 1) out[i] = 0;
    return len;
}

// --------------------------------------------------------------
// Kernel: RMSNorm
// --------------------------------------------------------------
//
// y[t, d] = x[t, d] * gamma[d] / sqrt(mean_d(x[t, :]^2) + eps)
//
// x:     bf16 [T, D]      row-major
// gamma: bf16 [D]
// out:   bf16 [T, D]      row-major
//
// Sum-of-squares accumulates in f32 for numerical stability (640-wide
// bf16 rows would lose ~6 bits of precision if accumulated in bf16).
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
            const v = bf16ToF32(x_ptr[row_base + d]);
            sumsq += v * v;
        }

        const inv_rms: f32 = 1.0 / @sqrt(sumsq * d_inv + eps);

        d = 0;
        while (d < Dz) : (d += 1) {
            const xv = bf16ToF32(x_ptr[row_base + d]);
            const gv = bf16ToF32(gamma_ptr[d]);
            out_ptr[row_base + d] = f32ToBf16(xv * inv_rms * gv);
        }
    }
}

// --------------------------------------------------------------
// Kernel: bf16 matmul with bias
// --------------------------------------------------------------
//
// out = x @ W.T + bias          // PyTorch `F.linear` convention
//   x:    bf16 [T, D]
//   W:    bf16 [N, D]           // HF stores linear weights as [out, in]
//   bias: bf16 [N]
//   out:  bf16 [T, N]
//
// Inner D-loop uses:
//   (1) 4-wide SIMD bf16→f32 widen + vector multiply (see `bf16x4ToF32x4`).
//   (2) Four independent @Vector(4, f32) accumulators, each covering a
//       separate 4-lane slice of D, unrolled 16-at-a-time. The four
//       accumulators break the serial add dependency chain, giving the
//       WASM engine (V8/SM/JSC) room to schedule FMA-ish throughput.
//   (3) Horizontal reduce at the end: (a0+a1+a2+a3)[0..3] summed to scalar.
//
// The accumulate order differs from a strictly-scalar reference, so the
// output is NOT bit-exact against the scalar-drain version this kernel
// used to ship. Parity contract is tolerance-based (rel_tol ≤ 1e-3 on
// the bf16 output); see `tests/fixtures/manifest.json`.
//
// `@mulAdd` is deliberately avoided — relaxed_madd has
// implementation-defined fusion, which would introduce further
// engine-dependent drift on top of the accumulator-order drift.

export fn matmul_bf16(
    x_ptr: [*]const u16,
    w_ptr: [*]const u16,
    bias_ptr: [*]const u16,
    out_ptr: [*]u16,
    T: u32,
    N: u32,
    D: u32,
) void {
    const Tz: usize = T;
    const Nz: usize = N;
    const Dz: usize = D;
    const LANES: usize = 4;
    const UNROLL: usize = 4; // 4 accumulators → 16 lanes per outer step

    var t: usize = 0;
    while (t < Tz) : (t += 1) {
        const x_base = t * Dz;
        const out_base = t * Nz;

        var n: usize = 0;
        while (n < Nz) : (n += 1) {
            const w_base = n * Dz;

            var a0: @Vector(4, f32) = @splat(0);
            var a1: @Vector(4, f32) = @splat(0);
            var a2: @Vector(4, f32) = @splat(0);
            var a3: @Vector(4, f32) = @splat(0);

            const step: usize = LANES * UNROLL; // 16
            var d: usize = 0;
            while (d + step <= Dz) : (d += step) {
                const x0: @Vector(4, u16) = x_ptr[x_base + d ..][0..LANES].*;
                const x1: @Vector(4, u16) = x_ptr[x_base + d + 4 ..][0..LANES].*;
                const x2: @Vector(4, u16) = x_ptr[x_base + d + 8 ..][0..LANES].*;
                const x3: @Vector(4, u16) = x_ptr[x_base + d + 12 ..][0..LANES].*;
                const w0: @Vector(4, u16) = w_ptr[w_base + d ..][0..LANES].*;
                const w1: @Vector(4, u16) = w_ptr[w_base + d + 4 ..][0..LANES].*;
                const w2: @Vector(4, u16) = w_ptr[w_base + d + 8 ..][0..LANES].*;
                const w3: @Vector(4, u16) = w_ptr[w_base + d + 12 ..][0..LANES].*;
                a0 += bf16x4ToF32x4(x0) * bf16x4ToF32x4(w0);
                a1 += bf16x4ToF32x4(x1) * bf16x4ToF32x4(w1);
                a2 += bf16x4ToF32x4(x2) * bf16x4ToF32x4(w2);
                a3 += bf16x4ToF32x4(x3) * bf16x4ToF32x4(w3);
            }

            // 4-lane tail: keep using the same accumulators so the tail
            // doesn't synchronise the pipeline.
            while (d + LANES <= Dz) : (d += LANES) {
                const xu: @Vector(4, u16) = x_ptr[x_base + d ..][0..LANES].*;
                const wu: @Vector(4, u16) = w_ptr[w_base + d ..][0..LANES].*;
                a0 += bf16x4ToF32x4(xu) * bf16x4ToF32x4(wu);
            }

            // Scalar tail for D not divisible by 4.
            const combined: @Vector(4, f32) = (a0 + a1) + (a2 + a3);
            var acc: f32 = combined[0] + combined[1] + combined[2] + combined[3];
            while (d < Dz) : (d += 1) {
                acc += bf16ToF32(x_ptr[x_base + d]) * bf16ToF32(w_ptr[w_base + d]);
            }

            acc += bf16ToF32(bias_ptr[n]);
            out_ptr[out_base + n] = f32ToBf16(acc);
        }
    }
}

// --------------------------------------------------------------
// Kernel: bf16 × int4-blockwise matmul with bias
// --------------------------------------------------------------
//
// out = x @ dequant(W).T + bias     // PyTorch F.linear convention
//   x:    bf16                [T, D]
//   W:    int4 blockwise sym  [N, D]      (logical shape)
//   bias: bf16                [N]
//   out:  bf16                [T, N]
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
//   per block: 4-wide SIMD load of 4 x-bf16 lanes + extract 4 int4
//     nibbles from 2 bytes, upcast both to f32, multiply. Drain lanes
//     into a scalar `block_sum` so the per-block sum order matches a
//     reference implementation. Then `acc += block_sum * scale_f32`.
//   across blocks: scalar accumulate into `acc` (each block has its
//     own scale; can't hoist).
// Final `acc + bias` rounded to bf16 via RNE.

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

/// Read a fp16 scale as f32 (same lossless widening trick as bf16, but with
/// fp16 semantics — we upcast via Zig's native f16 → f32 cast since fp16
/// has real exponent/mantissa bits).
inline fn fp16ToF32(bits: u16) f32 {
    const h: f16 = @bitCast(bits);
    return @floatCast(h);
}

/// Decode 32 int4 values at `block_d` of row `w_int4_row` into eight
/// `@Vector(4, f32)` chunks covering the block's 32 lanes.
/// Reused by both the TR=4 tiled path and the T-tail.
inline fn int4BlockDecode(w_int4_row: [*]const u8, block_d: usize) [8]@Vector(4, f32) {
    const packed_u: @Vector(16, u8) = w_int4_row[block_d / 2 ..][0..16].*;
    const packed_i: @Vector(16, i8) = @bitCast(packed_u);
    const lo: @Vector(16, i8) = (packed_i << @splat(4)) >> @splat(4);
    const hi: @Vector(16, i8) = packed_i >> @splat(4);
    var out: [8]@Vector(4, f32) = undefined;
    inline for (0..8) |chunk| {
        // Lane `chunk * 4 + c` in block corresponds to nibble `c` of
        // byte `chunk * 2 + c/2` if c is even, or the hi nibble if odd.
        const base_byte = chunk * 2;
        const slice: @Vector(4, i8) = @shuffle(
            i8, lo, hi,
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

export fn matmul_bf16_x_int4block(
    x_ptr: [*]const u16,   // bf16 [T, D]
    w_ptr: [*]const u8,    // packed [N, D/2] int4 followed by [N, D/32] fp16
    bias_ptr: [*]const u16, // bf16 [N]
    out_ptr: [*]u16,       // bf16 [T, N]
    T: u32,
    N: u32,
    D: u32,
) void {
    const Tz: usize = T;
    const Nz: usize = N;
    const Dz: usize = D;
    const BLOCK: usize = 32;
    const TR: usize = 4;

    const w_int4_stride: usize = Dz / 2;
    const w_scale_stride: usize = Dz / BLOCK;
    const scales_base: [*]const u16 = @ptrCast(@alignCast(w_ptr + Nz * w_int4_stride));

    // TR-tiled T loop: amortize int4 decode across 4 rows of X that
    // share the same W row. Trailing 0..3 rows run through the scalar
    // tail below.
    var tt: usize = 0;
    while (tt + TR <= Tz) : (tt += TR) {
        var n: usize = 0;
        while (n < Nz) : (n += 1) {
            const w_int4_row = w_ptr + n * w_int4_stride;
            const w_scale_row = scales_base + n * w_scale_stride;

            var acc0: f32 = 0.0;
            var acc1: f32 = 0.0;
            var acc2: f32 = 0.0;
            var acc3: f32 = 0.0;
            var b: usize = 0;
            while (b < w_scale_stride) : (b += 1) {
                const scale = fp16ToF32(w_scale_row[b]);
                const block_d = b * BLOCK;

                // Decode the block's 32 int4 values once; reuse across 4 T rows.
                const q_chunks: [8]@Vector(4, f32) = int4BlockDecode(w_int4_row, block_d);

                // One vector accumulator per T row (×2 for ILP).
                var v00: @Vector(4, f32) = @splat(0);
                var v01: @Vector(4, f32) = @splat(0);
                var v10: @Vector(4, f32) = @splat(0);
                var v11: @Vector(4, f32) = @splat(0);
                var v20: @Vector(4, f32) = @splat(0);
                var v21: @Vector(4, f32) = @splat(0);
                var v30: @Vector(4, f32) = @splat(0);
                var v31: @Vector(4, f32) = @splat(0);

                inline for (0..4) |k| {
                    const j = k * 8;
                    const qf0 = q_chunks[k * 2];
                    const qf1 = q_chunks[k * 2 + 1];
                    const x0_0: @Vector(4, u16) = x_ptr[(tt + 0) * Dz + block_d + j ..][0..4].*;
                    const x0_1: @Vector(4, u16) = x_ptr[(tt + 0) * Dz + block_d + j + 4 ..][0..4].*;
                    const x1_0: @Vector(4, u16) = x_ptr[(tt + 1) * Dz + block_d + j ..][0..4].*;
                    const x1_1: @Vector(4, u16) = x_ptr[(tt + 1) * Dz + block_d + j + 4 ..][0..4].*;
                    const x2_0: @Vector(4, u16) = x_ptr[(tt + 2) * Dz + block_d + j ..][0..4].*;
                    const x2_1: @Vector(4, u16) = x_ptr[(tt + 2) * Dz + block_d + j + 4 ..][0..4].*;
                    const x3_0: @Vector(4, u16) = x_ptr[(tt + 3) * Dz + block_d + j ..][0..4].*;
                    const x3_1: @Vector(4, u16) = x_ptr[(tt + 3) * Dz + block_d + j + 4 ..][0..4].*;
                    v00 += bf16x4ToF32x4(x0_0) * qf0;
                    v01 += bf16x4ToF32x4(x0_1) * qf1;
                    v10 += bf16x4ToF32x4(x1_0) * qf0;
                    v11 += bf16x4ToF32x4(x1_1) * qf1;
                    v20 += bf16x4ToF32x4(x2_0) * qf0;
                    v21 += bf16x4ToF32x4(x2_1) * qf1;
                    v30 += bf16x4ToF32x4(x3_0) * qf0;
                    v31 += bf16x4ToF32x4(x3_1) * qf1;
                }

                const c0 = v00 + v01;
                const c1 = v10 + v11;
                const c2 = v20 + v21;
                const c3 = v30 + v31;
                acc0 += ((c0[0] + c0[1]) + (c0[2] + c0[3])) * scale;
                acc1 += ((c1[0] + c1[1]) + (c1[2] + c1[3])) * scale;
                acc2 += ((c2[0] + c2[1]) + (c2[2] + c2[3])) * scale;
                acc3 += ((c3[0] + c3[1]) + (c3[2] + c3[3])) * scale;
            }

            const b_val = bf16ToF32(bias_ptr[n]);
            out_ptr[(tt + 0) * Nz + n] = f32ToBf16(acc0 + b_val);
            out_ptr[(tt + 1) * Nz + n] = f32ToBf16(acc1 + b_val);
            out_ptr[(tt + 2) * Nz + n] = f32ToBf16(acc2 + b_val);
            out_ptr[(tt + 3) * Nz + n] = f32ToBf16(acc3 + b_val);
        }
    }

    // T tail for T not divisible by 4. One row at a time, same decode
    // pattern without the cross-row amortization.
    while (tt < Tz) : (tt += 1) {
        const x_base = tt * Dz;
        const out_base = tt * Nz;

        var n: usize = 0;
        while (n < Nz) : (n += 1) {
            const w_int4_row = w_ptr + n * w_int4_stride;
            const w_scale_row = scales_base + n * w_scale_stride;

            var acc: f32 = 0.0;
            var b: usize = 0;
            while (b < w_scale_stride) : (b += 1) {
                const scale = fp16ToF32(w_scale_row[b]);
                const block_d = b * BLOCK;
                const q_chunks = int4BlockDecode(w_int4_row, block_d);

                var a0: @Vector(4, f32) = @splat(0);
                var a1: @Vector(4, f32) = @splat(0);
                inline for (0..4) |k| {
                    const j = k * 8;
                    const xu0: @Vector(4, u16) = x_ptr[x_base + block_d + j ..][0..4].*;
                    const xu1: @Vector(4, u16) = x_ptr[x_base + block_d + j + 4 ..][0..4].*;
                    a0 += bf16x4ToF32x4(xu0) * q_chunks[k * 2];
                    a1 += bf16x4ToF32x4(xu1) * q_chunks[k * 2 + 1];
                }
                const combined: @Vector(4, f32) = a0 + a1;
                const block_sum: f32 =
                    (combined[0] + combined[1]) + (combined[2] + combined[3]);
                acc += block_sum * scale;
            }

            acc += bf16ToF32(bias_ptr[n]);
            out_ptr[out_base + n] = f32ToBf16(acc);
        }
    }
}

// --------------------------------------------------------------
// Kernel: embedding lookup
// --------------------------------------------------------------
//
// out[t, d] = embed[ids[t], d]
//   embed: bf16 [V, D]   (V = vocab_size = 200 064, D = d_model = 640)
//   ids:   i32  [T]
//   out:   bf16 [T, D]
//
// A pure gather — no arithmetic. Sanity-checks the id against `V` and
// writes zeros for out-of-range ids rather than reading past the end
// of the embedding table. The model's pad_token_id (199 999) is in
// range so this branch only fires on bad input.

export fn embed_lookup(
    embed_ptr: [*]const u16,
    ids_ptr: [*]const i32,
    out_ptr: [*]u16,
    T: u32,
    V: u32,
    D: u32,
) void {
    const Tz: usize = T;
    const Dz: usize = D;

    var t: usize = 0;
    while (t < Tz) : (t += 1) {
        const out_base = t * Dz;
        const id = ids_ptr[t];
        if (id < 0 or @as(u32, @intCast(id)) >= V) {
            var d: usize = 0;
            while (d < Dz) : (d += 1) out_ptr[out_base + d] = 0;
            continue;
        }
        const row_base: usize = @as(usize, @intCast(id)) * Dz;
        var d: usize = 0;
        while (d < Dz) : (d += 1) {
            out_ptr[out_base + d] = embed_ptr[row_base + d];
        }
    }
}
