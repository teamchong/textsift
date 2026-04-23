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
//   (1) 4-wide SIMD bf16→f32 widen (see `bf16x4ToF32x4`).
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
// `@mulAdd` was tried and reverted — Zig's wasm-freestanding codegen
// has no hardware f32x4 FMA intrinsic, so it falls back to a scalar
// software fma that runs ~6× slower than the separate-mul-add vector
// form. `f32x4.relaxed_madd` would help here, but Zig has no builtin
// for it yet; inline asm or a future compiler update is needed.

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
    const TR: usize = 4;

    // TR-tiled T loop: amortize W row loads across 4 X rows. For each
    // tile of 4 consecutive T rows, one pass over W gives us 4 outputs
    // per n. Register footprint: TR × 2 accumulator vectors + 4 X
    // loads/iter + 2 W loads/iter.
    var tt: usize = 0;
    while (tt + TR <= Tz) : (tt += TR) {
        var n: usize = 0;
        while (n < Nz) : (n += 1) {
            const w_base = n * Dz;

            var a00: @Vector(4, f32) = @splat(0);
            var a01: @Vector(4, f32) = @splat(0);
            var a10: @Vector(4, f32) = @splat(0);
            var a11: @Vector(4, f32) = @splat(0);
            var a20: @Vector(4, f32) = @splat(0);
            var a21: @Vector(4, f32) = @splat(0);
            var a30: @Vector(4, f32) = @splat(0);
            var a31: @Vector(4, f32) = @splat(0);

            const step: usize = LANES * 2; // 8-wide step, two accs per T row
            var d: usize = 0;
            while (d + step <= Dz) : (d += step) {
                const w0 = bf16x4ToF32x4(w_ptr[w_base + d ..][0..LANES].*);
                const w1 = bf16x4ToF32x4(w_ptr[w_base + d + 4 ..][0..LANES].*);
                const x00 = bf16x4ToF32x4(x_ptr[(tt + 0) * Dz + d ..][0..LANES].*);
                const x01 = bf16x4ToF32x4(x_ptr[(tt + 0) * Dz + d + 4 ..][0..LANES].*);
                const x10 = bf16x4ToF32x4(x_ptr[(tt + 1) * Dz + d ..][0..LANES].*);
                const x11 = bf16x4ToF32x4(x_ptr[(tt + 1) * Dz + d + 4 ..][0..LANES].*);
                const x20 = bf16x4ToF32x4(x_ptr[(tt + 2) * Dz + d ..][0..LANES].*);
                const x21 = bf16x4ToF32x4(x_ptr[(tt + 2) * Dz + d + 4 ..][0..LANES].*);
                const x30 = bf16x4ToF32x4(x_ptr[(tt + 3) * Dz + d ..][0..LANES].*);
                const x31 = bf16x4ToF32x4(x_ptr[(tt + 3) * Dz + d + 4 ..][0..LANES].*);
                a00 += x00 * w0;
                a01 += x01 * w1;
                a10 += x10 * w0;
                a11 += x11 * w1;
                a20 += x20 * w0;
                a21 += x21 * w1;
                a30 += x30 * w0;
                a31 += x31 * w1;
            }

            // 4-lane tail for the last step if D % 8 != 0.
            while (d + LANES <= Dz) : (d += LANES) {
                const w0 = bf16x4ToF32x4(w_ptr[w_base + d ..][0..LANES].*);
                a00 += bf16x4ToF32x4(x_ptr[(tt + 0) * Dz + d ..][0..LANES].*) * w0;
                a10 += bf16x4ToF32x4(x_ptr[(tt + 1) * Dz + d ..][0..LANES].*) * w0;
                a20 += bf16x4ToF32x4(x_ptr[(tt + 2) * Dz + d ..][0..LANES].*) * w0;
                a30 += bf16x4ToF32x4(x_ptr[(tt + 3) * Dz + d ..][0..LANES].*) * w0;
            }

            const c0 = a00 + a01;
            const c1 = a10 + a11;
            const c2 = a20 + a21;
            const c3 = a30 + a31;
            var acc0: f32 = (c0[0] + c0[1]) + (c0[2] + c0[3]);
            var acc1: f32 = (c1[0] + c1[1]) + (c1[2] + c1[3]);
            var acc2: f32 = (c2[0] + c2[1]) + (c2[2] + c2[3]);
            var acc3: f32 = (c3[0] + c3[1]) + (c3[2] + c3[3]);

            while (d < Dz) : (d += 1) {
                const wv = bf16ToF32(w_ptr[w_base + d]);
                acc0 += bf16ToF32(x_ptr[(tt + 0) * Dz + d]) * wv;
                acc1 += bf16ToF32(x_ptr[(tt + 1) * Dz + d]) * wv;
                acc2 += bf16ToF32(x_ptr[(tt + 2) * Dz + d]) * wv;
                acc3 += bf16ToF32(x_ptr[(tt + 3) * Dz + d]) * wv;
            }

            const b_val = bf16ToF32(bias_ptr[n]);
            out_ptr[(tt + 0) * Nz + n] = f32ToBf16(acc0 + b_val);
            out_ptr[(tt + 1) * Nz + n] = f32ToBf16(acc1 + b_val);
            out_ptr[(tt + 2) * Nz + n] = f32ToBf16(acc2 + b_val);
            out_ptr[(tt + 3) * Nz + n] = f32ToBf16(acc3 + b_val);
        }
    }

    // T tail for T not divisible by TR. Same structure as the old
    // single-row path.
    while (tt < Tz) : (tt += 1) {
        const x_base = tt * Dz;
        const out_base = tt * Nz;

        var n: usize = 0;
        while (n < Nz) : (n += 1) {
            const w_base = n * Dz;

            var a0: @Vector(4, f32) = @splat(0);
            var a1: @Vector(4, f32) = @splat(0);
            var a2: @Vector(4, f32) = @splat(0);
            var a3: @Vector(4, f32) = @splat(0);

            const step: usize = LANES * 4;
            var d: usize = 0;
            while (d + step <= Dz) : (d += step) {
                a0 += bf16x4ToF32x4(x_ptr[x_base + d ..][0..LANES].*) *
                    bf16x4ToF32x4(w_ptr[w_base + d ..][0..LANES].*);
                a1 += bf16x4ToF32x4(x_ptr[x_base + d + 4 ..][0..LANES].*) *
                    bf16x4ToF32x4(w_ptr[w_base + d + 4 ..][0..LANES].*);
                a2 += bf16x4ToF32x4(x_ptr[x_base + d + 8 ..][0..LANES].*) *
                    bf16x4ToF32x4(w_ptr[w_base + d + 8 ..][0..LANES].*);
                a3 += bf16x4ToF32x4(x_ptr[x_base + d + 12 ..][0..LANES].*) *
                    bf16x4ToF32x4(w_ptr[w_base + d + 12 ..][0..LANES].*);
            }
            while (d + LANES <= Dz) : (d += LANES) {
                a0 += bf16x4ToF32x4(x_ptr[x_base + d ..][0..LANES].*) *
                    bf16x4ToF32x4(w_ptr[w_base + d ..][0..LANES].*);
            }
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

/// F32-output variant of `matmul_bf16`. Same arithmetic; stores the
/// accumulator + bias as f32 instead of rounding to bf16. Used when
/// the consumer runs in fp32 (router, expert chain).
export fn matmul_bf16_out_f32(
    x_ptr: [*]const u16,
    w_ptr: [*]const u16,
    bias_ptr: [*]const u16,
    out_ptr: [*]f32,
    T: u32,
    N: u32,
    D: u32,
) void {
    const Tz: usize = T;
    const Nz: usize = N;
    const Dz: usize = D;
    const LANES: usize = 4;
    const UNROLL: usize = 4;

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

            const step: usize = LANES * UNROLL;
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
            while (d + LANES <= Dz) : (d += LANES) {
                const xu: @Vector(4, u16) = x_ptr[x_base + d ..][0..LANES].*;
                const wu: @Vector(4, u16) = w_ptr[w_base + d ..][0..LANES].*;
                a0 += bf16x4ToF32x4(xu) * bf16x4ToF32x4(wu);
            }
            const combined: @Vector(4, f32) = (a0 + a1) + (a2 + a3);
            var acc: f32 = combined[0] + combined[1] + combined[2] + combined[3];
            while (d < Dz) : (d += 1) {
                acc += bf16ToF32(x_ptr[x_base + d]) * bf16ToF32(w_ptr[w_base + d]);
            }

            out_ptr[out_base + n] = acc + bf16ToF32(bias_ptr[n]);
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

/// Load 4 elements of an X-row as an f32 vector. `XType = u16` treats
/// the source as bf16 and widens; `XType = f32` is a direct load.
inline fn loadX4(comptime XType: type, base: [*]const XType, offset: usize) @Vector(4, f32) {
    if (XType == u16) {
        const xu: @Vector(4, u16) = base[offset..][0..4].*;
        return bf16x4ToF32x4(xu);
    } else if (XType == f32) {
        return base[offset..][0..4].*;
    } else @compileError("loadX4: unsupported XType");
}

/// Store an f32 accumulator lane to out. `OutType = u16` rounds to bf16;
/// `OutType = f32` is a direct store.
inline fn storeOut(comptime OutType: type, out: [*]OutType, idx: usize, val: f32) void {
    if (OutType == u16) {
        out[idx] = f32ToBf16(val);
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
/// three public exports (bf16→bf16, bf16→f32, f32→f32). Three variants
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
    const TR: usize = 4;

    const w_int4_stride: usize = Dz / 2;
    const w_scale_stride: usize = Dz / BLOCK;
    const w_zp_stride: usize = (w_scale_stride + 1) / 2;
    const w_ptr = w_int4_ptr;
    const scales_base: [*]const u16 = w_scales_ptr;

    // TR-tiled T loop: amortize int4 decode across 4 rows of X that
    // share the same W row. Trailing 0..3 rows run through the scalar
    // tail below.
    var tt: usize = 0;
    while (tt + TR <= Tz) : (tt += TR) {
        var n: usize = 0;
        while (n < Nz) : (n += 1) {
            const w_int4_row = w_ptr + n * w_int4_stride;
            const w_scale_row = scales_base + n * w_scale_stride;
            const w_zp_row: ?[*]const u8 = if (w_zp_ptr) |p| p + n * w_zp_stride else null;

            var acc0: f32 = 0.0;
            var acc1: f32 = 0.0;
            var acc2: f32 = 0.0;
            var acc3: f32 = 0.0;
            var b: usize = 0;
            while (b < w_scale_stride) : (b += 1) {
                const scale = fp16ToF32(w_scale_row[b]);
                const block_d = b * BLOCK;
                const zp: u8 = if (w_zp_row) |p| extractZp(p, b) else 0;

                // Decode the block's 32 int4 values once; reuse across 4 T rows.
                const q_chunks: [8]@Vector(4, f32) = int4BlockDecode(w_int4_row, block_d, zp);

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
                    const x0_0 = loadX4(XType, x_ptr, (tt + 0) * Dz + block_d + j);
                    const x0_1 = loadX4(XType, x_ptr, (tt + 0) * Dz + block_d + j + 4);
                    const x1_0 = loadX4(XType, x_ptr, (tt + 1) * Dz + block_d + j);
                    const x1_1 = loadX4(XType, x_ptr, (tt + 1) * Dz + block_d + j + 4);
                    const x2_0 = loadX4(XType, x_ptr, (tt + 2) * Dz + block_d + j);
                    const x2_1 = loadX4(XType, x_ptr, (tt + 2) * Dz + block_d + j + 4);
                    const x3_0 = loadX4(XType, x_ptr, (tt + 3) * Dz + block_d + j);
                    const x3_1 = loadX4(XType, x_ptr, (tt + 3) * Dz + block_d + j + 4);
                    v00 += x0_0 * qf0;
                    v01 += x0_1 * qf1;
                    v10 += x1_0 * qf0;
                    v11 += x1_1 * qf1;
                    v20 += x2_0 * qf0;
                    v21 += x2_1 * qf1;
                    v30 += x3_0 * qf0;
                    v31 += x3_1 * qf1;
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
            storeOut(OutType, out_ptr, (tt + 0) * Nz + n, acc0 + b_val);
            storeOut(OutType, out_ptr, (tt + 1) * Nz + n, acc1 + b_val);
            storeOut(OutType, out_ptr, (tt + 2) * Nz + n, acc2 + b_val);
            storeOut(OutType, out_ptr, (tt + 3) * Nz + n, acc3 + b_val);
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
            const w_zp_row: ?[*]const u8 = if (w_zp_ptr) |p| p + n * w_zp_stride else null;

            var acc: f32 = 0.0;
            var b: usize = 0;
            while (b < w_scale_stride) : (b += 1) {
                const scale = fp16ToF32(w_scale_row[b]);
                const block_d = b * BLOCK;
                const zp: u8 = if (w_zp_row) |p| extractZp(p, b) else 0;
                const q_chunks = int4BlockDecode(w_int4_row, block_d, zp);

                var a0: @Vector(4, f32) = @splat(0);
                var a1: @Vector(4, f32) = @splat(0);
                inline for (0..4) |k| {
                    const j = k * 8;
                    const xu0 = loadX4(XType, x_ptr, x_base + block_d + j);
                    const xu1 = loadX4(XType, x_ptr, x_base + block_d + j + 4);
                    a0 += xu0 * q_chunks[k * 2];
                    a1 += xu1 * q_chunks[k * 2 + 1];
                }
                const combined: @Vector(4, f32) = a0 + a1;
                const block_sum: f32 =
                    (combined[0] + combined[1]) + (combined[2] + combined[3]);
                acc += block_sum * scale;
            }

            acc += bf16ToF32(bias_ptr[n]);
            storeOut(OutType, out_ptr, out_base + n, acc);
        }
    }
}

/// bf16 x × int4-blockwise W + bias → bf16 out. ONNX MatMulNBits
/// semantics: uint4 weights, (q - zp) * scale dequant.
/// Pass `w_zp_ptr = 0` for symmetric-decode (treats zp = 0 for all blocks).
export fn matmul_bf16_x_int4block(
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

/// bf16 x × int4-blockwise W + bias → f32 out. Used for the MoE gate_up
/// matmul — upstream expert forward keeps the chain in fp32.
export fn matmul_bf16_x_int4block_out_f32(
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

// --------------------------------------------------------------
// Kernel: SwiGLU with clamp (privacy-filter variant)
// --------------------------------------------------------------
//
// Matches `OpenAIPrivacyFilterExperts._apply_gate` exactly:
//   gate, up = gate_up.chunk(2, dim=-1)
//   gate = gate.clamp(max=7.0)
//   up   = up.clamp(-7.0, 7.0)
//   glu  = gate * sigmoid(gate * 1.702)
//   out  = (up + 1.0) * glu
//
// All compute in f32 — the upstream expert forward runs in fp32
// throughout (`.float()` on every op). Input and output are f32.
//
// gate_up: [T, 2*D]   f32       (gate concatenated with up)
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

export fn swiglu_clamp_f32(
    gate_up_ptr: [*]const f32,
    out_ptr: [*]f32,
    T: u32,
    D: u32,
) void {
    const Tz: usize = T;
    const Dz: usize = D;
    const stride: usize = 2 * Dz;

    var t: usize = 0;
    while (t < Tz) : (t += 1) {
        const row_base = t * stride;
        const out_base = t * Dz;

        var d: usize = 0;
        while (d < Dz) : (d += 1) {
            var gate = gate_up_ptr[row_base + d];
            var up = gate_up_ptr[row_base + Dz + d];
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
// Kernel: elementwise bf16 add
// --------------------------------------------------------------
//
// out[i] = round_bf16(a[i] + b[i])
// Used for residual connections in the block forward pass. Upstream's
// `residual + hidden_states` is a bf16 + bf16 → bf16 with an implicit
// f32 widen inside — matched here.

export fn add_bf16(
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
        const sum: @Vector(4, f32) = bf16x4ToF32x4(av) + bf16x4ToF32x4(bv);
        out_ptr[i + 0] = f32ToBf16(sum[0]);
        out_ptr[i + 1] = f32ToBf16(sum[1]);
        out_ptr[i + 2] = f32ToBf16(sum[2]);
        out_ptr[i + 3] = f32ToBf16(sum[3]);
    }
    while (i < nz) : (i += 1) {
        out_ptr[i] = f32ToBf16(bf16ToF32(a_ptr[i]) + bf16ToF32(b_ptr[i]));
    }
}

// --------------------------------------------------------------
// Kernel: gather bf16 rows by int32 index
// --------------------------------------------------------------
//
// dst[i, :] = src[indices[i], :]   for i in 0..m
// Used by expert dispatch to pull the token rows assigned to one expert
// into a contiguous scratch before the matmul chain.

export fn gather_bf16(
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
            tgt_row[d ..][0..LANES].* = tgt + w_vec * val;
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
// Kernel: f32 → bf16 with optional scalar multiply
// --------------------------------------------------------------
//
// dst[i] = round_bf16(src[i] * scale)
// Used to finish the MoE accumulator: multiply by `num_experts_per_tok`
// and downcast to bf16 in one pass (saves a buffer).

export fn cast_f32_to_bf16_scaled(
    src_ptr: [*]const f32,
    dst_ptr: [*]u16,
    n: u32,
    scale: f32,
) void {
    const nz: usize = n;
    var i: usize = 0;
    while (i < nz) : (i += 1) {
        dst_ptr[i] = f32ToBf16(src_ptr[i] * scale);
    }
}

// --------------------------------------------------------------
// Kernel: in-place bf16 scale
// --------------------------------------------------------------
//
// `x[i] = bf16(bf16(x[i]) * scale)` for i in 0..n.
// Used by the attention composition to multiply Q and K by
// `head_dim^-0.25` after RoPE. Upstream applies this as an explicit
// Python multiply, one rounding per element — we match.

export fn scale_bf16_inplace(
    x_ptr: [*]u16,
    scale: f32,
    n: u32,
) void {
    const Nz: usize = n;
    const scale_vec: @Vector(4, f32) = @splat(scale);
    var i: usize = 0;
    while (i + 4 <= Nz) : (i += 4) {
        const xu: @Vector(4, u16) = x_ptr[i ..][0..4].*;
        const scaled: @Vector(4, f32) = bf16x4ToF32x4(xu) * scale_vec;
        x_ptr[i + 0] = f32ToBf16(scaled[0]);
        x_ptr[i + 1] = f32ToBf16(scaled[1]);
        x_ptr[i + 2] = f32ToBf16(scaled[2]);
        x_ptr[i + 3] = f32ToBf16(scaled[3]);
    }
    while (i < Nz) : (i += 1) {
        x_ptr[i] = f32ToBf16(bf16ToF32(x_ptr[i]) * scale);
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
//   q:     bf16 [T, H_q * head_dim]
//   k:     bf16 [T, H_kv * head_dim]
//   v:     bf16 [T, H_kv * head_dim]
//   sinks: f32  [H_q]                 (upstream keeps sinks in fp32)
//   mask:  u8   [T] or NULL (= 0 ptr) (1 = valid key, 0 = padding — skip)
//   out:   bf16 [T, H_q * head_dim]
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
        a0 += bf16x4ToF32x4(x0) * bf16x4ToF32x4(y0);
        a1 += bf16x4ToF32x4(x1) * bf16x4ToF32x4(y1);
        a2 += bf16x4ToF32x4(x2) * bf16x4ToF32x4(y2);
        a3 += bf16x4ToF32x4(x3) * bf16x4ToF32x4(y3);
    }
    while (i + LANES <= n) : (i += LANES) {
        const xu: @Vector(4, u16) = a[i ..][0..LANES].*;
        const yu: @Vector(4, u16) = b[i ..][0..LANES].*;
        a0 += bf16x4ToF32x4(xu) * bf16x4ToF32x4(yu);
    }
    const combined: @Vector(4, f32) = (a0 + a1) + (a2 + a3);
    var sum: f32 = combined[0] + combined[1] + combined[2] + combined[3];
    while (i < n) : (i += 1) {
        sum += bf16ToF32(a[i]) * bf16ToF32(b[i]);
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
        const new_acc = acc_vec + p_vec * bf16x4ToF32x4(vv);
        acc[i ..][0..LANES].* = new_acc;
    }
    while (i < n) : (i += 1) {
        acc[i] += p * bf16ToF32(v_ptr[i]);
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

            // exp(s - max) and running sum (including sink).
            var sum: f32 = 0.0;
            var i: usize = 0;
            while (i <= n_keys) : (i += 1) {
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
                out_ptr[out_base + d] = f32ToBf16(acc[d]);
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
// qk:   bf16 [T, H, head_dim]          row-major, in-place
// cos:  bf16 [T, head_dim/2]            yarn's mscale already baked in
// sin:  bf16 [T, head_dim/2]
//
// cos/sin are bf16 because that's what the upstream layer returns (it
// downcasts its f32 compute to the caller's dtype, which is bf16 in
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
                const c = bf16ToF32(cos_row[p]);
                const s = bf16ToF32(sin_row[p]);
                const a = bf16ToF32(qk_ptr[head_base + 2 * p]);
                const b = bf16ToF32(qk_ptr[head_base + 2 * p + 1]);
                // PyTorch eager bf16 arithmetic rounds between every op:
                // upcast → mul → round to bf16 → upcast → combine → round.
                // Three roundings total per lane, which produces bit-identical
                // output vs the reference but is slightly noisier than a
                // single-rounding f32 chain.
                const ac = bf16ToF32(f32ToBf16(a * c));
                const bs = bf16ToF32(f32ToBf16(b * s));
                const bc = bf16ToF32(f32ToBf16(b * c));
                const as_ = bf16ToF32(f32ToBf16(a * s));
                qk_ptr[head_base + 2 * p] = f32ToBf16(ac - bs);
                qk_ptr[head_base + 2 * p + 1] = f32ToBf16(bc + as_);
            }
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
