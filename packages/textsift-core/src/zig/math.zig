// Pure numerical helpers shared between the WASM exports and zig-test
// unit tests. Nothing here touches linear memory, globals, or wasm
// builtins — portable to any target.

const std = @import("std");

// bf16 = IEEE 754 binary32 with the low 16 mantissa bits truncated. We
// round f32→bf16 via round-to-nearest-ties-to-even, matching PyTorch's
// C10 `float_to_bfloat16_round_to_nearest_even`. Needed for bit-exact
// cross-backend parity with transformers.js (which consumes bf16
// weights via ORT Web's dequantization, which uses the same rounding).

pub inline fn bf16ToF32(b: u16) f32 {
    const bits: u32 = @as(u32, b) << 16;
    return @bitCast(bits);
}

pub inline fn f32ToBf16(f: f32) u16 {
    const bits: u32 = @bitCast(f);
    // NaN: preserve the top 16 bits so it stays NaN, but force a
    // non-zero mantissa so it doesn't collapse to infinity.
    if ((bits & 0x7F80_0000) == 0x7F80_0000 and (bits & 0x007F_FFFF) != 0) {
        return @intCast((bits | 0x0040_0000) >> 16);
    }
    // Round-to-nearest-even.
    const rounded = bits +% 0x7FFF +% ((bits >> 16) & 1);
    return @intCast(rounded >> 16);
}

/// Widen 4 bf16 lanes to 4 f32 lanes. Lossless (no rounding): bf16 is
/// the top 16 bits of binary32, so the widened value has zero low
/// mantissa bits. On wasm+simd128 this lowers to a single i32x4.shl.
pub inline fn bf16x4ToF32x4(u: @Vector(4, u16)) @Vector(4, f32) {
    const widened: @Vector(4, u32) = u;
    const shifted: @Vector(4, u32) = widened << @splat(16);
    return @bitCast(shifted);
}

// fp16 = IEEE 754 binary16 (1 sign + 5 exp + 10 mantissa, bias 15).
// Zig's `@floatCast(f16 ↔ f32)` still emits a libcall on wasm targets
// — LLVM doesn't have a native f16 ABI — which pulls the conversion
// out into a real function invocation inside hot loops. Do the bit
// manipulation explicitly so the compiler inlines it.

pub inline fn fp16ToF32(bits: u16) f32 {
    const u32_bits: u32 = bits;
    const sign_shifted: u32 = (u32_bits & 0x8000) << 16;
    const rest: u32 = u32_bits & 0x7fff;
    const biased: u32 = if (rest == 0) 0 else (rest << 13) +% (112 << 23);
    const result_bits: u32 = sign_shifted | biased;
    return @bitCast(result_bits);
}

pub inline fn f32ToFp16(f: f32) u16 {
    const u32_bits: u32 = @bitCast(f);
    const sign: u32 = (u32_bits >> 16) & 0x8000;
    const exp32: u32 = (u32_bits >> 23) & 0xff;
    const mant: u32 = u32_bits & 0x7fffff;
    if (exp32 == 0xff) {
        // inf or NaN
        return @intCast(sign | 0x7c00 | @as(u32, if (mant != 0) 0x200 else 0));
    }
    const exp16_signed: i32 = @as(i32, @intCast(exp32)) - 127 + 15;
    if (exp16_signed >= 0x1f) return @intCast(sign | 0x7c00);
    if (exp16_signed <= 0) {
        if (exp16_signed < -10) return @intCast(sign);
        const shift: u5 = @intCast(@as(u32, @intCast(14 - exp16_signed)));
        const mant24: u32 = mant | 0x800000;
        const rounded: u32 = (mant24 + (@as(u32, 1) << (shift - 1))) >> shift;
        return @intCast(sign | rounded);
    }
    // Normal: round-to-nearest-even of the top 10 mantissa bits.
    const lsb: u32 = (mant >> 13) & 1;
    var m10: u32 = (mant + 0xfff + lsb) >> 13;
    var e16: u32 = @intCast(exp16_signed);
    if (m10 >= 0x400) {
        m10 = 0;
        e16 += 1;
        if (e16 >= 0x1f) return @intCast(sign | 0x7c00);
    }
    return @intCast(sign | (e16 << 10) | m10);
}

pub inline fn fp16x4ToF32x4(u: @Vector(4, u16)) @Vector(4, f32) {
    // Zig/LLVM doesn't vectorise `@floatCast(@Vector(4, f16))` on wasm
    // (emits an invalid cast). Widen via integer bit-ops directly on
    // the v128 register: for any finite non-zero fp16, the f32 bit
    // pattern is `sign<<16 | ((bits & 0x7fff) << 13) + (112<<23)`.
    // Zero input must stay zero; mask the exp/mantissa word by
    // `non_sign != 0`. Subnormals and inf/NaN fall through to a
    // non-IEEE result, which is fine for trained-model tensors.
    const u32v: @Vector(4, u32) = u;
    const SIGN_MASK: @Vector(4, u32) = @splat(0x8000);
    const REST_MASK: @Vector(4, u32) = @splat(0x7fff);
    const EXP_BIAS: @Vector(4, u32) = @splat(112 << 23);
    const ZERO: @Vector(4, u32) = @splat(0);
    const sign_shifted = (u32v & SIGN_MASK) << @splat(16);
    const non_sign = u32v & REST_MASK;
    const biased = (non_sign << @splat(13)) +% EXP_BIAS;
    const biased_safe = @select(u32, non_sign != ZERO, biased, ZERO);
    const bits: @Vector(4, u32) = sign_shifted | biased_safe;
    return @bitCast(bits);
}

pub inline fn alignUp(x: usize, a: usize) usize {
    return (x + (a - 1)) & ~@as(usize, a - 1);
}

// Imported from src/c/fma.c (always_inline), built via `zig build-exe`
// with `-flto`. Emits a single `f32x4.relaxed_madd` op inlined into
// every caller. Zig's own `@mulAdd(@Vector(4, f32), …)` falls back to
// scalar per-lane software FMA on wasm. Returns `a * b + c` with
// fused-or-unfused rounding at V8's choice.
pub extern fn relaxed_madd_f32x4(a: @Vector(4, f32), b: @Vector(4, f32), c: @Vector(4, f32)) @Vector(4, f32);

// --------------------------------------------------------------
// Tests
// --------------------------------------------------------------

test "bf16 roundtrip — clean finite values are bit-exact" {
    // Values exactly representable in bf16: the low 16 mantissa bits are zero.
    const clean: [6]f32 = .{ 0.0, 1.0, -1.0, 2.5, -3.25, 1024.0 };
    for (clean) |f| {
        const b = f32ToBf16(f);
        const back = bf16ToF32(b);
        try std.testing.expectEqual(f, back);
    }
}

test "bf16 rounding — ties go to even" {
    // A value at exactly the halfway point between two bf16s. RNE rounds
    // to the even neighbour. Constructed: top 16 bits = 0x3F80 (1.0),
    // bottom 16 bits = 0x8000 (exact halfway). Expected: round to even =
    // down = 0x3F80.
    const halfway: f32 = @bitCast(@as(u32, 0x3F80_8000));
    try std.testing.expectEqual(@as(u16, 0x3F80), f32ToBf16(halfway));

    // And the next odd neighbour: 0x3F81_8000 — halfway between 0x3F81 (odd)
    // and 0x3F82 (even). RNE picks even → 0x3F82.
    const halfway_odd: f32 = @bitCast(@as(u32, 0x3F81_8000));
    try std.testing.expectEqual(@as(u16, 0x3F82), f32ToBf16(halfway_odd));
}

test "bf16 rounding — above halfway rounds up" {
    // 0x3F80_8001 is just above halfway → rounds up to 0x3F81.
    const above: f32 = @bitCast(@as(u32, 0x3F80_8001));
    try std.testing.expectEqual(@as(u16, 0x3F81), f32ToBf16(above));
}

test "bf16 NaN — preserves NaN rather than collapsing to inf" {
    // A signaling NaN: exponent all 1s, mantissa bit in the low half only.
    // Naive truncation would drop the mantissa and yield +inf (0x7F80).
    // Our NaN guard sets the top mantissa bit, producing a qNaN.
    const snan: f32 = @bitCast(@as(u32, 0x7F80_0001));
    const b = f32ToBf16(snan);
    try std.testing.expect(b != 0x7F80); // not collapsed to +inf
    try std.testing.expect((b & 0x7F80) == 0x7F80); // exponent still all-1s
    try std.testing.expect((b & 0x007F) != 0); // mantissa non-zero → NaN
}

test "bf16 infinity passes through" {
    const pinf: f32 = @bitCast(@as(u32, 0x7F80_0000));
    const ninf: f32 = @bitCast(@as(u32, 0xFF80_0000));
    try std.testing.expectEqual(@as(u16, 0x7F80), f32ToBf16(pinf));
    try std.testing.expectEqual(@as(u16, 0xFF80), f32ToBf16(ninf));
}

test "bf16x4ToF32x4 matches scalar bf16ToF32 lane-by-lane" {
    const samples: [4]u16 = .{ 0x0000, 0x3F80, 0xBF80, 0x4200 }; // 0, 1.0, -1.0, 32.0
    const v: @Vector(4, u16) = samples;
    const out = bf16x4ToF32x4(v);
    inline for (0..4) |i| {
        try std.testing.expectEqual(bf16ToF32(samples[i]), out[i]);
    }
}

test "fp16 roundtrip — exactly-representable finite values" {
    const clean: [6]f32 = .{ 0.0, 1.0, -1.0, 2.5, -3.25, 1024.0 };
    for (clean) |f| {
        try std.testing.expectEqual(f, fp16ToF32(f32ToFp16(f)));
    }
}

test "fp16x4ToF32x4 matches scalar lane-by-lane" {
    const samples: [4]u16 = .{ 0x0000, 0x3C00, 0xBC00, 0x5000 }; // 0, 1.0, -1.0, 32.0
    const v: @Vector(4, u16) = samples;
    const out = fp16x4ToF32x4(v);
    inline for (0..4) |i| {
        try std.testing.expectEqual(fp16ToF32(samples[i]), out[i]);
    }
}

test "alignUp rounds to multiple" {
    try std.testing.expectEqual(@as(usize, 0), alignUp(0, 16));
    try std.testing.expectEqual(@as(usize, 16), alignUp(1, 16));
    try std.testing.expectEqual(@as(usize, 16), alignUp(16, 16));
    try std.testing.expectEqual(@as(usize, 32), alignUp(17, 16));
    try std.testing.expectEqual(@as(usize, 64), alignUp(48, 64));
}

