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
// Conversions go through Zig's native `f16` type, which the backend
// lowers to inline integer bit-math on wasm — no libcall.

pub inline fn fp16ToF32(bits: u16) f32 {
    const h: f16 = @bitCast(bits);
    return @floatCast(h);
}

pub inline fn f32ToFp16(f: f32) u16 {
    const h: f16 = @floatCast(f);
    return @bitCast(h);
}

pub inline fn fp16x4ToF32x4(u: @Vector(4, u16)) @Vector(4, f32) {
    // Zig/LLVM doesn't vectorise `@floatCast(@Vector(4, f16))` on wasm
    // (emits an invalid cast). Inline an integer-ops widening: build the
    // f32 bit pattern from the fp16 bits and reinterpret. Handles the
    // finite-normal + zero cases model weights actually see. Subnormals
    // and inf/NaN fall through to a non-IEEE result, which is fine for
    // trained-model tensors.
    const u32v: @Vector(4, u32) = u;
    const sign_shifted: @Vector(4, u32) =
        (u32v & @as(@Vector(4, u32), @splat(0x8000))) << @splat(16);
    const exp_nib: @Vector(4, u32) =
        (u32v >> @splat(10)) & @as(@Vector(4, u32), @splat(0x1f));
    const mant_nib: @Vector(4, u32) =
        u32v & @as(@Vector(4, u32), @splat(0x3ff));
    // exp==0 → zero (for non-subnormals). exp+112 is the rebase from
    // fp16 bias 15 to f32 bias 127. Subnormals would need normalization
    // — we accept the drift (trained weights ≫ 2^-14).
    const exp_mask: @Vector(4, u32) =
        @select(u32, exp_nib == @as(@Vector(4, u32), @splat(0)),
            @as(@Vector(4, u32), @splat(0)),
            @as(@Vector(4, u32), @splat(0xFFFF_FFFF)));
    const f32_exp: @Vector(4, u32) =
        ((exp_nib + @as(@Vector(4, u32), @splat(112))) << @splat(23)) & exp_mask;
    const f32_mant: @Vector(4, u32) = (mant_nib << @splat(13)) & exp_mask;
    const bits: @Vector(4, u32) = sign_shifted | f32_exp | f32_mant;
    return @bitCast(bits);
}

pub inline fn alignUp(x: usize, a: usize) usize {
    return (x + (a - 1)) & ~@as(usize, a - 1);
}

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

