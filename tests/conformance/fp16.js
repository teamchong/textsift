// IEEE 754 binary16 (fp16) helpers — used by both the browser
// fixture-dump page and the Node-side conformance asserter to convert
// f32 ↔ f16 with round-to-nearest-even semantics. Keeping the
// conversion here in plain ESM means the browser HTML and Node tests
// see identical rounding behavior.

/** Round-to-nearest-even f32 → f16, returns a uint16. */
export function f32_to_f16(val) {
  const f = new Float32Array(1);
  f[0] = val;
  const u = new Uint32Array(f.buffer)[0];
  const sign = (u >>> 16) & 0x8000;
  const exp = (u >>> 23) & 0xff;
  const mant = u & 0x7fffff;
  if (exp === 0xff) return (sign | 0x7c00 | (mant ? 0x0200 : 0)) & 0xffff; // inf/NaN
  if (exp === 0) return sign & 0xffff; // ±0 / denormal flushed
  let e = exp - 127 + 15;
  if (e >= 0x1f) return (sign | 0x7c00) & 0xffff; // overflow → ±inf
  if (e <= 0) {
    // Subnormal — flush small values to ±0 (the model only uses normal range).
    return sign & 0xffff;
  }
  const m = mant >>> 13;
  const roundBit = (mant >>> 12) & 1;
  const sticky = (mant & 0xfff) !== 0;
  let h = sign | (e << 10) | m;
  if (roundBit && (sticky || (m & 1))) h++;
  return h & 0xffff;
}

/** f16 (uint16) → f32 — exact, no rounding (f16 ⊂ f32). */
export function f16_to_f32(h) {
  const sign = (h & 0x8000) >> 15;
  const exp = (h & 0x7c00) >> 10;
  const mant = h & 0x3ff;
  if (exp === 0) {
    if (mant === 0) return sign ? -0 : 0;
    return (sign ? -1 : 1) * Math.pow(2, -14) * (mant / 1024);
  }
  if (exp === 0x1f) return mant ? NaN : sign ? -Infinity : Infinity;
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + mant / 1024);
}

/** Pack a Float32Array into a Uint16Array of f16 bits. */
export function pack_f16(values) {
  const out = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = f32_to_f16(values[i]);
  return out;
}

/** Unpack a Uint16Array of f16 bits into a Float32Array. */
export function unpack_f16(bits) {
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i++) out[i] = f16_to_f32(bits[i]);
  return out;
}

/**
 * Seeded LCG → uniform float in [-1, 1). Deterministic across runs
 * so the browser-dumped fixtures and any future regenerations match.
 */
export function makeRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return (s / 0x100000000) * 2 - 1;
  };
}
