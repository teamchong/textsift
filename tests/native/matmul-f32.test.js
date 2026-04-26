// fp32 matmul on the GPU vs the same computation in JS. Bit-equality
// isn't guaranteed across (CPU, fp32-FMA, summation order), so we use
// a tight relative tolerance — fp32 matmul typically rounds to within
// ~1e-5 across implementations.
//
// This is the harness that the int4-fp16 conformance port (next
// milestone) inherits: same shape (deterministic input → reference
// output → assert close), just different reference source (CPU here,
// browser-WebGPU output binary for int4-fp16).

import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { strict as assert } from "node:assert";

const HERE = dirname(fileURLToPath(import.meta.url));
const NATIVE_PATH = resolve(HERE, "../../packages/textsift/dist/textsift-native.node");
const native = createRequire(import.meta.url)(NATIVE_PATH);

assert.equal(typeof native.matmulF32, "function");

// Seeded LCG so test inputs are deterministic across runs.
function lcg(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

function cpuMatmul(a, b, m, n, k) {
  const out = new Float32Array(m * n);
  for (let row = 0; row < m; row++) {
    for (let col = 0; col < n; col++) {
      let acc = 0;
      for (let i = 0; i < k; i++) acc += a[row * k + i] * b[i * n + col];
      out[row * n + col] = acc;
    }
  }
  return out;
}

function maxRelErr(actual, expected) {
  let max = 0;
  for (let i = 0; i < actual.length; i++) {
    const a = actual[i];
    const e = expected[i];
    const denom = Math.max(1e-6, Math.abs(e));
    const rel = Math.abs(a - e) / denom;
    if (rel > max) max = rel;
  }
  return max;
}

const cases = [
  { m: 4, n: 4, k: 4 },
  { m: 8, n: 16, k: 32 },
  { m: 32, n: 64, k: 128 },
];

for (const { m, n, k } of cases) {
  const rng = lcg(12345);
  const a = new Float32Array(m * k);
  const b = new Float32Array(k * n);
  for (let i = 0; i < a.length; i++) a[i] = rng() * 2 - 1;
  for (let i = 0; i < b.length; i++) b[i] = rng() * 2 - 1;

  const expected = cpuMatmul(a, b, m, n, k);
  const actual = native.matmulF32(a, b, m, n, k);
  const err = maxRelErr(actual, expected);
  console.log(`  ${m}×${k} · ${k}×${n}: max relative error ${err.toExponential(2)}`);
  // GPU and CPU sum reductions in different orders, and Metal/D3D
  // FMA (single-rounding) differs from CPU sequential mul+add (two
  // roundings). Drift scales with K. ~1e-2 is normal for K=128 fp32
  // matmul across implementations; the int4-fp16 conformance
  // milestone (next) compares native vs browser-WebGPU (same WGSL
  // on both sides) and there we expect bit equality.
  assert.ok(err < 1e-2, `matmul ${m}×${n}×${k} drift too large: ${err}`);
}

console.log("OK: native fp32 matmul matches CPU reference within tolerance");
