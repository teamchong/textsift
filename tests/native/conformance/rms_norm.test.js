// Conformance: native rms_norm dispatch (wgpu-native + Metal) vs the
// browser-dumped fixture (Chromium Dawn + Metal). Same WGSL on both
// sides — Naga vs Tint compile it to slightly different MSL, so we
// assert within 1 fp16 ULP per element rather than byte-equal.
//
// Fixture is dumped by `tests/conformance/dump-fixtures.spec.ts`
// from a deterministic seeded input. Re-running the dump regenerates
// the same bytes.

import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { strict as assert } from "node:assert";
import { f16_to_f32 } from "../../conformance/fp16.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const NATIVE_PATH = resolve(HERE, "../../../packages/textsift/dist/textsift-native.node");
const FIXTURE_DIR = resolve(HERE, "fixtures/rms_norm");

const native = createRequire(import.meta.url)(NATIVE_PATH);
assert.equal(typeof native.dispatchRmsnorm, "function");

function readFixture(name) {
  return new Uint8Array(readFileSync(resolve(FIXTURE_DIR, name)));
}

const meta = JSON.parse(readFileSync(resolve(FIXTURE_DIR, "meta.json"), "utf8"));
console.log(
  `[rms_norm] dims=${JSON.stringify(meta.dims)} dispatch=${JSON.stringify(meta.dispatch)} ` +
    `output=${meta.outputBytes}B (${meta.outputDtype})`,
);

const uniform = readFixture("uniform.bin");
const x = readFixture("x.bin");
const gamma = readFixture("gamma.bin");
const expected = readFixture("expected.bin");

const dispatchX = meta.dispatch[0];
const output = native.dispatchRmsnorm(uniform, x, gamma, dispatchX, expected.byteLength);

assert.equal(output.byteLength, expected.byteLength, "output length mismatch");

// Byte-equal check first — if true, no rounding drift between Dawn
// and Naga. Most kernels won't hit this since the compilers can
// reorder fp ops, but it's the gold standard so log when we get it.
let byteEqual = true;
for (let i = 0; i < expected.length; i++) {
  if (output[i] !== expected[i]) { byteEqual = false; break; }
}
if (byteEqual) {
  console.log(`OK: rms_norm byte-equal vs browser fixture (${expected.byteLength} bytes)`);
  process.exit(0);
}

// Fall back to fp16 ULP comparison. Decode pairs of bytes as f16,
// convert to f32, assert |delta| ≤ 1 ULP at the element's
// magnitude. ULP at value v ≈ 2^(floor(log2|v|) - 10).
const expU16 = new Uint16Array(expected.buffer, expected.byteOffset, expected.byteLength / 2);
const outU16 = new Uint16Array(output.buffer, output.byteOffset, output.byteLength / 2);

let maxAbsDelta = 0;
let maxUlpDelta = 0;
let mismatchCount = 0;
for (let i = 0; i < expU16.length; i++) {
  const a = f16_to_f32(outU16[i]);
  const e = f16_to_f32(expU16[i]);
  const abs = Math.abs(a - e);
  if (abs > maxAbsDelta) maxAbsDelta = abs;
  // ULP at e
  const mag = Math.max(Math.abs(e), Math.pow(2, -14));
  const ulp = Math.pow(2, Math.floor(Math.log2(mag)) - 10);
  const ulps = abs / ulp;
  if (ulps > maxUlpDelta) maxUlpDelta = ulps;
  if (ulps > 1) mismatchCount++;
}

console.log(
  `rms_norm (within-fp16-ULP comparison): ` +
    `maxAbsDelta=${maxAbsDelta.toExponential(2)}, ` +
    `maxUlpDelta=${maxUlpDelta.toFixed(2)}, ` +
    `>1ULP elements=${mismatchCount}/${expU16.length}`,
);
// Allow up to 1 ULP per element — any more indicates a kernel bug.
assert.ok(maxUlpDelta <= 1.001, `rms_norm drift > 1 ULP (max=${maxUlpDelta})`);
console.log(`OK: rms_norm within 1 fp16 ULP of browser fixture`);
