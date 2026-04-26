// Generic native conformance runner. Reads `meta.json` for a
// shader's binding shape, loads `uniform.bin` + `_uniform_<name>.bin`
// extras + `<input.name>.bin` files + `expected.bin` (+ optional
// `_output_initial.bin`), drives `native.dispatchByName`, asserts
// the output matches `expected.bin` byte-equal (or within 1 fp16
// ULP for f16 outputs where Naga vs Tint MSL gen drifts a single
// rounding bit).
//
// Each shader's conformance test is then a one-liner:
//   import { runConformance } from "./harness.js";
//   await runConformance("rms_norm");

import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { strict as assert } from "node:assert";
import { f16_to_f32 } from "../../conformance/fp16.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const NATIVE_PATH = resolve(HERE, "../../../packages/textsift/dist/textsift-native.node");
const FIXTURES_DIR = resolve(HERE, "fixtures");

const native = createRequire(import.meta.url)(NATIVE_PATH);
assert.equal(typeof native.dispatchByName, "function", "native.dispatchByName missing");

function readBin(file) {
  return new Uint8Array(readFileSync(file));
}

export function runConformance(shaderName) {
  const dir = resolve(FIXTURES_DIR, shaderName);
  const meta = JSON.parse(readFileSync(resolve(dir, "meta.json"), "utf8"));

  const uniform = readBin(resolve(dir, "uniform.bin"));
  const extraUniforms = meta.extraUniforms.map((eu) => ({
    binding: eu.binding,
    bytes: readBin(resolve(dir, `_uniform_${eu.name}.bin`)),
  }));
  const inputs = meta.inputs.map((i) => ({
    binding: i.binding,
    bytes: readBin(resolve(dir, `${i.name}.bin`)),
  }));
  const expected = readBin(resolve(dir, "expected.bin"));
  const output = {
    binding: meta.output.binding,
    byteLength: meta.output.byteLength,
  };
  if (meta.output.hasInitial) {
    output.initial = readBin(resolve(dir, "_output_initial.bin"));
  }

  console.log(
    `[${shaderName}] uniform=${meta.uniform.byteLength}B ` +
      `extras=${extraUniforms.length} inputs=${inputs.length} ` +
      `output=${meta.output.byteLength}B (${meta.output.dtype}) ` +
      `dispatch=${JSON.stringify(meta.dispatch)}`,
  );

  const actual = native.dispatchByName(
    shaderName,
    uniform,
    extraUniforms,
    inputs,
    output,
    meta.dispatch,
  );

  assert.equal(actual.byteLength, expected.byteLength, "output length mismatch");

  // Byte-equal check first.
  let byteEqual = true;
  for (let i = 0; i < expected.length; i++) {
    if (actual[i] !== expected[i]) {
      byteEqual = false;
      break;
    }
  }
  if (byteEqual) {
    console.log(`OK [${shaderName}] byte-equal vs browser fixture (${expected.byteLength} bytes)`);
    return { byteEqual: true };
  }

  // Falls back to dtype-aware comparison.
  if (meta.output.dtype === "f16") {
    const expU16 = new Uint16Array(expected.buffer, expected.byteOffset, expected.byteLength / 2);
    const actU16 = new Uint16Array(actual.buffer, actual.byteOffset, actual.byteLength / 2);
    let maxAbs = 0, maxUlp = 0, mismatch = 0;
    for (let i = 0; i < expU16.length; i++) {
      const a = f16_to_f32(actU16[i]);
      const e = f16_to_f32(expU16[i]);
      const abs = Math.abs(a - e);
      if (abs > maxAbs) maxAbs = abs;
      const mag = Math.max(Math.abs(e), Math.pow(2, -14));
      const ulp = Math.pow(2, Math.floor(Math.log2(mag)) - 10);
      const u = abs / ulp;
      if (u > maxUlp) maxUlp = u;
      if (u > 1) mismatch++;
    }
    console.log(
      `[${shaderName}] f16 drift: maxAbs=${maxAbs.toExponential(2)} ` +
        `maxUlp=${maxUlp.toFixed(2)} >1ULP=${mismatch}/${expU16.length}`,
    );
    assert.ok(maxUlp <= 1.001, `${shaderName}: drift > 1 fp16 ULP (max=${maxUlp})`);
    console.log(`OK [${shaderName}] within 1 fp16 ULP of browser fixture`);
    return { byteEqual: false, maxUlp };
  }
  if (meta.output.dtype === "f32") {
    const expF32 = new Float32Array(expected.buffer, expected.byteOffset, expected.byteLength / 4);
    const actF32 = new Float32Array(actual.buffer, actual.byteOffset, actual.byteLength / 4);
    let maxAbs = 0, maxRel = 0, mismatch = 0;
    for (let i = 0; i < expF32.length; i++) {
      const a = actF32[i];
      const e = expF32[i];
      const abs = Math.abs(a - e);
      if (abs > maxAbs) maxAbs = abs;
      const denom = Math.max(1e-6, Math.abs(e));
      const rel = abs / denom;
      if (rel > maxRel) maxRel = rel;
      if (rel > 1e-3) mismatch++;
    }
    console.log(
      `[${shaderName}] f32 drift: maxAbs=${maxAbs.toExponential(2)} ` +
        `maxRel=${maxRel.toExponential(2)} >1e-3=${mismatch}/${expF32.length}`,
    );
    assert.ok(maxRel <= 1e-2, `${shaderName}: relative drift > 1% (max=${maxRel})`);
    console.log(`OK [${shaderName}] within 1% relative drift of browser fixture`);
    return { byteEqual: false, maxRel };
  }
  // u32 / other: must be byte-equal.
  let firstDiff = -1;
  for (let i = 0; i < expected.length; i++) {
    if (actual[i] !== expected[i]) { firstDiff = i; break; }
  }
  throw new Error(
    `${shaderName}: output bytes differ at index ${firstDiff} ` +
      `(actual=${actual[firstDiff]}, expected=${expected[firstDiff]}); ` +
      `dtype=${meta.output.dtype} requires byte-equality`,
  );
}
