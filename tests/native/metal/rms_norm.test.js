// Conformance + bench: hand-written MSL rms_norm via Metal-direct
// vs the WGSL reference (browser-dumped fixture). If the MSL is
// correct AND faster than the wgpu-native path, this validates the
// Metal-direct strategy and we scale up to the remaining shaders.

import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { strict as assert } from "node:assert";
import { f16_to_f32 } from "../../conformance/fp16.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const NATIVE_PATH = resolve(HERE, "../../../packages/textsift/dist/textsift-native.node");
const FIXTURE = resolve(HERE, "../conformance/fixtures/rms_norm");
const native = createRequire(import.meta.url)(NATIVE_PATH);

assert.equal(typeof native.metalCreateBackend, "function", "metalCreateBackend missing");

const meta = JSON.parse(readFileSync(resolve(FIXTURE, "meta.json"), "utf8"));
const uniform = new Uint8Array(readFileSync(resolve(FIXTURE, "uniform.bin")));
const x = new Uint8Array(readFileSync(resolve(FIXTURE, "x.bin")));
const gamma = new Uint8Array(readFileSync(resolve(FIXTURE, "gamma.bin")));
const expected = new Uint8Array(readFileSync(resolve(FIXTURE, "expected.bin")));

console.log("[metal] creating backend");
const b = native.metalCreateBackend();
console.log("[metal] device:", native.metalDeviceName(b));

const xBuf = native.metalCreateBuffer(b, x);
const gammaBuf = native.metalCreateBuffer(b, gamma);
const yBuf = native.metalCreateBuffer(b, new Uint8Array(expected.byteLength));

// Dispatch: one threadgroup per row (T threadgroups), 64 threads each.
// dimsBuf: T (4), D (128), eps (1e-6), pad
const T = 4;
native.metalDispatchOneShot(
  b,
  "rms_norm",
  [
    { index: 0, bytes: uniform },     // dims
    { index: 1, bufPtr: xBuf },       // x
    { index: 2, bufPtr: gammaBuf },   // gamma
    { index: 3, bufPtr: yBuf },       // y (output)
  ],
  [T, 1, 1],   // grid = threadgroups
  [64, 1, 1],  // threadgroup size
);

const got = native.metalReadBuffer(yBuf, 0, expected.byteLength);

let byteEqual = true;
for (let i = 0; i < expected.length; i++) {
  if (got[i] !== expected[i]) { byteEqual = false; break; }
}
if (byteEqual) {
  console.log(`OK [metal/rms_norm] byte-equal vs browser fixture (${expected.byteLength} bytes)`);
} else {
  // fp16 ULP comparison
  const eU = new Uint16Array(expected.buffer, expected.byteOffset, expected.byteLength / 2);
  const gU = new Uint16Array(got.buffer, got.byteOffset, got.byteLength / 2);
  let maxUlp = 0, mismatch = 0;
  for (let i = 0; i < eU.length; i++) {
    const a = f16_to_f32(gU[i]);
    const e = f16_to_f32(eU[i]);
    const abs = Math.abs(a - e);
    const mag = Math.max(Math.abs(e), Math.pow(2, -14));
    const ulp = Math.pow(2, Math.floor(Math.log2(mag)) - 10);
    const u = abs / ulp;
    if (u > maxUlp) maxUlp = u;
    if (u > 1) mismatch++;
  }
  console.log(`[metal/rms_norm] f16 drift maxUlp=${maxUlp.toFixed(2)} >1ULP=${mismatch}/${eU.length}`);
  assert.ok(maxUlp <= 1.001, `metal rms_norm drift > 1 ULP (${maxUlp})`);
  console.log(`OK [metal/rms_norm] within 1 fp16 ULP of browser fixture`);
}

native.metalReleaseBuffer(xBuf);
native.metalReleaseBuffer(gammaBuf);
native.metalReleaseBuffer(yBuf);
native.metalDestroyBackend(b);
console.log("[metal] backend disposed");
