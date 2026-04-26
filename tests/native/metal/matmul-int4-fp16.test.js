// Conformance: hand-MSL matmul_int4_fp16_f16 vs browser fixture.

import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { strict as assert } from "node:assert";
import { f16_to_f32 } from "../../conformance/fp16.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const NATIVE_PATH = resolve(HERE, "../../../packages/textsift/dist/textsift-native.node");
const FIXTURE = resolve(HERE, "../conformance/fixtures/matmul_int4_fp16_f16");
const native = createRequire(import.meta.url)(NATIVE_PATH);

const meta = JSON.parse(readFileSync(resolve(FIXTURE, "meta.json"), "utf8"));
const uniform = new Uint8Array(readFileSync(resolve(FIXTURE, "uniform.bin")));
const x = new Uint8Array(readFileSync(resolve(FIXTURE, "x.bin")));
const w_int4 = new Uint8Array(readFileSync(resolve(FIXTURE, "w_int4.bin")));
const w_scales = new Uint8Array(readFileSync(resolve(FIXTURE, "w_scales.bin")));
const w_zp = new Uint8Array(readFileSync(resolve(FIXTURE, "w_zp.bin")));
const bias = new Uint8Array(readFileSync(resolve(FIXTURE, "bias.bin")));
const expected = new Uint8Array(readFileSync(resolve(FIXTURE, "expected.bin")));

const b = native.metalCreateBackend();
console.log("[metal] device:", native.metalDeviceName(b));

const xBuf = native.metalCreateBuffer(b, x);
const wBuf = native.metalCreateBuffer(b, w_int4);
const sBuf = native.metalCreateBuffer(b, w_scales);
const zBuf = native.metalCreateBuffer(b, w_zp);
const bBuf = native.metalCreateBuffer(b, bias);
const yBuf = native.metalCreateBuffer(b, new Uint8Array(expected.byteLength));

const grid = meta.dispatch; // [ceil(N/64), ceil(T/4), 1]
native.metalDispatchOneShot(
  b, "matmul_int4_fp16_f16",
  [
    { index: 0, bytes: uniform },
    { index: 1, bufPtr: xBuf },
    { index: 2, bufPtr: wBuf },
    { index: 3, bufPtr: sBuf },
    { index: 4, bufPtr: zBuf },
    { index: 5, bufPtr: bBuf },
    { index: 6, bufPtr: yBuf },
  ],
  grid, [64, 1, 1],
);
const got = native.metalReadBuffer(yBuf, 0, expected.byteLength);

let byteEqual = true;
for (let i = 0; i < expected.length; i++) if (got[i] !== expected[i]) { byteEqual = false; break; }
if (byteEqual) {
  console.log(`OK [metal/matmul_int4_fp16_f16] byte-equal vs browser (${expected.byteLength} bytes)`);
} else {
  const eU = new Uint16Array(expected.buffer, expected.byteOffset, expected.byteLength / 2);
  const gU = new Uint16Array(got.buffer, got.byteOffset, got.byteLength / 2);
  let maxUlp = 0, mismatch = 0;
  for (let i = 0; i < eU.length; i++) {
    const a = f16_to_f32(gU[i]); const e = f16_to_f32(eU[i]);
    const abs = Math.abs(a - e);
    const mag = Math.max(Math.abs(e), Math.pow(2, -14));
    const ulp = Math.pow(2, Math.floor(Math.log2(mag)) - 10);
    const u = abs / ulp;
    if (u > maxUlp) maxUlp = u;
    if (u > 1) mismatch++;
  }
  console.log(`[metal/matmul_int4_fp16_f16] f16 drift maxUlp=${maxUlp.toFixed(2)} >1ULP=${mismatch}/${eU.length}`);
  assert.ok(maxUlp <= 2.0, `metal matmul drift > 2 ULP (${maxUlp})`);
  console.log(`OK [metal/matmul_int4_fp16_f16] within 2 fp16 ULP of browser fixture`);
}

native.metalReleaseBuffer(xBuf);
native.metalReleaseBuffer(wBuf);
native.metalReleaseBuffer(sBuf);
native.metalReleaseBuffer(zBuf);
native.metalReleaseBuffer(bBuf);
native.metalReleaseBuffer(yBuf);
native.metalDestroyBackend(b);
