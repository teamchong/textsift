// Smoke test: Dawn-direct rms_norm via the dawn* NAPI surface.
// Validates the full WGSL → Tint → SPIR-V → Vulkan → execute path
// using the canonical browser shader source.

import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURE = resolve(HERE, "../conformance/fixtures/rms_norm");
const NATIVE_PATH = resolve(HERE, "../../../packages/textsift/dist/textsift-native.node");
const native = createRequire(import.meta.url)(NATIVE_PATH);

const handle = native.dawnCreateBackend();
console.log(`[dawn] device: ${native.dawnDeviceName(handle)}`);

const uniform = new Uint8Array(readFileSync(resolve(FIXTURE, "uniform.bin")));
const x = new Uint8Array(readFileSync(resolve(FIXTURE, "x.bin")));
const gamma = new Uint8Array(readFileSync(resolve(FIXTURE, "gamma.bin")));
const expected = new Uint8Array(readFileSync(resolve(FIXTURE, "expected.bin")));

const xBuf = native.dawnCreateBuffer(handle, x);
const gBuf = native.dawnCreateBuffer(handle, gamma);
const yBuf = native.dawnCreateEmptyBuffer(handle, expected.byteLength);

// WGSL `rms_norm` declares:
//   binding(0) uniform Dims
//   binding(1) storage X
//   binding(2) storage Gamma
//   binding(3) storage Y
// Bridge handles uniform internally; bindings[] is storage in WGSL-binding order.
native.dawnDispatchOneShot(
  handle,
  "rms_norm",
  [xBuf, gBuf, yBuf],
  uniform,
  [4, 1, 1],
);

const got = native.dawnReadBuffer(handle, yBuf, 0, expected.byteLength);

let byteEqual = true;
let firstDiff = -1;
for (let i = 0; i < expected.length; i++) {
  if (got[i] !== expected[i]) { byteEqual = false; firstDiff = i; break; }
}

if (byteEqual) {
  console.log(`[dawn/rms_norm] OK byte-equal (${expected.byteLength} bytes)`);
} else {
  let diffCount = 0;
  for (let i = 0; i < expected.length; i++) if (got[i] !== expected[i]) diffCount++;
  console.log(`[dawn/rms_norm] MISMATCH at byte ${firstDiff}: expected=${expected[firstDiff]} got=${got[firstDiff]}, ${diffCount}/${expected.length} differ`);
}

native.dawnReleaseBuffer(handle, xBuf);
native.dawnReleaseBuffer(handle, gBuf);
native.dawnReleaseBuffer(handle, yBuf);
native.dawnDestroyBackend(handle);

process.exit(byteEqual ? 0 : 1);
