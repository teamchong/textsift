// Smoke test: Vulkan-direct rms_norm via the vulkan* NAPI surface.
// Byte-equal vs the browser-dumped fixture proves:
//   - GLSL → SPIR-V codegen is correct
//   - bridge.c instance/device/buffer/pipeline/cmd flow works
//   - vulkan_backend.zig wrapper marshals JS args correctly
//   - napi.zig vulkan* surface plumbs everything through
//
// If this passes, the toolchain is ready for the remaining 14 kernels.

import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURE = resolve(HERE, "../conformance/fixtures/rms_norm");
const NATIVE_PATH = resolve(HERE, "../../../packages/textsift/dist/textsift-native.node");
const native = createRequire(import.meta.url)(NATIVE_PATH);

const handle = native.vulkanCreateBackend();
console.log(`[vulkan] device: ${native.vulkanDeviceName(handle)}`);

const uniform = new Uint8Array(readFileSync(resolve(FIXTURE, "uniform.bin")));
const x = new Uint8Array(readFileSync(resolve(FIXTURE, "x.bin")));
const gamma = new Uint8Array(readFileSync(resolve(FIXTURE, "gamma.bin")));
const expected = new Uint8Array(readFileSync(resolve(FIXTURE, "expected.bin")));

console.log(
  `[vulkan/rms_norm] uniform=${uniform.byteLength}B x=${x.byteLength}B ` +
    `gamma=${gamma.byteLength}B expected=${expected.byteLength}B`,
);

const xBuf = native.vulkanCreateBuffer(handle, x);
const gBuf = native.vulkanCreateBuffer(handle, gamma);
const yBuf = native.vulkanCreateEmptyBuffer(handle, expected.byteLength);

// Push-constant block carries Dims { T, D, eps, _pad } — same 16-byte
// layout as the WGSL `var<uniform>` binding (which we replaced with
// push constants on Vulkan; saves a buffer + descriptor slot).
// Bindings are ordered by GLSL `layout(set=0, binding=K)` slot:
//   binding 0: x   (readonly SSBO)
//   binding 1: gamma (readonly SSBO)
//   binding 2: y   (writable SSBO)
native.vulkanDispatchOneShot(
  handle,
  "rms_norm",
  [xBuf, gBuf, yBuf],
  uniform,
  [4, 1, 1],
);

const got = native.vulkanReadBuffer(handle, yBuf, 0, expected.byteLength);

let byteEqual = true;
let firstDiff = -1;
for (let i = 0; i < expected.length; i++) {
  if (got[i] !== expected[i]) {
    byteEqual = false;
    firstDiff = i;
    break;
  }
}

if (byteEqual) {
  console.log(`[vulkan/rms_norm] OK byte-equal (${expected.byteLength} bytes)`);
} else {
  console.log(`[vulkan/rms_norm] MISMATCH at byte ${firstDiff}: expected=${expected[firstDiff]} got=${got[firstDiff]}`);
  // Tally how many bytes differ in case it's drift not corruption
  let diffCount = 0;
  for (let i = 0; i < expected.length; i++) if (got[i] !== expected[i]) diffCount++;
  console.log(`  ${diffCount}/${expected.length} bytes differ`);
}

native.vulkanReleaseBuffer(handle, xBuf);
native.vulkanReleaseBuffer(handle, gBuf);
native.vulkanReleaseBuffer(handle, yBuf);
native.vulkanDestroyBackend(handle);

process.exit(byteEqual ? 0 : 1);
