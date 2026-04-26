// Verify the native binding can request a wgpu Device with the
// same shader-f16 + limit clamps the browser WebGpuBackend asks
// for. If this passes, we have a usable device on which to compile
// pipelines and run dispatches — the next step toward porting the
// WebGpuBackend.

import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { strict as assert } from "node:assert";

const HERE = dirname(fileURLToPath(import.meta.url));
const NATIVE_PATH = resolve(HERE, "../../packages/textsift/dist/textsift-native.node");
const native = createRequire(import.meta.url)(NATIVE_PATH);

assert.equal(typeof native.getDeviceInfo, "function", "expected getDeviceInfo export");
const info = native.getDeviceInfo();
console.log("device info:", info);

assert.equal(typeof info.adapter, "object");
assert.ok(info.adapter.device.length > 0, "adapter device name non-empty");

// shader-f16 is required for the int4 matmul path; if the call
// returned at all, the feature was negotiated successfully (otherwise
// probeDevice() would have thrown ShaderF16Unavailable).
assert.equal(typeof info.maxStorageBufferBindingSize, "bigint");
assert.equal(typeof info.maxBufferSize, "bigint");

// Every kernel needs at least 256 MiB of storage buffer to fit weight
// blocks; the browser path clamps to 1 GiB. Anything below 256 MiB
// would fail at warmup, so check the floor here.
const ONE_MIB = 1024n * 1024n;
assert.ok(
  info.maxStorageBufferBindingSize >= 256n * ONE_MIB,
  `maxStorageBufferBindingSize too small: ${info.maxStorageBufferBindingSize}`,
);

// Compute capabilities that matter for the WGSL kernels:
//   - workgroup_storage_size ≥ 16 KiB so banded_attention can stash
//     scores + softmax + reductions for one (t_query, h_query) tile
//   - invocations_per_workgroup ≥ 256 so matmul tiles fit
//   - workgroup_size_x ≥ 256 for the same reason
assert.ok(
  info.maxComputeWorkgroupStorageSize >= 16384,
  `compute workgroup storage too small: ${info.maxComputeWorkgroupStorageSize}`,
);
assert.ok(
  info.maxComputeInvocationsPerWorkgroup >= 256,
  `compute invocations/workgroup too small: ${info.maxComputeInvocationsPerWorkgroup}`,
);
assert.ok(
  info.maxComputeWorkgroupSizeX >= 256,
  `compute workgroup size x too small: ${info.maxComputeWorkgroupSizeX}`,
);

console.log("OK: native device meets the limits the browser path requires");
