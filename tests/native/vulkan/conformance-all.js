// Aggregate Vulkan-direct conformance: every shader in the registry,
// run via the vulkan* NAPI surface against the same browser-dumped
// fixtures the wgpu-native and Metal-direct harnesses use. A pass means
// the GLSL → SPIR-V → Vulkan compute pipeline is byte-equal (or
// within fp16/fp32 ε) to Tint's WebGPU output.
//
// Mapping notes vs the wgpu-native harness:
//   - Uniform buffer + extraUniforms collapse into one push-constant
//     block (concatenated in binding order).
//   - SSBO bindings start at slot 0 (not 1), since we removed the
//     uniform-at-binding-0 indirection. Each kernel's slot order =
//     (inputs sorted by binding ascending) followed by the output.
//   - Initial state for in-out buffers (`_sum_out_init`, `_out_idx_init`,
//     `_output_initial.bin`) is loaded into the buffer at create time.

import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { f16_to_f32 } from "../../conformance/fp16.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const NATIVE_PATH = resolve(HERE, "../../../packages/textsift/dist/textsift-native.node");
const FIXTURES_DIR = resolve(HERE, "../conformance/fixtures");
const native = createRequire(import.meta.url)(NATIVE_PATH);

function readBin(file) {
  return new Uint8Array(readFileSync(file));
}

// Per-kernel fp16 ULP tolerance overrides. Default is 1 ULP (strict).
// Some kernels have intrinsic fp16 hardware variance that's not a kernel
// bug — list them here with a justified looser bound.
const F16_ULP_TOLERANCE = {
  // rope_apply computes `a*c - b*s` which cancels when the rotation
  // angle is small. Even with f32 intermediates (we promote in the
  // GLSL — see rope_apply.comp.glsl), the final fp16 narrow-conversion
  // routine differs across drivers: real GPUs vs Mesa llvmpipe (CI's
  // software renderer) round differently in the cancellation region.
  // Measured drift is ~16 ULPs on llvmpipe, ≤1 ULP on real GPUs. Allow
  // 32 to absorb cross-driver variance without hiding kernel bugs;
  // absolute drift remains ~1e-3 either way.
  rope_apply: 32,
};

function runVulkanConformance(name, handle) {
  const dir = resolve(FIXTURES_DIR, name);
  const meta = JSON.parse(readFileSync(resolve(dir, "meta.json"), "utf8"));

  // Build push-constant payload: uniform.bin || _uniform_<name>.bin (in binding order).
  const uniformBytes = readBin(resolve(dir, "uniform.bin"));
  const extras = (meta.extraUniforms || []).slice().sort((a, b) => a.binding - b.binding);
  const extraBytes = extras.map((eu) => readBin(resolve(dir, `_uniform_${eu.name}.bin`)));
  const pushTotal = uniformBytes.byteLength + extraBytes.reduce((s, b) => s + b.byteLength, 0);
  const push = new Uint8Array(pushTotal);
  push.set(uniformBytes, 0);
  let off = uniformBytes.byteLength;
  for (const b of extraBytes) {
    push.set(b, off);
    off += b.byteLength;
  }

  // Build SSBO list. Vulkan slot order = WGSL binding number minus the
  // number of uniform bindings collapsed into push constants (always 1
  // for `dims`, plus 1 per extraUniform). For rope_apply the WGSL output
  // (qk, binding 1) precedes its inputs (cos_tab/sin_tab at 2/3), so
  // we can't just append output last — sort everything together by
  // WGSL binding to get the correct slot order.
  const allEntries = [];
  for (const i of meta.inputs || []) {
    const buf = native.vulkanCreateBuffer(handle, readBin(resolve(dir, `${i.name}.bin`)));
    allEntries.push({ binding: i.binding, buf, isOutput: false });
  }
  let outBuf;
  if (meta.output.hasInitial) {
    outBuf = native.vulkanCreateBuffer(handle, readBin(resolve(dir, "_output_initial.bin")));
  } else {
    outBuf = native.vulkanCreateEmptyBuffer(handle, meta.output.byteLength);
  }
  allEntries.push({ binding: meta.output.binding, buf: outBuf, isOutput: true });
  allEntries.sort((a, b) => a.binding - b.binding);
  const bindings = allEntries.map((e) => e.buf);
  const inputBufs = allEntries.filter((e) => !e.isOutput).map((e) => e.buf);

  console.log(
    `[${name}] uniform=${uniformBytes.byteLength}B extras=${extras.length} ` +
      `inputs=${(meta.inputs || []).length} output=${meta.output.byteLength}B (${meta.output.dtype}) ` +
      `dispatch=${JSON.stringify(meta.dispatch)}`,
  );

  native.vulkanDispatchOneShot(handle, name, bindings, push, meta.dispatch);

  const actual = native.vulkanReadBuffer(handle, outBuf, 0, meta.output.byteLength);
  const expected = readBin(resolve(dir, "expected.bin"));

  for (const buf of inputBufs) native.vulkanReleaseBuffer(handle, buf);
  native.vulkanReleaseBuffer(handle, outBuf);

  if (actual.byteLength !== expected.byteLength) {
    throw new Error(`output length mismatch: got ${actual.byteLength}, expected ${expected.byteLength}`);
  }

  let byteEqual = true;
  for (let i = 0; i < expected.length; i++) {
    if (actual[i] !== expected[i]) {
      byteEqual = false;
      break;
    }
  }
  if (byteEqual) {
    console.log(`OK [${name}] byte-equal vs browser fixture (${expected.byteLength} bytes)`);
    return { byteEqual: true };
  }

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
    const ulpLimit = F16_ULP_TOLERANCE[name] ?? 1;
    console.log(
      `[${name}] f16 drift: maxAbs=${maxAbs.toExponential(2)} ` +
        `maxUlp=${maxUlp.toFixed(2)} >1ULP=${mismatch}/${expU16.length} (limit=${ulpLimit})`,
    );
    if (maxUlp > ulpLimit + 0.001) {
      throw new Error(`${name}: drift > ${ulpLimit} fp16 ULP (max=${maxUlp})`);
    }
    console.log(`OK [${name}] within ${ulpLimit} fp16 ULP of browser fixture`);
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
      `[${name}] f32 drift: maxAbs=${maxAbs.toExponential(2)} ` +
        `maxRel=${maxRel.toExponential(2)} >1e-3=${mismatch}/${expF32.length}`,
    );
    if (maxRel > 1e-2) {
      throw new Error(`${name}: relative drift > 1% (max=${maxRel})`);
    }
    console.log(`OK [${name}] within 1% relative drift of browser fixture`);
    return { byteEqual: false, maxRel };
  }
  // u32 / int / other: must be byte-equal
  let firstDiff = -1;
  for (let i = 0; i < expected.length; i++) {
    if (actual[i] !== expected[i]) { firstDiff = i; break; }
  }
  throw new Error(
    `${name}: bytes differ at index ${firstDiff} ` +
      `(actual=${actual[firstDiff]}, expected=${expected[firstDiff]}); ` +
      `dtype=${meta.output.dtype} requires byte-equality`,
  );
}

const SHADERS = [
  "rms_norm",
  "zero_f32",
  "cast_fp16_to_f32",
  "cast_f32_to_fp16_scaled",
  "add_fp16",
  "swiglu_clamp",
  "rope_apply",
  "matmul_int4_fp16_f16",
  "matmul_int4_f32_f32",
  "embed_lookup_int4",
  "add_rmsnorm_fp16_to_f32",
  "router_topk",
  "banded_attention",
  "qmoe_gate_up",
  "qmoe_down_scatter",
];

const handle = native.vulkanCreateBackend();
console.log(`[vulkan] device: ${native.vulkanDeviceName(handle)}\n`);

let failed = 0;
const skip = (process.env.SKIP ?? "").split(",").filter(Boolean);
const only = process.env.ONLY ? process.env.ONLY.split(",") : null;
for (const name of SHADERS) {
  if (skip.includes(name)) { console.log(`SKIP [${name}]`); continue; }
  if (only && !only.includes(name)) continue;
  try {
    runVulkanConformance(name, handle);
  } catch (e) {
    failed++;
    console.error(`FAIL [${name}]: ${e.message}\n`);
  }
}
native.vulkanDestroyBackend(handle);

if (failed > 0) {
  console.error(`\n${failed} shader(s) failed Vulkan-direct conformance`);
  process.exit(1);
}
console.log(`\nALL ${SHADERS.length} shaders pass Vulkan-direct conformance`);
