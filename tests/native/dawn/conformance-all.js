// Aggregate Dawn-direct conformance: every shader in the registry,
// run via the dawn* NAPI surface against the same browser-dumped
// fixtures the Vulkan-direct harness uses. A pass means Dawn's Tint
// produces byte-equal (or within fp ε) output for our hand-written
// WGSL kernels.

import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { f16_to_f32 } from "../../conformance/fp16.js";

// Surface the actual platform/architecture/load failure on Windows
// (where pwsh's default invocation has been swallowing late stderr)
// so future CI runs aren't silent on .node load problems.
process.on("uncaughtException", (e) => {
  console.error(`[dawn-conformance] uncaught: ${e.stack ?? e}`);
  process.exit(1);
});
console.error(`[dawn-conformance] node=${process.version} platform=${process.platform} arch=${process.arch}`);

const HERE = dirname(fileURLToPath(import.meta.url));
const NATIVE_PATH = resolve(HERE, "../../../packages/textsift/dist/textsift-native.node");
const FIXTURES_DIR = resolve(HERE, "../conformance/fixtures");
console.error(`[dawn-conformance] loading native from: ${NATIVE_PATH}`);
let native;
try {
  native = createRequire(import.meta.url)(NATIVE_PATH);
} catch (e) {
  console.error(`[dawn-conformance] failed to load .node: ${e.stack ?? e}`);
  process.exit(1);
}
console.error(`[dawn-conformance] native loaded ok`);

function readBin(file) {
  return new Uint8Array(readFileSync(file));
}

function runDawnConformance(name, handle) {
  const dir = resolve(FIXTURES_DIR, name);
  const meta = JSON.parse(readFileSync(resolve(dir, "meta.json"), "utf8"));

  // Build uniform payload from main + extras, in binding order.
  const uniformBytes = readBin(resolve(dir, "uniform.bin"));
  const extras = (meta.extraUniforms || []).slice().sort((a, b) => a.binding - b.binding);
  const extraBytes = extras.map((eu) => readBin(resolve(dir, `_uniform_${eu.name}.bin`)));
  const pushTotal = uniformBytes.byteLength + extraBytes.reduce((s, b) => s + b.byteLength, 0);
  const uni = new Uint8Array(pushTotal);
  uni.set(uniformBytes, 0);
  let off = uniformBytes.byteLength;
  for (const b of extraBytes) {
    uni.set(b, off);
    off += b.byteLength;
  }

  // For Dawn: WGSL uniform stays at binding 0 (handled by the bridge),
  // storage buffers are at WGSL bindings 1..N+1. Sort all storage
  // entries (inputs + output) by binding ascending to match WGSL
  // declaration order.
  const allEntries = [];
  for (const i of meta.inputs || []) {
    const buf = native.dawnCreateBuffer(handle, readBin(resolve(dir, `${i.name}.bin`)));
    allEntries.push({ binding: i.binding, buf, isOutput: false });
  }
  let outBuf;
  if (meta.output.hasInitial) {
    outBuf = native.dawnCreateBuffer(handle, readBin(resolve(dir, "_output_initial.bin")));
  } else {
    outBuf = native.dawnCreateEmptyBuffer(handle, meta.output.byteLength);
  }
  allEntries.push({ binding: meta.output.binding, buf: outBuf, isOutput: true });
  allEntries.sort((a, b) => a.binding - b.binding);
  const bindings = allEntries.map((e) => e.buf);

  console.log(
    `[${name}] uniform=${uniformBytes.byteLength}B extras=${extras.length} ` +
      `inputs=${(meta.inputs || []).length} output=${meta.output.byteLength}B (${meta.output.dtype}) ` +
      `dispatch=${JSON.stringify(meta.dispatch)}`,
  );

  native.dawnDispatchOneShot(handle, name, bindings, uni, meta.dispatch);

  const actual = native.dawnReadBuffer(handle, outBuf, 0, meta.output.byteLength);
  const expected = readBin(resolve(dir, "expected.bin"));

  for (const e of allEntries) native.dawnReleaseBuffer(handle, e.buf);

  let byteEqual = true;
  for (let i = 0; i < expected.length; i++) {
    if (actual[i] !== expected[i]) { byteEqual = false; break; }
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
    console.log(`[${name}] f16 drift: maxAbs=${maxAbs.toExponential(2)} maxUlp=${maxUlp.toFixed(2)} >1ULP=${mismatch}/${expU16.length}`);
    if (maxUlp > 1.001) throw new Error(`${name}: drift > 1 fp16 ULP (max=${maxUlp})`);
    console.log(`OK [${name}] within 1 fp16 ULP of browser fixture`);
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
    console.log(`[${name}] f32 drift: maxAbs=${maxAbs.toExponential(2)} maxRel=${maxRel.toExponential(2)} >1e-3=${mismatch}/${expF32.length}`);
    if (maxRel > 1e-2) throw new Error(`${name}: relative drift > 1% (max=${maxRel})`);
    console.log(`OK [${name}] within 1% relative drift of browser fixture`);
    return { byteEqual: false, maxRel };
  }
  let firstDiff = -1;
  for (let i = 0; i < expected.length; i++) {
    if (actual[i] !== expected[i]) { firstDiff = i; break; }
  }
  throw new Error(`${name}: bytes differ at index ${firstDiff} (a=${actual[firstDiff]}, e=${expected[firstDiff]}); dtype=${meta.output.dtype}`);
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

const handle = native.dawnCreateBackend();
console.log(`[dawn] device: ${native.dawnDeviceName(handle)}\n`);

let failed = 0;
const skip = (process.env.SKIP ?? "").split(",").filter(Boolean);
const only = process.env.ONLY ? process.env.ONLY.split(",") : null;
for (const name of SHADERS) {
  if (skip.includes(name)) { console.log(`SKIP [${name}]`); continue; }
  if (only && !only.includes(name)) continue;
  try {
    runDawnConformance(name, handle);
  } catch (e) {
    failed++;
    console.error(`FAIL [${name}]: ${e.message}\n`);
  }
}
native.dawnDestroyBackend(handle);

if (failed > 0) {
  console.error(`\n${failed} shader(s) failed Dawn-direct conformance`);
  process.exit(1);
}
console.log(`\nALL ${SHADERS.length} shaders pass Dawn-direct conformance`);
