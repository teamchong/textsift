#!/usr/bin/env node
// Kernel parity runner. Loads dist/pii.wasm, replays each kernel against
// fixtures in tests/fixtures/, compares output byte-for-byte against the
// expected output captured from PyTorch. Regression vs the baseline count
// in manifest.json exits non-zero.
//
// Run after `npm run build:zig`:
//
//     node scripts/verify-kernels.mjs
//
// No dependencies. Node 20+.

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, "..");
const WASM_PATH = join(REPO, "dist", "pii.wasm");
const FIXTURES = join(REPO, "tests", "fixtures");

const GREEN = "\x1b[32m";
const RED = "\x1b[31m";
const DIM = "\x1b[2m";
const RESET = "\x1b[0m";

async function loadWasm() {
  const bytes = await readFile(WASM_PATH);
  const { instance } = await WebAssembly.instantiate(bytes, {});
  // Deliberately skip heap_init — lazy init inside alloc/reset/heap_mark_now
  // is the contract we rely on. If this harness is the first thing to call
  // alloc, the first call must still succeed.
  return instance.exports;
}

// Any alloc can grow linear memory and detach previously-taken views;
// always re-derive views from `memory.buffer` after alloc.
function view(ex) {
  return new DataView(ex.memory.buffer);
}
function u8(ex) {
  return new Uint8Array(ex.memory.buffer);
}

async function loadWeights(ex) {
  const blob = await readFile(join(FIXTURES, "pii-weights.bin"));
  const ptr = ex.alloc(blob.byteLength);
  if (ptr === 0) throw new Error("weights alloc OOM");
  u8(ex).set(blob, ptr);
  const rc = ex.weights_load(ptr, blob.byteLength);
  if (rc !== 0) throw new Error(`weights_load returned ${rc}`);
  ex.heap_mark_now();

  const shapeBuf = ex.alloc(16);
  const nameBuf = ex.alloc(64);
  const map = new Map();
  const n = ex.weights_count();
  const dec = new TextDecoder();
  for (let i = 0; i < n; i++) {
    ex.weights_shape(i, shapeBuf);
    const ndim = ex.weights_ndim(i);
    const shape = Array.from(
      new Uint32Array(ex.memory.buffer, shapeBuf, 4).slice(0, ndim),
    );
    const nameLen = ex.weights_name(i, nameBuf);
    const name = dec.decode(new Uint8Array(ex.memory.buffer, nameBuf, nameLen));
    map.set(name, {
      ptr: ex.weights_data_ptr(i),
      size: ex.weights_data_size(i),
      shape,
      dtype: ex.weights_dtype(i),
    });
  }
  // Drop parsing scratch; weights stay below the mark.
  ex.reset();
  return map;
}

async function loadFixture(name) {
  return await readFile(join(FIXTURES, name));
}

function allocAndCopy(ex, bytes) {
  const ptr = ex.alloc(bytes.byteLength);
  if (ptr === 0) throw new Error(`alloc OOM ${bytes.byteLength}`);
  u8(ex).set(bytes, ptr);
  return ptr;
}

function compareU16(ex, outPtr, expectedBytes) {
  const got = new Uint16Array(ex.memory.buffer, outPtr, expectedBytes.byteLength >>> 1);
  const want = new Uint16Array(
    expectedBytes.buffer,
    expectedBytes.byteOffset,
    expectedBytes.byteLength >>> 1,
  );
  let matches = 0;
  let firstMismatch = -1;
  for (let i = 0; i < want.length; i++) {
    if (got[i] === want[i]) matches++;
    else if (firstMismatch === -1) firstMismatch = i;
  }
  return { total: want.length, matches, firstMismatch, got, want };
}

function bf16ToF32(u) {
  const buf = new ArrayBuffer(4);
  new DataView(buf).setUint32(0, u << 16, true);
  return new DataView(buf).getFloat32(0, true);
}

async function runRmsNorm(ex, weights, spec) {
  const x = await loadFixture(spec.x);
  const gamma = weights.get(spec.gamma_tensor);
  if (!gamma) throw new Error(`missing tensor ${spec.gamma_tensor}`);
  const expected = await loadFixture(spec.expected);
  const xPtr = allocAndCopy(ex, x);
  const outPtr = ex.alloc(expected.byteLength);
  ex.rms_norm(xPtr, gamma.ptr, outPtr, spec.T, spec.D, spec.eps);
  return compareU16(ex, outPtr, expected);
}

async function runMatmul(ex, weights, spec) {
  const x = await loadFixture(spec.x);
  const w = weights.get(spec.weight_tensor);
  const b = weights.get(spec.bias_tensor);
  if (!w || !b) throw new Error(`missing tensor ${spec.weight_tensor}/${spec.bias_tensor}`);
  const expected = await loadFixture(spec.expected);
  const xPtr = allocAndCopy(ex, x);
  const outPtr = ex.alloc(expected.byteLength);
  ex.matmul_bf16(xPtr, w.ptr, b.ptr, outPtr, spec.T, spec.N, spec.D);
  return compareU16(ex, outPtr, expected);
}

async function runEmbed(ex, weights, spec) {
  const ids = await loadFixture(spec.ids);
  const embed = weights.get(spec.embed_tensor);
  if (!embed) throw new Error(`missing tensor ${spec.embed_tensor}`);
  const expected = await loadFixture(spec.expected);
  const idsPtr = allocAndCopy(ex, ids);
  const outPtr = ex.alloc(expected.byteLength);
  ex.embed_lookup(embed.ptr, idsPtr, outPtr, spec.T, spec.V, spec.D);
  return compareU16(ex, outPtr, expected);
}

const RUNNERS = {
  rms_norm: runRmsNorm,
  matmul_bf16: runMatmul,
  embed_lookup: runEmbed,
};

async function main() {
  const manifestBuf = await readFile(join(FIXTURES, "manifest.json"));
  const manifest = JSON.parse(new TextDecoder().decode(manifestBuf));
  const ex = await loadWasm();
  const weights = await loadWeights(ex);

  let failures = 0;
  for (const spec of manifest.kernels) {
    const runner = RUNNERS[spec.name];
    if (!runner) {
      console.log(`${RED}? ${spec.name}${RESET}  no runner`);
      failures++;
      continue;
    }
    const { total, matches, firstMismatch, got, want } = await runner(ex, weights, spec);
    const pass =
      matches === total ||
      (matches >= spec.baseline_matches && total === spec.baseline_total);
    const tag = pass ? `${GREEN}PASS${RESET}` : `${RED}FAIL${RESET}`;
    const baseline =
      spec.baseline_matches === spec.baseline_total
        ? "bit-exact"
        : `${spec.baseline_matches}/${spec.baseline_total}`;
    let detail = `${matches}/${total}  (baseline ${baseline})`;
    if (!pass && firstMismatch >= 0) {
      const g = got[firstMismatch];
      const w = want[firstMismatch];
      const gf = bf16ToF32(g).toPrecision(6);
      const wf = bf16ToF32(w).toPrecision(6);
      detail += `\n    first mismatch @ ${firstMismatch}: got 0x${g.toString(16).padStart(4, "0")} (${gf}), want 0x${w.toString(16).padStart(4, "0")} (${wf})`;
    }
    console.log(`${tag}  ${spec.name.padEnd(14)} ${detail}`);
    if (!pass) failures++;
    // Per-kernel scratch drops off on next reset; weights survive.
    ex.reset();
  }

  if (failures > 0) {
    console.log(`\n${RED}${failures} failure(s)${RESET}`);
    process.exit(1);
  }
  console.log(`\n${DIM}all kernels at or above baseline.${RESET}`);
}

await main();
