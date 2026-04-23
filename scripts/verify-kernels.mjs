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
import { attentionForward } from "../src/js/inference/attention.ts";
import { buildRopeTables } from "../src/js/inference/rope.ts";

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
      dataOffset: ex.weights_data_ptr(i),
      size: ex.weights_data_size(i),
      dataSize: ex.weights_data_size(i),
      shape,
      dtype: ex.weights_dtype(i),
      name,
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

/**
 * Tolerance-based comparison for kernels where bit-exact is not the
 * contract (e.g. int4 dequant + matmul, where PyTorch's fp32 linear
 * uses a different accumulation order than our per-block kernel).
 * Returns {total, maxAbs, maxRel, rms, fails}. fails = # lanes where
 * |got - want| > max(abs_tol, rel_tol * |want|).
 */
function compareBf16Tolerance(ex, outPtr, expectedBytes, { absTol = 0, relTol }) {
  const got = new Uint16Array(ex.memory.buffer, outPtr, expectedBytes.byteLength >>> 1);
  const want = new Uint16Array(
    expectedBytes.buffer,
    expectedBytes.byteOffset,
    expectedBytes.byteLength >>> 1,
  );
  let maxAbs = 0;
  let maxRel = 0;
  let sumSq = 0;
  let fails = 0;
  let firstFail = -1;
  for (let i = 0; i < want.length; i++) {
    const g = bf16ToF32(got[i]);
    const w = bf16ToF32(want[i]);
    const abs = Math.abs(g - w);
    const rel = Math.abs(w) > 0 ? abs / Math.abs(w) : abs;
    if (abs > maxAbs) maxAbs = abs;
    if (rel > maxRel) maxRel = rel;
    sumSq += abs * abs;
    const tol = Math.max(absTol, relTol * Math.abs(w));
    if (abs > tol) {
      fails++;
      if (firstFail === -1) firstFail = i;
    }
  }
  return {
    total: want.length,
    fails,
    firstFail,
    maxAbs,
    maxRel,
    rms: Math.sqrt(sumSq / want.length),
    got,
    want,
  };
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

async function runAttentionForward(ex, weights, spec) {
  const hidden = await loadFixture(spec.hidden);
  const expected = await loadFixture(spec.expected);
  const hiddenPtr = allocAndCopy(ex, hidden);

  // RoPE tables: compute via the JS rope helper, copy into WASM memory.
  const { cos, sin } = buildRopeTables(
    {
      headDim: spec.head_dim,
      theta: spec.rope.theta,
      factor: spec.rope.factor,
      originalMaxPositionEmbeddings: spec.rope.original_max_position_embeddings,
      betaFast: spec.rope.beta_fast,
      betaSlow: spec.rope.beta_slow,
      truncate: spec.rope.truncate,
    },
    spec.T,
  );
  const cosBytes = new Uint8Array(cos.buffer, cos.byteOffset, cos.byteLength);
  const sinBytes = new Uint8Array(sin.buffer, sin.byteOffset, sin.byteLength);
  const cosPtr = allocAndCopy(ex, cosBytes);
  const sinPtr = allocAndCopy(ex, sinBytes);

  const outPtr = ex.alloc(expected.byteLength);

  const w = (name) => {
    const t = weights.get(name);
    if (!t) throw new Error(`missing tensor: ${name}`);
    return t;
  };

  attentionForward(
    ex, hiddenPtr, outPtr,
    {
      qProj: w("model.layers.0.self_attn.q_proj.weight"),
      qProjBias: w("model.layers.0.self_attn.q_proj.bias"),
      kProj: w("model.layers.0.self_attn.k_proj.weight"),
      kProjBias: w("model.layers.0.self_attn.k_proj.bias"),
      vProj: w("model.layers.0.self_attn.v_proj.weight"),
      vProjBias: w("model.layers.0.self_attn.v_proj.bias"),
      oProj: w("model.layers.0.self_attn.o_proj.weight"),
      oProjBias: w("model.layers.0.self_attn.o_proj.bias"),
      sinks: w("model.layers.0.self_attn.sinks"),
    },
    {
      hiddenSize: spec.hidden_size,
      numHeads: spec.H_q,
      numKvHeads: spec.H_kv,
      headDim: spec.head_dim,
      slidingWindow: spec.window,
    },
    { ropeCosPtr: cosPtr, ropeSinPtr: sinPtr },
    spec.T,
  );

  return {
    tolerance: { relTol: spec.rel_tol, absTol: spec.abs_tol ?? 0 },
    ...compareBf16Tolerance(ex, outPtr, expected, {
      relTol: spec.rel_tol, absTol: spec.abs_tol ?? 0,
    }),
  };
}

async function runBandedAttention(ex, _weights, spec) {
  const q = await loadFixture(spec.q);
  const k = await loadFixture(spec.k);
  const v = await loadFixture(spec.v);
  const sinks = await loadFixture(spec.sinks);
  const expected = await loadFixture(spec.expected);
  const qPtr = allocAndCopy(ex, q);
  const kPtr = allocAndCopy(ex, k);
  const vPtr = allocAndCopy(ex, v);
  const sinksPtr = allocAndCopy(ex, sinks);
  const outPtr = ex.alloc(expected.byteLength);
  ex.banded_attention(qPtr, kPtr, vPtr, sinksPtr, outPtr,
    spec.T, spec.H_q, spec.H_kv, spec.head_dim, spec.window);
  return {
    tolerance: { relTol: spec.rel_tol, absTol: spec.abs_tol ?? 0 },
    ...compareBf16Tolerance(ex, outPtr, expected, {
      relTol: spec.rel_tol, absTol: spec.abs_tol ?? 0,
    }),
  };
}

async function runMatmulOutF32(ex, weights, spec) {
  const x = await loadFixture(spec.x);
  const w = weights.get(spec.weight_tensor);
  const b = weights.get(spec.bias_tensor);
  if (!w || !b) throw new Error(`missing tensor ${spec.weight_tensor}/${spec.bias_tensor}`);
  const expected = await loadFixture(spec.expected);
  const xPtr = allocAndCopy(ex, x);
  const outPtr = ex.alloc(expected.byteLength);
  ex.matmul_bf16_out_f32(xPtr, w.ptr, b.ptr, outPtr, spec.T, spec.N, spec.D);
  const got = new Float32Array(ex.memory.buffer, outPtr, expected.byteLength >>> 2);
  const want = new Float32Array(
    expected.buffer, expected.byteOffset, expected.byteLength >>> 2,
  );
  let maxAbs = 0, maxRel = 0, sumSq = 0, fails = 0, firstFail = -1;
  for (let i = 0; i < want.length; i++) {
    const abs = Math.abs(got[i] - want[i]);
    const rel = Math.abs(want[i]) > 0 ? abs / Math.abs(want[i]) : abs;
    if (abs > maxAbs) maxAbs = abs;
    if (rel > maxRel) maxRel = rel;
    sumSq += abs * abs;
    const tol = Math.max(spec.abs_tol ?? 0, spec.rel_tol * Math.abs(want[i]));
    if (abs > tol) {
      fails++;
      if (firstFail === -1) firstFail = i;
    }
  }
  return {
    tolerance: { relTol: spec.rel_tol, absTol: spec.abs_tol ?? 0 },
    total: want.length, fails, firstFail, maxAbs, maxRel,
    rms: Math.sqrt(sumSq / want.length),
    got: new Uint16Array(0), want: new Uint16Array(0),
  };
}

async function runTopk(ex, _weights, spec) {
  const x = await loadFixture(spec.x);
  const expectedIdx = await loadFixture(spec.expected_idx);
  const expectedVal = await loadFixture(spec.expected_val);
  const xPtr = allocAndCopy(ex, x);
  const idxBytes = spec.rows * spec.k * 4;
  const valBytes = spec.rows * spec.k * 4;
  const idxPtr = ex.alloc(idxBytes);
  const valPtr = ex.alloc(valBytes);
  ex.topk_partial_f32(xPtr, idxPtr, valPtr, spec.rows, spec.cols, spec.k);
  const gotIdx = new Int32Array(ex.memory.buffer, idxPtr, idxBytes >>> 2);
  const wantIdx = new Int32Array(
    expectedIdx.buffer, expectedIdx.byteOffset, expectedIdx.byteLength >>> 2,
  );
  const gotVal = new Float32Array(ex.memory.buffer, valPtr, valBytes >>> 2);
  const wantVal = new Float32Array(
    expectedVal.buffer, expectedVal.byteOffset, expectedVal.byteLength >>> 2,
  );
  let idxFails = 0, valFails = 0, firstFail = -1;
  let maxAbs = 0;
  for (let i = 0; i < wantIdx.length; i++) {
    if (gotIdx[i] !== wantIdx[i]) {
      idxFails++;
      if (firstFail === -1) firstFail = i;
    }
    const abs = Math.abs(gotVal[i] - wantVal[i]);
    if (abs > maxAbs) maxAbs = abs;
    if (abs > (spec.val_abs_tol ?? 0)) valFails++;
  }
  // Synthesize a tolerance-shaped result for the shared printer.
  return {
    tolerance: { relTol: spec.val_abs_tol ?? 0, absTol: spec.val_abs_tol ?? 0 },
    total: wantIdx.length,
    fails: idxFails + valFails,
    firstFail,
    maxAbs,
    maxRel: 0,
    rms: 0,
    got: new Uint16Array(0),
    want: new Uint16Array(0),
  };
}

async function runSwigluF32(ex, _weights, spec) {
  const x = await loadFixture(spec.x);
  const expected = await loadFixture(spec.expected);
  const xPtr = allocAndCopy(ex, x);
  const outPtr = ex.alloc(expected.byteLength);
  ex.swiglu_clamp_f32(xPtr, outPtr, spec.T, spec.D);
  const got = new Float32Array(ex.memory.buffer, outPtr, expected.byteLength >>> 2);
  const want = new Float32Array(
    expected.buffer,
    expected.byteOffset,
    expected.byteLength >>> 2,
  );
  let maxAbs = 0, maxRel = 0, sumSq = 0, fails = 0, firstFail = -1;
  for (let i = 0; i < want.length; i++) {
    const abs = Math.abs(got[i] - want[i]);
    const rel = Math.abs(want[i]) > 0 ? abs / Math.abs(want[i]) : abs;
    if (abs > maxAbs) maxAbs = abs;
    if (rel > maxRel) maxRel = rel;
    sumSq += abs * abs;
    const tol = Math.max(spec.abs_tol ?? 0, spec.rel_tol * Math.abs(want[i]));
    if (abs > tol) {
      fails++;
      if (firstFail === -1) firstFail = i;
    }
  }
  return {
    tolerance: { relTol: spec.rel_tol, absTol: spec.abs_tol ?? 0 },
    total: want.length, fails, firstFail, maxAbs, maxRel,
    rms: Math.sqrt(sumSq / want.length),
    got: new Uint16Array(0), want: new Uint16Array(0),
  };
}

async function runSoftmaxF32(ex, _weights, spec) {
  const x = await loadFixture(spec.x);
  const expected = await loadFixture(spec.expected);
  const xPtr = allocAndCopy(ex, x);
  const outPtr = ex.alloc(expected.byteLength);
  ex.softmax_f32(xPtr, outPtr, spec.rows, spec.cols);
  // Compare as f32 with tolerance — exp has >1 ULP drift vs libm.
  const got = new Float32Array(ex.memory.buffer, outPtr, expected.byteLength >>> 2);
  const want = new Float32Array(
    expected.buffer,
    expected.byteOffset,
    expected.byteLength >>> 2,
  );
  let maxAbs = 0;
  let maxRel = 0;
  let sumSq = 0;
  let fails = 0;
  let firstFail = -1;
  for (let i = 0; i < want.length; i++) {
    const g = got[i];
    const w = want[i];
    const abs = Math.abs(g - w);
    const rel = Math.abs(w) > 0 ? abs / Math.abs(w) : abs;
    if (abs > maxAbs) maxAbs = abs;
    if (rel > maxRel) maxRel = rel;
    sumSq += abs * abs;
    const tol = Math.max(spec.abs_tol ?? 0, spec.rel_tol * Math.abs(w));
    if (abs > tol) {
      fails++;
      if (firstFail === -1) firstFail = i;
    }
  }
  return {
    tolerance: { relTol: spec.rel_tol, absTol: spec.abs_tol ?? 0 },
    total: want.length,
    fails,
    firstFail,
    maxAbs,
    maxRel,
    rms: Math.sqrt(sumSq / want.length),
    got: new Uint16Array(0),
    want: new Uint16Array(0),
  };
}

async function runRopeApply(ex, _weights, spec) {
  const qkIn = await loadFixture(spec.qk_in);
  const cos = await loadFixture(spec.cos);
  const sin = await loadFixture(spec.sin);
  const expected = await loadFixture(spec.expected);
  const qkPtr = allocAndCopy(ex, qkIn);
  const cosPtr = allocAndCopy(ex, cos);
  const sinPtr = allocAndCopy(ex, sin);
  ex.rope_apply(qkPtr, cosPtr, sinPtr, spec.T, spec.H, spec.head_dim);
  return compareU16(ex, qkPtr, expected);
}

async function runMatmulInt4(ex, _weights, spec) {
  // int4 matmul uses a standalone packed-weight fixture rather than a
  // tensor from the weight blob, because our blob only carries bf16 of
  // score.weight — the quantized equivalent is generated specifically
  // for this test.
  const x = await loadFixture(spec.x);
  const w = await loadFixture(spec.w);
  const bias = await loadFixture(spec.bias);
  const expected = await loadFixture(spec.expected);
  const xPtr = allocAndCopy(ex, x);
  const wPtr = allocAndCopy(ex, w);
  const bPtr = allocAndCopy(ex, bias);
  const outPtr = ex.alloc(expected.byteLength);
  ex.matmul_bf16_x_int4block(xPtr, wPtr, bPtr, outPtr, spec.T, spec.N, spec.D);
  return {
    tolerance: { relTol: spec.rel_tol, absTol: spec.abs_tol ?? 0 },
    ...compareBf16Tolerance(ex, outPtr, expected, {
      relTol: spec.rel_tol,
      absTol: spec.abs_tol ?? 0,
    }),
  };
}

const RUNNERS = {
  rms_norm: runRmsNorm,
  matmul_bf16: runMatmul,
  embed_lookup: runEmbed,
  matmul_bf16_x_int4block: runMatmulInt4,
  rope_apply: runRopeApply,
  softmax_f32: runSoftmaxF32,
  swiglu_clamp_f32: runSwigluF32,
  matmul_bf16_out_f32: runMatmulOutF32,
  topk_partial_f32: runTopk,
  banded_attention: runBandedAttention,
  attention_forward: runAttentionForward,
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
    const result = await runner(ex, weights, spec);
    const label = spec.name.padEnd(24);

    if (result.tolerance !== undefined) {
      // Tolerance-based comparison.
      const { fails, total, maxAbs, maxRel, rms, firstFail, got, want } = result;
      const pass = fails === 0;
      const tag = pass ? `${GREEN}PASS${RESET}` : `${RED}FAIL${RESET}`;
      let detail =
        `tol rel<${result.tolerance.relTol}  ` +
        `fails=${fails}/${total}  maxAbs=${maxAbs.toExponential(2)}  ` +
        `maxRel=${maxRel.toExponential(2)}  rms=${rms.toExponential(2)}`;
      if (!pass && firstFail >= 0) {
        const g = got[firstFail];
        const w = want[firstFail];
        const gf = bf16ToF32(g).toPrecision(6);
        const wf = bf16ToF32(w).toPrecision(6);
        detail += `\n    first fail @ ${firstFail}: got ${gf}, want ${wf}`;
      }
      console.log(`${tag}  ${label} ${detail}`);
      if (!pass) failures++;
    } else {
      // Bit-count comparison (baseline-gated).
      const { total, matches, firstMismatch, got, want } = result;
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
      console.log(`${tag}  ${label} ${detail}`);
      if (!pass) failures++;
    }
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
