// Aggregate Metal conformance: every kernel runs against its
// browser-dumped WGSL fixture, asserts byte-equal (or within fp
// tolerance for fp16/f32). Each entry maps the fixture's
// uniform/inputs/output to Metal buffer indices + dispatch shape.

import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { f16_to_f32 } from "../../conformance/fp16.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const NATIVE_PATH = resolve(HERE, "../../../packages/textsift/dist/textsift-native.node");
const FIXTURES = resolve(HERE, "../conformance/fixtures");
const native = createRequire(import.meta.url)(NATIVE_PATH);

function readBin(p) { return new Uint8Array(readFileSync(p)); }

// For each kernel: binding spec + grid/threadgroup formula.
// `bindings` is the list of bindings IN ORDER of metal buffer
// indices (0..N-1). Each spec is one of:
//   { from: "uniform" }      — reads uniform.bin into setBytes
//   { from: "extra:<name>" } — reads _uniform_<name>.bin into setBytes
//   { from: "input:<name>" } — reads <name>.bin into a buffer
//   { from: "output", elements?, dtype? }  — output buffer
const SHADERS = [
  { name: "rms_norm",
    bindings: [
      { from: "uniform" },
      { from: "input:x" },
      { from: "input:gamma" },
      { from: "output" },
    ],
    grid: (m) => [m.dispatch[0], 1, 1],
    tg: () => [64, 1, 1],
  },
  { name: "zero_f32",
    bindings: [
      { from: "uniform" },
      { from: "output" },
    ],
    grid: (m) => [m.dispatch[0], 1, 1],
    tg: () => [64, 1, 1],
  },
  { name: "cast_fp16_to_f32",
    bindings: [
      { from: "uniform" },
      { from: "input:src" },
      { from: "output" },
    ],
    grid: (m) => [m.dispatch[0], 1, 1],
    tg: () => [64, 1, 1],
  },
  { name: "cast_f32_to_fp16_scaled",
    bindings: [
      { from: "uniform" },
      { from: "extra:scale" },
      { from: "input:src" },
      { from: "output" },
    ],
    grid: (m) => [m.dispatch[0], 1, 1],
    tg: () => [64, 1, 1],
  },
  { name: "add_fp16",
    bindings: [
      { from: "uniform" },
      { from: "input:a" },
      { from: "input:b" },
      { from: "output" },
    ],
    grid: (m) => [m.dispatch[0], 1, 1],
    tg: () => [64, 1, 1],
  },
  { name: "swiglu_clamp",
    bindings: [
      { from: "uniform" },
      { from: "input:gate_up" },
      { from: "output" },
    ],
    grid: (m) => [m.dispatch[0], 1, 1],
    tg: () => [64, 1, 1],
  },
  { name: "rope_apply",
    bindings: [
      { from: "uniform" },
      // qk lives at binding 1, read+write. Browser fixture seeded
      // qk via outputInitial; we upload that into a buffer and pass
      // it as the "output" position too.
      { from: "output_initial_buffer" },
      { from: "input:cos_tab" },
      { from: "input:sin_tab" },
    ],
    grid: (m) => [m.dispatch[0], 1, 1],
    tg: () => [64, 1, 1],
  },
  { name: "matmul_int4_fp16_f16",
    bindings: [
      { from: "uniform" },
      { from: "input:x" },
      { from: "input:w_int4" },
      { from: "input:w_scales" },
      { from: "input:w_zp" },
      { from: "input:bias" },
      { from: "output" },
    ],
    grid: (m) => m.dispatch,
    tg: () => [64, 1, 1],
  },
  { name: "matmul_int4_f32_f32",
    bindings: [
      { from: "uniform" },
      { from: "input:x" },
      { from: "input:w_int4" },
      { from: "input:w_scales" },
      { from: "input:w_zp" },
      { from: "input:bias" },
      { from: "output" },
    ],
    // Match the browser fixture's dispatch (which is what generated
    // expected.bin). The WGSL only uses gid.x, so only the .x extent
    // matters — but expected.bin reflects what the browser actually
    // dispatched (one row at a time in the model's real call sites).
    grid: (m) => m.dispatch,
    tg: () => [64, 1, 1],
  },
  { name: "embed_lookup_int4",
    bindings: [
      { from: "uniform" },
      { from: "input:embed_int4" },
      { from: "input:embed_scales" },
      { from: "input:embed_zp" },
      { from: "input:ids" },
      { from: "output" },
    ],
    grid: (m) => [m.dispatch[0], 1, 1],
    tg: () => [64, 1, 1],
  },
  { name: "add_rmsnorm_fp16_to_f32",
    bindings: [
      { from: "uniform" },
      { from: "input:a" },
      { from: "input:b" },
      { from: "input:gamma" },
      { from: "input:_sum_out_init" }, // sum_out (writable) — fixture seeded with zeros
      { from: "output" },
    ],
    grid: (m) => [m.dispatch[0], 1, 1],
    tg: () => [64, 1, 1],
  },
  { name: "router_topk",
    bindings: [
      { from: "uniform" },
      { from: "input:logits" },
      { from: "input:_out_idx_init" }, // out_idx (writable storage)
      { from: "output" },               // out_scores
    ],
    grid: (m) => [m.dispatch[0], 1, 1],
    tg: () => [64, 1, 1],
  },
  { name: "banded_attention",
    bindings: [
      { from: "uniform" },
      { from: "input:q" },
      { from: "input:k" },
      { from: "input:v" },
      { from: "input:sinks" },
      { from: "input:mask" },
      { from: "output" },
    ],
    grid: (m) => m.dispatch,
    tg: () => [64, 1, 1],
  },
  { name: "qmoe_gate_up",
    bindings: [
      { from: "uniform" },
      { from: "input:x" },
      { from: "input:routing_idx" },
      { from: "input:w_int4" },
      { from: "input:w_scales" },
      { from: "input:w_zp" },
      { from: "input:bias" },
      { from: "output" },
    ],
    grid: (m) => m.dispatch,
    tg: () => [64, 1, 1],
  },
  { name: "qmoe_down_scatter",
    bindings: [
      { from: "uniform" },
      { from: "input:glu" },
      { from: "input:routing_idx" },
      { from: "input:routing_scores" },
      { from: "input:w_int4" },
      { from: "input:w_scales" },
      { from: "input:w_zp" },
      { from: "input:bias" },
      { from: "output" },
    ],
    grid: (m) => m.dispatch,
    tg: () => [64, 1, 1],
  },
];

function loadFixture(name) {
  const dir = resolve(FIXTURES, name);
  return {
    meta: JSON.parse(readFileSync(resolve(dir, "meta.json"), "utf8")),
    uniform: readBin(resolve(dir, "uniform.bin")),
    expected: readBin(resolve(dir, "expected.bin")),
    extras: Object.fromEntries(
      JSON.parse(readFileSync(resolve(dir, "meta.json"), "utf8"))
        .extraUniforms.map((eu) => [eu.name, readBin(resolve(dir, `_uniform_${eu.name}.bin`))]),
    ),
    inputs: Object.fromEntries(
      JSON.parse(readFileSync(resolve(dir, "meta.json"), "utf8"))
        .inputs.map((i) => [i.name, readBin(resolve(dir, `${i.name}.bin`))]),
    ),
    initial: (() => {
      try { return readBin(resolve(dir, "_output_initial.bin")); }
      catch { return null; }
    })(),
  };
}

function compareOutput(name, dtype, expected, got) {
  let byteEqual = true;
  for (let i = 0; i < expected.length; i++) if (got[i] !== expected[i]) { byteEqual = false; break; }
  if (byteEqual) {
    return { ok: true, kind: "byte-equal" };
  }
  if (dtype === "f16") {
    const eU = new Uint16Array(expected.buffer, expected.byteOffset, expected.byteLength / 2);
    const gU = new Uint16Array(got.buffer, got.byteOffset, got.byteLength / 2);
    let maxUlp = 0;
    for (let i = 0; i < eU.length; i++) {
      const a = f16_to_f32(gU[i]); const e = f16_to_f32(eU[i]);
      const abs = Math.abs(a - e);
      const mag = Math.max(Math.abs(e), Math.pow(2, -14));
      const ulp = Math.pow(2, Math.floor(Math.log2(mag)) - 10);
      const u = abs / ulp;
      if (u > maxUlp) maxUlp = u;
    }
    return { ok: maxUlp <= 2.001, kind: `f16 maxUlp=${maxUlp.toFixed(2)}` };
  }
  if (dtype === "f32") {
    const eF = new Float32Array(expected.buffer, expected.byteOffset, expected.byteLength / 4);
    const gF = new Float32Array(got.buffer, got.byteOffset, got.byteLength / 4);
    let maxRel = 0;
    for (let i = 0; i < eF.length; i++) {
      const abs = Math.abs(gF[i] - eF[i]);
      const rel = abs / Math.max(1e-6, Math.abs(eF[i]));
      if (rel > maxRel) maxRel = rel;
    }
    return { ok: maxRel <= 1e-2, kind: `f32 maxRel=${maxRel.toExponential(2)}` };
  }
  return { ok: false, kind: "byte mismatch" };
}

const b = native.metalCreateBackend();
console.log("[metal] device:", native.metalDeviceName(b));

let passed = 0, failed = 0;
for (const sh of SHADERS) {
  let f;
  try { f = loadFixture(sh.name); }
  catch (e) { console.log(`SKIP [${sh.name}] no fixture: ${e.message}`); continue; }

  const buffers = []; // metal buffer pointers to release
  const bindings = [];
  let outBuf = null;
  let outByteLen = f.expected.byteLength;

  for (let bi = 0; bi < sh.bindings.length; bi++) {
    const spec = sh.bindings[bi];
    if (spec.from === "uniform") {
      bindings.push({ index: bi, bytes: f.uniform });
    } else if (spec.from.startsWith("extra:")) {
      const ename = spec.from.slice("extra:".length);
      bindings.push({ index: bi, bytes: f.extras[ename] });
    } else if (spec.from.startsWith("input:")) {
      const iname = spec.from.slice("input:".length);
      const buf = native.metalCreateBuffer(b, f.inputs[iname]);
      buffers.push(buf);
      bindings.push({ index: bi, bufPtr: buf });
    } else if (spec.from === "output") {
      const init = f.initial ?? new Uint8Array(outByteLen);
      outBuf = native.metalCreateBuffer(b, init);
      buffers.push(outBuf);
      bindings.push({ index: bi, bufPtr: outBuf });
    } else if (spec.from === "output_initial_buffer") {
      // For rope_apply: the qk binding IS the output, pre-seeded
      // from the fixture's outputInitial.
      outBuf = native.metalCreateBuffer(b, f.initial);
      buffers.push(outBuf);
      bindings.push({ index: bi, bufPtr: outBuf });
    }
  }

  const grid = sh.grid(f.meta);
  const tg = sh.tg(f.meta);

  try {
    native.metalDispatchOneShot(b, sh.name, bindings, grid, tg);
    const got = native.metalReadBuffer(outBuf, 0, outByteLen);
    const r = compareOutput(sh.name, f.meta.output.dtype, f.expected, got);
    if (r.ok) {
      console.log(`OK   [${sh.name}] ${r.kind}`);
      passed++;
    } else {
      console.log(`FAIL [${sh.name}] ${r.kind}`);
      failed++;
    }
  } catch (e) {
    console.log(`FAIL [${sh.name}] ${e.message}`);
    failed++;
  }

  for (const ptr of buffers) native.metalReleaseBuffer(ptr);
}

native.metalDestroyBackend(b);
console.log(`\n${passed}/${passed + failed} kernels pass Metal conformance`);
process.exit(failed > 0 ? 1 : 0);
