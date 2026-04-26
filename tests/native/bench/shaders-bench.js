// Native per-shader microbench (mirrors tests/conformance/bench-shaders.html).
// Reuses the same fixtures, same dispatch shape, same iteration count
// so the medians line up against the browser baseline.
//
// Reads the browser bench JSON dropped at fixtures/_browser-bench.json
// (run `npx playwright test conformance/bench-shaders.spec.ts` first)
// and prints a side-by-side comparison.

import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const NATIVE_PATH = resolve(HERE, "../../../packages/textsift/dist/textsift-native.node");
const FIXTURES = resolve(HERE, "../conformance/fixtures");
const BROWSER_JSON = resolve(FIXTURES, "_browser-bench.json");

const native = createRequire(import.meta.url)(NATIVE_PATH);

const SHADERS = [
  "rms_norm", "zero_f32", "cast_fp16_to_f32", "cast_f32_to_fp16_scaled",
  "add_fp16", "swiglu_clamp", "rope_apply",
  "matmul_int4_fp16_f16", "matmul_int4_f32_f32",
  "embed_lookup_int4", "add_rmsnorm_fp16_to_f32", "router_topk",
  "banded_attention", "qmoe_gate_up", "qmoe_down_scatter",
];

const WARMUP = 5;
const ITERS = 50;

function readBin(file) {
  return new Uint8Array(readFileSync(file));
}

function loadFixture(name) {
  const dir = resolve(FIXTURES, name);
  const meta = JSON.parse(readFileSync(resolve(dir, "meta.json"), "utf8"));
  return {
    meta,
    uniform: readBin(resolve(dir, "uniform.bin")),
    extraUniforms: meta.extraUniforms.map((eu) => ({
      binding: eu.binding,
      bytes: readBin(resolve(dir, `_uniform_${eu.name}.bin`)),
    })),
    inputs: meta.inputs.map((i) => ({
      binding: i.binding,
      bytes: readBin(resolve(dir, `${i.name}.bin`)),
    })),
    outputInitial: meta.output.hasInitial ? readBin(resolve(dir, "_output_initial.bin")) : undefined,
  };
}

const CHAIN_LEN = parseInt(process.env.CHAIN ?? "1", 10);

function bench(name, backend, f) {
  const out = {
    binding: f.meta.output.binding,
    byteLength: f.meta.output.byteLength,
    chainLen: CHAIN_LEN,
  };
  if (f.outputInitial) out.initial = f.outputInitial;
  const dispatch = f.meta.dispatch;

  // Warmup batch (separate call so first-call setup isn't in the timed run).
  native.benchDispatch(backend, name, f.uniform, f.extraUniforms, f.inputs, out, dispatch, WARMUP);
  // Timed batch — Zig measures per-iter time, returns Float64Array of ms.
  const samples = native.benchDispatch(backend, name, f.uniform, f.extraUniforms, f.inputs, out, dispatch, ITERS);
  const sorted = Array.from(samples).sort((a, b) => a - b);
  return { median: sorted[Math.floor(sorted.length / 2)], min: sorted[0] };
}

const browser = (() => {
  try { return JSON.parse(readFileSync(BROWSER_JSON, "utf8")); }
  catch { return null; }
})();
if (!browser) {
  console.warn("no browser bench JSON; run `npx playwright test conformance/bench-shaders.spec.ts` first");
}

const backend = native.createBackend();
console.log(`created backend handle=${backend}\n`);

const results = {};
for (const name of SHADERS) {
  const f = loadFixture(name);
  const r = bench(name, backend, f);
  const b = browser?.[name];
  const ratio = b ? (r.median / b.median).toFixed(2) : "—";
  console.log(
    `  ${name.padEnd(28)} ` +
      `native median=${r.median.toFixed(3)}ms ` +
      `browser median=${(b?.median ?? NaN).toFixed(3)}ms ` +
      `native/browser=${ratio}x`,
  );
  results[name] = { native: r, browser: b ?? null, ratio: b ? r.median / b.median : null };
}

native.destroyBackend(backend);

// Aggregate: geomean ratio across shaders that have both numbers.
const ratios = Object.values(results).map((r) => r.ratio).filter((r) => r != null);
if (ratios.length > 0) {
  const logSum = ratios.reduce((s, r) => s + Math.log(r), 0);
  const geomean = Math.exp(logSum / ratios.length);
  console.log(`\ngeomean native/browser across ${ratios.length} shaders: ${geomean.toFixed(2)}x`);
  if (geomean < 1) console.log(`  → native is ${(1 / geomean).toFixed(2)}× faster than browser per dispatch`);
  else console.log(`  → native is ${geomean.toFixed(2)}× slower than browser per dispatch`);
}
