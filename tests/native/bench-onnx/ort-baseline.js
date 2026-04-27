// Cross-platform ORT-baseline bench. Loads model_q4f16.onnx via
// onnxruntime-node (CPU provider) and runs T forwards. Same model
// the textsift native path runs; same hardware as the runner. The
// realistic alternative for "Node.js dev who doesn't reach for
// textsift" is `npm i onnxruntime-node` + write your own loader, so
// this number is what most readers actually trade against.
//
// Source of the model: textsift's own on-disk cache. The integration
// test (filter-redact.test.js) populates it; this bench reuses the
// same file. No manual staging.
//
// Output (single line, parsed by bench-summary.mjs):
//   ORT_BASELINE_RESULT { "T": 32, "median_ms": 800.5, "min_ms": 780.1 }

import * as ort from "onnxruntime-node";
import { readdirSync, statSync } from "node:fs";
import { resolve } from "node:path";
import { homedir } from "node:os";

const T = parseInt(process.env.T ?? "32", 10);
const ITERS = parseInt(process.env.ITERS ?? "10", 10);
const WARMUP = parseInt(process.env.WARMUP ?? "3", 10);

function findModelInCache() {
  // Mirrors `getCacheRoot` from packages/textsift/src/native/loader.ts.
  // We don't import it here to avoid pulling the whole textsift bundle
  // into the bench process. Three rules:
  //   1. $TEXTSIFT_CACHE_DIR
  //   2. $XDG_CACHE_HOME/textsift
  //   3. ~/.cache/textsift
  const env = process.env.TEXTSIFT_CACHE_DIR;
  const xdg = process.env.XDG_CACHE_HOME;
  const root = env && env.length > 0
    ? resolve(env)
    : xdg && xdg.length > 0
      ? resolve(xdg, "textsift")
      : resolve(homedir(), ".cache", "textsift");

  let entries;
  try { entries = readdirSync(root); }
  catch { throw new Error(`textsift cache not found at ${root} — run filter-redact integration test first to populate it`); }

  for (const sub of entries) {
    const candidate = resolve(root, sub, "model_q4f16.onnx");
    try {
      const st = statSync(candidate);
      if (st.isFile() && st.size > 0) return candidate;
    } catch { /* continue */ }
  }
  throw new Error(`no model_q4f16.onnx found under ${root}/*/`);
}

const modelPath = findModelInCache();
console.log(`[ort] loading ${modelPath}…`);

const t0 = performance.now();
const session = await ort.InferenceSession.create(modelPath, {
  executionProviders: ["cpu"],
  graphOptimizationLevel: "all",
  intraOpNumThreads: 0,
});
console.log(`[ort] loaded in ${(performance.now() - t0).toFixed(0)}ms`);
console.log(`[ort] inputs:  ${session.inputNames.join(", ")}`);

// Build feeds: token ids + attention mask. ONNX export uses int64.
const ids64 = new BigInt64Array(T);
for (let i = 0; i < T; i++) ids64[i] = BigInt((i * 7919) % 200000);
const mask64 = new BigInt64Array(T).fill(1n);

const buildFeeds = (kind) => {
  const feeds = {};
  for (const name of session.inputNames) {
    const lower = name.toLowerCase();
    const dtype = kind === "int64" ? "int64" : "int32";
    const ids = kind === "int64" ? ids64 : new Int32Array(T).map((_, i) => (i * 7919) % 200000);
    const mask = kind === "int64" ? mask64 : new Int32Array(T).fill(1);
    if (lower.includes("input_id") || lower === "ids" || lower === "tokens") {
      feeds[name] = new ort.Tensor(dtype, ids, [1, T]);
    } else if (lower.includes("attention_mask") || lower === "mask") {
      feeds[name] = new ort.Tensor(dtype, mask, [1, T]);
    } else {
      const zeros = kind === "int64" ? new BigInt64Array(T) : new Int32Array(T);
      feeds[name] = new ort.Tensor(dtype, zeros, [1, T]);
    }
  }
  return feeds;
};

let feeds;
try {
  feeds = buildFeeds("int64");
  await session.run(feeds);
} catch (e64) {
  console.log(`[ort] int64 failed (${e64.message.split("\n")[0]}); trying int32`);
  feeds = buildFeeds("int32");
  await session.run(feeds);
}

for (let i = 0; i < WARMUP; i++) {
  const wt0 = performance.now();
  await session.run(feeds);
  console.log(`[ort] warmup ${i}: ${(performance.now() - wt0).toFixed(1)}ms`);
}

const samples = [];
for (let i = 0; i < ITERS; i++) {
  const t = performance.now();
  await session.run(feeds);
  samples.push(performance.now() - t);
}
samples.sort((a, b) => a - b);
const median = samples[Math.floor(ITERS / 2)];
const min = samples[0];

await session.release?.();

console.log(`[ort/cpu] forward T=${T}: median=${median.toFixed(1)}ms min=${min.toFixed(1)}ms (${ITERS} iters)`);
console.log(`ORT_BASELINE_RESULT ${JSON.stringify({ T, median_ms: median, min_ms: min })}`);
