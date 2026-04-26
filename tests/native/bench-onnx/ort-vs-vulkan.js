// ONNX Runtime Node vs textsift Vulkan-direct on the same Linux box.
//
// Goal: answer "is textsift faster than how a typical Node.js dev would
// otherwise run this model?" The realistic Node.js alternative is
// onnxruntime-node loading the canonical model_q4f16.onnx artifact from
// huggingface.co/openai/privacy-filter — exact same model, exact same
// hardware, different runtime.
//
// Note on hardware: this Linux box has Intel Iris Xe (no CUDA). ORT
// falls back to its CPU execution provider, which is what most
// Node.js users without an NVIDIA GPU would see. That's the fair
// comparison for the textsift target user.
//
// Run:
//   T=32 node tests/native/bench-onnx/ort-vs-vulkan.js
//
// Optional env: ITERS, WARMUP, MODEL_PATH (defaults to /tmp/textsift-bench/...)

import * as ort from "onnxruntime-node";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { VulkanForward } from "../forward-vulkan.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const MODEL_PATH = process.env.MODEL_PATH ?? "/tmp/textsift-bench/openai-privacy-filter/model_q4f16.onnx";

const T = parseInt(process.env.T ?? "32", 10);
const ITERS = parseInt(process.env.ITERS ?? "10", 10);
const WARMUP = parseInt(process.env.WARMUP ?? "3", 10);

console.log(`config: T=${T} ITERS=${ITERS} WARMUP=${WARMUP}\n`);

// ── ONNX Runtime Node, CPU provider ─────────────────────────────────
async function benchOrt() {
  console.log(`[ort] loading ${MODEL_PATH}…`);
  const t0 = performance.now();
  const session = await ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ["cpu"],
    graphOptimizationLevel: "all",
    intraOpNumThreads: 0, // auto
  });
  console.log(`[ort] loaded in ${(performance.now() - t0).toFixed(0)}ms`);
  console.log(`[ort] inputs:  ${session.inputNames.join(", ")}`);
  console.log(`[ort] outputs: ${session.outputNames.join(", ")}`);

  // Detect input dtype (int32 vs int64) by trying both. HF ONNX exports
  // typically use int64; some custom exports use int32.
  const ids32 = new Int32Array(T);
  for (let i = 0; i < T; i++) ids32[i] = (i * 7919) % 200000;
  const mask32 = new Int32Array(T).fill(1);
  const ids64 = new BigInt64Array(T);
  for (let i = 0; i < T; i++) ids64[i] = BigInt(ids32[i]);
  const mask64 = new BigInt64Array(T).fill(1n);

  const buildFeeds = (kind) => {
    const feeds = {};
    for (const name of session.inputNames) {
      const ids = kind === "int64" ? ids64 : ids32;
      const mask = kind === "int64" ? mask64 : mask32;
      const dtype = kind === "int64" ? "int64" : "int32";
      const lower = name.toLowerCase();
      if (lower.includes("input_id") || lower === "ids" || lower === "tokens") {
        feeds[name] = new ort.Tensor(dtype, ids, [1, T]);
      } else if (lower.includes("attention_mask") || lower === "mask") {
        feeds[name] = new ort.Tensor(dtype, mask, [1, T]);
      } else {
        // Best-effort: pass zeros at the right shape for any other input.
        feeds[name] = new ort.Tensor(dtype, kind === "int64" ? new BigInt64Array(T) : new Int32Array(T), [1, T]);
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

  // Warmup
  for (let i = 0; i < WARMUP; i++) {
    const wt0 = performance.now();
    await session.run(feeds);
    console.log(`[ort] warmup ${i}: ${(performance.now() - wt0).toFixed(1)}ms`);
  }

  // Bench
  const samples = [];
  for (let i = 0; i < ITERS; i++) {
    const t = performance.now();
    await session.run(feeds);
    samples.push(performance.now() - t);
  }
  samples.sort((a, b) => a - b);
  const median = samples[Math.floor(ITERS / 2)];
  const min = samples[0];
  console.log(`[ort/cpu] forward T=${T}: median=${median.toFixed(1)}ms min=${min.toFixed(1)}ms (${ITERS} iters)`);
  await session.release?.();
  return { median, min };
}

// ── textsift Vulkan-direct ──────────────────────────────────────────
function benchVulkan() {
  console.log(`\n[vulkan] starting up…`);
  const t0 = performance.now();
  const fwd = new VulkanForward();
  fwd.ensureScratch(T);
  console.log(`[vulkan] backend + weights uploaded in ${(performance.now() - t0).toFixed(0)}ms`);

  const ids = new Int32Array(T);
  for (let i = 0; i < T; i++) ids[i] = (i * 7919) % 200000;
  const mask = new Uint8Array(T).fill(1);

  for (let i = 0; i < WARMUP; i++) {
    const wt0 = performance.now();
    fwd.forward(ids, mask);
    console.log(`[vulkan] warmup ${i}: ${(performance.now() - wt0).toFixed(1)}ms`);
  }

  const samples = [];
  for (let i = 0; i < ITERS; i++) {
    const t = performance.now();
    fwd.forward(ids, mask);
    samples.push(performance.now() - t);
  }
  samples.sort((a, b) => a - b);
  const median = samples[Math.floor(ITERS / 2)];
  const min = samples[0];
  console.log(`[vulkan-direct] forward T=${T}: median=${median.toFixed(1)}ms min=${min.toFixed(1)}ms (${ITERS} iters)`);
  fwd.dispose();
  return { median, min };
}

// ── run both, report ratio ──────────────────────────────────────────
const ort_res = await benchOrt();
const vk_res = benchVulkan();

console.log(`
${"=".repeat(60)}
Same model (openai/privacy-filter model_q4f16.onnx), same hardware:

  ONNX Runtime Node (CPU):  ${ort_res.median.toFixed(1)} ms median  (${ort_res.min.toFixed(1)} ms min)
  textsift Vulkan-direct:   ${vk_res.median.toFixed(1)} ms median  (${vk_res.min.toFixed(1)} ms min)

  speedup (median): ${(ort_res.median / vk_res.median).toFixed(2)}× faster
${"=".repeat(60)}
`);
