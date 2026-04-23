#!/usr/bin/env node
// End-to-end wall-clock benchmark for WasmBackend vs TransformersJsBackend.
// Runs both on a handful of input lengths; reports warm forward time.
//
//     bun scripts/bench-e2e.mjs --scratch-dir /tmp/pii-wasm-e2e
//
// Requires the full blob produced by `gen-full-parity-fixture.py`.

import { parseArgs } from "node:util";
import { join } from "node:path";
import { WasmBackend } from "../src/js/backends/wasm.ts";
import { TransformersJsBackend } from "../src/js/backends/transformers-js.ts";
import { ModelLoader } from "../src/js/model/loader.ts";
import { Tokenizer } from "../src/js/model/tokenizer.ts";

const SEEDS = [
  "Hi, my name is John Smith.",  // ~10 tokens
  "Please email me at jane.doe@example.com about invoice 4511 or call (555) 123-4567.",
  "My SSN is 123-45-6789 and my bank account is 9876543210. I live at 456 Oak Avenue, " +
    "Boston, MA 02134. Call me at (617) 555-0199 or text my cell at 617.555.0123.",
];

const WARMUP = 2;
const ITERS = 5;

function fmtMs(ns) { return (ns / 1e6).toFixed(1); }

async function measureBackend(backend, tokenIds, mask, label) {
  for (let i = 0; i < WARMUP; i++) await backend.forward(tokenIds, mask);
  const samples = [];
  for (let i = 0; i < ITERS; i++) {
    const t0 = process.hrtime.bigint();
    await backend.forward(tokenIds, mask);
    samples.push(Number(process.hrtime.bigint() - t0));
  }
  samples.sort((a, b) => a - b);
  const median = samples[Math.floor(samples.length / 2)];
  const min = samples[0];
  console.log(`  ${label.padEnd(22)} median=${fmtMs(median).padStart(7)} ms  min=${fmtMs(min).padStart(7)} ms`);
  return median;
}

async function main() {
  const { values } = parseArgs({
    options: { "scratch-dir": { type: "string" } },
  });
  const scratch = values["scratch-dir"];
  if (!scratch) { console.error("--scratch-dir required"); process.exit(2); }
  const blobPath = join(scratch, "pii-weights-full.bin");

  const loader = new ModelLoader({ source: "openai/privacy-filter" });
  const bundle = await loader.load();
  const tokenizer = await Tokenizer.fromBundle(bundle);

  console.log("warming up both backends (+ heavy first-call JIT/caching)…");
  const tjs = new TransformersJsBackend({ bundle, quantization: "int4", device: "auto" });
  await tjs.warmup();
  const wasm = new WasmBackend({ weightsUrl: new URL(`file://${blobPath}`), bundle, quantization: "int4", device: "wasm" });
  await wasm.warmup();

  for (const text of SEEDS) {
    const enc = tokenizer.encode(text);
    console.log(`\nT=${enc.tokenIds.length}  input="${text.slice(0, 60)}${text.length > 60 ? "…" : ""}"`);
    const tjsMs = await measureBackend(tjs, enc.tokenIds, enc.attentionMask, "transformers.js");
    const wasmMs = await measureBackend(wasm, enc.tokenIds, enc.attentionMask, "wasm (Stage 1)");
    const speedup = tjsMs / wasmMs;
    console.log(`  speedup wasm vs tjs:   ${speedup.toFixed(2)}x`);
  }

  tjs.dispose();
  wasm.dispose();
}

await main();
