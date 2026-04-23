#!/usr/bin/env node
// Cross-backend parity: WasmBackend vs TransformersJsBackend on identical
// inputs. transformers.js is the Stage-0 reference; agreement on argmax
// classes confirms the Stage-1 backend is safe to swap in under the
// public API.
//
//     bun scripts/verify-xbackend-parity.mjs --scratch-dir /tmp/pii-wasm-e2e
//
// Requires the full blob at `<scratch-dir>/pii-weights-full.bin` (see
// gen-full-parity-fixture.py). transformers.js lazily fetches the ONNX
// model from HF Hub on first run and caches locally.

import { readFile } from "node:fs/promises";
import { parseArgs } from "node:util";
import { join } from "node:path";
import { WasmBackend } from "../src/js/backends/wasm.ts";
import { TransformersJsBackend } from "../src/js/backends/transformers-js.ts";
import { ModelLoader } from "../src/js/model/loader.ts";
import { Tokenizer } from "../src/js/model/tokenizer.ts";

const SAMPLE = "Please email me at jane.doe@example.com about invoice 4511 "
  + "or call (555) 123-4567. My mailing address is 123 Main Street, "
  + "Springfield, IL 62704.";

async function main() {
  const { values } = parseArgs({
    options: { "scratch-dir": { type: "string" } },
  });
  const scratch = values["scratch-dir"];
  if (!scratch) {
    console.error("--scratch-dir required (holds pii-weights-full.bin)");
    process.exit(2);
  }
  const blobPath = join(scratch, "pii-weights-full.bin");

  // --- Shared tokenization ----------------------------------------
  console.log("[xback] loading tokenizer + config from HF Hub …");
  const loader = new ModelLoader({ source: "openai/privacy-filter" });
  const bundle = await loader.load();
  const tokenizer = await Tokenizer.fromBundle(bundle);
  const encoded = tokenizer.encode(SAMPLE);
  const tokenIds = encoded.tokenIds;
  const mask = encoded.attentionMask;
  const T = tokenIds.length;
  console.log(`[xback] tokenized T=${T}`);

  // --- Stage-0 backend (transformers.js) --------------------------
  console.log("[xback] warming transformers.js backend …");
  const tjs = new TransformersJsBackend({
    bundle, quantization: "int4", device: "auto",
  });
  await tjs.warmup();
  const tjsT0 = process.hrtime.bigint();
  const tjsLogits = await tjs.forward(tokenIds, mask);
  const tjsMs = Number(process.hrtime.bigint() - tjsT0) / 1e6;
  console.log(`[xback] transformers.js forward: ${tjsMs.toFixed(1)} ms`);
  tjs.dispose();

  // --- Stage-1 backend (Zig+WASM) ---------------------------------
  console.log("[xback] warming Zig+WASM backend …");
  const wasm = new WasmBackend({
    weightsUrl: new URL(`file://${blobPath}`),
    bundle, quantization: "int4", device: "wasm",
  });
  await wasm.warmup();
  const wasmT0 = process.hrtime.bigint();
  const wasmLogits = await wasm.forward(tokenIds, mask);
  const wasmMs = Number(process.hrtime.bigint() - wasmT0) / 1e6;
  console.log(`[xback] Zig+WASM forward: ${wasmMs.toFixed(1)} ms`);
  wasm.dispose();

  // --- Compare ----------------------------------------------------
  if (tjsLogits.data.length !== wasmLogits.data.length) {
    console.error(`length mismatch: tjs ${tjsLogits.data.length} vs wasm ${wasmLogits.data.length}`);
    process.exit(1);
  }
  let maxAbs = 0, maxRel = 0, sumSq = 0;
  for (let i = 0; i < tjsLogits.data.length; i++) {
    const abs = Math.abs(tjsLogits.data[i] - wasmLogits.data[i]);
    const rel = Math.abs(tjsLogits.data[i]) > 0 ? abs / Math.abs(tjsLogits.data[i]) : abs;
    if (abs > maxAbs) maxAbs = abs;
    if (rel > maxRel) maxRel = rel;
    sumSq += abs * abs;
  }
  const rms = Math.sqrt(sumSq / tjsLogits.data.length);
  console.log(`[xback] logit drift  maxAbs=${maxAbs.toExponential(2)}  maxRel=${maxRel.toExponential(2)}  rms=${rms.toExponential(2)}`);

  // KPI: argmax agreement per token.
  const C = 33;
  let match = 0;
  for (let t = 0; t < T; t++) {
    let tBest = 0, tMax = tjsLogits.data[t * C];
    let wBest = 0, wMax = wasmLogits.data[t * C];
    for (let c = 1; c < C; c++) {
      if (tjsLogits.data[t * C + c] > tMax) { tMax = tjsLogits.data[t * C + c]; tBest = c; }
      if (wasmLogits.data[t * C + c] > wMax) { wMax = wasmLogits.data[t * C + c]; wBest = c; }
    }
    if (tBest === wBest) match++;
  }
  const pass = match === T;
  console.log(`[xback] ${pass ? "PASS" : "FAIL"}  argmax agreement: ${match}/${T}`);
  if (!pass) process.exit(1);
}

await main();
