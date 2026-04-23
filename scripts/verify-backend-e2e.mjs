#!/usr/bin/env node
// End-to-end verification of the WasmBackend. Exercises the full
// public-shaped path: load .wasm + weight blob, warmup, forward on token
// IDs, read logits, compare to a PyTorch reference.
//
// Against the committed truncated blob (tests/fixtures/pii-weights.bin,
// 1 layer + 4 experts) this is a fast run. Against a full blob produced
// by `scripts/convert_weights.py --full --quant-transpose mlp.experts.gate_up_proj --quant-transpose mlp.experts.down_proj`
// it would exercise all 8 blocks — requires generating the blob + a
// matching logits fixture separately.
//
//     bun scripts/verify-backend-e2e.mjs

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { WasmBackend } from "../src/js/backends/wasm.ts";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, "..", "tests", "fixtures");

function bf16ToF32(u) {
  const buf = new ArrayBuffer(4);
  new DataView(buf).setUint32(0, u << 16, true);
  return new DataView(buf).getFloat32(0, true);
}

async function main() {
  const idsBuf = await readFile(join(FIXTURES, "mdl_input_ids.i32"));
  const ids = new Int32Array(
    idsBuf.buffer.slice(idsBuf.byteOffset, idsBuf.byteOffset + idsBuf.byteLength),
  );

  const expectedBuf = await readFile(join(FIXTURES, "mdl_logits.bf16"));
  const expectedU16 = new Uint16Array(
    expectedBuf.buffer.slice(expectedBuf.byteOffset, expectedBuf.byteOffset + expectedBuf.byteLength),
  );

  const weightsPath = join(FIXTURES, "pii-weights.bin");
  const weightsUrl = new URL(`file://${weightsPath}`);

  const backend = new WasmBackend({
    weightsUrl,
    bundle: undefined,
    quantization: "int4",
    device: "wasm",
  });
  await backend.warmup();

  const dummyMask = new Uint8Array(ids.length).fill(1);
  const logits = await backend.forward(ids, dummyMask);

  // Compare f32 outputs to the bf16-upcast expected.
  if (logits.data.length !== expectedU16.length) {
    throw new Error(`shape mismatch: got ${logits.data.length}, expected ${expectedU16.length}`);
  }
  let maxAbs = 0, maxRel = 0, sumSq = 0, fails = 0;
  const RELTOL = 0.2, ABSTOL = 1.5;
  let firstFailIdx = -1;
  for (let i = 0; i < logits.data.length; i++) {
    const g = logits.data[i];
    const w = bf16ToF32(expectedU16[i]);
    const abs = Math.abs(g - w);
    const rel = Math.abs(w) > 0 ? abs / Math.abs(w) : abs;
    if (abs > maxAbs) maxAbs = abs;
    if (rel > maxRel) maxRel = rel;
    sumSq += abs * abs;
    if (abs > Math.max(ABSTOL, RELTOL * Math.abs(w))) {
      fails++;
      if (firstFailIdx === -1) firstFailIdx = i;
    }
  }
  const rms = Math.sqrt(sumSq / logits.data.length);
  const status = fails === 0 ? "PASS" : "FAIL";
  console.log(
    `WasmBackend e2e  ${status}  fails=${fails}/${logits.data.length}  ` +
    `maxAbs=${maxAbs.toExponential(2)}  maxRel=${maxRel.toExponential(2)}  rms=${rms.toExponential(2)}`,
  );
  if (fails > 0) {
    console.log(`    first fail @ ${firstFailIdx}: got ${logits.data[firstFailIdx].toPrecision(6)}, want ${bf16ToF32(expectedU16[firstFailIdx]).toPrecision(6)}`);
    process.exit(1);
  }
  backend.dispose();
}

await main();
