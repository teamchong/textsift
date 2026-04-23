#!/usr/bin/env node
// Parity check for the JS-side YARN RoPE precompute (src/js/inference/rope.ts)
// against the cos/sin tables produced by the PyTorch reference during
// `npm run gen:fixtures`. Runs before any WASM kernel tests so a broken
// precompute is caught early rather than surfacing as end-to-end noise.
//
//     node scripts/verify-rope-precompute.mjs
//
// Bun-only: we import the TS source directly. Node users should transpile
// first or run via tsx.

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { buildRopeTables } from "../src/js/inference/rope.ts";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, "..", "tests", "fixtures");

// From tests/fixtures/manifest.json rope_apply entry.
const T = 8;
const HEAD_DIM = 64;

// From openai/privacy-filter config.json.
const YARN_CONFIG = {
  headDim: HEAD_DIM,
  theta: 150000.0,
  factor: 32.0,
  originalMaxPositionEmbeddings: 4096,
  betaFast: 32.0,
  betaSlow: 1.0,
  truncate: false,
};

async function loadU16(name) {
  const buf = await readFile(join(FIXTURES, name));
  return new Uint16Array(
    buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
  );
}

function compareU16Bits(got, want, label) {
  let mismatches = 0;
  let firstMismatch = -1;
  for (let i = 0; i < want.length; i++) {
    if (got[i] !== want[i]) {
      mismatches++;
      if (firstMismatch === -1) firstMismatch = i;
    }
  }
  const matches = want.length - mismatches;
  console.log(`${label.padEnd(18)} ${matches}/${want.length} bit-exact`);
  if (mismatches > 0) {
    console.log(
      `    first @ ${firstMismatch}: got 0x${got[firstMismatch].toString(16)}, want 0x${want[firstMismatch].toString(16)}`,
    );
  }
  return mismatches === 0;
}

async function main() {
  const { cos, sin } = buildRopeTables(YARN_CONFIG, T);
  const cosRef = await loadU16("rope_cos.bf16");
  const sinRef = await loadU16("rope_sin.bf16");

  if (cos.length !== cosRef.length || sin.length !== sinRef.length) {
    console.error(`length mismatch: cos ${cos.length} vs ${cosRef.length}`);
    process.exit(1);
  }

  const cosPass = compareU16Bits(cos, cosRef, "rope cos bf16");
  const sinPass = compareU16Bits(sin, sinRef, "rope sin bf16");

  if (!cosPass || !sinPass) {
    console.error("rope precompute diverges from PyTorch reference");
    process.exit(1);
  }
  console.log("\nrope precompute matches PyTorch reference bit-exactly.");
}

await main();
