#!/usr/bin/env node
// Micro-benchmark for the Stage 1 kernels. Runs each kernel across a
// few Phase-D-representative shapes and reports median ns/call + derived
// GFLOPS for the dense kernels. Pure timing; does not check numerics —
// use `npm run verify:kernels` for parity.
//
//     node scripts/bench-kernels.mjs
//
// Shapes chosen to mirror upstream privacy-filter dims:
//   D (d_model) = 640
//   Attention head dims: Q=14×64=896, KV=2×64=128
//   Router = 128 experts
//   Classifier head = 33 classes
// No dependencies. Node 20+.

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, "..");
const WASM_PATH = join(REPO, "dist", "pii.wasm");

const WARMUP = 5;
const ITERS = 50;

async function loadWasm() {
  const bytes = await readFile(WASM_PATH);
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance.exports;
}

function median(xs) {
  const sorted = [...xs].sort((a, b) => a - b);
  const mid = sorted.length >>> 1;
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

function fmtNs(ns) {
  if (ns < 1000) return `${ns.toFixed(0)} ns`;
  if (ns < 1_000_000) return `${(ns / 1000).toFixed(2)} µs`;
  return `${(ns / 1_000_000).toFixed(3)} ms`;
}

function time(fn) {
  const t0 = process.hrtime.bigint();
  fn();
  return Number(process.hrtime.bigint() - t0);
}

async function main() {
  const ex = await loadWasm();

  const shapes = [
    { name: "classifier head", T: 8, N: 33, D: 640 },
    { name: "router         ", T: 8, N: 128, D: 640 },
    { name: "K/V projection  ", T: 128, N: 128, D: 640 },
    { name: "Q projection    ", T: 128, N: 896, D: 640 },
    { name: "O projection    ", T: 128, N: 640, D: 896 },
  ];

  console.log(`matmul_bf16 — SIMD (${ITERS} iters, ${WARMUP} warmup)`);
  console.log(`${"shape".padEnd(34)} ${"ns/call".padStart(12)} ${"GFLOPS".padStart(9)}`);
  for (const s of shapes) {
    const xBytes = s.T * s.D * 2;
    const wBytes = s.N * s.D * 2;
    const bBytes = s.N * 2;
    const oBytes = s.T * s.N * 2;
    const xPtr = ex.alloc(xBytes);
    const wPtr = ex.alloc(wBytes);
    const bPtr = ex.alloc(bBytes);
    const oPtr = ex.alloc(oBytes);
    // Non-zero inputs so the compiler can't elide work. bf16 0x3F80 = 1.0.
    new Uint16Array(ex.memory.buffer, xPtr, xBytes >>> 1).fill(0x3F80);
    new Uint16Array(ex.memory.buffer, wPtr, wBytes >>> 1).fill(0x3F00);
    new Uint16Array(ex.memory.buffer, bPtr, bBytes >>> 1).fill(0);

    for (let i = 0; i < WARMUP; i++) ex.matmul_bf16(xPtr, wPtr, bPtr, oPtr, s.T, s.N, s.D);

    const samples = [];
    for (let i = 0; i < ITERS; i++) {
      samples.push(time(() => ex.matmul_bf16(xPtr, wPtr, bPtr, oPtr, s.T, s.N, s.D)));
    }
    const med = median(samples);
    // FLOPs: 2 * T * N * D (mul + add per output element per D step).
    const flops = 2 * s.T * s.N * s.D;
    const gflops = flops / med;
    const label = `T=${s.T} N=${s.N} D=${s.D} (${s.name.trim()})`;
    console.log(`${label.padEnd(34)} ${fmtNs(med).padStart(12)} ${gflops.toFixed(2).padStart(9)}`);
    ex.reset();
  }

  console.log(`\nrms_norm`);
  console.log(`${"shape".padEnd(34)} ${"ns/call".padStart(12)}`);
  for (const { T, D, name } of [
    { T: 8, D: 640, name: "8 tokens" },
    { T: 128, D: 640, name: "128 tokens" },
    { T: 1024, D: 640, name: "1024 tokens" },
  ]) {
    const xBytes = T * D * 2;
    const gBytes = D * 2;
    const oBytes = T * D * 2;
    const xPtr = ex.alloc(xBytes);
    const gPtr = ex.alloc(gBytes);
    const oPtr = ex.alloc(oBytes);
    new Uint16Array(ex.memory.buffer, xPtr, xBytes >>> 1).fill(0x3F80);
    new Uint16Array(ex.memory.buffer, gPtr, gBytes >>> 1).fill(0x3F80);

    for (let i = 0; i < WARMUP; i++) ex.rms_norm(xPtr, gPtr, oPtr, T, D, 1e-5);
    const samples = [];
    for (let i = 0; i < ITERS; i++) {
      samples.push(time(() => ex.rms_norm(xPtr, gPtr, oPtr, T, D, 1e-5)));
    }
    const label = `T=${T} D=${D} (${name})`;
    console.log(`${label.padEnd(34)} ${fmtNs(median(samples)).padStart(12)}`);
    ex.reset();
  }

  console.log(`\nembed_lookup`);
  console.log(`${"shape".padEnd(34)} ${"ns/call".padStart(12)}`);
  for (const { T, V, D, name } of [
    { T: 8, V: 1024, D: 640, name: "8 tokens" },
    { T: 128, V: 1024, D: 640, name: "128 tokens" },
    { T: 1024, V: 1024, D: 640, name: "1024 tokens" },
  ]) {
    const iBytes = T * 4;
    const eBytes = V * D * 2;
    const oBytes = T * D * 2;
    const iPtr = ex.alloc(iBytes);
    const ePtr = ex.alloc(eBytes);
    const oPtr = ex.alloc(oBytes);
    const ids = new Int32Array(ex.memory.buffer, iPtr, T);
    for (let k = 0; k < T; k++) ids[k] = k % V;
    new Uint16Array(ex.memory.buffer, ePtr, eBytes >>> 1).fill(0x3F80);

    for (let i = 0; i < WARMUP; i++) ex.embed_lookup(ePtr, iPtr, oPtr, T, V, D);
    const samples = [];
    for (let i = 0; i < ITERS; i++) {
      samples.push(time(() => ex.embed_lookup(ePtr, iPtr, oPtr, T, V, D)));
    }
    const label = `T=${T} V=${V} D=${D} (${name})`;
    console.log(`${label.padEnd(34)} ${fmtNs(median(samples)).padStart(12)}`);
    ex.reset();
  }
}

await main();
