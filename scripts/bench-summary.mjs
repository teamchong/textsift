#!/usr/bin/env node
// Run forward + ORT comparison benches and write results to:
//   - $GITHUB_STEP_SUMMARY (markdown table for the job summary)
//   - bench-results.json (uploaded as artifact for cross-OS aggregation)
//
// Env in:
//   OS_FAMILY = linux | darwin | windows
//   BACKEND   = vulkan | metal | dawn

import { spawnSync } from "node:child_process";
import { writeFileSync, appendFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";

const OS_FAMILY = process.env.OS_FAMILY || "unknown";
const BACKEND = process.env.BACKEND || "unknown";
const GITHUB_STEP_SUMMARY = process.env.GITHUB_STEP_SUMMARY || "/dev/stdout";

const FORWARD_BY_BACKEND = {
  vulkan: "tests/native/forward-vulkan.js",
  metal: "tests/native/forward-metal.js",
  dawn: "tests/native/forward-dawn.js",
};

const forwardJs = FORWARD_BY_BACKEND[BACKEND];
if (!forwardJs) {
  console.error(`bench-summary: unknown BACKEND="${BACKEND}"`);
  process.exit(1);
}

function runForward(T) {
  const t0 = Date.now();
  const out = spawnSync(
    "node",
    [forwardJs],
    {
      env: { ...process.env, T: String(T) },
      encoding: "utf8",
      timeout: 5 * 60 * 1000,
    },
  );
  if (out.status !== 0) {
    console.error(`forward T=${T} failed:\n${out.stderr}`);
    return null;
  }
  // Parse "forward latency T=32: median=27.7ms, min=25.8ms, over 10 iters"
  const m = out.stdout.match(/forward latency T=(\d+):\s*median=([\d.]+)ms,\s*min=([\d.]+)ms/);
  if (!m) {
    console.error(`forward T=${T}: could not parse output:\n${out.stdout}`);
    return null;
  }
  return { T: Number(m[1]), median: Number(m[2]), min: Number(m[3]), wall: Date.now() - t0 };
}

const results = [];
for (const T of [7, 25, 32, 80]) {
  const r = runForward(T);
  if (r) results.push(r);
}

// End-to-end PrivacyFilter integration latency. Runs first because it
// populates the model cache that the ORT baseline reads from.
let e2eRedactMs = null;
const integ = spawnSync("node", ["tests/native/integration/filter-redact.test.js"], {
  encoding: "utf8",
  timeout: 5 * 60 * 1000,
});
if (integ.status === 0) {
  const m = integ.stdout.match(/redact\(\) = ([\d.]+) ms/);
  if (m) e2eRedactMs = Number(m[1]);
}

// ORT baseline runs on every OS — same model bytes, same hardware,
// different inference engine. The realistic comparison: "what would
// I get if I just `npm i onnxruntime-node` and wrote the loader
// myself?" — across Linux/Mac/Windows.
let ortMedian = null;
{
  const ortPath = resolve("tests/native/bench-onnx/ort-baseline.js");
  if (existsSync(ortPath)) {
    const out = spawnSync("node", [ortPath], {
      env: { ...process.env, T: "32", ITERS: "10", WARMUP: "3" },
      encoding: "utf8",
      timeout: 10 * 60 * 1000,
    });
    if (out.status === 0) {
      const m = out.stdout.match(/ORT_BASELINE_RESULT\s+(\{[^}]+\})/);
      if (m) {
        try { ortMedian = JSON.parse(m[1]).median_ms ?? null; }
        catch { /* ignore parse errors */ }
      }
    } else if (out.stderr) {
      console.error(`[bench-summary] ort-baseline failed:\n${out.stderr}`);
    }
  }
}

// ── Write JSON for aggregation ──
const summary = {
  os_family: OS_FAMILY,
  backend: BACKEND,
  ts: new Date().toISOString(),
  forward: results,
  ort_node_cpu_ms: ortMedian,
  e2e_redact_ms: e2eRedactMs,
};
writeFileSync("bench-results.json", JSON.stringify(summary, null, 2));

// ── Markdown for $GITHUB_STEP_SUMMARY ──
const lines = [];
lines.push(`## ${OS_FAMILY} / ${BACKEND}-direct`);
lines.push("");
lines.push(`Forward latency at T tokens (one column = how long a single \`detect()\` / \`redact()\` call's GPU compute takes for an input of T tokens). Lower is better. **Interactive UI threshold ≈ 100 ms; chat/tooltip threshold ≈ 50 ms; "feels instant" ≈ 16 ms.**`);
lines.push("");

const fwd32 = results.find((r) => r.T === 32);
const ortRatio = (ortMedian !== null && fwd32) ? (ortMedian / fwd32.median) : null;

lines.push(`| T | textsift (${BACKEND}-direct) | ORT Node CPU baseline | speedup |`);
lines.push(`|---:|---:|---:|---:|`);
for (const r of results) {
  // ORT was only measured at T=32; show "—" otherwise rather than
  // pretending we have a per-T baseline.
  const ortCell = (r.T === 32 && ortMedian !== null) ? `${ortMedian.toFixed(0)} ms` : "—";
  const speedupCell = (r.T === 32 && ortRatio !== null) ? `**${ortRatio.toFixed(1)}× faster**` : "—";
  lines.push(`| ${r.T} | ${r.median.toFixed(1)} ms | ${ortCell} | ${speedupCell} |`);
}
lines.push("");

if (ortMedian !== null && fwd32) {
  lines.push(`**Interpretation (T=32):** textsift native is ${ortRatio.toFixed(1)}× faster than the realistic Node.js alternative on the same hardware. ORT Node CPU is what most devs reach for first via \`npm i onnxruntime-node\` + a hand-rolled loader.`);
  lines.push("");
}

if (e2eRedactMs !== null) {
  const e2eClass = e2eRedactMs < 50 ? "feels instant"
    : e2eRedactMs < 100 ? "interactive"
    : e2eRedactMs < 250 ? "noticeable but acceptable"
    : "user-visible delay";
  lines.push(`**End-to-end \`PrivacyFilter.redact()\`:** ${e2eRedactMs.toFixed(1)} ms (${e2eClass}) — 122-char input with 4 PII spans, includes BPE tokenization + forward + Viterbi + span replacement.`);
  lines.push("");
}

lines.push(`<details><summary>Raw min/median per T</summary>`);
lines.push("");
lines.push(`| T | median | min |`);
lines.push(`|---:|---:|---:|`);
for (const r of results) {
  lines.push(`| ${r.T} | ${r.median.toFixed(1)} ms | ${r.min.toFixed(1)} ms |`);
}
lines.push("");
lines.push(`</details>`);
lines.push("");

appendFileSync(GITHUB_STEP_SUMMARY, lines.join("\n") + "\n");

console.log(lines.join("\n"));
