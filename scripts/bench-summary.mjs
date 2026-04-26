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

// ORT comparison only on Linux (others lack the test harness for now).
let ortMedian = null;
if (OS_FAMILY === "linux") {
  const ortPath = resolve("tests/native/bench-onnx/ort-vs-vulkan.js");
  if (existsSync(ortPath)) {
    const out = spawnSync("node", [ortPath], {
      env: { ...process.env, T: "32", ITERS: "10", WARMUP: "3" },
      encoding: "utf8",
      timeout: 10 * 60 * 1000,
    });
    if (out.status === 0) {
      const m = out.stdout.match(/ONNX Runtime Node \(CPU\):\s+([\d.]+)\s+ms/);
      if (m) ortMedian = Number(m[1]);
    }
  }
}

// End-to-end PrivacyFilter integration latency.
let e2eRedactMs = null;
const integ = spawnSync("node", ["tests/native/integration/filter-redact.test.js"], {
  encoding: "utf8",
  timeout: 5 * 60 * 1000,
});
if (integ.status === 0) {
  const m = integ.stdout.match(/redact\(\) = ([\d.]+) ms/);
  if (m) e2eRedactMs = Number(m[1]);
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
lines.push(`| T | median | min |`);
lines.push(`|---:|---:|---:|`);
for (const r of results) {
  lines.push(`| ${r.T} | ${r.median.toFixed(1)} ms | ${r.min.toFixed(1)} ms |`);
}
lines.push("");
if (ortMedian !== null) {
  const fwd32 = results.find((r) => r.T === 32);
  if (fwd32) {
    const speedup = (ortMedian / fwd32.median).toFixed(1);
    lines.push(`**ORT Node CPU (T=32):** ${ortMedian.toFixed(1)} ms — textsift native is **${speedup}× faster**`);
    lines.push("");
  }
}
if (e2eRedactMs !== null) {
  lines.push(`**End-to-end \`PrivacyFilter.redact()\`:** ${e2eRedactMs.toFixed(1)} ms (122-char input, 4 PII spans)`);
  lines.push("");
}
appendFileSync(GITHUB_STEP_SUMMARY, lines.join("\n") + "\n");

console.log(lines.join("\n"));
