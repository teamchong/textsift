#!/usr/bin/env node
// Read per-OS bench-results.json files and emit a combined markdown table
// for the workflow's overall summary. Argv[1] = directory containing the
// bench-${os_family}/ artifact subdirs.

import { readdirSync, readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";

const dir = process.argv[2] || "bench-artifacts";

const all = [];
for (const entry of readdirSync(dir)) {
  const p = resolve(dir, entry, "bench-results.json");
  try {
    const raw = readFileSync(p, "utf8");
    all.push(JSON.parse(raw));
  } catch (_e) {
    // skip incomplete artifacts
  }
}

if (all.length === 0) {
  console.log("# Bench aggregate\n\n_No bench results available._");
  process.exit(0);
}

console.log("# Cross-platform benchmark");
console.log("");
console.log("Same model (`openai/privacy-filter` model_q4f16.onnx), one runner per OS family.");
console.log("");

// Forward-pass table.
const Ts = [7, 25, 32, 80];
console.log("## Forward latency (`forward()` only, median of 10 iters)");
console.log("");
console.log(`| OS | backend |${Ts.map((T) => ` T=${T} `).join("|")}|`);
console.log(`|---|---|${Ts.map(() => "---:").join("|")}|`);
for (const a of all.sort((x, y) => x.os_family.localeCompare(y.os_family))) {
  const cells = Ts.map((T) => {
    const r = a.forward.find((f) => f.T === T);
    return r ? ` ${r.median.toFixed(1)} ms ` : ` — `;
  });
  console.log(`| ${a.os_family} | ${a.backend}-direct |${cells.join("|")}|`);
}
console.log("");

// ORT comparison (Linux-only typically).
const withOrt = all.filter((a) => a.ort_node_cpu_ms !== null);
if (withOrt.length > 0) {
  console.log("## vs. ONNX Runtime Node CPU at T=32");
  console.log("");
  console.log(`| OS | textsift native | ORT Node CPU | speedup |`);
  console.log(`|---|---:|---:|---:|`);
  for (const a of withOrt) {
    const fwd32 = a.forward.find((f) => f.T === 32);
    if (!fwd32 || a.ort_node_cpu_ms === null) continue;
    const speedup = (a.ort_node_cpu_ms / fwd32.median).toFixed(1);
    console.log(`| ${a.os_family} | ${fwd32.median.toFixed(1)} ms | ${a.ort_node_cpu_ms.toFixed(1)} ms | **${speedup}× faster** |`);
  }
  console.log("");
}

// End-to-end redact() latency.
const withE2E = all.filter((a) => a.e2e_redact_ms !== null);
if (withE2E.length > 0) {
  console.log("## End-to-end `PrivacyFilter.redact()`");
  console.log("");
  console.log("122-char input with 4 PII spans (email × 2, phone, person).");
  console.log("");
  console.log(`| OS | redact() latency |`);
  console.log(`|---|---:|`);
  for (const a of withE2E.sort((x, y) => x.os_family.localeCompare(y.os_family))) {
    console.log(`| ${a.os_family} | ${a.e2e_redact_ms.toFixed(1)} ms |`);
  }
  console.log("");
}
