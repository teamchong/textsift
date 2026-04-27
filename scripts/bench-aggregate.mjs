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
console.log("Same model (`openai/privacy-filter` model_q4f16.onnx), one runner per OS family. Lower latency = better.");
console.log("");
console.log("**How to read these numbers:**");
console.log("- < 16 ms feels instant (60 fps frame budget)");
console.log("- < 50 ms feels interactive (chat tooltip threshold)");
console.log("- < 100 ms is the standard interactive UI threshold");
console.log("- ≥ 250 ms is a user-visible delay");
console.log("");

// Headline: textsift native vs ORT Node CPU at T=32 — the realistic
// "what would I use otherwise" comparison, with explicit speedup.
const withOrt = all.filter((a) => a.ort_node_cpu_ms !== null);
if (withOrt.length > 0) {
  console.log("## vs. the realistic Node baseline (ORT Node CPU, same model)");
  console.log("");
  console.log("ORT Node CPU is what most devs reach for if they don't use textsift: `npm i onnxruntime-node` and write the loader by hand. Same model bytes, same hardware as the runner — just a different inference engine.");
  console.log("");
  console.log(`| OS | textsift native @ T=32 | ORT Node CPU @ T=32 | speedup |`);
  console.log(`|---|---:|---:|---:|`);
  for (const a of withOrt.sort((x, y) => x.os_family.localeCompare(y.os_family))) {
    const fwd32 = a.forward.find((f) => f.T === 32);
    if (!fwd32 || a.ort_node_cpu_ms === null) continue;
    const speedup = (a.ort_node_cpu_ms / fwd32.median).toFixed(1);
    console.log(`| ${a.os_family} (${a.backend}) | ${fwd32.median.toFixed(1)} ms | ${a.ort_node_cpu_ms.toFixed(0)} ms | **${speedup}× faster** |`);
  }
  console.log("");
}

// Forward-pass detail table.
const Ts = [7, 25, 32, 80];
console.log("## textsift forward latency by input size");
console.log("");
console.log("Pure GPU-compute time per forward (median of 10 iters). Excludes tokenization + Viterbi + redaction overhead — see the end-to-end section below for those.");
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

// End-to-end redact() latency.
const withE2E = all.filter((a) => a.e2e_redact_ms !== null);
if (withE2E.length > 0) {
  console.log("## End-to-end `PrivacyFilter.redact()`");
  console.log("");
  console.log("122-char input with 4 PII spans (email × 2, phone, person). Includes BPE tokenization + forward + Viterbi + span replacement — the full call cost a real app pays.");
  console.log("");
  console.log(`| OS | redact() latency | feel |`);
  console.log(`|---|---:|---|`);
  for (const a of withE2E.sort((x, y) => x.os_family.localeCompare(y.os_family))) {
    const ms = a.e2e_redact_ms;
    const feel = ms < 50 ? "feels instant"
      : ms < 100 ? "interactive"
      : ms < 250 ? "noticeable but acceptable"
      : "user-visible delay";
    console.log(`| ${a.os_family} | ${ms.toFixed(1)} ms | ${feel} |`);
  }
  console.log("");
}

console.log("## What this comparison does NOT cover");
console.log("");
console.log("- **PyTorch CUDA on NVIDIA datacenter GPUs** — datacenter ML stacks will outrun textsift; that's a different lane (different audience, different deployment cost).");
console.log("- **PyTorch CPU/MPS via the Python `transformers` library** — the conventional ML-engineer baseline; lives outside the JS ecosystem and adds a multi-GB pip install.");
console.log("");
console.log("textsift's lane is **JS/Node apps that need PII detection without leaving the user's machine**, where the realistic alternative is the ORT Node CPU number above. If you're building a Python data pipeline on a 4090, use PyTorch.");
