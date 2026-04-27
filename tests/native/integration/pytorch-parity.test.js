// Conformance test: textsift's span output for each fixture input
// must match the PyTorch reference (committed to
// tests/conformance/pytorch/fixtures.json).
//
// The fixtures are produced by tests/conformance/pytorch/generate-fixtures.py
// against `openai/privacy-filter` via HuggingFace transformers — that's the
// canonical PyTorch path. If textsift's WASM/native backends agree with
// those spans, the implementation is span-equivalent to PyTorch.
//
// Skip cleanly if `fixtures.json` hasn't been generated yet, so a fresh
// clone without Python+torch installed doesn't fail this check; the
// regenerator instructions live in tests/conformance/pytorch/README.md.

import { readFileSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { PrivacyFilter } from "../../../packages/textsift/dist/index.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = resolve(HERE, "../../conformance/pytorch/fixtures.json");

if (!existsSync(FIXTURES)) {
  console.log(
    `[pytorch-parity] SKIP: ${FIXTURES} not generated yet. ` +
      `Run tests/conformance/pytorch/generate-fixtures.py to populate it.`,
  );
  process.exit(0);
}

const fixtures = JSON.parse(readFileSync(FIXTURES, "utf8"));
if (!Array.isArray(fixtures) || fixtures.length === 0) {
  console.error(`[pytorch-parity] FAIL: ${FIXTURES} is empty or malformed`);
  process.exit(1);
}

console.log(`[pytorch-parity] checking ${fixtures.length} fixtures...`);

const filter = await PrivacyFilter.create();

let mismatches = 0;
for (const fx of fixtures) {
  const { text, spans: expected } = fx;
  const { spans: actual } = await filter.detect(text);

  // Compare label + start + end. PyTorch spans are absolute char
  // offsets, same as textsift's. We don't compare text/marker
  // because those are derived from offsets.
  const expectedKey = expected
    .map((s) => `${s.label}:${s.start}-${s.end}`)
    .sort()
    .join(",");
  const actualKey = actual
    .map((s) => `${s.label}:${s.start}-${s.end}`)
    .sort()
    .join(",");

  if (expectedKey === actualKey) {
    console.log(`OK  "${text.slice(0, 50)}…" (${expected.length} spans)`);
  } else {
    mismatches++;
    console.error(
      `FAIL "${text.slice(0, 50)}…"\n` +
        `  PyTorch: ${expectedKey || "(none)"}\n` +
        `  textsift: ${actualKey || "(none)"}`,
    );
  }
}

filter.dispose();

if (mismatches > 0) {
  console.error(
    `\n${mismatches}/${fixtures.length} fixtures disagree with PyTorch reference`,
  );
  process.exit(1);
}
console.log(`\n${fixtures.length}/${fixtures.length} fixtures agree with PyTorch reference`);
