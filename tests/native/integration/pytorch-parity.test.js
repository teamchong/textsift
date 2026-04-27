// Conformance test: textsift's span output for each fixture input
// must match the canonical ONNX reference (committed to
// tests/conformance/pytorch/fixtures.json).
//
// "Canonical ONNX reference" = the same `model_q4f16.onnx` file
// textsift loads, run through ONNX Runtime, decoded with the same
// Viterbi+biases textsift uses (loaded from the model's
// `viterbi_calibration.json`). That removes both the precision
// confounder (fp32 vs q4f16) and the decoder confounder (argmax vs
// Viterbi) from the comparison; what remains is pure "does
// textsift's reimplementation match the canonical pipeline?"
//
// Fixtures are produced by tests/conformance/pytorch/generate-fixtures.py
// (kept under that directory name for historical reasons — the script
// no longer touches PyTorch).
//
// Skip cleanly if `fixtures.json` hasn't been generated yet, so a
// fresh clone without Python installed doesn't fail this check; the
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
        `  reference: ${expectedKey || "(none)"}\n` +
        `  textsift: ${actualKey || "(none)"}`,
    );
  }
}

filter.dispose();

if (mismatches > 0) {
  console.error(
    `\n${mismatches}/${fixtures.length} fixtures disagree with ONNX reference`,
  );
  process.exit(1);
}
console.log(`\n${fixtures.length}/${fixtures.length} fixtures agree with ONNX reference`);
