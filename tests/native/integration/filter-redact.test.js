// End-to-end integration test for textsift native PrivacyFilter on Node.
//
// Loads the real openai/privacy-filter model_q4f16.onnx (cached on disk),
// initializes the platform-native backend (vulkan*/metal*/dawn*), runs
// redact() on a string with PII, and asserts the output is sanely
// redacted.
//
// Run: node tests/native/integration/filter-redact.test.js
//      (the .node binary must be built first via scripts/build-native.sh)

import { PrivacyFilter } from "../../../packages/textsift/dist/index.js";

const TEST_INPUT =
  "Hi Alice, my email is alice@example.com and my phone is +1-555-0123. " +
  "Please send the report to bob@gmail.com by tomorrow.";

const t0 = performance.now();
console.log("[filter] creating PrivacyFilter...");
const filter = await PrivacyFilter.create();
const tCreate = performance.now() - t0;
console.log(`[filter] ready in ${tCreate.toFixed(0)} ms`);

const t1 = performance.now();
const result = await filter.redact(TEST_INPUT);
const tRedact = performance.now() - t1;
console.log(`[filter] redact() = ${tRedact.toFixed(1)} ms`);

console.log("\n=== INPUT ===");
console.log(TEST_INPUT);
console.log("\n=== REDACTED ===");
console.log(result.redactedText);
console.log("\n=== SPANS ===");
for (const span of result.spans) {
  console.log(`  [${span.start}..${span.end}] ${span.label}: "${TEST_INPUT.slice(span.start, span.end)}"`);
}
console.log(`\n=== SUMMARY ===`);
console.log(`  containsPii: ${result.containsPii}`);
console.log(`  spans: ${result.spans.length}`);

filter.dispose();
console.log("\n[filter] disposed cleanly");

// Sanity assertions.
if (!result.containsPii) {
  console.error("FAIL: expected containsPii=true (input has emails + phone)");
  process.exit(1);
}
if (result.spans.length < 2) {
  console.error(`FAIL: expected ≥2 spans (2 emails + phone), got ${result.spans.length}`);
  process.exit(1);
}
console.log("\n✓ end-to-end PrivacyFilter on Node works");
