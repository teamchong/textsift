// End-to-end test for markerPresets.faker — runs the real model
// through PrivacyFilter, swaps the default markers for faker output,
// asserts spans get realistic-looking fake values and consistency
// holds across mentions of the same input text.

import { PrivacyFilter, markerPresets } from "../../../packages/textsift/dist/index.js";

const TEST_INPUT =
  "Hi Alice, my email is alice@example.com and my phone is +1-555-0123. " +
  "Please send the report to bob@gmail.com by tomorrow. " +
  "Alice will follow up.";

console.log("[filter] creating PrivacyFilter with faker markers...");
const filter = await PrivacyFilter.create({ markers: markerPresets.faker() });

const result = await filter.redact(TEST_INPUT);

console.log("\n=== INPUT ===");
console.log(TEST_INPUT);
console.log("\n=== FAKER OUTPUT ===");
console.log(result.redactedText);
console.log("\n=== SPANS ===");
for (const span of result.spans) {
  console.log(`  [${span.start}..${span.end}] ${span.label}: "${span.text}" → "${span.marker}"`);
}

filter.dispose();
console.log("\n[filter] disposed");

// Assertions.
let fail = 0;
function check(name, cond, hint) {
  if (cond) { console.log(`OK   ${name}`); }
  else      { console.log(`FAIL ${name}${hint ? ": " + hint : ""}`); fail++; }
}

const personSpans = result.spans.filter((s) => s.label === "private_person");
const emailSpans  = result.spans.filter((s) => s.label === "private_email");
const phoneSpans  = result.spans.filter((s) => s.label === "private_phone");

// 1. The redacted output should NOT contain any of the original PII.
check(
  "redacted text does not contain original email 'alice@example.com'",
  !result.redactedText.includes("alice@example.com"),
);
check(
  "redacted text does not contain original email 'bob@gmail.com'",
  !result.redactedText.includes("bob@gmail.com"),
);
check(
  "redacted text does not contain original phone '+1-555-0123'",
  !result.redactedText.includes("+1-555-0123"),
);

// 2. The redacted output should contain faker-style values
//    (fictional 555-01XX phone, example.com email).
check(
  "redacted text contains a 555-01XX fake phone",
  /\+1-555-01\d{2}/.test(result.redactedText),
  result.redactedText,
);
check(
  "redacted text contains an example.com fake email",
  /@example\.com/.test(result.redactedText),
  result.redactedText,
);

// 3. Consistency: if "Alice" appears twice as `private_person`, both
//    mentions should map to the same fake name. (The model may or may
//    not detect both — only assert if it did.)
const aliceSpans = personSpans.filter((s) => s.text === "Alice");
if (aliceSpans.length >= 2) {
  const markers = new Set(aliceSpans.map((s) => s.marker));
  check(
    "all 'Alice' mentions map to the same fake name",
    markers.size === 1,
    `got ${markers.size} distinct fakes: ${[...markers].join(", ")}`,
  );
} else {
  console.log(`SKIP same-text consistency (model only detected ${aliceSpans.length} 'Alice' mention)`);
}

// 4. Markers are not the default `[label]` form (faker is active).
const usedDefaultMarker = result.spans.some(
  (s) => s.marker === `[${s.label}]` && s.label !== "secret",
);
check(
  "no default [label] markers in output (faker is active)",
  !usedDefaultMarker,
  result.spans.filter((s) => s.marker === `[${s.label}]`).map((s) => s.label).join(", "),
);

// 5. At minimum we expect some PII detected.
check("model detected ≥1 PII span", result.spans.length >= 1);

console.log(`\n${fail === 0 ? "✓" : "✗"} faker integration: ${fail === 0 ? "PASS" : `${fail} failure(s)`}`);
process.exit(fail === 0 ? 0 : 1);
