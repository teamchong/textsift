// Unit tests for the SARIF v2.1.0 converter. Pure data transform —
// no model needed.

import { strict as assert } from "node:assert";

const { toSarif, detectResultToSarif } = await import(
  "../../packages/textsift/src/sarif.ts"
);

let pass = 0, fail = 0;
function check(name, fn) {
  try { fn(); console.log(`OK   ${name}`); pass++; }
  catch (e) { console.log(`FAIL ${name}: ${e.message}`); fail++; }
}

const span = (label, source, severity, start, end, text, confidence = 0.95) => ({
  label, source, severity, start, end, text,
  marker: `[${label}]`, confidence,
});

check("toSarif: empty findings → valid empty log", () => {
  const out = toSarif([]);
  assert.equal(out.version, "2.1.0");
  assert.equal(out.runs.length, 1);
  assert.equal(out.runs[0].results.length, 0);
  assert.equal(out.runs[0].tool.driver.rules.length, 0);
  assert.equal(out.runs[0].tool.driver.name, "textsift");
});

check("toSarif: model span → error result with rule + region", () => {
  const out = toSarif([{
    uri: "src/test.ts",
    text: "Hi Alice, email alice@example.com",
    spans: [span("private_email", "model", undefined, 16, 33, "alice@example.com")],
  }]);
  const result = out.runs[0].results[0];
  assert.equal(result.ruleId, "private_email");
  assert.equal(result.level, "error");
  assert.match(result.message.text, /Found private_email/);
  const loc = result.locations[0].physicalLocation;
  assert.equal(loc.artifactLocation.uri, "src/test.ts");
  assert.equal(loc.region.startLine, 1);
  assert.equal(loc.region.startColumn, 17, `expected col 17 (1-based offset 16 + 1), got ${loc.region.startColumn}`);
  assert.equal(loc.region.endLine, 1);
  assert.equal(loc.region.endColumn, 34);
});

check("toSarif: rule span with severity:warn → warning level", () => {
  const out = toSarif([{
    uri: "x", text: "secret here",
    spans: [span("MY_TOKEN", "rule", "warn", 0, 6, "secret")],
  }]);
  assert.equal(out.runs[0].results[0].level, "warning");
});

check("toSarif: rule span with severity:block → error level", () => {
  const out = toSarif([{
    uri: "x", text: "ghp_abc",
    spans: [span("GITHUB_PAT_CLASSIC", "rule", "block", 0, 7, "ghp_abc")],
  }]);
  assert.equal(out.runs[0].results[0].level, "error");
});

check("toSarif: modelSpanLevel override downgrades model spans", () => {
  const out = toSarif([{
    uri: "x", text: "Alice",
    spans: [span("private_person", "model", undefined, 0, 5, "Alice")],
  }], { modelSpanLevel: "warning" });
  assert.equal(out.runs[0].results[0].level, "warning");
});

check("toSarif: modelSpanLevel does NOT affect rule spans", () => {
  const out = toSarif([{
    uri: "x", text: "Alice ghp_xyz",
    spans: [
      span("private_person", "model", undefined, 0, 5, "Alice"),
      span("GITHUB_PAT_CLASSIC", "rule", "block", 6, 13, "ghp_xyz"),
    ],
  }], { modelSpanLevel: "warning" });
  assert.equal(out.runs[0].results[0].level, "warning"); // model
  assert.equal(out.runs[0].results[1].level, "error");   // rule (untouched)
});

check("toSarif: rules array deduplicated by id", () => {
  const out = toSarif([{
    uri: "x", text: "Alice and Bob",
    spans: [
      span("private_person", "model", undefined, 0, 5, "Alice"),
      span("private_person", "model", undefined, 10, 13, "Bob"),
    ],
  }]);
  const rules = out.runs[0].tool.driver.rules;
  assert.equal(rules.length, 1, `expected 1 rule, got ${rules.length}`);
  assert.equal(rules[0].id, "private_person");
});

check("toSarif: line/col across newlines computed correctly", () => {
  // "Alice" on line 1, "Bob" on line 3
  const text = "Hi Alice\n\nbye Bob";
  const out = toSarif([{
    uri: "x", text,
    spans: [
      span("private_person", "model", undefined, 3, 8, "Alice"),
      span("private_person", "model", undefined, 14, 17, "Bob"),
    ],
  }]);
  const r1 = out.runs[0].results[0].locations[0].physicalLocation.region;
  const r2 = out.runs[0].results[1].locations[0].physicalLocation.region;
  assert.equal(r1.startLine, 1);
  assert.equal(r2.startLine, 3, `Bob should be on line 3, got ${r2.startLine}`);
  assert.equal(r2.startColumn, 5);
});

check("toSarif: long span text truncated in message", () => {
  const long = "a".repeat(200);
  const out = toSarif([{
    uri: "x", text: long,
    spans: [span("secret", "model", undefined, 0, 200, long)],
  }]);
  assert.match(out.runs[0].results[0].message.text, /\.\.\."$/);
  assert.ok(out.runs[0].results[0].message.text.length < 200);
});

check("toSarif: partialFingerprints stable for same input", () => {
  const args = [{
    uri: "x", text: "alice@example.com",
    spans: [span("private_email", "model", undefined, 0, 17, "alice@example.com")],
  }];
  const a = toSarif(args).runs[0].results[0].partialFingerprints?.primaryLocationLineHash;
  const b = toSarif(args).runs[0].results[0].partialFingerprints?.primaryLocationLineHash;
  assert.ok(a, "fingerprint should be set");
  assert.equal(a, b, "fingerprint should be deterministic");
});

check("toSarif: rule spans without model_rules entry get synthesised metadata", () => {
  const out = toSarif([{
    uri: "x", text: "ghp_xyz",
    spans: [span("CUSTOM_INTERNAL_TOKEN", "rule", "block", 0, 7, "ghp_xyz")],
  }]);
  const rules = out.runs[0].tool.driver.rules;
  assert.equal(rules[0].id, "CUSTOM_INTERNAL_TOKEN");
  assert.equal(rules[0].name, "CUSTOM_INTERNAL_TOKEN");
  assert.match(rules[0].shortDescription.text, /custom rule/);
});

check("detectResultToSarif: convenience wrapper for single-file results", () => {
  const result = {
    input: "Alice",
    spans: [span("private_person", "model", undefined, 0, 5, "Alice")],
    summary: {},
    containsPii: true,
  };
  const out = detectResultToSarif(result, "src/test.ts");
  assert.equal(out.runs[0].results[0].locations[0].physicalLocation.artifactLocation.uri, "src/test.ts");
});

check("toSarif: schema URL points at the canonical SARIF 2.1.0 schema", () => {
  const out = toSarif([]);
  assert.equal(out.$schema, "https://json.schemastore.org/sarif-2.1.0.json");
});

console.log(`\n${pass}/${pass + fail} passed`);
process.exit(fail === 0 ? 0 : 1);
