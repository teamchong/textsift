// Pure-logic unit test for markerPresets.faker — no GPU, no model.
// Builds a synthetic DetectedSpan list and asserts the strategy
// returns realistic-looking fakes that are consistent within an
// instance.

import { strict as assert } from "node:assert";
import { markerPresets } from "../../packages/textsift/dist/index.js";

function span(label, text, start = 0) {
  return {
    label,
    source: "model",
    start,
    end: start + text.length,
    text,
    marker: "",
    confidence: 0.9,
  };
}

let pass = 0, fail = 0;
function check(name, fn) {
  try { fn(); console.log(`OK   ${name}`); pass++; }
  catch (e) { console.log(`FAIL ${name}: ${e.message}`); fail++; }
}

check("faker() returns a function", () => {
  const m = markerPresets.faker();
  assert.equal(typeof m, "function");
});

check("private_person → realistic name", () => {
  const m = markerPresets.faker();
  const out = m(span("private_person", "John Smith"), 0);
  assert.match(out, /^[A-Z][a-z]+ [A-Z][a-z]+$/, `got "${out}"`);
});

check("private_email → email format", () => {
  const m = markerPresets.faker();
  const out = m(span("private_email", "john@example.com"), 0);
  assert.match(out, /^[a-z]+(?:\.[a-z]+)*@example\.com$/, `got "${out}"`);
});

check("private_phone → fictional 555-01XX number", () => {
  const m = markerPresets.faker();
  const out = m(span("private_phone", "+1-415-555-1234"), 0);
  assert.match(out, /^\+1-555-01\d{2}$/, `got "${out}"`);
});

check("account_number → test BIN", () => {
  const m = markerPresets.faker();
  const out = m(span("account_number", "4242424242424242"), 0);
  assert.match(out, /^4111-1111-1111-\d{4}$/, `got "${out}"`);
});

check("private_date → ISO date", () => {
  const m = markerPresets.faker();
  const out = m(span("private_date", "December 1, 2024"), 0);
  assert.match(out, /^\d{4}-\d{2}-\d{2}$/, `got "${out}"`);
});

check("private_url → example.com", () => {
  const m = markerPresets.faker();
  const out = m(span("private_url", "https://acme.internal/foo"), 0);
  assert.match(out, /^https:\/\/example\.com\/path\/\d+$/, `got "${out}"`);
});

check("private_address → Springfield template", () => {
  const m = markerPresets.faker();
  const out = m(span("private_address", "123 Real St, Real City, CA 99999"), 0);
  assert.match(out, /^\d+ Main St, Springfield, IL \d{5}$/, `got "${out}"`);
});

check("secret → not faked (security footgun)", () => {
  const m = markerPresets.faker();
  const out = m(span("secret", "ghp_abcdefghijklmnopqrstuvwxyzABCDEF1234"), 0);
  assert.equal(out, "[secret]");
});

check("same input text → same fake (within instance)", () => {
  const m = markerPresets.faker();
  const a = m(span("private_person", "John Smith"), 0);
  const b = m(span("private_person", "John Smith"), 1);
  assert.equal(a, b, "John Smith twice should give same fake");
});

check("different input text → different fake", () => {
  const m = markerPresets.faker();
  const a = m(span("private_person", "John Smith"), 0);
  const b = m(span("private_person", "Jane Doe"), 1);
  assert.notEqual(a, b, "different names should give different fakes");
});

check("fresh faker() instance → fresh state", () => {
  const m1 = markerPresets.faker();
  const m2 = markerPresets.faker();
  // Both start at index 0, so first call to each gives the same result.
  // That's expected — each instance starts fresh.
  const a = m1(span("private_person", "John"), 0);
  const b = m2(span("private_person", "Mary"), 0);
  assert.equal(a, b, "two fresh instances both pick pool[0]");
});

check("counter increments per category independently", () => {
  const m = markerPresets.faker();
  const p1 = m(span("private_person", "A"), 0);
  const p2 = m(span("private_person", "B"), 1);
  const e1 = m(span("private_email", "a@x.com"), 2);
  // Person counter is at 2; email counter starts fresh at 0.
  assert.notEqual(p1, p2, "two distinct people → two distinct fakes");
  assert.match(e1, /alice\.anderson@example\.com/);
});

check("unknown label → falls back to [label]", () => {
  const m = markerPresets.faker();
  const out = m(span("custom_thing", "foo"), 0);
  assert.equal(out, "[custom_thing]");
});

console.log(`\n${pass}/${pass + fail} passed`);
process.exit(fail === 0 ? 0 : 1);
