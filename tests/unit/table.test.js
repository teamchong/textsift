// Unit test for classifyColumns / redactTable using a fake detect
// implementation. No model needed — we substitute `detect()` with
// regex rules that mirror what the real model would emit for the
// test inputs. This is the right abstraction layer to test: the
// table orchestration is pure JS over `detect()` results.

import { strict as assert } from "node:assert";

// Import the table impl directly from source. bun handles the .ts
// natively (this test runs via `bun tests/unit/table.test.js`); the
// PrivacyFilter integration test hits the same code through the
// public method.
const { classifyColumns, redactTable } = await import(
  "../../packages/textsift/src/browser/inference/table.ts"
);

// Fake `detect` that emits spans for emails / phones / names by
// regex. Matches the shape of what the real model would produce on
// the test inputs.
function fakeDetect(text) {
  const spans = [];
  const emailRe = /[a-z0-9._-]+@[a-z0-9.-]+\.[a-z]{2,}/gi;
  const phoneRe = /\+?\d[\d\s().-]{8,}\d/g;
  const nameRe = /\b(Alice|Bob|Carol|David|Eve|Frank|Grace|Hank)(?:\s[A-Z][a-z]+)?\b/g;
  let m;
  while ((m = emailRe.exec(text)) !== null) {
    spans.push({
      label: "private_email", source: "model", start: m.index, end: m.index + m[0].length,
      text: m[0], marker: "[private_email]", confidence: 0.95,
    });
  }
  while ((m = phoneRe.exec(text)) !== null) {
    spans.push({
      label: "private_phone", source: "model", start: m.index, end: m.index + m[0].length,
      text: m[0], marker: "[private_phone]", confidence: 0.95,
    });
  }
  while ((m = nameRe.exec(text)) !== null) {
    spans.push({
      label: "private_person", source: "model", start: m.index, end: m.index + m[0].length,
      text: m[0], marker: "[private_person]", confidence: 0.95,
    });
  }
  return Promise.resolve({
    input: text,
    spans,
    summary: {},
    containsPii: spans.length > 0,
  });
}

// Fake `redact` built on fakeDetect — splice markers in.
async function fakeRedact(text, opts = {}) {
  const det = await fakeDetect(text);
  const sorted = [...det.spans].sort((a, b) => a.start - b.start);
  let out = text;
  // splice from end so earlier offsets don't shift
  for (let i = sorted.length - 1; i >= 0; i--) {
    const s = sorted[i];
    let marker = s.marker;
    if (typeof opts.markers === "function") {
      const r = opts.markers(s, i);
      if (r !== null && r !== undefined) marker = r;
    } else if (opts.markers && typeof opts.markers === "object") {
      const m = opts.markers[s.label];
      if (m !== null && m !== undefined) marker = m;
    }
    out = out.slice(0, s.start) + marker + out.slice(s.end);
  }
  return { input: text, redactedText: out, spans: det.spans, summary: {}, containsPii: det.spans.length > 0 };
}

let pass = 0, fail = 0;
function check(name, fn) {
  return Promise.resolve()
    .then(fn)
    .then(() => { console.log(`OK   ${name}`); pass++; })
    .catch((e) => { console.log(`FAIL ${name}: ${e.message}`); fail++; });
}

const ROWS_WITH_HEADER = [
  ["id",  "name",         "email",                 "amount"],
  ["1",   "Alice Carter", "alice@example.com",     "100"],
  ["2",   "Bob Davis",    "bob@example.com",       "250"],
  ["3",   "Carol Evans",  "carol@example.com",     "175"],
  ["4",   "David Frank",  "david@example.com",     "300"],
];

const ROWS_NO_HEADER = ROWS_WITH_HEADER.slice(1);

await check("classifyColumns: headerRow:true picks up header names", async () => {
  const cols = await classifyColumns(fakeDetect, ROWS_WITH_HEADER, { headerRow: true });
  assert.equal(cols.length, 4);
  assert.equal(cols[0].header, "id");
  assert.equal(cols[1].header, "name");
  assert.equal(cols[2].header, "email");
  assert.equal(cols[3].header, "amount");
});

await check("classifyColumns: detects PII columns + skips numeric columns", async () => {
  const cols = await classifyColumns(fakeDetect, ROWS_WITH_HEADER, { headerRow: true });
  assert.equal(cols[0].label, null,            `id col should be null, got ${cols[0].label}`);
  assert.equal(cols[1].label, "private_person", `name col should be person, got ${cols[1].label}`);
  assert.equal(cols[2].label, "private_email",  `email col should be email, got ${cols[2].label}`);
  assert.equal(cols[3].label, null,            `amount col should be null, got ${cols[3].label}`);
});

await check("classifyColumns: confidence reflects sample match rate", async () => {
  const cols = await classifyColumns(fakeDetect, ROWS_WITH_HEADER, { headerRow: true });
  assert.equal(cols[1].confidence, 1.0, "all 4 names matched");
  assert.equal(cols[2].confidence, 1.0, "all 4 emails matched");
  assert.equal(cols[3].confidence, 0,   "no amount cells matched");
});

await check("classifyColumns: explicit headers option", async () => {
  const cols = await classifyColumns(fakeDetect, ROWS_NO_HEADER, {
    headers: ["id", "name", "email", "amount"],
  });
  assert.equal(cols[0].header, "id");
  assert.equal(cols[1].header, "name");
});

await check("classifyColumns: explicit headers length mismatch throws", async () => {
  await assert.rejects(
    () => classifyColumns(fakeDetect, ROWS_NO_HEADER, { headers: ["one", "two"] }),
    /headers length .* does not match/,
  );
});

await check("classifyColumns: empty rows → empty result", async () => {
  const cols = await classifyColumns(fakeDetect, []);
  assert.deepEqual(cols, []);
});

await check("classifyColumns: respects sampleSize", async () => {
  const big = Array.from({ length: 1000 }, (_, i) => [
    `${i}`, `Alice ${i}`, `alice${i}@example.com`, `${i * 10}`,
  ]);
  const cols = await classifyColumns(fakeDetect, big, { sampleSize: 5 });
  // samples is per-column; we should have at most 5
  assert.ok(cols[1].samples <= 5, `expected ≤5 samples, got ${cols[1].samples}`);
  // confidence still calculated correctly
  assert.equal(cols[1].label, "private_person");
});

await check("classifyColumns: empty cells skipped from sampling", async () => {
  const sparse = [
    ["",                    ""],
    ["alice@example.com",   ""],
    ["",                    ""],
    ["bob@example.com",     ""],
    ["carol@example.com",   ""],
  ];
  const cols = await classifyColumns(fakeDetect, sparse);
  assert.equal(cols[0].samples, 3, "3 non-empty email cells");
  assert.equal(cols[0].label, "private_email");
  assert.equal(cols[1].samples, 0, "all empty");
  assert.equal(cols[1].label, null);
});

await check("redactTable mode=redact: PII cells get markers, others pass through", async () => {
  const out = await redactTable(fakeDetect, fakeRedact, undefined, ROWS_WITH_HEADER, {
    headerRow: true,
  });
  // Header row + 4 data rows
  assert.equal(out.length, 5);
  assert.deepEqual(out[0], ["id", "name", "email", "amount"]);
  // Row 1: id passes, name + email redacted, amount passes
  assert.equal(out[1][0], "1");
  assert.equal(out[1][1], "[private_person]");
  assert.equal(out[1][2], "[private_email]");
  assert.equal(out[1][3], "100");
});

await check("redactTable mode=drop_column: PII columns omitted entirely", async () => {
  const out = await redactTable(fakeDetect, fakeRedact, undefined, ROWS_WITH_HEADER, {
    headerRow: true,
    mode: "drop_column",
  });
  assert.equal(out[0].length, 2, "only id + amount remain");
  assert.deepEqual(out[0], ["id", "amount"]);
  assert.deepEqual(out[1], ["1", "100"]);
});

await check("redactTable mode=synth: cells get realistic fakes", async () => {
  const out = await redactTable(fakeDetect, fakeRedact, undefined, ROWS_WITH_HEADER, {
    headerRow: true,
    mode: "synth",
  });
  // Names should look real, not be `[private_person]`
  for (let r = 1; r < out.length; r++) {
    assert.notEqual(out[r][1], "[private_person]");
    assert.notEqual(out[r][2], "[private_email]");
    assert.match(out[r][1], /^[A-Z][a-z]+ [A-Z][a-z]+$/, `expected fake name, got "${out[r][1]}"`);
    assert.match(out[r][2], /@example\.com$/);
  }
});

await check("redactTable mode=synth: same input → same fake within table", async () => {
  // Repeat Alice on row 4 to check consistency.
  const repeated = [...ROWS_WITH_HEADER];
  repeated.push(["5", "Alice Carter", "alice@example.com", "999"]);
  const out = await redactTable(fakeDetect, fakeRedact, undefined, repeated, {
    headerRow: true, mode: "synth",
  });
  // Row 1 is "Alice Carter", row 5 is also "Alice Carter".
  // Note: data starts at out[1] (after header row).
  assert.equal(out[1][1], out[5][1], "both Alice Carter mentions → same fake");
  assert.equal(out[1][2], out[5][2], "both alice@example.com mentions → same fake");
});

await check("redactTable: pre-supplied classifications skip the classify step", async () => {
  let detectCalls = 0;
  const countingDetect = (text, opts) => { detectCalls++; return fakeDetect(text, opts); };
  const classifications = [
    { index: 0, label: null, confidence: 0, samples: 0, labelCounts: {} },
    { index: 1, label: "private_person", confidence: 1.0, samples: 4, labelCounts: { private_person: 4 } },
    { index: 2, label: "private_email", confidence: 1.0, samples: 4, labelCounts: { private_email: 4 } },
    { index: 3, label: null, confidence: 0, samples: 0, labelCounts: {} },
  ];
  const out = await redactTable(countingDetect, fakeRedact, undefined, ROWS_WITH_HEADER, {
    headerRow: true,
    classifications,
  });
  // classify step skipped → countingDetect should not be invoked at
  // all from the column-sampling path (the per-cell redact path
  // doesn't route through this wrapper either, so 0 is correct).
  // What we're really checking: passing classifications avoids the
  // sampling work that would otherwise hit detect ~16 times here.
  assert.equal(detectCalls, 0, `expected 0 detect calls when classifications pre-supplied, got ${detectCalls}`);
  assert.equal(out.length, 5);
});

await Promise.resolve(); // flush any pending async log
console.log(`\n${pass}/${pass + fail} passed`);
process.exit(fail === 0 ? 0 : 1);
