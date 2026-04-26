// End-to-end integration test for `classifyColumns` + `redactTable`
// running through the real model on Node native. Loads a small CSV-
// shaped dataset, classifies columns, redacts in three modes, and
// asserts the results pass.

import { PrivacyFilter } from "../../../packages/textsift/dist/index.js";

const ROWS = [
  ["id", "name",         "email",                  "phone",          "amount"],
  ["1",  "Alice Carter", "alice@example.com",      "+1-617-555-0199", "100"],
  ["2",  "Bob Davis",    "bob@example.com",        "+1-415-555-0188", "250"],
  ["3",  "Carol Evans",  "carol@example.com",      "+1-212-555-0144", "175"],
  ["4",  "David Frank",  "david@example.com",      "+1-617-555-0123", "300"],
  ["5",  "Alice Carter", "alice@example.com",      "+1-617-555-0199", "999"],  // repeat for consistency check
];

console.log("[filter] creating PrivacyFilter...");
const filter = await PrivacyFilter.create();
console.log("[filter] ready");

// 1. classifyColumns
console.log("\n=== classifyColumns ===");
const cols = await filter.classifyColumns(ROWS, { headerRow: true });
for (const c of cols) {
  console.log(`  [${c.index}] ${c.header}: ${c.label ?? "(none)"} (conf=${c.confidence.toFixed(2)}, samples=${c.samples})`);
}

let fail = 0;
function check(name, cond, hint) {
  if (cond) console.log(`OK   ${name}`);
  else      { console.log(`FAIL ${name}${hint ? ": " + hint : ""}`); fail++; }
}

check("id column → no PII",             cols[0].label === null);
check("name column → private_person",   cols[1].label === "private_person");
check("email column → private_email",   cols[2].label === "private_email");
check("phone column → private_phone",   cols[3].label === "private_phone");
check("amount column → no PII",         cols[4].label === null);
check("name confidence high",           cols[1].confidence >= 0.7, `conf=${cols[1].confidence}`);
check("email confidence high",          cols[2].confidence >= 0.7, `conf=${cols[2].confidence}`);
check("phone confidence high",          cols[3].confidence >= 0.7, `conf=${cols[3].confidence}`);

// 2. redactTable mode=redact
console.log("\n=== redactTable mode=redact ===");
const redactedRedact = await filter.redactTable(ROWS, {
  headerRow: true,
  classifications: cols,  // skip re-classification
});
for (const row of redactedRedact) console.log("  ", row.join(" | "));

check("redact: header preserved",       redactedRedact[0].join(",") === "id,name,email,phone,amount");
check("redact: id passes through",      redactedRedact[1][0] === "1");
check("redact: amount passes through",  redactedRedact[1][4] === "100");
check("redact: name has marker",        redactedRedact[1][1].includes("[private_person]"));
check("redact: email has marker",       redactedRedact[1][2].includes("[private_email]"));
check("redact: phone has marker",       redactedRedact[1][3].includes("[private_phone]"));

// 3. redactTable mode=drop_column
console.log("\n=== redactTable mode=drop_column ===");
const redactedDrop = await filter.redactTable(ROWS, {
  headerRow: true,
  mode: "drop_column",
  classifications: cols,
});
for (const row of redactedDrop) console.log("  ", row.join(" | "));

check("drop_column: only id+amount in header", redactedDrop[0].length === 2 && redactedDrop[0].includes("id") && redactedDrop[0].includes("amount"));
check("drop_column: data rows have 2 cols",    redactedDrop[1].length === 2);
check("drop_column: numeric values preserved", redactedDrop[1][0] === "1" && redactedDrop[1][1] === "100");

// 4. redactTable mode=synth
console.log("\n=== redactTable mode=synth ===");
const redactedSynth = await filter.redactTable(ROWS, {
  headerRow: true,
  mode: "synth",
  classifications: cols,
});
for (const row of redactedSynth) console.log("  ", row.join(" | "));

// Names should look real, not be `[private_person]`
const nameCells = redactedSynth.slice(1).map((r) => r[1]);
const emailCells = redactedSynth.slice(1).map((r) => r[2]);
const phoneCells = redactedSynth.slice(1).map((r) => r[3]);

check("synth: names are realistic",    nameCells.every((n) => /[A-Z][a-z]+ [A-Z][a-z]+/.test(n)),  nameCells.join(", "));
check("synth: emails are realistic",   emailCells.every((e) => /@example\.com/.test(e)),            emailCells.join(", "));
check("synth: phones are 555-01XX",    phoneCells.every((p) => /\+1-555-01\d{2}/.test(p)),          phoneCells.join(", "));
check("synth: no [private_*] markers", !redactedSynth.some((r) => r.some((c) => c.includes("[private"))));

// Consistency: row 1 and row 5 both have "Alice Carter" / "alice@example.com" / "+1-617-555-0199".
// Synth should map them to the same fake values.
check(
  "synth: repeated name → same fake",
  redactedSynth[1][1] === redactedSynth[5][1],
  `row1=${redactedSynth[1][1]}, row5=${redactedSynth[5][1]}`,
);
check(
  "synth: repeated email → same fake",
  redactedSynth[1][2] === redactedSynth[5][2],
  `row1=${redactedSynth[1][2]}, row5=${redactedSynth[5][2]}`,
);
check(
  "synth: repeated phone → same fake",
  redactedSynth[1][3] === redactedSynth[5][3],
  `row1=${redactedSynth[1][3]}, row5=${redactedSynth[5][3]}`,
);

filter.dispose();

console.log(`\n${fail === 0 ? "✓" : "✗"} table integration: ${fail === 0 ? "PASS" : `${fail} failure(s)`}`);
process.exit(fail === 0 ? 0 : 1);
