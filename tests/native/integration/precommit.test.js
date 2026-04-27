// Subprocess test for the pre-commit hook entry script. Verifies the
// industry-standard interface that pre-commit (https://pre-commit.com)
// expects: pass file paths as argv, exit non-zero on findings, exit 0
// on clean files. Also covers --warn-only, severity env, and the
// extension/size filters.

import { spawn } from "node:child_process";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { strict as assert } from "node:assert";

const HOOK = resolve(import.meta.dirname, "../../../packages/textsift/scripts/precommit.js");

function run(args, extraEnv = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn("node", [HOOK, ...args], {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, ...extraEnv },
    });
    let stdout = "", stderr = "";
    child.stdout.on("data", (d) => { stdout += d.toString(); });
    child.stderr.on("data", (d) => { stderr += d.toString(); });
    child.on("error", reject);
    child.on("close", (code) => resolve({ stdout, stderr, code: code ?? 0 }));
  });
}

let pass = 0, fail = 0;
async function check(name, fn) {
  try { await fn(); console.log(`OK   ${name}`); pass++; }
  catch (e) { console.log(`FAIL ${name}: ${e.message}`); fail++; }
}

const work = await mkdtemp(resolve(tmpdir(), "textsift-precommit-test-"));

// Pre-create test files.
const cleanFile = resolve(work, "clean.txt");
const piiFile = resolve(work, "pii.txt");
const secretFile = resolve(work, "secret.txt");
const binaryFile = resolve(work, "binary.png");
const lockFile = resolve(work, "package-lock.json");
const bigFile = resolve(work, "big.txt");

await writeFile(cleanFile,  "Hello world, just text here.");
await writeFile(piiFile,    "Hi Alice, my email is alice@example.com");
await writeFile(secretFile, "ghp_abcdefghijklmnopqrstuvwxyzABCDEF1234");
await writeFile(binaryFile, Buffer.from([0x89, 0x50, 0x4e, 0x47]));   // PNG header
await writeFile(lockFile,   '{"lockfileVersion":3}');
// 2 MB file (exceeds the 1 MB default cap).
await writeFile(bigFile,    "x".repeat(2_000_000));

await check("--help prints usage and exits 0", async () => {
  const r = await run(["--help"]);
  assert.equal(r.code, 0);
  assert.match(r.stdout, /pre-commit hook/);
});

await check("no files argv → exit 0 (no-op)", async () => {
  const r = await run([]);
  assert.equal(r.code, 0);
});

await check("clean file only → exit 0, no output", async () => {
  const r = await run([cleanFile]);
  assert.equal(r.code, 0);
  assert.equal(r.stdout, "");
  assert.equal(r.stderr, "");
});

await check("PII file → exit 1, finding(s) printed to stderr", async () => {
  const r = await run([piiFile]);
  assert.equal(r.code, 1);
  assert.match(r.stderr, /blocking finding/);
  assert.match(r.stderr, /private_email/);
  assert.match(r.stderr, /alice@example\.com/);
});

await check("secret file → exit 1 (secrets preset enabled by default)", async () => {
  const r = await run([secretFile]);
  assert.equal(r.code, 1);
  assert.match(r.stderr, /GITHUB_PAT_CLASSIC|ghp_/);
});

await check("--warn-only → exit 0 even with PII findings", async () => {
  const r = await run(["--warn-only", piiFile]);
  assert.equal(r.code, 0);
  // Findings still printed for visibility
  assert.match(r.stderr, /blocking finding/);
});

await check("SEVERITY=warn → model spans skipped, rule-block spans still block", async () => {
  // PII file: only model spans → exit 0
  const r1 = await run([piiFile], { TEXTSIFT_PRECOMMIT_SEVERITY: "warn" });
  assert.equal(r1.code, 0, `pii file: stderr=${r1.stderr}`);
  // Secret file: rule span with block severity → still exit 1
  const r2 = await run([secretFile], { TEXTSIFT_PRECOMMIT_SEVERITY: "warn" });
  assert.equal(r2.code, 1);
});

await check("--allow-list shows scan/skip decisions, exits 0", async () => {
  const r = await run([
    "--allow-list",
    cleanFile, piiFile, binaryFile, lockFile, bigFile,
  ]);
  assert.equal(r.code, 0);
  assert.match(r.stdout, /SCAN .*clean\.txt/);
  assert.match(r.stdout, /SCAN .*pii\.txt/);
  assert.match(r.stdout, /SKIP .*binary\.png/);
  assert.match(r.stdout, /SKIP .*big\.txt/);
});

await check("binary + lock + huge files filtered out, only PII detected", async () => {
  const r = await run([
    cleanFile, piiFile, binaryFile, lockFile, bigFile,
  ]);
  assert.equal(r.code, 1);
  // Only the PII file is reported; binary/lock/big skipped silently.
  assert.match(r.stderr, /private_email/);
  assert.doesNotMatch(r.stderr, /binary\.png|package-lock\.json|big\.txt/);
});

await check("custom MAX_BYTES env shrinks the size filter", async () => {
  // Set MAX to 10 bytes — clean file (~28 bytes) gets filtered out.
  const r = await run([cleanFile], { TEXTSIFT_PRECOMMIT_MAX_BYTES: "10" });
  assert.equal(r.code, 0);
});

await check("TEXTSIFT_PRECOMMIT_SECRETS=0 drops the regex-rule labels", async () => {
  // The secret file gets flagged as GITHUB_PAT_CLASSIC by the regex
  // preset; with the preset disabled, the model itself may still
  // catch it (secret is one of the 8 model categories), but the
  // GITHUB_PAT_CLASSIC label specifically should not appear.
  const r = await run([secretFile], { TEXTSIFT_PRECOMMIT_SECRETS: "0" });
  assert.doesNotMatch(r.stderr, /GITHUB_PAT_CLASSIC/);
});

await check("GITHUB_ACTIONS=true emits ::error annotations on stdout", async () => {
  const r = await run([piiFile], { GITHUB_ACTIONS: "true" });
  assert.equal(r.code, 1);
  // Annotation format: ::error file=...,line=...,col=...,title=...::Found "..."
  assert.match(r.stdout, /^::error file=.+,line=\d+,col=\d+,title=textsift PII \(.+\)::Found ".+"/m);
  // Should have one annotation per blocking span (PII file has 2: person + email).
  const lines = r.stdout.split("\n").filter((l) => l.startsWith("::"));
  assert.equal(lines.length, 2, `expected 2 annotations, got ${lines.length}: ${r.stdout}`);
});

await check("GITHUB_ACTIONS=true + --warn-only emits ::warning annotations", async () => {
  const r = await run(["--warn-only", piiFile], { GITHUB_ACTIONS: "true" });
  assert.equal(r.code, 0);
  assert.match(r.stdout, /^::warning file=/m);
  assert.doesNotMatch(r.stdout, /^::error /m);
});

await check("annotations escape newlines/percent in preview text", async () => {
  // Span text with a newline + percent shouldn't break the workflow command parser.
  const trickyFile = resolve(work, "tricky.txt");
  await writeFile(trickyFile, "Email me at alice@example.com\n50%");
  const r = await run([trickyFile], { GITHUB_ACTIONS: "true" });
  assert.equal(r.code, 1);
  // Each annotation should be on a single line.
  for (const l of r.stdout.split("\n")) {
    if (!l.startsWith("::")) continue;
    assert.doesNotMatch(l, /\n/);
    assert.doesNotMatch(l, /%[^:]/, `unescaped percent in: ${l}`);
  }
});

await rm(work, { recursive: true, force: true });

console.log(`\n${pass}/${pass + fail} passed`);
process.exit(fail === 0 ? 0 : 1);
