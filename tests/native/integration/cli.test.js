// CLI smoke test — runs the bundled cli.js as a subprocess across
// the documented surfaces. Each subprocess shares the same model
// cache (the model is pre-warmed by the integration tests above).

import { spawn } from "node:child_process";
import { mkdtemp, writeFile, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { strict as assert } from "node:assert";

const CLI = resolve(import.meta.dirname, "../../../packages/textsift/dist/cli.js");

function run(args, stdin = "") {
  return new Promise((resolve, reject) => {
    const child = spawn("node", [CLI, ...args], { stdio: ["pipe", "pipe", "pipe"] });
    let stdout = "", stderr = "";
    child.stdout.on("data", (d) => { stdout += d.toString(); });
    child.stderr.on("data", (d) => { stderr += d.toString(); });
    child.on("error", reject);
    child.on("close", (code) => resolve({ stdout, stderr, code: code ?? 0 }));
    if (stdin) child.stdin.write(stdin);
    child.stdin.end();
  });
}

let pass = 0, fail = 0;
async function check(name, fn) {
  try { await fn(); console.log(`OK   ${name}`); pass++; }
  catch (e) { console.log(`FAIL ${name}: ${e.message}`); fail++; }
}

const work = await mkdtemp(resolve(tmpdir(), "textsift-cli-test-"));

await check("--help prints usage", async () => {
  const r = await run(["--help"]);
  assert.equal(r.code, 0);
  assert.match(r.stdout, /textsift .* — local-first PII/);
  assert.match(r.stdout, /COMMANDS/);
  assert.match(r.stdout, /redact/);
});

await check("--version prints version", async () => {
  const r = await run(["--version"]);
  assert.equal(r.code, 0);
  assert.match(r.stdout, /^textsift \d/);
});

await check("unknown command exits 2", async () => {
  const r = await run(["banana"]);
  assert.equal(r.code, 2);
  assert.match(r.stderr, /unknown command/);
});

await check("redact via stdin", async () => {
  const r = await run(["redact", "--no-prompt"], "Hi Alice, my email is alice@example.com");
  assert.equal(r.code, 0);
  assert.match(r.stdout, /\[private_person\]/);
  assert.match(r.stdout, /\[private_email\]/);
  assert.doesNotMatch(r.stdout, /alice@example\.com/);
});

await check("redact --synth produces realistic fakes", async () => {
  const r = await run(["redact", "--synth", "--no-prompt"], "Hi Alice, my email is alice@example.com");
  assert.equal(r.code, 0);
  assert.doesNotMatch(r.stdout, /\[private_/);
  assert.match(r.stdout, /@example\.com/);
});

await check("redact --in-place writes to file", async () => {
  const file = resolve(work, "input.txt");
  await writeFile(file, "Hi Alice, alice@example.com");
  const r = await run(["redact", file, "--in-place", "--no-prompt"]);
  assert.equal(r.code, 0);
  const written = await readFile(file, "utf8");
  assert.match(written, /\[private_person\]/);
  assert.match(r.stderr, /wrote .* redactions to/);
});

await check("detect emits JSON shape", async () => {
  const r = await run(["detect", "--no-prompt"], "Email me at alice@example.com");
  assert.equal(r.code, 0);
  const parsed = JSON.parse(r.stdout);
  assert.equal(parsed.containsPii, true);
  assert.ok(parsed.spans.length >= 1);
  assert.equal(parsed.spans[0].label, "private_email");
});

await check("detect --jsonl emits one span per line", async () => {
  const r = await run(["detect", "--jsonl", "--no-prompt"], "Hi Alice (alice@example.com)");
  assert.equal(r.code, 0);
  const lines = r.stdout.trim().split("\n").filter(Boolean);
  for (const l of lines) {
    const span = JSON.parse(l);
    assert.ok(span.label, `expected label on each line, got: ${l}`);
  }
  assert.ok(lines.length >= 1);
});

await check("classify CSV emits per-column JSON", async () => {
  const file = resolve(work, "people.csv");
  await writeFile(file,
    "id,name,email,amount\n" +
    "1,Alice Carter,alice@example.com,100\n" +
    "2,Bob Davis,bob@example.com,250\n",
  );
  const r = await run(["classify", file, "--header", "--no-prompt"]);
  assert.equal(r.code, 0, r.stderr);
  const cols = JSON.parse(r.stdout);
  assert.equal(cols.length, 4);
  assert.equal(cols[0].label, null);                 // id
  assert.equal(cols[1].label, "private_person");
  assert.equal(cols[2].label, "private_email");
  assert.equal(cols[3].label, null);                 // amount
});

await check("table mode=synth swaps PII cells with realistic fakes", async () => {
  const file = resolve(work, "table.csv");
  await writeFile(file,
    "id,name,email,amount\n" +
    "1,Alice Carter,alice@example.com,100\n" +
    "2,Bob Davis,bob@example.com,250\n",
  );
  const r = await run(["table", file, "--header", "--mode", "synth", "--no-prompt"]);
  assert.equal(r.code, 0, r.stderr);
  const lines = r.stdout.trim().split("\n");
  assert.equal(lines[0], "id,name,email,amount");
  // Row 1 should have realistic fake name + email but original id + amount
  const row1 = lines[1].split(",");
  assert.equal(row1[0], "1");
  assert.notEqual(row1[1], "Alice Carter");
  assert.match(row1[1], /^[A-Z][a-z]+ [A-Z][a-z]+$/);
  assert.match(row1[2], /@example\.com$/);
  assert.equal(row1[3], "100");
});

await check("table mode=drop_column omits PII columns", async () => {
  const file = resolve(work, "table-drop.csv");
  await writeFile(file,
    "id,name,email,amount\n" +
    "1,Alice Carter,alice@example.com,100\n" +
    "2,Bob Davis,bob@example.com,250\n",
  );
  const r = await run(["table", file, "--header", "--mode", "drop_column", "--no-prompt"]);
  assert.equal(r.code, 0, r.stderr);
  const lines = r.stdout.trim().split("\n");
  assert.equal(lines[0], "id,amount");
  assert.equal(lines[1], "1,100");
  assert.equal(lines[2], "2,250");
});

await check("cache info subcommand emits JSON", async () => {
  const r = await run(["cache", "info"]);
  assert.equal(r.code, 0);
  const parsed = JSON.parse(r.stdout);
  assert.ok(typeof parsed.cacheDir === "string");
  assert.ok(typeof parsed.totalBytes === "number");
});

await check("cache without subcommand exits 2", async () => {
  const r = await run(["cache"]);
  assert.equal(r.code, 2);
  assert.match(r.stderr, /requires `info` or `clear`/);
});

await check("--offline + cold cacheDir exits with clear error", async () => {
  // Use a fresh empty cache dir → offline mode should fail loudly.
  const cacheDir = resolve(work, "fake-cache");
  const r = await run([
    "redact", "--offline", "--cache-dir", cacheDir, "--no-prompt",
  ], "Hi Alice");
  assert.notEqual(r.code, 0);
  assert.match(r.stderr, /TEXTSIFT_OFFLINE|cache miss|offline/i);
});

await rm(work, { recursive: true, force: true });

console.log(`\n${pass}/${pass + fail} passed`);
process.exit(fail === 0 ? 0 : 1);
