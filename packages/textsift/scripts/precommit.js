#!/usr/bin/env node
/**
 * pre-commit framework hook entry. Receives staged file paths as
 * argv from pre-commit; loads PrivacyFilter once; scans each file;
 * exits 1 if any blocking PII is found.
 *
 * Loading the model once and scanning N files in-process beats the
 * CLI-per-file approach by ~30 sec per commit (cold-start latency
 * × N files); with this we pay it once.
 */

import { readFile, stat } from "node:fs/promises";
import { extname, relative } from "node:path";
import { argv, env, stderr, stdout, exit, cwd } from "node:process";

// ── Config ──

const MAX_BYTES = Number(env.TEXTSIFT_PRECOMMIT_MAX_BYTES ?? 1_000_000); // 1 MB
const SEVERITY = (env.TEXTSIFT_PRECOMMIT_SEVERITY ?? "block").toLowerCase();
const ENABLE_SECRETS = (env.TEXTSIFT_PRECOMMIT_SECRETS ?? "1") !== "0";

// Extensions known to be generated / binary / not worth scanning.
// Pre-commit's `exclude` regex in `.pre-commit-hooks.yaml` is the
// first line of defence; this is the second so the hook script
// behaves sanely if invoked directly.
const SKIP_EXT = new Set([
  ".lock", ".lockfile",
  ".min.js", ".min.css", ".min.html",
  ".wasm", ".node", ".dylib", ".so", ".dll", ".a", ".lib", ".o",
  ".onnx", ".safetensors", ".gguf", ".pt", ".pth", ".bin", ".pkl",
  ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".svg",
  ".mp3", ".mp4", ".webm", ".mov", ".avi", ".wav", ".flac", ".ogg",
  ".pdf", ".zip", ".tar", ".gz", ".7z", ".rar",
  ".woff", ".woff2", ".ttf", ".otf", ".eot",
  ".class", ".jar", ".war",
]);

// ── Args ──

const args = argv.slice(2);
let warnOnly = false;
let allowList = false;
const files = [];
for (let i = 0; i < args.length; i++) {
  const a = args[i];
  if (a === "--warn-only") warnOnly = true;
  else if (a === "--severity") { /* env supersedes; advance past value */ i++; }
  else if (a === "--allow-list") allowList = true;
  else if (a === "--help" || a === "-h") {
    stdout.write([
      "textsift-pii-scan — pre-commit hook entry",
      "",
      "Usage: precommit.js [options] <file1> [file2 …]",
      "",
      "Options:",
      "  --warn-only          Always exit 0 even if PII found.",
      "  --allow-list         Print files that would be skipped (debug).",
      "",
      "Env vars:",
      "  TEXTSIFT_PRECOMMIT_MAX_BYTES   Skip files larger than N bytes (default 1000000).",
      "  TEXTSIFT_PRECOMMIT_SEVERITY    'block' (default), 'warn', or 'all'.",
      "  TEXTSIFT_PRECOMMIT_SECRETS     '0' to disable the secrets rule preset.",
      "  TEXTSIFT_OFFLINE / TEXTSIFT_MODEL_PATH / etc — same as the CLI loader flags.",
      "",
    ].join("\n"));
    exit(0);
  }
  else if (a.startsWith("-")) {
    stderr.write(`textsift-pii-scan: unknown flag "${a}". --help for usage.\n`);
    exit(2);
  } else {
    files.push(a);
  }
}

if (files.length === 0) {
  // pre-commit invoked with no files (e.g. --files filter excluded
  // everything). That's a clean no-op.
  exit(0);
}

// ── Filter staged files ──

const candidates = [];
const skipped = [];

for (const path of files) {
  const ext = extname(path).toLowerCase();
  if (SKIP_EXT.has(ext)) { skipped.push({ path, reason: `skip extension ${ext}` }); continue; }
  let st;
  try { st = await stat(path); }
  catch (e) { skipped.push({ path, reason: `stat: ${e.message}` }); continue; }
  if (!st.isFile()) { skipped.push({ path, reason: "not a regular file" }); continue; }
  if (st.size > MAX_BYTES) {
    skipped.push({ path, reason: `${(st.size / 1024 / 1024).toFixed(1)} MB > ${MAX_BYTES / 1024 / 1024} MB limit` });
    continue;
  }
  if (st.size === 0) { skipped.push({ path, reason: "empty" }); continue; }
  candidates.push({ path, bytes: st.size });
}

if (allowList) {
  for (const c of candidates) stdout.write(`SCAN ${c.path} (${c.bytes} bytes)\n`);
  for (const s of skipped)    stdout.write(`SKIP ${s.path}: ${s.reason}\n`);
  exit(0);
}

if (candidates.length === 0) {
  exit(0);
}

// ── Load model once + scan all candidates ──

const { PrivacyFilter } = await import("textsift");

let filter;
try {
  filter = await PrivacyFilter.create({
    backend: "auto",
    presets: ENABLE_SECRETS ? ["secrets"] : undefined,
    // CI defaults: respect TEXTSIFT_OFFLINE if set, never prompt.
    // First-run on a dev box: warn loudly so the user knows what's
    // about to download.
  });
} catch (err) {
  stderr.write(`textsift-pii-scan: failed to load PrivacyFilter: ${err.message}\n`);
  stderr.write("textsift-pii-scan: run `npx textsift download` first to pre-warm the model cache.\n");
  exit(1);
}

const findings = [];

function lineColOf(text, offset) {
  let line = 1, col = 1;
  for (let i = 0; i < offset && i < text.length; i++) {
    if (text[i] === "\n") { line++; col = 1; } else { col++; }
  }
  return { line, col };
}

function spanIsBlocking(span) {
  // Severity default: model spans always block; rule spans block
  // only when the rule's declared severity is "block" (the secrets
  // preset uses "block" universally).
  if (SEVERITY === "all") return true;
  if (SEVERITY === "warn") return span.severity === "block";
  // SEVERITY === "block" (default)
  if (span.source === "model") return true;
  return span.severity === "block";
}

for (const { path } of candidates) {
  const text = await readFile(path, "utf8");
  const result = await filter.detect(text);
  if (result.spans.length === 0) continue;

  const blockingSpans = result.spans.filter(spanIsBlocking);
  if (blockingSpans.length === 0) continue;

  findings.push({
    path,
    spans: blockingSpans.map((s) => ({
      label: s.label,
      text: s.text,
      ...lineColOf(text, s.start),
      severity: s.severity,
      source: s.source,
    })),
  });
}

filter.dispose();

// ── Report + exit ──

if (findings.length === 0) {
  exit(0);
}

const isTty = stderr.isTTY;
const RED = isTty ? "\x1b[31m" : "";
const YEL = isTty ? "\x1b[33m" : "";
const DIM = isTty ? "\x1b[2m"  : "";
const RST = isTty ? "\x1b[0m"  : "";

const total = findings.reduce((n, f) => n + f.spans.length, 0);
stderr.write(`\n${RED}textsift-pii-scan: ${total} blocking finding(s) across ${findings.length} file(s)${RST}\n\n`);

for (const f of findings) {
  const rel = relative(cwd(), f.path);
  for (const s of f.spans) {
    const tag = `${RED}${s.label}${RST}`;
    const where = `${DIM}${rel}:${s.line}:${s.col}${RST}`;
    const preview = s.text.length > 80 ? s.text.slice(0, 77) + "..." : s.text;
    stderr.write(`  ${tag}  ${where}  ${YEL}"${preview}"${RST}\n`);
  }
}

stderr.write([
  "",
  `${DIM}Bypass with \`git commit --no-verify\` (do not commit real PII).${RST}`,
  `${DIM}Adjust severity with TEXTSIFT_PRECOMMIT_SEVERITY=warn|all|block.${RST}`,
  "",
].join("\n"));

exit(warnOnly ? 0 : 1);
