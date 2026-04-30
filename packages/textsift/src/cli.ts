#!/usr/bin/env node
/**
 * textsift CLI — `npx textsift <subcommand>`. Wraps the same
 * PrivacyFilter the lib exports, mapping subcommands and argv to
 * method calls. Reads stdin / file paths, writes stdout / in-place,
 * supports cache management, all via the existing optionalDependencies
 * native binding (Mac=Metal, Linux=Vulkan, Win=Dawn) with WASM fallback.
 */

import { readFile, writeFile } from "node:fs/promises";
import { readFileSync } from "node:fs";
import { createInterface } from "node:readline";
import { stdin, stdout, stderr, exit, argv, env } from "node:process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { PrivacyFilter, markerPresets } from "./index.js";
import { getCacheInfo, clearCache, getCacheRoot } from "./native/loader.js";
import type { CreateOptions, RedactTableMode } from "./browser/types.js";
import type { LoaderProgress } from "./native/loader.js";

// Read the live version from the umbrella package.json at runtime so
// `textsift --version` and the CLI banner stay in sync with what's
// actually installed. Hardcoding the string means a package.json bump
// without a CLI source edit silently leaves the wrong version on
// `--help` (caught during the v0.1.1 smoke test).
const VERSION: string = ((): string => {
  try {
    const here = dirname(fileURLToPath(import.meta.url));
    const pkg = JSON.parse(readFileSync(resolve(here, "../package.json"), "utf8"));
    return pkg.version;
  } catch {
    return "unknown";
  }
})();
const HELP = `textsift ${VERSION} — local-first PII detection / redaction

USAGE
  textsift <command> [args] [flags]

COMMANDS
  redact [file]              Redact PII from text. File or stdin → stdout.
  detect [file]              Detect PII; emit spans as JSON. File or stdin → stdout.
  table [file]               Redact a CSV. Per-column classification + per-cell redaction.
  classify [file]            Classify CSV columns (JSON output).
  download                   Pre-warm the cache; download model if missing.
  cache info                 Show cache location + size + entries.
  cache clear                Wipe the cache.

REDACT / DETECT FLAGS
  --in-place                 Write back to <file> instead of stdout.
  --secrets                  Enable the built-in "secrets" rule preset.
  --synth                    Faker mode: realistic fakes instead of [label] markers.
  --min-confidence <N>       Drop spans with confidence < N (0..1). Default: 0.
  --jsonl                    detect: emit one span per line (jq-friendly).
  --sarif                    detect: emit SARIF v2.1.0 (for GitHub Code Scanning, etc.).

TABLE / CLASSIFY FLAGS
  --header                   First row is column headers (default: data only).
  --mode <m>                 redact (default) | synth | drop_column.
  --sample-size <N>          Cells to sample per column for classify (default: 50).

LOADER FLAGS  (env-var equivalents in parens)
  --cache-dir <path>         Override cache root.    (TEXTSIFT_CACHE_DIR)
  --model <path>             Use a pre-staged ONNX file; skip cache+fetch.
                                                     (TEXTSIFT_MODEL_PATH)
  --model-source <url>       Override the default HF Hub URL.
                                                     (TEXTSIFT_MODEL_SOURCE)
  --offline                  Fail loudly on cache miss (CI / air-gapped).
                                                     (TEXTSIFT_OFFLINE)
  --no-prompt                Don't ask before downloading on first run.

EXAMPLES
  echo "Hi Alice, alice@example.com" | textsift redact
  textsift redact ./customer.txt --in-place
  textsift table ./customers.csv --header --mode synth > clean.csv
  textsift classify ./customers.csv --header
  textsift detect ./log.txt --jsonl | jq 'select(.label == "private_email")'
  cat secrets.txt | textsift redact --secrets
  TEXTSIFT_OFFLINE=1 textsift redact ./file.txt   # CI: fail if not pre-cached
  textsift download                                # pre-warm in CI before --offline runs
`;

interface ParsedArgs {
  cmd: string | null;
  sub: string | null;
  positional: string[];
  flags: Record<string, string | boolean>;
}

function parseArgs(args: string[]): ParsedArgs {
  const out: ParsedArgs = { cmd: null, sub: null, positional: [], flags: {} };
  if (args.length === 0) return out;
  out.cmd = args[0]!;
  let i = 1;
  // Two-word commands ("cache info", "cache clear").
  if (out.cmd === "cache" && args[1] && !args[1]!.startsWith("-")) {
    out.sub = args[1]!;
    i = 2;
  }
  while (i < args.length) {
    const a = args[i]!;
    if (a.startsWith("--")) {
      const eq = a.indexOf("=");
      if (eq !== -1) {
        out.flags[a.slice(2, eq)] = a.slice(eq + 1);
      } else {
        const key = a.slice(2);
        const next = args[i + 1];
        if (next && !next.startsWith("-")) {
          out.flags[key] = next;
          i++;
        } else {
          out.flags[key] = true;
        }
      }
    } else if (a === "-") {
      out.positional.push("-");
    } else if (a.startsWith("-") && a.length > 1) {
      // short flags collapsed for v0; only long flags are documented.
      stderr.write(`textsift: unknown short flag "${a}". Use --help.\n`);
      exit(2);
    } else {
      out.positional.push(a);
    }
    i++;
  }
  return out;
}

function flagBool(flags: Record<string, string | boolean>, name: string): boolean {
  return flags[name] === true || flags[name] === "true";
}
function flagStr(flags: Record<string, string | boolean>, name: string): string | undefined {
  const v = flags[name];
  if (typeof v === "string") return v;
  return undefined;
}

async function readInput(positional: string[]): Promise<{ text: string; path: string | null }> {
  const arg = positional[0];
  if (arg && arg !== "-") {
    return { text: await readFile(arg, "utf8"), path: arg };
  }
  // Read all of stdin.
  const chunks: Buffer[] = [];
  for await (const chunk of stdin) chunks.push(chunk as Buffer);
  return { text: Buffer.concat(chunks).toString("utf8"), path: null };
}

function loaderOptsFromFlags(flags: Record<string, string | boolean>): Partial<CreateOptions> {
  const opts: Partial<CreateOptions> = {};
  const cacheDir = flagStr(flags, "cache-dir");
  if (cacheDir) opts.cacheDir = cacheDir;
  const modelPath = flagStr(flags, "model");
  if (modelPath) opts.modelPath = modelPath;
  if (flagBool(flags, "offline")) opts.offline = true;
  const modelSource = flagStr(flags, "model-source") ?? env.TEXTSIFT_MODEL_SOURCE;
  if (modelSource) opts.modelSource = modelSource;
  return opts;
}

function presetsFromFlags(flags: Record<string, string | boolean>): string[] | undefined {
  return flagBool(flags, "secrets") ? ["secrets"] : undefined;
}

function minConfidenceFromFlags(flags: Record<string, string | boolean>): number | undefined {
  const v = flagStr(flags, "min-confidence");
  if (v === undefined) return undefined;
  const n = Number(v);
  if (!Number.isFinite(n) || n < 0 || n > 1) {
    stderr.write(`textsift: --min-confidence must be a number between 0 and 1, got "${v}"\n`);
    exit(2);
  }
  return n;
}

function progressBar(loaded: number, total: number): string {
  const w = 30;
  const pct = total > 0 ? loaded / total : 0;
  const filled = Math.floor(pct * w);
  const bar = "█".repeat(filled) + "░".repeat(w - filled);
  const mb = (n: number) => (n / 1024 / 1024).toFixed(1);
  return `[${bar}] ${(pct * 100).toFixed(1)}% ${mb(loaded)}/${mb(total)} MB`;
}

async function confirmDownload(): Promise<boolean> {
  if (!stdin.isTTY) return false; // non-interactive: don't auto-download
  const rl = createInterface({ input: stdin, output: stderr });
  return new Promise((resolve) => {
    rl.question(
      "textsift: model not cached. Download ~770 MB from HuggingFace? [Y/n] ",
      (answer) => {
        rl.close();
        resolve(answer === "" || answer.trim().toLowerCase().startsWith("y"));
      },
    );
  });
}

/**
 * Build a PrivacyFilter wired with the CLI's loader overrides + a
 * progress reporter on stderr. Pulls `--no-prompt` from flags so
 * non-interactive callers (CI) can skip the TTY prompt entirely.
 */
async function buildFilter(
  flags: Record<string, string | boolean>,
  extra: Partial<CreateOptions> = {},
): Promise<PrivacyFilter> {
  const loaderOpts = loaderOptsFromFlags(flags);
  // First-run TTY prompt: only when not offline, no --model, no --no-prompt,
  // cache empty, and stdin is a TTY.
  if (!loaderOpts.offline && !loaderOpts.modelPath && !flagBool(flags, "no-prompt")) {
    const cacheInfo = await getCacheInfo({ cacheDir: loaderOpts.cacheDir });
    if (cacheInfo.entries.length === 0 && stdin.isTTY) {
      const ok = await confirmDownload();
      if (!ok) {
        stderr.write("textsift: cancelled.\n");
        exit(1);
      }
    }
  }
  // Render a simple progress bar during download. The lib already
  // surfaces these events via onProgress; we just route to stderr.
  let lastPct = -1;
  const onProgress = (e: ReturnType<typeof Object.create> & {
    stage?: string; loaded?: number; total?: number; bytes?: number;
  }): void => {
    const stage = e.stage as LoaderProgress["stage"] | undefined;
    if (stage === "download-start") stderr.write("downloading…\n");
    else if (stage === "download-progress" && e.total) {
      const pct = Math.floor((e.loaded ?? 0) / e.total * 100);
      if (pct !== lastPct) {
        stderr.write(`\r${progressBar(e.loaded ?? 0, e.total)}`);
        lastPct = pct;
      }
    } else if (stage === "download-done") {
      stderr.write(`\rdownload-done: ${(((e.bytes ?? 0)) / 1024 / 1024).toFixed(1)} MB\n`);
    }
  };
  return PrivacyFilter.create({
    backend: "auto",
    ...loaderOpts,
    ...extra,
    onProgress,
  });
}

// ── CSV parse / serialize (RFC 4180 minimal: quoted fields + escaped quotes) ──

function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; }
        else inQuotes = false;
      } else field += c;
    } else {
      if (c === '"') inQuotes = true;
      else if (c === ",") { row.push(field); field = ""; }
      else if (c === "\n") { row.push(field); rows.push(row); row = []; field = ""; }
      else if (c === "\r") { /* skip */ }
      else field += c;
    }
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

function serializeCsv(rows: string[][]): string {
  return rows.map((row) => row.map(quoteField).join(",")).join("\n") + "\n";
}

function quoteField(s: string): string {
  if (s.includes(",") || s.includes('"') || s.includes("\n")) {
    return `"${s.replace(/"/g, '""')}"`;
  }
  return s;
}

// ── Subcommand handlers ──

async function cmdRedact(args: ParsedArgs): Promise<void> {
  const { text, path } = await readInput(args.positional);
  const filter = await buildFilter(args.flags, {
    presets: presetsFromFlags(args.flags),
    markers: flagBool(args.flags, "synth") ? markerPresets.faker() : undefined,
    minConfidence: minConfidenceFromFlags(args.flags),
  });
  const result = await filter.redact(text);
  filter.dispose();
  if (flagBool(args.flags, "in-place") && path) {
    await writeFile(path, result.redactedText);
    stderr.write(`textsift: wrote ${result.spans.length} redactions to ${path}\n`);
  } else {
    stdout.write(result.redactedText);
  }
}

async function cmdDetect(args: ParsedArgs): Promise<void> {
  const { text, path } = await readInput(args.positional);
  const filter = await buildFilter(args.flags, {
    presets: presetsFromFlags(args.flags),
    minConfidence: minConfidenceFromFlags(args.flags),
  });
  const result = await filter.detect(text);
  filter.dispose();
  if (flagBool(args.flags, "sarif")) {
    const { detectResultToSarif } = await import("./sarif.js");
    stdout.write(JSON.stringify(detectResultToSarif(result, path ?? "<stdin>"), null, 2) + "\n");
  } else if (flagBool(args.flags, "jsonl")) {
    for (const s of result.spans) stdout.write(JSON.stringify(s) + "\n");
  } else {
    stdout.write(JSON.stringify({
      containsPii: result.containsPii,
      spans: result.spans,
      summary: result.summary,
    }, null, 2) + "\n");
  }
}

async function cmdTable(args: ParsedArgs): Promise<void> {
  const { text, path } = await readInput(args.positional);
  const rows = parseCsv(text);
  const filter = await buildFilter(args.flags, {
    presets: presetsFromFlags(args.flags),
    minConfidence: minConfidenceFromFlags(args.flags),
  });
  const mode = (flagStr(args.flags, "mode") ?? "redact") as RedactTableMode;
  const sampleSize = flagStr(args.flags, "sample-size");
  const out = await filter.redactTable(rows, {
    headerRow: flagBool(args.flags, "header"),
    mode,
    ...(sampleSize ? { sampleSize: Number(sampleSize) } : {}),
  });
  filter.dispose();
  const csv = serializeCsv(out);
  if (flagBool(args.flags, "in-place") && path) {
    await writeFile(path, csv);
    stderr.write(`textsift: wrote ${out.length} rows to ${path}\n`);
  } else {
    stdout.write(csv);
  }
}

async function cmdClassify(args: ParsedArgs): Promise<void> {
  const { text } = await readInput(args.positional);
  const rows = parseCsv(text);
  const filter = await buildFilter(args.flags, {
    minConfidence: minConfidenceFromFlags(args.flags),
  });
  const sampleSize = flagStr(args.flags, "sample-size");
  const cols = await filter.classifyColumns(rows, {
    headerRow: flagBool(args.flags, "header"),
    ...(sampleSize ? { sampleSize: Number(sampleSize) } : {}),
  });
  filter.dispose();
  stdout.write(JSON.stringify(cols, null, 2) + "\n");
}

async function cmdDownload(args: ParsedArgs): Promise<void> {
  // buildFilter triggers the warmup → if cache cold, downloads now.
  // Force prompts off so the user gets the bytes without questions.
  args.flags["no-prompt"] = true;
  const filter = await buildFilter(args.flags);
  filter.dispose();
  stderr.write("textsift: model ready in cache.\n");
}

async function cmdCacheInfo(args: ParsedArgs): Promise<void> {
  const cacheDir = flagStr(args.flags, "cache-dir");
  const info = await getCacheInfo(cacheDir ? { cacheDir } : {});
  const root = getCacheRoot(cacheDir ? { cacheDir } : {});
  const out = {
    cacheDir: root,
    totalBytes: info.totalBytes,
    totalMB: +(info.totalBytes / 1024 / 1024).toFixed(2),
    entries: info.entries.map((e) => ({
      source: e.source,
      totalBytes: e.totalBytes,
      totalMB: +(e.totalBytes / 1024 / 1024).toFixed(2),
      files: e.files,
    })),
  };
  stdout.write(JSON.stringify(out, null, 2) + "\n");
}

async function cmdCacheClear(args: ParsedArgs): Promise<void> {
  const cacheDir = flagStr(args.flags, "cache-dir");
  const result = await clearCache(cacheDir ? { cacheDir } : {});
  stderr.write(`textsift: removed ${result.removed} files (${(result.bytes / 1024 / 1024).toFixed(1)} MB)\n`);
}

// ── Main ──

async function main(): Promise<void> {
  const args = parseArgs(argv.slice(2));

  if (!args.cmd || flagBool(args.flags, "help") || args.cmd === "--help" || args.cmd === "-h") {
    stdout.write(HELP);
    exit(0);
  }
  if (flagBool(args.flags, "version") || args.cmd === "--version" || args.cmd === "-v") {
    stdout.write(`textsift ${VERSION}\n`);
    exit(0);
  }

  switch (args.cmd) {
    case "redact":   await cmdRedact(args);   break;
    case "detect":   await cmdDetect(args);   break;
    case "table":    await cmdTable(args);    break;
    case "classify": await cmdClassify(args); break;
    case "download": await cmdDownload(args); break;
    case "cache":
      if (args.sub === "info")  await cmdCacheInfo(args);
      else if (args.sub === "clear") await cmdCacheClear(args);
      else { stderr.write("textsift: cache requires `info` or `clear`. See --help.\n"); exit(2); }
      break;
    default:
      stderr.write(`textsift: unknown command "${args.cmd}". See --help.\n`);
      exit(2);
  }
}

main().catch((err: Error) => {
  stderr.write(`textsift: ${err.message}\n`);
  exit(1);
});
