// Node-side ONNX bytes loader with on-disk cache.
//
// The browser path stores model_q4f16.onnx + .onnx_data in OPFS. On Node
// we cache to $XDG_CACHE_HOME/textsift/<sha-of-source>/{onnx,onnx_data}.
// First call downloads via fetch(); subsequent calls hit the disk cache.
//
// Override knobs (CLI / lib): `cacheDir` for an alternate cache root,
// `modelPath` to skip cache+fetch and read a pre-staged file, `offline`
// to fail loudly on cache miss instead of fetching. Env vars
// (TEXTSIFT_CACHE_DIR / TEXTSIFT_MODEL_PATH / TEXTSIFT_OFFLINE) supply
// defaults so the CLI can set them once and the lib auto-picks them up.

import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile, stat, readdir, rm } from "node:fs/promises";
import { homedir } from "node:os";
import { resolve } from "node:path";

/** Per-call options that override env-var defaults. */
export interface LoaderOptions {
  /**
   * Cache root override. Falls back to `$TEXTSIFT_CACHE_DIR`, then
   * `$XDG_CACHE_HOME/textsift`, then `~/.cache/textsift`.
   */
  cacheDir?: string;
  /**
   * Absolute path to a pre-staged `model_q4f16.onnx` (the graph file).
   * If set, skips both the cache and the network. The companion
   * `.onnx_data` file is expected at the same path with `_data`
   * appended (e.g. `/abs/model_q4f16.onnx` + `/abs/model_q4f16.onnx_data`).
   * Falls back to `$TEXTSIFT_MODEL_PATH`.
   */
  modelPath?: string;
  /**
   * If true, fail loudly on cache miss instead of fetching. For CI /
   * air-gapped use. Falls back to `$TEXTSIFT_OFFLINE` being set to a
   * truthy value.
   */
  offline?: boolean;
  /** Reported during long downloads. */
  onProgress?: (event: LoaderProgress) => void;
  /** Aborts a pending download. */
  signal?: AbortSignal;
}

export type LoaderProgress =
  | { stage: "cache-hit"; bytes: number; path: string }
  | { stage: "download-start"; url: string }
  | { stage: "download-progress"; loaded: number; total: number; url: string }
  | { stage: "download-done"; bytes: number; url: string };

function cacheRoot(opts: LoaderOptions): string {
  if (opts.cacheDir && opts.cacheDir.length > 0) return resolve(opts.cacheDir);
  const env = process.env.TEXTSIFT_CACHE_DIR;
  if (env && env.length > 0) return resolve(env);
  const xdg = process.env.XDG_CACHE_HOME;
  if (xdg && xdg.length > 0) return resolve(xdg, "textsift");
  return resolve(homedir(), ".cache", "textsift");
}

function shortHash(s: string): string {
  return createHash("sha256").update(s).digest("hex").slice(0, 16);
}

function envBool(name: string): boolean {
  const v = process.env[name];
  if (!v) return false;
  return v !== "" && v !== "0" && v.toLowerCase() !== "false";
}

function isOffline(opts: LoaderOptions): boolean {
  return opts.offline === true || envBool("TEXTSIFT_OFFLINE");
}

function envModelPath(opts: LoaderOptions): string | undefined {
  if (opts.modelPath) return opts.modelPath;
  const env = process.env.TEXTSIFT_MODEL_PATH;
  if (env && env.length > 0) return env;
  return undefined;
}

async function readFileBytes(path: string): Promise<Uint8Array> {
  const buf = await readFile(path);
  return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
}

async function cachedFetch(
  url: string,
  cachePath: string,
  opts: LoaderOptions,
): Promise<Uint8Array> {
  // Fast path: cache hit.
  try {
    const st = await stat(cachePath);
    if (st.isFile() && st.size > 0) {
      opts.onProgress?.({ stage: "cache-hit", bytes: st.size, path: cachePath });
      return readFileBytes(cachePath);
    }
  } catch (_e) {
    // miss → either fetch or fail per offline mode
  }
  if (isOffline(opts)) {
    throw new Error(
      `textsift: cache miss for ${cachePath} and TEXTSIFT_OFFLINE is set. ` +
        `Run \`npx textsift download\` first, or unset TEXTSIFT_OFFLINE.`,
    );
  }
  // Slow path: fetch + persist with progress reporting.
  opts.onProgress?.({ stage: "download-start", url });
  const res = await fetch(url, opts.signal ? { signal: opts.signal } : {});
  if (!res.ok) {
    throw new Error(`fetch ${url} → ${res.status} ${res.statusText}`);
  }

  // Stream the response so we can report progress. Falls back to a
  // single arrayBuffer() if the response has no body stream.
  const total = Number(res.headers.get("content-length") ?? 0);
  let bytes: Uint8Array;
  if (res.body && total > 0 && opts.onProgress) {
    const reader = res.body.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      opts.onProgress({ stage: "download-progress", loaded, total, url });
    }
    bytes = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
  } else {
    const ab = await res.arrayBuffer();
    bytes = new Uint8Array(ab);
  }

  await mkdir(resolve(cachePath, ".."), { recursive: true });
  await writeFile(cachePath, bytes);
  opts.onProgress?.({ stage: "download-done", bytes: bytes.byteLength, url });
  return bytes;
}

/** Fetch the model_q4f16.onnx graph file, caching on disk. */
export async function fetchOnnxGraph(
  modelSource: string,
  opts: LoaderOptions = {},
): Promise<Uint8Array> {
  const local = envModelPath(opts);
  if (local) return readFileBytes(local);
  const base = modelSource.endsWith("/") ? modelSource : `${modelSource}/`;
  const url = `${base}onnx/model_q4f16.onnx`;
  const cacheDir = resolve(cacheRoot(opts), shortHash(base));
  return cachedFetch(url, resolve(cacheDir, "model_q4f16.onnx"), opts);
}

/** Fetch the external data sidecar, caching on disk. */
export async function fetchOnnxExtData(
  modelSource: string,
  opts: LoaderOptions = {},
): Promise<Uint8Array> {
  const local = envModelPath(opts);
  if (local) {
    // Expect the sidecar at the same path with `_data` appended.
    return readFileBytes(`${local}_data`);
  }
  const base = modelSource.endsWith("/") ? modelSource : `${modelSource}/`;
  const url = `${base}onnx/model_q4f16.onnx_data`;
  const cacheDir = resolve(cacheRoot(opts), shortHash(base));
  return cachedFetch(url, resolve(cacheDir, "model_q4f16.onnx_data"), opts);
}

/** Fetch tokenizer.json, caching on disk. */
export async function fetchTokenizerJson(
  modelSource: string,
  opts: LoaderOptions = {},
): Promise<string> {
  const base = modelSource.endsWith("/") ? modelSource : `${modelSource}/`;
  const url = `${base}tokenizer.json`;
  const cacheDir = resolve(cacheRoot(opts), shortHash(base));
  const bytes = await cachedFetch(url, resolve(cacheDir, "tokenizer.json"), opts);
  return new TextDecoder("utf-8").decode(bytes);
}

// ── Cache management (used by the CLI's `cache info` / `cache clear`) ──

export interface CacheEntryInfo {
  /** Per-source subdirectory (the sha-of-URL hash). */
  source: string;
  /** Files within this subdirectory. */
  files: { name: string; bytes: number }[];
  totalBytes: number;
}

export interface CacheInfo {
  cacheDir: string;
  entries: CacheEntryInfo[];
  totalBytes: number;
}

/** Walk the cache directory and report its contents. */
export async function getCacheInfo(opts: LoaderOptions = {}): Promise<CacheInfo> {
  const root = cacheRoot(opts);
  const out: CacheInfo = { cacheDir: root, entries: [], totalBytes: 0 };
  let topEntries: string[];
  try {
    topEntries = await readdir(root);
  } catch {
    return out; // cache dir doesn't exist yet → empty
  }
  for (const source of topEntries) {
    const sourceDir = resolve(root, source);
    let files: string[];
    try {
      files = await readdir(sourceDir);
    } catch { continue; }
    const entry: CacheEntryInfo = { source, files: [], totalBytes: 0 };
    for (const name of files) {
      try {
        const st = await stat(resolve(sourceDir, name));
        if (!st.isFile()) continue;
        entry.files.push({ name, bytes: st.size });
        entry.totalBytes += st.size;
      } catch { /* skip */ }
    }
    out.entries.push(entry);
    out.totalBytes += entry.totalBytes;
  }
  return out;
}

/**
 * Wipe the entire textsift cache directory. Returns total bytes freed.
 * Safe to call on a non-existent cache (returns 0).
 */
export async function clearCache(opts: LoaderOptions = {}): Promise<{
  removed: number;
  bytes: number;
}> {
  const info = await getCacheInfo(opts);
  if (info.entries.length === 0) return { removed: 0, bytes: 0 };
  await rm(info.cacheDir, { recursive: true, force: true });
  let removed = 0;
  for (const e of info.entries) removed += e.files.length;
  return { removed, bytes: info.totalBytes };
}

/** The cache root path that would be used given the provided options. */
export function getCacheRoot(opts: LoaderOptions = {}): string {
  return cacheRoot(opts);
}
