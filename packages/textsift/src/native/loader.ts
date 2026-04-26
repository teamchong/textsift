// Node-side ONNX bytes loader with on-disk cache.
//
// The browser path stores model_q4f16.onnx + .onnx_data in OPFS. On Node
// we cache to $XDG_CACHE_HOME/textsift/<sha-of-source>/{onnx,onnx_data}.
// First call downloads via fetch(); subsequent calls hit the disk cache.

import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { resolve } from "node:path";

function cacheRoot(): string {
  const xdg = process.env.XDG_CACHE_HOME;
  if (xdg && xdg.length > 0) return resolve(xdg, "textsift");
  return resolve(homedir(), ".cache", "textsift");
}

function shortHash(s: string): string {
  return createHash("sha256").update(s).digest("hex").slice(0, 16);
}

async function cachedFetch(url: string, cachePath: string): Promise<Uint8Array> {
  // Fast path: cache hit.
  try {
    const st = await stat(cachePath);
    if (st.isFile() && st.size > 0) {
      const buf = await readFile(cachePath);
      return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    }
  } catch (_e) {
    // miss → download
  }
  // Slow path: fetch + persist.
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`fetch ${url} → ${res.status} ${res.statusText}`);
  }
  const ab = await res.arrayBuffer();
  const bytes = new Uint8Array(ab);
  await mkdir(resolve(cachePath, ".."), { recursive: true });
  await writeFile(cachePath, bytes);
  return bytes;
}

/** Fetch the model_q4f16.onnx graph file, caching on disk. */
export async function fetchOnnxGraph(modelSource: string): Promise<Uint8Array> {
  const base = modelSource.endsWith("/") ? modelSource : `${modelSource}/`;
  const url = `${base}onnx/model_q4f16.onnx`;
  const cacheDir = resolve(cacheRoot(), shortHash(base));
  return cachedFetch(url, resolve(cacheDir, "model_q4f16.onnx"));
}

/** Fetch the external data sidecar, caching on disk. */
export async function fetchOnnxExtData(modelSource: string): Promise<Uint8Array> {
  const base = modelSource.endsWith("/") ? modelSource : `${modelSource}/`;
  const url = `${base}onnx/model_q4f16.onnx_data`;
  const cacheDir = resolve(cacheRoot(), shortHash(base));
  return cachedFetch(url, resolve(cacheDir, "model_q4f16.onnx_data"));
}

/** Fetch tokenizer.json, caching on disk. */
export async function fetchTokenizerJson(modelSource: string): Promise<string> {
  const base = modelSource.endsWith("/") ? modelSource : `${modelSource}/`;
  const url = `${base}tokenizer.json`;
  const cacheDir = resolve(cacheRoot(), shortHash(base));
  const bytes = await cachedFetch(url, resolve(cacheDir, "tokenizer.json"));
  return new TextDecoder("utf-8").decode(bytes);
}
