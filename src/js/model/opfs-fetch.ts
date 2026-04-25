/**
 * Persistent weight fetch backed by OPFS.
 *
 * Cache API caps out around ~500 MB per origin in practice (varies by
 * browser); the ~770 MB `.onnx_data` blob in `model_q4f16.onnx` blows
 * that quota consistently. OPFS (Origin Private File System) has looser
 * limits — Chromium advertises ~60% of free disk, Safari 1 GB — and is
 * the intended storage for large binary blobs in web apps.
 *
 * The same OPFS directory is shared by every backend in this package,
 * so a user who switches backends (wasm ↔ webgpu) doesn't retrigger a
 * download. Cache entry names are deterministic URL hashes so distinct
 * model sources (e.g. enterprise mirrors) don't collide.
 *
 * Failure modes — OPFS unavailable, quota exceeded, write error — all
 * fall through silently to plain `fetch`. A user with a constrained
 * storage environment gets correct behaviour, just without the
 * second-visit fast path.
 */

const OPFS_DIR = "textsift-v1";

function opfsKey(url: string): string {
  // FNV-1a on the URL + the trailing basename → human-readable + unique.
  let h = 2166136261 >>> 0;
  for (let i = 0; i < url.length; i++) {
    h = Math.imul(h ^ url.charCodeAt(i), 16777619) >>> 0;
  }
  const last = url.split("/").pop() || "blob";
  return `${last}.${h.toString(36)}`;
}

async function opfsRoot(): Promise<FileSystemDirectoryHandle | null> {
  if (typeof navigator === "undefined" || !navigator.storage?.getDirectory) {
    return null;
  }
  try {
    const root = await navigator.storage.getDirectory();
    return await root.getDirectoryHandle(OPFS_DIR, { create: true });
  } catch {
    return null;
  }
}

/**
 * Fetch `url` and return its bytes. On browsers with OPFS support,
 * caches the payload under a deterministic name so repeat visits skip
 * the network.
 */
export async function fetchBytesCached(url: string): Promise<ArrayBuffer> {
  const dir = await opfsRoot();
  if (dir) {
    const name = opfsKey(url);
    try {
      const handle = await dir.getFileHandle(name);
      const file = await handle.getFile();
      if (file.size > 0) {
        return await file.arrayBuffer();
      }
    } catch {
      // Entry doesn't exist — fall through to fetch.
    }
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`fetchBytesCached: fetch ${url} → ${resp.status} ${resp.statusText}`);
    }
    const bytes = await resp.arrayBuffer();
    try {
      const writeHandle = await dir.getFileHandle(name, { create: true });
      const writable = await writeHandle.createWritable();
      await writable.write(bytes);
      await writable.close();
    } catch {
      // Quota exceeded / write failure — serve the in-memory bytes and
      // let the next fetch re-download. Non-fatal.
    }
    return bytes;
  }
  const r = await fetch(url);
  if (!r.ok) {
    throw new Error(`fetchBytesCached: fetch ${url} → ${r.status} ${r.statusText}`);
  }
  return r.arrayBuffer();
}
