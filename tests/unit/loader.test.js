// Unit test for the Node-side loader's new options: cacheDir,
// modelPath, offline, and cache management. Doesn't hit the network
// — uses tmpdir + small synthetic byte payloads. Run via `bun`
// because we import .ts source directly.

import { strict as assert } from "node:assert";
import { mkdtemp, mkdir, rm, writeFile, readFile, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

const loader = await import(
  "../../packages/textsift/src/native/loader.ts"
);

let pass = 0, fail = 0;
function check(name, fn) {
  return Promise.resolve()
    .then(fn)
    .then(() => { console.log(`OK   ${name}`); pass++; })
    .catch((e) => { console.log(`FAIL ${name}: ${e.message}`); fail++; });
}

const root = await mkdtemp(resolve(tmpdir(), "textsift-loader-test-"));

// Persistent env state can affect later tests; snapshot + restore.
const envSnapshot = {
  TEXTSIFT_CACHE_DIR: process.env.TEXTSIFT_CACHE_DIR,
  TEXTSIFT_MODEL_PATH: process.env.TEXTSIFT_MODEL_PATH,
  TEXTSIFT_OFFLINE: process.env.TEXTSIFT_OFFLINE,
  XDG_CACHE_HOME: process.env.XDG_CACHE_HOME,
};
function restoreEnv() {
  for (const [k, v] of Object.entries(envSnapshot)) {
    if (v === undefined) delete process.env[k];
    else process.env[k] = v;
  }
}

await check("getCacheRoot: explicit cacheDir wins", () => {
  process.env.TEXTSIFT_CACHE_DIR = "/from/env";
  process.env.XDG_CACHE_HOME = "/from/xdg";
  const path = loader.getCacheRoot({ cacheDir: "/from/opts" });
  assert.equal(path, "/from/opts");
  restoreEnv();
});

await check("getCacheRoot: env var beats XDG", () => {
  process.env.TEXTSIFT_CACHE_DIR = "/from/env";
  process.env.XDG_CACHE_HOME = "/from/xdg";
  const path = loader.getCacheRoot({});
  assert.equal(path, "/from/env");
  restoreEnv();
});

await check("getCacheRoot: XDG when no opts/env override", () => {
  delete process.env.TEXTSIFT_CACHE_DIR;
  process.env.XDG_CACHE_HOME = "/from/xdg";
  const path = loader.getCacheRoot({});
  assert.equal(path, "/from/xdg/textsift");
  restoreEnv();
});

await check("getCacheRoot: ~/.cache/textsift fallback", () => {
  delete process.env.TEXTSIFT_CACHE_DIR;
  delete process.env.XDG_CACHE_HOME;
  const path = loader.getCacheRoot({});
  assert.match(path, /\.cache\/textsift$/);
  restoreEnv();
});

await check("modelPath option: reads local file, skips cache+fetch", async () => {
  const dir = resolve(root, "modelpath");
  await mkdir(dir, { recursive: true });
  const file = resolve(dir, "graph.onnx");
  const data = new Uint8Array([1, 2, 3, 4, 5]);
  await writeFile(file, data);

  const bytes = await loader.fetchOnnxGraph(
    "https://nonexistent.example.com/repo",
    { modelPath: file },
  );
  assert.deepEqual([...bytes], [1, 2, 3, 4, 5]);
});

await check("modelPath: companion .onnx_data read with _data suffix", async () => {
  const dir = resolve(root, "modelpath-data");
  await mkdir(dir, { recursive: true });
  const file = resolve(dir, "graph.onnx");
  await writeFile(file, new Uint8Array([1]));
  await writeFile(`${file}_data`, new Uint8Array([9, 9, 9]));

  const bytes = await loader.fetchOnnxExtData(
    "https://nonexistent.example.com/repo",
    { modelPath: file },
  );
  assert.deepEqual([...bytes], [9, 9, 9]);
});

await check("offline: cache hit still returns bytes", async () => {
  const cacheDir = resolve(root, "offline-hit");
  // Pre-stage a "cache" entry so cachedFetch's hit path fires.
  // The hash is internal so we have to use the real fetchOnnxGraph
  // path — easier: write directly under the expected sha-hashed dir.
  // Use a known modelSource to derive the sub-path predictably.
  const src = "https://example.com/test1/";
  // We know the loader does sha256(src).slice(0,16) and then puts
  // model_q4f16.onnx in there.
  const { createHash } = await import("node:crypto");
  const hash = createHash("sha256").update(src).digest("hex").slice(0, 16);
  const fileDir = resolve(cacheDir, hash);
  await mkdir(fileDir, { recursive: true });
  await writeFile(resolve(fileDir, "model_q4f16.onnx"), new Uint8Array([7, 7, 7]));

  const bytes = await loader.fetchOnnxGraph(src, { cacheDir, offline: true });
  assert.deepEqual([...bytes], [7, 7, 7]);
});

await check("offline: cache miss throws clear error", async () => {
  const cacheDir = resolve(root, "offline-miss");
  await assert.rejects(
    () => loader.fetchOnnxGraph("https://example.com/never-cached/", {
      cacheDir, offline: true,
    }),
    /TEXTSIFT_OFFLINE is set|cache miss/i,
  );
});

await check("TEXTSIFT_OFFLINE env var triggers offline mode", async () => {
  const cacheDir = resolve(root, "offline-env");
  process.env.TEXTSIFT_OFFLINE = "1";
  await assert.rejects(
    () => loader.fetchOnnxGraph("https://example.com/x/", { cacheDir }),
    /TEXTSIFT_OFFLINE/,
  );
  restoreEnv();
});

await check("TEXTSIFT_OFFLINE=0 does NOT trigger offline mode", async () => {
  const cacheDir = resolve(root, "offline-zero");
  process.env.TEXTSIFT_OFFLINE = "0";
  // Should attempt fetch and get a network error, not the "offline" error.
  await assert.rejects(
    () => loader.fetchOnnxGraph("https://nonexistent.invalid/x/", { cacheDir }),
    (err) => !/TEXTSIFT_OFFLINE/.test(err.message),
  );
  restoreEnv();
});

await check("getCacheInfo: empty cache dir → empty result", async () => {
  const cacheDir = resolve(root, "info-empty");
  const info = await loader.getCacheInfo({ cacheDir });
  assert.equal(info.entries.length, 0);
  assert.equal(info.totalBytes, 0);
});

await check("getCacheInfo: lists entries + totals", async () => {
  const cacheDir = resolve(root, "info-populated");
  await mkdir(resolve(cacheDir, "abc123"), { recursive: true });
  await mkdir(resolve(cacheDir, "def456"), { recursive: true });
  await writeFile(resolve(cacheDir, "abc123", "model_q4f16.onnx"), new Uint8Array(100));
  await writeFile(resolve(cacheDir, "abc123", "model_q4f16.onnx_data"), new Uint8Array(200));
  await writeFile(resolve(cacheDir, "def456", "model_q4f16.onnx"), new Uint8Array(50));

  const info = await loader.getCacheInfo({ cacheDir });
  assert.equal(info.entries.length, 2);
  assert.equal(info.totalBytes, 350);
  const abc = info.entries.find((e) => e.source === "abc123");
  assert.ok(abc);
  assert.equal(abc.totalBytes, 300);
  assert.equal(abc.files.length, 2);
});

await check("clearCache: removes everything + reports counts", async () => {
  const cacheDir = resolve(root, "clear-test");
  await mkdir(resolve(cacheDir, "abc123"), { recursive: true });
  await writeFile(resolve(cacheDir, "abc123", "model_q4f16.onnx"), new Uint8Array(100));
  await writeFile(resolve(cacheDir, "abc123", "model_q4f16.onnx_data"), new Uint8Array(200));

  const result = await loader.clearCache({ cacheDir });
  assert.equal(result.removed, 2);
  assert.equal(result.bytes, 300);

  // Cache dir should now not exist (clearCache rms it recursively).
  await assert.rejects(() => stat(cacheDir), { code: "ENOENT" });
});

await check("clearCache: empty cache → 0/0, no error", async () => {
  const cacheDir = resolve(root, "clear-empty");
  const result = await loader.clearCache({ cacheDir });
  assert.equal(result.removed, 0);
  assert.equal(result.bytes, 0);
});

// Cleanup
await rm(root, { recursive: true, force: true });

console.log(`\n${pass}/${pass + fail} passed`);
process.exit(fail === 0 ? 0 : 1);
