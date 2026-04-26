// Microbench: Metal-direct rms_norm vs wgpu-native rms_norm.
// Both run the same shape (4 threadgroups × 64 threads, T=4 D=128).
// Measures pure dispatch + GPU compute + readback time per iter.

import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const NATIVE_PATH = resolve(HERE, "../../../packages/textsift/dist/textsift-native.node");
const FIXTURE = resolve(HERE, "../conformance/fixtures/rms_norm");
const native = createRequire(import.meta.url)(NATIVE_PATH);

const meta = JSON.parse(readFileSync(resolve(FIXTURE, "meta.json"), "utf8"));
const uniform = new Uint8Array(readFileSync(resolve(FIXTURE, "uniform.bin")));
const x = new Uint8Array(readFileSync(resolve(FIXTURE, "x.bin")));
const gamma = new Uint8Array(readFileSync(resolve(FIXTURE, "gamma.bin")));
const expected = new Uint8Array(readFileSync(resolve(FIXTURE, "expected.bin")));

const WARMUP = 10;
const ITERS = 100;

function median(xs) { const s = [...xs].sort((a,b)=>a-b); return s[Math.floor(s.length/2)]; }

// ── Metal-direct ──
{
  const b = native.metalCreateBackend();
  const xBuf = native.metalCreateBuffer(b, x);
  const gammaBuf = native.metalCreateBuffer(b, gamma);
  const yBuf = native.metalCreateBuffer(b, new Uint8Array(expected.byteLength));
  const T = 4;

  for (let i = 0; i < WARMUP; i++) {
    native.metalDispatchOneShot(
      b, "rms_norm",
      [
        { index: 0, bytes: uniform },
        { index: 1, bufPtr: xBuf },
        { index: 2, bufPtr: gammaBuf },
        { index: 3, bufPtr: yBuf },
      ],
      [T, 1, 1], [64, 1, 1],
    );
  }
  const samples = [];
  for (let i = 0; i < ITERS; i++) {
    const t0 = performance.now();
    native.metalDispatchOneShot(
      b, "rms_norm",
      [
        { index: 0, bytes: uniform },
        { index: 1, bufPtr: xBuf },
        { index: 2, bufPtr: gammaBuf },
        { index: 3, bufPtr: yBuf },
      ],
      [T, 1, 1], [64, 1, 1],
    );
    samples.push(performance.now() - t0);
  }
  const med = median(samples);
  console.log(`metal/rms_norm   median=${med.toFixed(3)}ms min=${Math.min(...samples).toFixed(3)}ms`);

  native.metalReleaseBuffer(xBuf);
  native.metalReleaseBuffer(gammaBuf);
  native.metalReleaseBuffer(yBuf);
  native.metalDestroyBackend(b);
}

// ── wgpu-native (Naga) ──
{
  const b = native.createBackend();
  const xBuf = native.createBuffer(b, x);
  const gammaBuf = native.createBuffer(b, gamma);
  const T = 4;
  const out = { binding: 3, byteLength: expected.byteLength };

  for (let i = 0; i < WARMUP; i++) {
    native.dispatchByBuffers(
      b, "rms_norm", uniform, [],
      [
        { binding: 1, bufPtr: xBuf, byteLen: x.byteLength },
        { binding: 2, bufPtr: gammaBuf, byteLen: gamma.byteLength },
      ],
      out, [T, 1, 1],
    );
  }
  const samples = [];
  for (let i = 0; i < ITERS; i++) {
    const t0 = performance.now();
    native.dispatchByBuffers(
      b, "rms_norm", uniform, [],
      [
        { binding: 1, bufPtr: xBuf, byteLen: x.byteLength },
        { binding: 2, bufPtr: gammaBuf, byteLen: gamma.byteLength },
      ],
      out, [T, 1, 1],
    );
    samples.push(performance.now() - t0);
  }
  const med = median(samples);
  console.log(`wgpu/rms_norm    median=${med.toFixed(3)}ms min=${Math.min(...samples).toFixed(3)}ms`);

  native.releaseBuffer(xBuf);
  native.releaseBuffer(gammaBuf);
  native.destroyBackend(b);
}
