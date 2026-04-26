// Verify the persistent buffer API: create a buffer, run a kernel
// that reads from it (without re-uploading per call), confirm the
// output is the same as the conformance fixture. This is the
// foundation the e2e forward orchestration uses — model weights
// upload once at warmup, then every forward references them by
// pointer.

import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { strict as assert } from "node:assert";

const HERE = dirname(fileURLToPath(import.meta.url));
const NATIVE_PATH = resolve(HERE, "../../packages/textsift/dist/textsift-native.node");
const FIXTURE = resolve(HERE, "conformance/fixtures/rms_norm");
const native = createRequire(import.meta.url)(NATIVE_PATH);

assert.equal(typeof native.createBuffer, "function");
assert.equal(typeof native.releaseBuffer, "function");
assert.equal(typeof native.dispatchByBuffers, "function");
assert.equal(typeof native.readBuffer, "function");

const meta = JSON.parse(readFileSync(resolve(FIXTURE, "meta.json"), "utf8"));
const uniform = new Uint8Array(readFileSync(resolve(FIXTURE, "uniform.bin")));
const x = new Uint8Array(readFileSync(resolve(FIXTURE, "x.bin")));
const gamma = new Uint8Array(readFileSync(resolve(FIXTURE, "gamma.bin")));
const expected = new Uint8Array(readFileSync(resolve(FIXTURE, "expected.bin")));

const backend = native.createBackend();

// Upload x and gamma to persistent buffers — they live across calls.
const xBuf = native.createBuffer(backend, x);
const gammaBuf = native.createBuffer(backend, gamma);

// Run rms_norm using buffer references (not bytes).
const output = native.dispatchByBuffers(
  backend,
  "rms_norm",
  uniform,
  [], // no extra uniforms
  [
    { binding: 1, bufPtr: xBuf, byteLen: x.byteLength },
    { binding: 2, bufPtr: gammaBuf, byteLen: gamma.byteLength },
  ],
  { binding: 3, byteLength: expected.byteLength },
  meta.dispatch,
);

assert.equal(output.byteLength, expected.byteLength);
let match = true;
for (let i = 0; i < expected.length; i++) {
  if (output[i] !== expected[i]) { match = false; break; }
}
assert.ok(match, "rms_norm output via persistent buffers must match fixture");

native.releaseBuffer(xBuf);
native.releaseBuffer(gammaBuf);
native.destroyBackend(backend);

console.log("OK: persistent buffer API works (rms_norm byte-equal vs fixture)");
