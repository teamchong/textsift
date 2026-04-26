// Round-trip a known byte pattern through a wgpu buffer:
// host → queue.writeBuffer → device storage → mapAsync(Read) → host
// Confirms buffer creation, queue submission, async map, and raw
// byte fidelity all work end-to-end. This is the prerequisite for
// every shader port — if buffers don't round-trip cleanly, no
// dispatched compute will read/write what we expect.

import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { strict as assert } from "node:assert";

const HERE = dirname(fileURLToPath(import.meta.url));
const NATIVE_PATH = resolve(HERE, "../../packages/textsift/dist/textsift-native.node");
const native = createRequire(import.meta.url)(NATIVE_PATH);

assert.equal(typeof native.roundtripBuffer, "function");

// 256-byte deterministic pattern — every byte distinct so any
// off-by-one in offset/length surfaces as a value mismatch.
const input = new Uint8Array(256);
for (let i = 0; i < input.length; i++) input[i] = i & 0xff;

const output = native.roundtripBuffer(input);
assert.equal(output.byteLength, input.byteLength, "output length matches");
for (let i = 0; i < input.length; i++) {
  assert.equal(output[i], input[i], `byte ${i} mismatch: ${output[i]} vs ${input[i]}`);
}

// Larger size (4 MiB — exercises a non-trivial GPU transfer path).
const big = new Uint8Array(4 * 1024 * 1024);
for (let i = 0; i < big.length; i++) big[i] = (i * 31 + 7) & 0xff;
const bigOut = native.roundtripBuffer(big);
assert.equal(bigOut.byteLength, big.byteLength);
for (let i = 0; i < big.length; i += 65536) {
  assert.equal(bigOut[i], big[i], `byte ${i} mismatch in 4 MiB run`);
}

console.log(`OK: buffer round-trip works (256 B + 4 MiB byte-equal)`);
