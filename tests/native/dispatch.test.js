// First compute dispatch: a "double everything" kernel that proves
// WGSL compile + auto-layout pipeline + bind groups + dispatch +
// readback all work together. The next milestone is a real matmul
// with conformance against the browser path; this is the simplest
// kernel that exercises every link in that chain.

import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { strict as assert } from "node:assert";

const HERE = dirname(fileURLToPath(import.meta.url));
const NATIVE_PATH = resolve(HERE, "../../packages/textsift/dist/textsift-native.node");
const native = createRequire(import.meta.url)(NATIVE_PATH);

assert.equal(typeof native.dispatchDouble, "function");

const input = new Float32Array(1024);
for (let i = 0; i < input.length; i++) input[i] = i + 0.5;

const output = native.dispatchDouble(input);
assert.equal(output.length, input.length);

for (let i = 0; i < input.length; i++) {
  const expected = input[i] * 2;
  assert.equal(output[i], expected, `index ${i}: got ${output[i]} expected ${expected}`);
}

console.log(`OK: dispatchDouble correct over ${input.length} f32 elements`);
