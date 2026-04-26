// Verify the Zig→.node→wgpu-native pipeline end-to-end. Loads the
// compiled native module, calls getAdapterInfo(), confirms a real
// GPU adapter was found. If this passes, Node can talk to wgpu-native
// through our Zig bindings — and the WGSL kernel port can begin.

import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { strict as assert } from "node:assert";

const HERE = dirname(fileURLToPath(import.meta.url));
const NODE_PATH = resolve(HERE, "../../packages/textsift/dist/textsift-native.node");
const native = createRequire(import.meta.url)(NODE_PATH);

console.log("loaded native module, exports:", Object.keys(native));
assert.equal(typeof native.getAdapterInfo, "function", "expected getAdapterInfo export");

const info = native.getAdapterInfo();
console.log("adapter info:", info);
assert.equal(typeof info, "object", "getAdapterInfo() should return an object");
assert.equal(typeof info.vendor, "string");
assert.equal(typeof info.device, "string");
assert.ok(info.device.length > 0, "device name should be non-empty");

console.log("OK: Node ↔ wgpu-native via Zig works on this host");
