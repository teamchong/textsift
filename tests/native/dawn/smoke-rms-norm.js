// Smoke test: dawn.node (npm `webgpu`) running our WGSL rms_norm kernel
// against the browser-dumped fixture. If this is byte-equal, the entire
// existing WebGpuBackend.forward() should run unmodified through Dawn
// in Node — Tint codegen, no Naga overhead.

import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURE = resolve(HERE, "../conformance/fixtures/rms_norm");
const SHADER_PATH = resolve(HERE, "../../../packages/textsift/src/native/shaders/rms_norm.wgsl");

const require = createRequire(import.meta.url);
const { create, globals } = require("webgpu");

Object.assign(globalThis, globals);
const gpu = create([]);

const adapter = await gpu.requestAdapter();
if (!adapter) throw new Error("no adapter");
const info = adapter.info ?? (await adapter.requestAdapterInfo?.());
console.log("[dawn] adapter:", info?.vendor, info?.architecture, info?.device || info?.description);

const features = ["shader-f16"];
const device = await adapter.requestDevice({ requiredFeatures: features });

const wgsl = readFileSync(SHADER_PATH, "utf8");
const module = device.createShaderModule({ code: wgsl });

const meta = JSON.parse(readFileSync(resolve(FIXTURE, "meta.json"), "utf8"));
const uniform = new Uint8Array(readFileSync(resolve(FIXTURE, "uniform.bin")));
const x = new Uint8Array(readFileSync(resolve(FIXTURE, "x.bin")));
const gamma = new Uint8Array(readFileSync(resolve(FIXTURE, "gamma.bin")));
const expected = new Uint8Array(readFileSync(resolve(FIXTURE, "expected.bin")));

const T = 4;

const uBuf = device.createBuffer({ size: uniform.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
device.queue.writeBuffer(uBuf, 0, uniform);

const xBuf = device.createBuffer({ size: x.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
device.queue.writeBuffer(xBuf, 0, x);

const gBuf = device.createBuffer({ size: gamma.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
device.queue.writeBuffer(gBuf, 0, gamma);

const yBuf = device.createBuffer({ size: expected.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

const pipeline = device.createComputePipeline({
  layout: "auto",
  compute: { module, entryPoint: "main" },
});

const bg = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: uBuf } },
    { binding: 1, resource: { buffer: xBuf } },
    { binding: 2, resource: { buffer: gBuf } },
    { binding: 3, resource: { buffer: yBuf } },
  ],
});

const enc = device.createCommandEncoder();
const pass = enc.beginComputePass();
pass.setPipeline(pipeline);
pass.setBindGroup(0, bg);
pass.dispatchWorkgroups(T, 1, 1);
pass.end();

const readBuf = device.createBuffer({ size: expected.byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
enc.copyBufferToBuffer(yBuf, 0, readBuf, 0, expected.byteLength);
device.queue.submit([enc.finish()]);

await readBuf.mapAsync(GPUMapMode.READ);
const got = new Uint8Array(readBuf.getMappedRange().slice(0));
readBuf.unmap();

let byteEqual = true;
for (let i = 0; i < expected.length; i++) if (got[i] !== expected[i]) { byteEqual = false; break; }

console.log(`[dawn/rms_norm] ${byteEqual ? "OK byte-equal" : "MISMATCH"} (${expected.byteLength} bytes)`);
if (!byteEqual) {
  // Print first diff
  for (let i = 0; i < expected.length; i++) {
    if (got[i] !== expected[i]) {
      console.log(`  first diff at byte ${i}: expected=${expected[i]} got=${got[i]}`);
      break;
    }
  }
  process.exit(1);
}
