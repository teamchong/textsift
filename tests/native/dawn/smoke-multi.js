// Multi-dispatch smoke: same Dawn instance, encode 5 simple dispatches
// in one pass, see if mapAsync still resolves. Isolates whether the
// hang is the volume of dispatches or specific kernels.

import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const SHADERS = resolve(HERE, "../../../packages/textsift/src/native/shaders");

const require = createRequire(import.meta.url);
const { create, globals } = require("webgpu");
Object.assign(globalThis, globals);

const gpu = create([]);
const adapter = await gpu.requestAdapter();
const dev = await adapter.requestDevice({ requiredFeatures: ["shader-f16"] });

const wgsl = readFileSync(resolve(SHADERS, "rms_norm.wgsl"), "utf8");
const m = dev.createShaderModule({ code: wgsl });
const p = dev.createComputePipeline({ layout: "auto", compute: { module: m, entryPoint: "main" } });

const T = 4, D = 128;
const u = new Uint8Array(16);
const dv = new DataView(u.buffer);
dv.setUint32(0, T, true); dv.setUint32(4, D, true);
new Float32Array(u.buffer, 8, 1)[0] = 1e-6;

const xBytes = new Uint8Array(T * D * 2); for (let i = 0; i < xBytes.length; i++) xBytes[i] = i & 0xff;
const gBytes = new Uint8Array(D * 2);     for (let i = 0; i < gBytes.length; i++) gBytes[i] = i & 0xff;

const xBuf = dev.createBuffer({ size: T * D * 2, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
dev.queue.writeBuffer(xBuf, 0, xBytes);
const gBuf = dev.createBuffer({ size: D * 2, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
dev.queue.writeBuffer(gBuf, 0, gBytes);
const yBuf = dev.createBuffer({ size: T * D * 2, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

const enc = dev.createCommandEncoder();
const pass = enc.beginComputePass();

// Run rms_norm 5 times in the same pass with fresh uniform buffers each time
const uniforms = [];
for (let i = 0; i < 5; i++) {
  const ub = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(ub, 0, u);
  uniforms.push(ub);
  const bg = dev.createBindGroup({
    layout: p.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: ub } },
      { binding: 1, resource: { buffer: xBuf } },
      { binding: 2, resource: { buffer: gBuf } },
      { binding: 3, resource: { buffer: yBuf } },
    ],
  });
  pass.setPipeline(p);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(T, 1, 1);
}
pass.end();

const readBuf = dev.createBuffer({ size: T * D * 2, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
enc.copyBufferToBuffer(yBuf, 0, readBuf, 0, T * D * 2);

dev.queue.submit([enc.finish()]);
console.log("submitted, mapping...");
await readBuf.mapAsync(GPUMapMode.READ);
console.log("mapped OK, byteLength=", readBuf.getMappedRange().byteLength);
readBuf.unmap();
process.exit(0);
