// End-to-end Dawn (npm `webgpu`) forward pass using synthetic weights.
//
// Same dispatch sequence as forward-metal.js / forward.js — different
// API. This is intended to validate the Linux/Windows ship path:
// Dawn's Tint WGSL→MSL/SPIR-V/HLSL codegen runs the same WGSL the
// browser uses.
//
// ⚠️  KNOWN ISSUE (as of session 2026-04-26):
// Heavy 130-dispatch-per-pass forwards reproducibly hang inside
// `await readBuf.mapAsync(...)`. Smoke tests (1–5 dispatches in
// tests/native/dawn/smoke-*.js) work fine. Tiny forwards (T=4 with
// MAX_LAYERS=1) work. Full forwards (T≥4 with MAX_LAYERS≥4) hang.
//
// Things that DID NOT fix the hang on macOS:
//  - `await new Promise(r => setImmediate(r))` between forwards
//  - `await new Promise(r => setTimeout(r, 0))` between forwards
//  - Polling `mapAsync` resolution via setImmediate/setTimeout
//  - `await dev.queue.onSubmittedWorkDone()` before mapAsync
//  - `createComputePipelineAsync` instead of sync createComputePipeline
//  - Bind group + uniform buffer caching across forwards
//  - Splitting per-layer into separate submits
//  - Cooldown delays between Node processes
//
// Hypothesis: this is either a Mac-specific bug in dawn.node's Metal
// path or a deadlock in the setImmediate-based event pump under heavy
// command-buffer load. Linux/Windows behavior unknown — the Linux
// agent should run this same file on real Linux GPU and report.
//
// Apples-to-apples vs forward-metal.js: same RNG-seeded synthetic
// weights, same dispatch shapes, same shader source.

import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { PF } from "../forward.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const SHADERS = resolve(HERE, "../../../packages/textsift/src/native/shaders");

const require = createRequire(import.meta.url);
const { create, globals } = require("webgpu");
Object.assign(globalThis, globals);

function makeRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}
function rngBytes(seed, n) {
  const out = new Uint8Array(n);
  const rng = makeRng(seed);
  for (let i = 0; i < n; i++) out[i] = Math.floor(rng() * 256);
  return out;
}
function dimsBuf(values) {
  const buf = new ArrayBuffer(Math.max(16, ((values.length * 4 + 15) >> 4) << 4));
  const v = new DataView(buf);
  for (let i = 0; i < values.length; i++) {
    if (values[i].t === "u32") v.setUint32(i * 4, values[i].v, true);
    else v.setFloat32(i * 4, values[i].v, true);
  }
  return new Uint8Array(buf);
}
const u32 = (v) => ({ t: "u32", v });
const f32v = (v) => ({ t: "f32", v });

const WGSL = Object.fromEntries(
  ["embed_lookup_int4","rms_norm","matmul_int4_fp16_f16","matmul_int4_f32_f32",
   "rope_apply","banded_attention","add_fp16","cast_fp16_to_f32",
   "cast_f32_to_fp16_scaled","router_topk","qmoe_gate_up","swiglu_clamp",
   "qmoe_down_scatter","add_rmsnorm_fp16_to_f32","zero_f32"]
    .map((n) => [n, readFileSync(resolve(SHADERS, `${n}.wgsl`), "utf8")]),
);

export class DawnForward {
  constructor() { this.cfg = { ...PF }; }

  async warmup() {
    const gpu = create([]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("no adapter");
    const info = adapter.info ?? {};
    this.device = await adapter.requestDevice({
      requiredFeatures: ["shader-f16"],
      requiredLimits: {
        maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 1024 * 1024 * 1024),
        maxBufferSize: Math.min(adapter.limits.maxBufferSize, 1024 * 1024 * 1024),
        maxStorageBuffersPerShaderStage: 10,
      },
    });
    this.device.lost?.then((info) => console.error("[dawn] device lost:", info));
    if (typeof this.device.addEventListener === "function") {
      this.device.addEventListener("uncapturederror", (ev) => {
        console.error("WebGPU error:", ev.error?.message);
      });
    }
    console.log(`[dawn] adapter: ${info.vendor ?? ""} ${info.architecture ?? ""} ${info.device ?? info.description ?? ""}`);

    // Compile pipelines async so Tint's WGSL→MSL translation happens
    // before the first forward(). Sync createComputePipeline can defer
    // shader compilation lazily, which then blocks on the first dispatch.
    this.pipelines = {};
    for (const [name, code] of Object.entries(WGSL)) {
      const m = this.device.createShaderModule({ label: `${name}.wgsl`, code });
      this.pipelines[name] = await this.device.createComputePipelineAsync({
        label: name, layout: "auto", compute: { module: m, entryPoint: "main" },
      });
      // Yield between pipeline compiles so Dawn's setImmediate-based
      // event pump can process compile events.
      await new Promise((r) => setImmediate(r));
    }
    console.log(`[dawn] all ${Object.keys(this.pipelines).length} pipelines compiled`);
    this._uploadWeights();
  }

  _addBuf(name, bytes, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) {
    const padded = (bytes.byteLength + 3) & ~3;
    const buf = this.device.createBuffer({ size: Math.max(16, padded), usage });
    if (padded === bytes.byteLength) {
      this.device.queue.writeBuffer(buf, 0, bytes);
    } else {
      const p = new Uint8Array(padded); p.set(bytes);
      this.device.queue.writeBuffer(buf, 0, p);
    }
    this.weights.set(name, { buf, size: padded });
  }

  _uploadWeights() {
    const cfg = this.cfg;
    const D = cfg.hiddenSize, Hq = cfg.numHeads, Hkv = cfg.numKvHeads, hd = cfg.headDim;
    const dff = cfg.intermediateSize, E = cfg.numExperts, V = cfg.vocabSize;
    const F16 = 2, F32 = 4, BLOCK = 32;
    let s = 100;
    this.weights = new Map();

    const embedBlocks = D / BLOCK;
    const embedZpPerRow = (embedBlocks + 1) >>> 1;
    this._addBuf("embed.int4", rngBytes(s++, V * D / 2));
    this._addBuf("embed.scales", rngBytes(s++, V * embedBlocks * F16));
    this._addBuf("embed.zp", rngBytes(s++, V * embedZpPerRow));

    for (let L = 0; L < cfg.numLayers; L++) {
      const tag = (k) => `layers.${L}.${k}`;
      this._addBuf(tag("input_layernorm"), rngBytes(s++, D * F16));
      this._addBuf(tag("post_attention_layernorm"), rngBytes(s++, D * F16));

      const projDims = {
        q_proj: { N: Hq * hd, K: D }, k_proj: { N: Hkv * hd, K: D },
        v_proj: { N: Hkv * hd, K: D }, o_proj: { N: D, K: Hq * hd },
      };
      for (const [proj, { N, K }] of Object.entries(projDims)) {
        const blocks = K / BLOCK;
        const zpPerRow = (blocks + 1) >>> 1;
        this._addBuf(tag(`attn.${proj}.int4`), rngBytes(s++, N * K / 2));
        this._addBuf(tag(`attn.${proj}.scales`), rngBytes(s++, N * blocks * F16));
        this._addBuf(tag(`attn.${proj}.zp`), rngBytes(s++, N * zpPerRow));
        this._addBuf(tag(`attn.${proj}.bias`), rngBytes(s++, N * F16));
      }

      this._addBuf(tag("attn.sinks"), rngBytes(s++, Hq * F32));

      const routerBlocks = D / BLOCK;
      const routerZp = (routerBlocks + 1) >>> 1;
      this._addBuf(tag("router.int4"), rngBytes(s++, E * D / 2));
      this._addBuf(tag("router.scales"), rngBytes(s++, E * routerBlocks * F16));
      this._addBuf(tag("router.zp"), rngBytes(s++, E * routerZp));
      this._addBuf(tag("router.bias"), rngBytes(s++, E * F16));

      const gateUpBlocks = D / BLOCK;
      const gateUpZp = (gateUpBlocks + 1) >>> 1;
      this._addBuf(tag("experts.gate_up.int4"), rngBytes(s++, E * (2 * dff) * D / 2));
      this._addBuf(tag("experts.gate_up.scales"), rngBytes(s++, E * (2 * dff) * gateUpBlocks * F16));
      const guzp = new Uint8Array(E * (2 * dff) * gateUpZp); guzp.fill(0x88);
      this._addBuf(tag("experts.gate_up.zp"), guzp);
      this._addBuf(tag("experts.gate_up.bias"), rngBytes(s++, E * (2 * dff) * F16));

      const downBlocks = dff / BLOCK;
      const downZp = (downBlocks + 1) >>> 1;
      this._addBuf(tag("experts.down.int4"), rngBytes(s++, E * D * dff / 2));
      this._addBuf(tag("experts.down.scales"), rngBytes(s++, E * D * downBlocks * F16));
      const dzp = new Uint8Array(E * D * downZp); dzp.fill(0x88);
      this._addBuf(tag("experts.down.zp"), dzp);
      this._addBuf(tag("experts.down.bias"), rngBytes(s++, E * D * F16));
    }

    this._addBuf("final_norm", rngBytes(s++, D * F16));
    const scoreBlocks = D / BLOCK;
    const scoreZp = (scoreBlocks + 1) >>> 1;
    this._addBuf("score.int4", rngBytes(s++, cfg.numClasses * D / 2));
    this._addBuf("score.scales", rngBytes(s++, cfg.numClasses * scoreBlocks * F16));
    this._addBuf("score.zp", rngBytes(s++, cfg.numClasses * scoreZp));
    this._addBuf("score.bias", new Uint8Array(cfg.numClasses * F16));

    let total = 0;
    for (const v of this.weights.values()) total += v.size;
    console.log(`[DawnForward] weights uploaded: ${this.weights.size} tensors, ${(total / 1024 / 1024).toFixed(1)} MB`);
  }

  ensureScratch(maxT) {
    if (maxT <= (this.maxT ?? 0) && this.scratch) return;
    if (this.scratch) for (const b of Object.values(this.scratch)) b.buf.destroy?.();
    this.maxT = maxT;
    const cfg = this.cfg;
    const D = cfg.hiddenSize, Hq = cfg.numHeads, Hkv = cfg.numKvHeads, hd = cfg.headDim;
    const dff = cfg.intermediateSize, K = cfg.numExpertsPerTok;
    const E = cfg.numExperts, NC = cfg.numClasses;
    const F16 = 2, F32 = 4;
    const T = maxT;
    const empty = (bytes, usage) => ({
      buf: this.device.createBuffer({
        size: Math.max(16, (bytes + 3) & ~3),
        usage: usage ?? (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
      }),
      size: bytes,
    });
    const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const ST_SRC = ST | GPUBufferUsage.COPY_SRC;
    this.scratch = {
      idsBuf: empty(T * 4, ST),
      maskBuf: empty(T, ST),
      cosBuf: empty(T * (hd / 2) * F16, ST),
      sinBuf: empty(T * (hd / 2) * F16, ST),
      h0: empty(T * D * F16, ST),
      h1: empty(T * D * F16, ST),
      normed1: empty(T * D * F16, ST),
      hiddenF32: empty(T * D * F32, ST),
      qBuf: empty(T * Hq * hd * F16, ST),
      kBuf: empty(T * Hkv * hd * F16, ST),
      vBuf: empty(T * Hkv * hd * F16, ST),
      attnOut: empty(T * Hq * hd * F16, ST),
      oOut: empty(T * D * F16, ST),
      routerLogits: empty(T * E * F32, ST),
      routingIdx: empty(T * K * 4, ST),
      routingScores: empty(T * K * F32, ST),
      acc: empty(T * D * F32, ST),
      gateUp: empty(T * K * (2 * dff) * F32, ST),
      glu: empty(T * K * dff * F32, ST),
      moeOut: empty(T * D * F16, ST),
      logitsOut: empty(T * NC * F32, ST_SRC),
    };
  }

  _ropeTables(T) {
    const cfg = this.cfg;
    const half = cfg.headDim / 2;
    const cos = new Float32Array(T * half);
    const sin = new Float32Array(T * half);
    const theta = 150000.0;
    for (let t = 0; t < T; t++) {
      for (let p = 0; p < half; p++) {
        const angle = t / Math.pow(theta, (2 * p) / cfg.headDim);
        cos[t * half + p] = Math.cos(angle);
        sin[t * half + p] = Math.sin(angle);
      }
    }
    return { cos, sin };
  }

  _packF16(arr) {
    const out = new Uint16Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      const buf = new Float32Array([v]);
      const u = new Uint32Array(buf.buffer)[0];
      const sign = (u >>> 16) & 0x8000;
      const exp = (u >>> 23) & 0xff;
      const mant = u & 0x7fffff;
      if (exp === 0xff) { out[i] = sign | 0x7c00 | (mant ? 0x200 : 0); continue; }
      if (exp === 0) { out[i] = sign; continue; }
      let e = exp - 127 + 15;
      if (e >= 0x1f) { out[i] = sign | 0x7c00; continue; }
      if (e <= 0) { out[i] = sign; continue; }
      const m = mant >>> 13;
      const r = (mant >>> 12) & 1;
      const sticky = (mant & 0xfff) !== 0;
      let h = sign | (e << 10) | m;
      if (r && (sticky || (m & 1))) h++;
      out[i] = h & 0xffff;
    }
    return new Uint8Array(out.buffer);
  }

  async forward(tokenIds, attentionMask) {
    const cfg = this.cfg;
    const T = tokenIds.length;
    if (T > (this.maxT ?? 0)) this.ensureScratch(T);
    const D = cfg.hiddenSize, Hq = cfg.numHeads, Hkv = cfg.numKvHeads, hd = cfg.headDim;
    const dff = cfg.intermediateSize;
    const Kp = cfg.numExpertsPerTok;
    const E = cfg.numExperts;
    const F16 = 2, F32 = 4;
    const s = this.scratch;
    const dev = this.device;
    const wbuf = (name) => this.weights.get(name).buf;

    const tokenBytes = new Uint8Array(tokenIds.buffer, tokenIds.byteOffset, T * 4);
    dev.queue.writeBuffer(s.idsBuf.buf, 0, tokenBytes);
    const maskPadded = new Uint8Array(Math.max(4, (T + 3) & ~3));
    maskPadded.set(attentionMask);
    dev.queue.writeBuffer(s.maskBuf.buf, 0, maskPadded);

    const { cos, sin } = this._ropeTables(T);
    const cosBytes = this._packF16(cos);
    const sinBytes = this._packF16(sin);
    const cosPad = new Uint8Array(Math.max(4, (cosBytes.byteLength + 3) & ~3)); cosPad.set(cosBytes);
    const sinPad = new Uint8Array(Math.max(4, (sinBytes.byteLength + 3) & ~3)); sinPad.set(sinBytes);
    dev.queue.writeBuffer(s.cosBuf.buf, 0, cosPad);
    dev.queue.writeBuffer(s.sinBuf.buf, 0, sinPad);

    let useMask = 0;
    for (let i = 0; i < T; i++) { if (attentionMask[i] !== 1) { useMask = 1; break; } }

    // Pre-zero the MoE accumulator each forward.
    const accInit = new Uint8Array(T * D * F32);
    dev.queue.writeBuffer(s.acc.buf, 0, accInit);

    this._tEncStart = performance.now();
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();

    const uniformBuffers = [];
    const SKIP = (process.env.SKIP ?? "").split(",").filter(Boolean);
    const dispatch = (name, uniform, bufBindings, grid) => {
      if (SKIP.includes(name)) return;
      const p = this.pipelines[name];
      const u = dev.createBuffer({
        size: Math.max(16, uniform.byteLength),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      dev.queue.writeBuffer(u, 0, uniform);
      uniformBuffers.push(u);
      const entries = [{ binding: 0, resource: { buffer: u } }];
      for (const b of bufBindings) entries.push({ binding: b.binding, resource: { buffer: b.buf } });
      const bg = dev.createBindGroup({ layout: p.getBindGroupLayout(0), entries });
      pass.setPipeline(p);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(grid[0], grid[1], grid[2]);
    };

    // Embed → h0
    dispatch(
      "embed_lookup_int4",
      dimsBuf([u32(T), u32(cfg.vocabSize), u32(D), u32(0)]),
      [
        { binding: 1, buf: wbuf("embed.int4") },
        { binding: 2, buf: wbuf("embed.scales") },
        { binding: 3, buf: wbuf("embed.zp") },
        { binding: 4, buf: s.idsBuf.buf },
        { binding: 5, buf: s.h0.buf },
      ],
      [Math.ceil(T * D / 64), 1, 1],
    );

    const maxL = parseInt(process.env.MAX_LAYERS ?? `${cfg.numLayers}`, 10);
    let hCur = s.h0.buf, hAlt = s.h1.buf;
    for (let L = 0; L < maxL; L++) {
      const tag = (k) => `layers.${L}.${k}`;

      dispatch(
        "rms_norm",
        dimsBuf([u32(T), u32(D), f32v(cfg.rmsNormEps), u32(0)]),
        [
          { binding: 1, buf: hCur },
          { binding: 2, buf: wbuf(tag("input_layernorm")) },
          { binding: 3, buf: s.normed1.buf },
        ],
        [T, 1, 1],
      );

      const matmul = (weightsBase, xPtr, yPtr, N, K) => {
        dispatch(
          "matmul_int4_fp16_f16",
          dimsBuf([u32(T), u32(N), u32(K), u32(0)]),
          [
            { binding: 1, buf: xPtr },
            { binding: 2, buf: wbuf(`${weightsBase}.int4`) },
            { binding: 3, buf: wbuf(`${weightsBase}.scales`) },
            { binding: 4, buf: wbuf(`${weightsBase}.zp`) },
            { binding: 5, buf: wbuf(`${weightsBase}.bias`) },
            { binding: 6, buf: yPtr },
          ],
          [Math.ceil(N / 64), Math.ceil(T / 4), 1],
        );
      };
      matmul(tag("attn.q_proj"), s.normed1.buf, s.qBuf.buf, Hq * hd, D);
      matmul(tag("attn.k_proj"), s.normed1.buf, s.kBuf.buf, Hkv * hd, D);
      matmul(tag("attn.v_proj"), s.normed1.buf, s.vBuf.buf, Hkv * hd, D);

      const rope = (qkPtr, heads) => {
        const half = hd / 2;
        dispatch(
          "rope_apply",
          dimsBuf([u32(T), u32(heads), u32(hd), u32(0)]),
          [
            { binding: 1, buf: qkPtr },
            { binding: 2, buf: s.cosBuf.buf },
            { binding: 3, buf: s.sinBuf.buf },
          ],
          [Math.ceil(T * heads * half / 64), 1, 1],
        );
      };
      rope(s.qBuf.buf, Hq);
      rope(s.kBuf.buf, Hkv);

      dispatch(
        "banded_attention",
        dimsBuf([
          u32(T), u32(Hq), u32(Hkv), u32(hd),
          u32(cfg.slidingWindow), u32(useMask), u32(0), u32(0),
        ]),
        [
          { binding: 1, buf: s.qBuf.buf },
          { binding: 2, buf: s.kBuf.buf },
          { binding: 3, buf: s.vBuf.buf },
          { binding: 4, buf: wbuf(tag("attn.sinks")) },
          { binding: 5, buf: s.maskBuf.buf },
          { binding: 6, buf: s.attnOut.buf },
        ],
        [T * Hq, 1, 1],
      );

      matmul(tag("attn.o_proj"), s.attnOut.buf, s.oOut.buf, D, Hq * hd);

      dispatch(
        "add_fp16",
        dimsBuf([u32(T * D), u32(0), u32(0), u32(0)]),
        [
          { binding: 1, buf: hCur },
          { binding: 2, buf: s.oOut.buf },
          { binding: 3, buf: s.normed1.buf },
        ],
        [Math.ceil(T * D / 64), 1, 1],
      );

      dispatch(
        "rms_norm",
        dimsBuf([u32(T), u32(D), f32v(cfg.rmsNormEps), u32(0)]),
        [
          { binding: 1, buf: s.normed1.buf },
          { binding: 2, buf: wbuf(tag("post_attention_layernorm")) },
          { binding: 3, buf: s.moeOut.buf },
        ],
        [T, 1, 1],
      );

      dispatch(
        "cast_fp16_to_f32",
        dimsBuf([u32(T * D), u32(0), u32(0), u32(0)]),
        [
          { binding: 1, buf: s.moeOut.buf },
          { binding: 2, buf: s.hiddenF32.buf },
        ],
        [Math.ceil(T * D / 64), 1, 1],
      );

      dispatch(
        "matmul_int4_f32_f32",
        dimsBuf([u32(T), u32(E), u32(D), u32(0)]),
        [
          { binding: 1, buf: s.hiddenF32.buf },
          { binding: 2, buf: wbuf(tag("router.int4")) },
          { binding: 3, buf: wbuf(tag("router.scales")) },
          { binding: 4, buf: wbuf(tag("router.zp")) },
          { binding: 5, buf: wbuf(tag("router.bias")) },
          { binding: 6, buf: s.routerLogits.buf },
        ],
        [Math.ceil(E / 64), Math.ceil(T / 4), 1],
      );

      dispatch(
        "router_topk",
        dimsBuf([u32(T), u32(E), u32(Kp), u32(0)]),
        [
          { binding: 1, buf: s.routerLogits.buf },
          { binding: 2, buf: s.routingIdx.buf },
          { binding: 3, buf: s.routingScores.buf },
        ],
        [Math.ceil(T / 64), 1, 1],
      );

      dispatch(
        "qmoe_gate_up",
        dimsBuf([
          u32(T), u32(Kp), u32(2 * dff), u32(D),
          u32(0), u32(0), u32(0), u32(0),
        ]),
        [
          { binding: 1, buf: s.hiddenF32.buf },
          { binding: 2, buf: s.routingIdx.buf },
          { binding: 3, buf: wbuf(tag("experts.gate_up.int4")) },
          { binding: 4, buf: wbuf(tag("experts.gate_up.scales")) },
          { binding: 5, buf: wbuf(tag("experts.gate_up.zp")) },
          { binding: 6, buf: wbuf(tag("experts.gate_up.bias")) },
          { binding: 7, buf: s.gateUp.buf },
        ],
        [T * Kp, Math.ceil((2 * dff) / 64), 1],
      );

      dispatch(
        "swiglu_clamp",
        dimsBuf([u32(T * Kp), u32(dff), u32(0), u32(0)]),
        [
          { binding: 1, buf: s.gateUp.buf },
          { binding: 2, buf: s.glu.buf },
        ],
        [Math.ceil(T * Kp * dff / 64), 1, 1],
      );

      dispatch(
        "qmoe_down_scatter",
        dimsBuf([
          u32(T), u32(Kp), u32(D), u32(dff),
          u32(0), u32(0), u32(0), u32(0),
        ]),
        [
          { binding: 1, buf: s.glu.buf },
          { binding: 2, buf: s.routingIdx.buf },
          { binding: 3, buf: s.routingScores.buf },
          { binding: 4, buf: wbuf(tag("experts.down.int4")) },
          { binding: 5, buf: wbuf(tag("experts.down.scales")) },
          { binding: 6, buf: wbuf(tag("experts.down.zp")) },
          { binding: 7, buf: wbuf(tag("experts.down.bias")) },
          { binding: 8, buf: s.acc.buf },
        ],
        [T * Kp, Math.ceil(D / 64), 1],
      );

      // cast_f32_to_fp16_scaled has TWO uniform bindings (dims + scale).
      {
        const p = this.pipelines["cast_f32_to_fp16_scaled"];
        const dimsU = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const scaleU = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        dev.queue.writeBuffer(dimsU, 0, dimsBuf([u32(T * D), u32(0), u32(0), u32(0)]));
        const scaleBuf = new ArrayBuffer(16);
        new Float32Array(scaleBuf, 0, 1).set([Kp]);
        dev.queue.writeBuffer(scaleU, 0, new Uint8Array(scaleBuf));
        uniformBuffers.push(dimsU, scaleU);
        const bg = dev.createBindGroup({
          layout: p.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: dimsU } },
            { binding: 1, resource: { buffer: scaleU } },
            { binding: 2, resource: { buffer: s.acc.buf } },
            { binding: 3, resource: { buffer: s.moeOut.buf } },
          ],
        });
        pass.setPipeline(p);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(T * D / 64), 1, 1);
      }

      dispatch(
        "add_fp16",
        dimsBuf([u32(T * D), u32(0), u32(0), u32(0)]),
        [
          { binding: 1, buf: s.normed1.buf },
          { binding: 2, buf: s.moeOut.buf },
          { binding: 3, buf: hAlt },
        ],
        [Math.ceil(T * D / 64), 1, 1],
      );

      const tmp = hCur; hCur = hAlt; hAlt = tmp;
    }

    dispatch(
      "rms_norm",
      dimsBuf([u32(T), u32(D), f32v(cfg.rmsNormEps), u32(0)]),
      [
        { binding: 1, buf: hCur },
        { binding: 2, buf: wbuf("final_norm") },
        { binding: 3, buf: s.normed1.buf },
      ],
      [T, 1, 1],
    );

    dispatch(
      "cast_fp16_to_f32",
      dimsBuf([u32(T * D), u32(0), u32(0), u32(0)]),
      [
        { binding: 1, buf: s.normed1.buf },
        { binding: 2, buf: s.hiddenF32.buf },
      ],
      [Math.ceil(T * D / 64), 1, 1],
    );

    dispatch(
      "matmul_int4_f32_f32",
      dimsBuf([u32(T), u32(cfg.numClasses), u32(D), u32(0)]),
      [
        { binding: 1, buf: s.hiddenF32.buf },
        { binding: 2, buf: wbuf("score.int4") },
        { binding: 3, buf: wbuf("score.scales") },
        { binding: 4, buf: wbuf("score.zp") },
        { binding: 5, buf: wbuf("score.bias") },
        { binding: 6, buf: s.logitsOut.buf },
      ],
      [Math.ceil(cfg.numClasses / 64), Math.ceil(T / 4), 1],
    );

    pass.end();
    if (process.env.DEBUG) console.log("[dawn] pass.end done");
    const readBuf = dev.createBuffer({
      size: T * cfg.numClasses * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    enc.copyBufferToBuffer(s.logitsOut.buf, 0, readBuf, 0, T * cfg.numClasses * 4);
    if (process.env.DEBUG) console.log("[dawn] copyBufferToBuffer done");

    const tEncEnd = performance.now();
    dev.queue.submit([enc.finish()]);
    // Pump Dawn's event loop while waiting for the map. setTimeout
    // gives the event loop a full tick to drain accumulated work.
    let mapped = false;
    const mapPromise = readBuf.mapAsync(GPUMapMode.READ).then(() => { mapped = true; });
    while (!mapped) {
      await new Promise((r) => setTimeout(r, 0));
    }
    await mapPromise;
    if (process.env.DEBUG) console.log("[dawn] mapped");
    const got = new Uint8Array(readBuf.getMappedRange().slice(0));
    readBuf.unmap();
    readBuf.destroy();
    for (const u of uniformBuffers) u.destroy();
    const tSubEnd = performance.now();

    if (process.env.PROFILE) {
      this._lastEncodeMs = tEncEnd - this._tEncStart;
      this._lastSubmitMs = tSubEnd - tEncEnd;
    }
    return new Float32Array(got.buffer.slice(0));
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const fwd = new DawnForward();
  await fwd.warmup();
  const T = parseInt(process.env.T ?? "32", 10);
  fwd.ensureScratch(T);
  console.log(`[DawnForward] scratch allocated for T=${T}`);
  const ids = new Int32Array(T);
  for (let i = 0; i < ids.length; i++) ids[i] = (i * 7919) % PF.vocabSize;
  const mask = new Uint8Array(T).fill(1);

  const ITERS = parseInt(process.env.ITERS ?? "10", 10);
  const WARMUP = parseInt(process.env.WARMUP ?? "3", 10);
  console.log(`[bench] WARMUP=${WARMUP} ITERS=${ITERS}`);
  // Dawn-Node pumps wgpuInstanceProcessEvents via setImmediate. A tight
  // await chain starves the immediate phase and `mapAsync` never resolves
  // after a few iters. Yielding to setTimeout between forwards lets
  // Dawn's event pump drain — setImmediate alone is sometimes not enough.
  const yieldToEventLoop = () => new Promise((r) => setTimeout(r, 0));

  for (let i = 0; i < WARMUP; i++) {
    await yieldToEventLoop();
    const t0 = performance.now();
    await fwd.forward(ids, mask);
    console.log(`  warmup ${i}: ${(performance.now() - t0).toFixed(1)} ms`);
  }

  const N = ITERS;
  const samples = [];
  const encs = [], subs = [];
  for (let i = 0; i < N; i++) {
    await yieldToEventLoop();
    const t0 = performance.now();
    const logits = await fwd.forward(ids, mask);
    samples.push(performance.now() - t0);
    console.log(`  iter ${i}: ${samples[i].toFixed(1)} ms`);
    if (process.env.PROFILE) {
      encs.push(fwd._lastEncodeMs);
      subs.push(fwd._lastSubmitMs);
    }
    if (i === 0) console.log(`[DawnForward] logits shape=(${T}, ${PF.numClasses}) len=${logits.length}`);
  }
  samples.sort((a, b) => a - b);
  console.log(
    `[DawnForward] forward latency T=${T}: ` +
      `median=${samples[Math.floor(N/2)].toFixed(1)}ms, ` +
      `min=${samples[0].toFixed(1)}ms, ` +
      `over ${N} iters`,
  );
  if (process.env.PROFILE) {
    encs.sort((a, b) => a - b);
    subs.sort((a, b) => a - b);
    console.log(
      `  phase breakdown: encode median=${encs[Math.floor(N/2)].toFixed(1)}ms, ` +
        `submit+readback median=${subs[Math.floor(N/2)].toFixed(1)}ms`,
    );
  }
  process.exit(0);
}
