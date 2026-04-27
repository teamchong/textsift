// NodeBackend — implements browser/backends/abstract.ts's InferenceBackend
// against the metal*/vulkan*/dawn* NAPI surfaces in textsift-native.node.
//
// Architecture mirror of WebGpuBackend (browser/backends/webgpu.ts) — same
// weight name mapping, same dispatch sequence, same logits-out shape — but
// uses the platform-native fast path instead of navigator.gpu.
//
// Platform routing (matches the comptime gates in src/native/napi.zig):
//   darwin → metal*  (hand-written MSL via Obj-C bridge)
//   linux  → vulkan* (hand-written GLSL → SPIR-V via glslangValidator)
//   win32  → dawn*   (Tint → D3D12 via Dawn's backend selection)

import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import type { InferenceBackend, Logits, BackendConstructionOptions } from "../browser/backends/abstract.js";
import { parseOnnxGraph, resolveTensorBytes, type OnnxTensorRef } from "../browser/model/onnx-reader.js";
import { fetchOnnxGraph, fetchOnnxExtData } from "./loader.js";

// ── platform routing ────────────────────────────────────────────────

type Platform = "vulkan" | "dawn" | "metal";

function detectPlatform(): Platform {
  switch (process.platform) {
    case "darwin": return "metal";
    case "linux":  return "vulkan";
    case "win32":  return "dawn";
    default:
      throw new Error(`textsift native: ${process.platform} not supported`);
  }
}

interface NativeApi {
  // Lifecycle
  createBackend(): bigint;
  destroyBackend(handle: bigint): void;
  deviceName(handle: bigint): string;
  // Buffers
  createBuffer(handle: bigint, bytes: Uint8Array): bigint;
  createEmptyBuffer(handle: bigint, byteLen: number): bigint;
  releaseBuffer(handle: bigint, buf: bigint): void;
  writeBuffer(handle: bigint, buf: bigint, offset: number, bytes: Uint8Array): void;
  readBuffer(handle: bigint, buf: bigint, offset: number, byteLen: number): Uint8Array;
  // Encoder
  beginEncoder(handle: bigint): bigint;
  enqueueDispatch(
    encoder: bigint,
    name: string,
    bindings: bigint[],
    uniformData: Uint8Array,
    grid: [number, number, number],
  ): void;
  submitAndReadback(
    encoder: bigint,
    outBuf: bigint,
    offset: number,
    byteLen: number,
  ): Uint8Array;
}

function nativeApi(platform: Platform, native: any): NativeApi {
  switch (platform) {
    case "vulkan":
      return {
        createBackend:     () => native.vulkanCreateBackend(),
        destroyBackend:    (h) => native.vulkanDestroyBackend(h),
        deviceName:        (h) => native.vulkanDeviceName(h),
        createBuffer:      (h, b) => native.vulkanCreateBuffer(h, b),
        createEmptyBuffer: (h, n) => native.vulkanCreateEmptyBuffer(h, n),
        releaseBuffer:     (h, b) => native.vulkanReleaseBuffer(h, b),
        writeBuffer:       (h, b, o, by) => native.vulkanWriteBuffer(h, b, o, by),
        readBuffer:        (h, b, o, n) => native.vulkanReadBuffer(h, b, o, n),
        beginEncoder:      (h) => native.vulkanBeginEncoder(h),
        enqueueDispatch:   (e, n, bs, u, g) => native.vulkanEnqueueDispatch(e, n, bs, u, g),
        submitAndReadback: (e, b, o, n) => native.vulkanSubmitAndReadback(e, b, o, n),
      };
    case "dawn":
      return {
        createBackend:     () => native.dawnCreateBackend(),
        destroyBackend:    (h) => native.dawnDestroyBackend(h),
        deviceName:        (h) => native.dawnDeviceName(h),
        createBuffer:      (h, b) => native.dawnCreateBuffer(h, b),
        createEmptyBuffer: (h, n) => native.dawnCreateEmptyBuffer(h, n),
        releaseBuffer:     (h, b) => native.dawnReleaseBuffer(h, b),
        writeBuffer:       (h, b, o, by) => native.dawnWriteBuffer(h, b, o, by),
        readBuffer:        (h, b, o, n) => native.dawnReadBuffer(h, b, o, n),
        beginEncoder:      (h) => native.dawnBeginEncoder(h),
        enqueueDispatch:   (e, n, bs, u, g) => native.dawnEnqueueDispatch(e, n, bs, u, g),
        submitAndReadback: (e, b, o, n) => native.dawnSubmitAndReadback(e, b, o, n),
      };
    case "metal":
      // Metal's NAPI surface uses the {index, bytes/bufPtr} binding shape
      // from the original Mac port. Adapt to the unified bindings + uniform
      // shape this NodeBackend uses by injecting the uniform at index 0
      // and the storage buffers at indices 1..N+1.
      return {
        createBackend:     () => native.metalCreateBackend(),
        destroyBackend:    (h) => native.metalDestroyBackend(h),
        deviceName:        (h) => native.metalDeviceName(h),
        createBuffer:      (h, b) => native.metalCreateBuffer(h, b),
        createEmptyBuffer: (h, n) => native.metalCreateEmptyBuffer(h, n),
        releaseBuffer:     (_h, b) => native.metalReleaseBuffer(b),
        writeBuffer:       (_h, b, o, by) => native.metalWriteBuffer(b, o, by),
        readBuffer:        (_h, b, o, n) => native.metalReadBuffer(b, o, n),
        beginEncoder:      (h) => native.metalBeginEncoder(h),
        enqueueDispatch:   (e, name, bs, u, g) => {
          // `cast_f32_to_fp16_scaled` is the only kernel with two
          // uniform bindings (dims at [[buffer(0)]], scale at
          // [[buffer(1)]]). The unified API concatenates them into
          // one 32-byte blob (Vulkan/Dawn use push constants and
          // don't care). Split for Metal so each lands at the right
          // binding slot.
          const bindings: any[] = [];
          let storageStart = 1;
          if (name === "cast_f32_to_fp16_scaled" && u.byteLength === 32) {
            bindings.push({ index: 0, bytes: u.subarray(0, 16) });
            bindings.push({ index: 1, bytes: u.subarray(16, 32) });
            storageStart = 2;
          } else {
            bindings.push({ index: 0, bytes: u });
          }
          for (let i = 0; i < bs.length; i++) {
            bindings.push({ index: storageStart + i, bufPtr: bs[i] });
          }
          native.metalEnqueueDispatch(e, name, bindings, g, [64, 1, 1]);
        },
        submitAndReadback: (e, b, o, n) => native.metalSubmitAndReadback(e, b, o, n),
      };
  }
}

// ── ONNX → buffer name mapping (mirrors WebGpuBackend.uploadWeights) ──

/**
 * Given the parsed ONNX graph + external data, upload every required
 * tensor to a backend buffer with the same name our forward dispatch
 * sequence uses. Returns a Map keyed by buffer-name → handle.
 */
function uploadWeights(
  api: NativeApi,
  handle: bigint,
  graph: Map<string, OnnxTensorRef>,
  extData: Uint8Array,
): { buffers: Map<string, bigint>; numLayers: number } {
  const buffers = new Map<string, bigint>();
  const bytesOf = (name: string): Uint8Array => {
    const t = graph.get(name);
    if (!t) throw new Error(`uploadWeights: missing ONNX tensor "${name}"`);
    return resolveTensorBytes(t, extData);
  };
  const upload = (key: string, bytes: Uint8Array): void => {
    const padded = (bytes.byteLength + 3) & ~3;
    let buf: Uint8Array;
    if (padded === bytes.byteLength) {
      buf = bytes;
    } else {
      buf = new Uint8Array(padded);
      buf.set(bytes);
    }
    buffers.set(key, api.createBuffer(handle, buf));
  };

  // Embed table.
  upload("embed.int4",   bytesOf("model_embed_tokens_weight_quant"));
  upload("embed.scales", bytesOf("model_embed_tokens_weight_scales"));
  upload("embed.zp",     bytesOf("model_embed_tokens_weight_zp"));

  // Detect numLayers by probing for `model.layers.${L}.input_layernorm.weight`.
  let numLayers = 0;
  for (let L = 0; L < 64; L++) {
    if (graph.has(`model.layers.${L}.input_layernorm.weight`)) numLayers = L + 1;
  }
  if (numLayers === 0) throw new Error("uploadWeights: no transformer layers found in ONNX");

  // Final norm. ONNX names it under layers.${numLayers}.final_norm_layernorm.
  upload(
    "final_norm",
    bytesOf(`model.layers.${numLayers}.final_norm_layernorm.weight`),
  );

  // Classifier head. No score bias in ONNX; synthesize zero fp16 buffer.
  const scoreQuant = graph.get("model_score_MatMul_weight_quant");
  if (!scoreQuant) throw new Error("uploadWeights: missing score MatMul weight");
  upload("score.int4",   bytesOf("model_score_MatMul_weight_quant"));
  upload("score.scales", bytesOf("model_score_MatMul_weight_scales"));
  upload("score.zp",     bytesOf("model_score_MatMul_weight_zp"));
  const numClasses = scoreQuant.shape[0]!;
  upload("score.bias", new Uint8Array(numClasses * 2));

  // Per-layer weights.
  for (let L = 0; L < numLayers; L++) {
    for (const n of ["input_layernorm", "post_attention_layernorm"] as const) {
      upload(`layers.${L}.${n}`, bytesOf(`model.layers.${L}.${n}.weight`));
    }
    for (const proj of ["q_proj", "k_proj", "v_proj", "o_proj"] as const) {
      const qBase = `model_layers_${L}_attn_${proj}_MatMul`;
      const k = `layers.${L}.attn.${proj}`;
      upload(`${k}.int4`,   bytesOf(`${qBase}_weight_quant`));
      upload(`${k}.scales`, bytesOf(`${qBase}_weight_scales`));
      upload(`${k}.zp`,     bytesOf(`${qBase}_weight_zp`));
      upload(`${k}.bias`,   bytesOf(`model.layers.${L}.attn.${proj}.Add.bias`));
    }
    upload(`layers.${L}.attn.sinks`, bytesOf(`model.layers.${L}.attn.sinks`));
    {
      const rBase = `/model/layers_${L}/moe/router/MatMul`;
      const k = `layers.${L}.router`;
      upload(`${k}.int4`,   bytesOf(`${rBase}_weight_fp32_quant`));
      // Router scales are f32 in ONNX — shader expects f16, down-convert.
      upload(`${k}.scales`, f32ToFp16Bytes(bytesOf(`${rBase}_weight_fp32_scales`)));
      upload(`${k}.zp`,     bytesOf(`${rBase}_weight_fp32_zp`));
      upload(`${k}.bias`,   f32ToFp16Bytes(bytesOf(`/model/layers.${L}/moe/router/Add.bias_fp32`)));
    }
    for (const [onnxProj, ourKey] of [
      ["gate_up_proj", "gate_up"] as const,
      ["down_proj",    "down"]    as const,
    ]) {
      const eBase = `model_layers_${L}_moe_experts_${onnxProj}`;
      const k = `layers.${L}.experts.${ourKey}`;
      const quantShape = graph.get(`${eBase}_weight_quant`)!.shape;
      const scalesShape = graph.get(`${eBase}_weight_scales`)!.shape;
      const nBlocks = scalesShape[2]!;
      const zpPerRow = (nBlocks + 1) >>> 1;
      const Ect = quantShape[0]!;
      const Nrows = quantShape[1]!;
      const zp = new Uint8Array(Ect * Nrows * zpPerRow);
      zp.fill(0x88);  // expert quantization always uses zp=8 (mid-point of 4-bit)
      upload(`${k}.int4`,   bytesOf(`${eBase}_weight_quant`));
      upload(`${k}.scales`, bytesOf(`${eBase}_weight_scales`));
      upload(`${k}.zp`,     zp);
      upload(`${k}.bias`,   bytesOf(`model.layers.${L}.moe.experts.${onnxProj}.bias`));
    }
  }

  return { buffers, numLayers };
}

/** f32 buffer → fp16 buffer, IEEE round-to-nearest-even with stuck-bit handling. */
function f32ToFp16Bytes(f32: Uint8Array): Uint8Array {
  const n = f32.byteLength / 4;
  const f32v = new Float32Array(f32.buffer, f32.byteOffset, n);
  const out = new Uint16Array(n);
  for (let i = 0; i < n; i++) {
    const v = f32v[i]!;
    const buf = new Float32Array([v]);
    const u = new Uint32Array(buf.buffer)[0]!;
    const sign = (u >>> 16) & 0x8000;
    const exp = (u >>> 23) & 0xff;
    const mant = u & 0x7fffff;
    if (exp === 0xff) { out[i] = sign | 0x7c00 | (mant ? 0x200 : 0); continue; }
    if (exp === 0)    { out[i] = sign; continue; }
    let e = exp - 127 + 15;
    if (e >= 0x1f) { out[i] = sign | 0x7c00; continue; }
    if (e <= 0)    { out[i] = sign; continue; }
    const m = mant >>> 13;
    const r = (mant >>> 12) & 1;
    const sticky = (mant & 0xfff) !== 0;
    let h = sign | (e << 10) | m;
    if (r && (sticky || (m & 1))) h++;
    out[i] = h & 0xffff;
  }
  return new Uint8Array(out.buffer);
}

// ── dispatch helpers ────────────────────────────────────────────────

function dimsBuf(values: ReadonlyArray<{ t: "u32" | "f32"; v: number }>): Uint8Array {
  const buf = new ArrayBuffer(Math.max(16, ((values.length * 4 + 15) >> 4) << 4));
  const v = new DataView(buf);
  for (let i = 0; i < values.length; i++) {
    if (values[i]!.t === "u32") v.setUint32(i * 4, values[i]!.v, true);
    else v.setFloat32(i * 4, values[i]!.v, true);
  }
  return new Uint8Array(buf);
}
const u32 = (v: number) => ({ t: "u32" as const, v });
const f32v = (v: number) => ({ t: "f32" as const, v });

function concatBytes(a: Uint8Array, b: Uint8Array): Uint8Array {
  const out = new Uint8Array(a.byteLength + b.byteLength);
  out.set(a, 0);
  out.set(b, a.byteLength);
  return out;
}

function packF16(arr: Float32Array): Uint8Array {
  const out = new Uint16Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i]!;
    const buf = new Float32Array([v]);
    const u = new Uint32Array(buf.buffer)[0]!;
    const sign = (u >>> 16) & 0x8000;
    const exp = (u >>> 23) & 0xff;
    const mant = u & 0x7fffff;
    if (exp === 0xff) { out[i] = sign | 0x7c00 | (mant ? 0x200 : 0); continue; }
    if (exp === 0)    { out[i] = sign; continue; }
    let e = exp - 127 + 15;
    if (e >= 0x1f) { out[i] = sign | 0x7c00; continue; }
    if (e <= 0)    { out[i] = sign; continue; }
    const m = mant >>> 13;
    const r = (mant >>> 12) & 1;
    const sticky = (mant & 0xfff) !== 0;
    let h = sign | (e << 10) | m;
    if (r && (sticky || (m & 1))) h++;
    out[i] = h & 0xffff;
  }
  return new Uint8Array(out.buffer);
}

// ── NodeBackend ─────────────────────────────────────────────────────

const HERE = dirname(fileURLToPath(import.meta.url));

/**
 * Resolve the .node binary. Production install:
 *   `@textsift/native-${platform}-${arch}/textsift-native.node`
 *   (one of the optionalDependencies; npm picks the right one at install).
 * Dev / monorepo:
 *   `<this dir>/textsift-native.node` (built by scripts/build-native.sh).
 */
function resolveNativePath(): string {
  const triple = `${process.platform}-${process.arch}`;
  const subpackage = `@textsift/native-${triple}/textsift-native.node`;
  try {
    return createRequire(import.meta.url).resolve(subpackage);
  } catch {
    return resolve(HERE, "./textsift-native.node");
  }
}
const NATIVE_PATH = resolveNativePath();

interface ScratchBuffers {
  idsBuf: bigint;
  maskBuf: bigint;
  cosBuf: bigint;
  sinBuf: bigint;
  h0: bigint;
  h1: bigint;
  normed1: bigint;
  hiddenF32: bigint;
  qBuf: bigint;
  kBuf: bigint;
  vBuf: bigint;
  attnOut: bigint;
  oOut: bigint;
  routerLogits: bigint;
  routingIdx: bigint;
  routingScores: bigint;
  acc: bigint;
  gateUp: bigint;
  glu: bigint;
  moeOut: bigint;
  logitsOut: bigint;
}

export class NodeBackend implements InferenceBackend {
  readonly name: string;
  private platform: Platform;
  private api: NativeApi;
  private handle: bigint = 0n;
  private weights: Map<string, bigint> | null = null;
  private numLayers = 0;
  private scratch: ScratchBuffers | null = null;
  private maxT = 0;
  private opts: BackendConstructionOptions;
  private cfg: {
    hiddenSize: number;
    numHeads: number;
    numKvHeads: number;
    headDim: number;
    slidingWindow: number;
    intermediateSize: number;
    numExpertsPerTok: number;
    numExperts: number;
    rmsNormEps: number;
    numClasses: number;
    vocabSize: number;
  };

  constructor(opts: BackendConstructionOptions) {
    this.opts = opts;
    this.platform = detectPlatform();
    this.name = `native-${this.platform}`;
    const native = createRequire(import.meta.url)(NATIVE_PATH);
    this.api = nativeApi(this.platform, native);
    // Production model dimensions — must match openai/privacy-filter config.
    this.cfg = {
      hiddenSize: 640,
      numHeads: 14,
      numKvHeads: 2,
      headDim: 64,
      slidingWindow: 128,
      intermediateSize: 640,
      numExpertsPerTok: 4,
      numExperts: 128,
      rmsNormEps: 1e-5,
      numClasses: 33,
      vocabSize: 200000,
    };
  }

  async warmup(): Promise<void> {
    if (this.handle !== 0n) return;  // idempotent — base PrivacyFilter calls this twice
    this.handle = this.api.createBackend();

    // Pull ONNX from model source (HTTP + on-disk cache) and parse.
    const loaderOpts = {
      cacheDir: this.opts.cacheDir,
      modelPath: this.opts.modelPath,
      offline: this.opts.offline,
    };
    const [graphBytes, extBytes] = await Promise.all([
      fetchOnnxGraph(this.opts.bundle.modelSource, loaderOpts),
      fetchOnnxExtData(this.opts.bundle.modelSource, loaderOpts),
    ]);
    const graph = parseOnnxGraph(graphBytes);
    const { buffers, numLayers } = uploadWeights(this.api, this.handle, graph, extBytes);
    this.weights = buffers;
    this.numLayers = numLayers;

    // Sanity: vocabSize/numExperts from real weights.
    const embedShape = graph.get("model_embed_tokens_weight_quant")!.shape;
    this.cfg.vocabSize = embedShape[0]!;
    const routerShape = graph.get("/model/layers_0/moe/router/MatMul_weight_fp32_quant")!.shape;
    this.cfg.numExperts = routerShape[0]!;
  }

  async forward(tokenIds: Int32Array, attentionMask: Uint8Array): Promise<Logits> {
    if (this.handle === 0n || !this.weights) {
      throw new Error("NodeBackend.forward: call warmup() first");
    }
    const T = tokenIds.length;
    if (T > this.maxT) this.ensureScratch(T);
    const cfg = this.cfg;
    const D = cfg.hiddenSize;
    const Hq = cfg.numHeads, Hkv = cfg.numKvHeads, hd = cfg.headDim;
    const dff = cfg.intermediateSize;
    const Kp = cfg.numExpertsPerTok;
    const E = cfg.numExperts;
    const F32 = 4;
    const s = this.scratch!;
    const wbuf = (name: string): bigint => {
      const b = this.weights!.get(name);
      if (b === undefined) throw new Error(`NodeBackend: missing weight buffer "${name}"`);
      return b;
    };
    const api = this.api;

    // Upload tokenIds, mask, cos/sin tables.
    const tokenBytes = new Uint8Array(tokenIds.buffer, tokenIds.byteOffset, T * 4);
    api.writeBuffer(this.handle, s.idsBuf, 0, tokenBytes);
    const maskPadded = new Uint8Array(Math.max(4, (T + 3) & ~3));
    maskPadded.set(attentionMask);
    api.writeBuffer(this.handle, s.maskBuf, 0, maskPadded);

    const half = hd / 2;
    const cos = new Float32Array(T * half);
    const sin = new Float32Array(T * half);
    const theta = 150000.0;
    for (let t = 0; t < T; t++) {
      for (let p = 0; p < half; p++) {
        const angle = t / Math.pow(theta, (2 * p) / hd);
        cos[t * half + p] = Math.cos(angle);
        sin[t * half + p] = Math.sin(angle);
      }
    }
    const cosBytes = packF16(cos);
    const sinBytes = packF16(sin);
    const cosPad = new Uint8Array(Math.max(4, (cosBytes.byteLength + 3) & ~3)); cosPad.set(cosBytes);
    const sinPad = new Uint8Array(Math.max(4, (sinBytes.byteLength + 3) & ~3)); sinPad.set(sinBytes);
    api.writeBuffer(this.handle, s.cosBuf, 0, cosPad);
    api.writeBuffer(this.handle, s.sinBuf, 0, sinPad);

    let useMask = 0;
    for (let i = 0; i < T; i++) { if (attentionMask[i] !== 1) { useMask = 1; break; } }

    // Pre-zero the MoE accumulator each forward (down_scatter atomic-adds).
    api.writeBuffer(this.handle, s.acc, 0, new Uint8Array(T * D * F32));

    const enc = api.beginEncoder(this.handle);

    const dispatch = (
      name: string,
      uniform: Uint8Array,
      bufBindings: bigint[],
      grid: [number, number, number],
    ): void => {
      api.enqueueDispatch(enc, name, bufBindings, uniform, grid);
    };

    // Embed → h0
    dispatch(
      "embed_lookup_int4",
      dimsBuf([u32(T), u32(cfg.vocabSize), u32(D), u32(0)]),
      [wbuf("embed.int4"), wbuf("embed.scales"), wbuf("embed.zp"), s.idsBuf, s.h0],
      [Math.ceil(T * D / 64), 1, 1],
    );

    let hCur = s.h0, hAlt = s.h1;
    for (let L = 0; L < this.numLayers; L++) {
      const tag = (k: string) => `layers.${L}.${k}`;

      dispatch(
        "rms_norm",
        dimsBuf([u32(T), u32(D), f32v(cfg.rmsNormEps), u32(0)]),
        [hCur, wbuf(tag("input_layernorm")), s.normed1],
        [T, 1, 1],
      );

      const matmul = (weightsBase: string, xPtr: bigint, yPtr: bigint, N: number, K: number) => {
        dispatch(
          "matmul_int4_fp16_f16",
          dimsBuf([u32(T), u32(N), u32(K), u32(0)]),
          [
            xPtr,
            wbuf(`${weightsBase}.int4`),
            wbuf(`${weightsBase}.scales`),
            wbuf(`${weightsBase}.zp`),
            wbuf(`${weightsBase}.bias`),
            yPtr,
          ],
          [Math.ceil(N / 64), Math.ceil(T / 4), 1],
        );
      };
      matmul(tag("attn.q_proj"), s.normed1, s.qBuf, Hq * hd, D);
      matmul(tag("attn.k_proj"), s.normed1, s.kBuf, Hkv * hd, D);
      matmul(tag("attn.v_proj"), s.normed1, s.vBuf, Hkv * hd, D);

      const rope = (qkPtr: bigint, heads: number) => {
        dispatch(
          "rope_apply",
          dimsBuf([u32(T), u32(heads), u32(hd), u32(0)]),
          [qkPtr, s.cosBuf, s.sinBuf],
          [Math.ceil(T * heads * (hd / 2) / 64), 1, 1],
        );
      };
      rope(s.qBuf, Hq);
      rope(s.kBuf, Hkv);

      dispatch(
        "banded_attention",
        dimsBuf([
          u32(T), u32(Hq), u32(Hkv), u32(hd),
          u32(cfg.slidingWindow), u32(useMask), u32(0), u32(0),
        ]),
        [s.qBuf, s.kBuf, s.vBuf, wbuf(tag("attn.sinks")), s.maskBuf, s.attnOut],
        [T * Hq, 1, 1],
      );

      matmul(tag("attn.o_proj"), s.attnOut, s.oOut, D, Hq * hd);

      dispatch(
        "add_fp16",
        dimsBuf([u32(T * D), u32(0), u32(0), u32(0)]),
        [hCur, s.oOut, s.normed1],
        [Math.ceil(T * D / 64), 1, 1],
      );

      dispatch(
        "rms_norm",
        dimsBuf([u32(T), u32(D), f32v(cfg.rmsNormEps), u32(0)]),
        [s.normed1, wbuf(tag("post_attention_layernorm")), s.moeOut],
        [T, 1, 1],
      );

      dispatch(
        "cast_fp16_to_f32",
        dimsBuf([u32(T * D), u32(0), u32(0), u32(0)]),
        [s.moeOut, s.hiddenF32],
        [Math.ceil(T * D / 64), 1, 1],
      );

      dispatch(
        "matmul_int4_f32_f32",
        dimsBuf([u32(T), u32(E), u32(D), u32(0)]),
        [
          s.hiddenF32,
          wbuf(tag("router.int4")),
          wbuf(tag("router.scales")),
          wbuf(tag("router.zp")),
          wbuf(tag("router.bias")),
          s.routerLogits,
        ],
        // matmul_int4_f32_f32 uses 1D gl_GlobalInvocationID.x for tn = t*N + n
        [Math.ceil(T * E / 64), 1, 1],
      );

      dispatch(
        "router_topk",
        dimsBuf([u32(T), u32(E), u32(Kp), u32(0)]),
        [s.routerLogits, s.routingIdx, s.routingScores],
        [Math.ceil(T / 64), 1, 1],
      );

      dispatch(
        "qmoe_gate_up",
        dimsBuf([
          u32(T), u32(Kp), u32(2 * dff), u32(D),
          u32(0), u32(0), u32(0), u32(0),
        ]),
        [
          s.hiddenF32,
          s.routingIdx,
          wbuf(tag("experts.gate_up.int4")),
          wbuf(tag("experts.gate_up.scales")),
          wbuf(tag("experts.gate_up.zp")),
          wbuf(tag("experts.gate_up.bias")),
          s.gateUp,
        ],
        [T * Kp, Math.ceil((2 * dff) / 64), 1],
      );

      dispatch(
        "swiglu_clamp",
        dimsBuf([u32(T * Kp), u32(dff), u32(0), u32(0)]),
        [s.gateUp, s.glu],
        [Math.ceil(T * Kp * dff / 64), 1, 1],
      );

      dispatch(
        "qmoe_down_scatter",
        dimsBuf([
          u32(T), u32(Kp), u32(D), u32(dff),
          u32(0), u32(0), u32(0), u32(0),
        ]),
        [
          s.glu,
          s.routingIdx,
          s.routingScores,
          wbuf(tag("experts.down.int4")),
          wbuf(tag("experts.down.scales")),
          wbuf(tag("experts.down.zp")),
          wbuf(tag("experts.down.bias")),
          s.acc,
        ],
        [T * Kp, Math.ceil(D / 64), 1],
      );

      const scaleBuf = (() => {
        const b = new ArrayBuffer(16);
        new Float32Array(b, 0, 1).set([Kp]);
        return new Uint8Array(b);
      })();
      dispatch(
        "cast_f32_to_fp16_scaled",
        concatBytes(dimsBuf([u32(T * D), u32(0), u32(0), u32(0)]), scaleBuf),
        [s.acc, s.moeOut],
        [Math.ceil(T * D / 64), 1, 1],
      );

      dispatch(
        "add_fp16",
        dimsBuf([u32(T * D), u32(0), u32(0), u32(0)]),
        [s.normed1, s.moeOut, hAlt],
        [Math.ceil(T * D / 64), 1, 1],
      );

      const tmp = hCur; hCur = hAlt; hAlt = tmp;
    }

    // Final norm + classifier.
    dispatch(
      "rms_norm",
      dimsBuf([u32(T), u32(D), f32v(cfg.rmsNormEps), u32(0)]),
      [hCur, wbuf("final_norm"), s.normed1],
      [T, 1, 1],
    );
    dispatch(
      "cast_fp16_to_f32",
      dimsBuf([u32(T * D), u32(0), u32(0), u32(0)]),
      [s.normed1, s.hiddenF32],
      [Math.ceil(T * D / 64), 1, 1],
    );
    dispatch(
      "matmul_int4_f32_f32",
      dimsBuf([u32(T), u32(cfg.numClasses), u32(D), u32(0)]),
      [
        s.hiddenF32,
        wbuf("score.int4"),
        wbuf("score.scales"),
        wbuf("score.zp"),
        wbuf("score.bias"),
        s.logitsOut,
      ],
      [Math.ceil(T * cfg.numClasses / 64), 1, 1],
    );

    const logitsBytes = api.submitAndReadback(enc, s.logitsOut, 0, T * cfg.numClasses * 4);
    return {
      data: new Float32Array(
        logitsBytes.buffer.slice(
          logitsBytes.byteOffset,
          logitsBytes.byteOffset + logitsBytes.byteLength,
        ),
      ),
      sequenceLength: T,
      numClasses: cfg.numClasses,
    };
  }

  dispose(): void {
    if (this.handle === 0n) return;
    if (this.weights) {
      for (const buf of this.weights.values()) this.api.releaseBuffer(this.handle, buf);
      this.weights = null;
    }
    if (this.scratch) {
      for (const buf of Object.values(this.scratch)) this.api.releaseBuffer(this.handle, buf);
      this.scratch = null;
    }
    this.api.destroyBackend(this.handle);
    this.handle = 0n;
  }

  private ensureScratch(maxT: number): void {
    if (maxT <= this.maxT && this.scratch) return;
    if (this.scratch) {
      for (const buf of Object.values(this.scratch)) this.api.releaseBuffer(this.handle, buf);
    }
    this.maxT = maxT;
    const cfg = this.cfg;
    const D = cfg.hiddenSize;
    const Hq = cfg.numHeads, Hkv = cfg.numKvHeads, hd = cfg.headDim;
    const dff = cfg.intermediateSize, K = cfg.numExpertsPerTok;
    const E = cfg.numExperts, NC = cfg.numClasses;
    const F16 = 2, F32 = 4;
    const T = maxT;
    const empty = (bytes: number): bigint =>
      this.api.createEmptyBuffer(this.handle, Math.max(16, (bytes + 3) & ~3));
    this.scratch = {
      idsBuf:        empty(T * 4),
      maskBuf:       empty(T),
      cosBuf:        empty(T * (hd / 2) * F16),
      sinBuf:        empty(T * (hd / 2) * F16),
      h0:            empty(T * D * F16),
      h1:            empty(T * D * F16),
      normed1:       empty(T * D * F16),
      hiddenF32:     empty(T * D * F32),
      qBuf:          empty(T * Hq * hd * F16),
      kBuf:          empty(T * Hkv * hd * F16),
      vBuf:          empty(T * Hkv * hd * F16),
      attnOut:       empty(T * Hq * hd * F16),
      oOut:          empty(T * D * F16),
      routerLogits:  empty(T * E * F32),
      routingIdx:    empty(T * K * 4),
      routingScores: empty(T * K * F32),
      acc:           empty(T * D * F32),
      gateUp:        empty(T * K * (2 * dff) * F32),
      glu:           empty(T * K * dff * F32),
      moeOut:        empty(T * D * F16),
      logitsOut:     empty(T * NC * F32),
    };
  }
}
