/**
 * Stage 1 backend: custom Zig+WASM inference engine.
 *
 * This file is the JS side of the bridge; the Zig side lives at
 * `src/zig/wasm_exports.zig` and compiles to `dist/pii.wasm` via
 * `npm run build:zig`.
 *
 * Loads weights directly from the upstream ONNX graph shipped on HF Hub
 * (`onnx/model_q4f16.onnx` + `.onnx_data`). No custom weight format;
 * shares the same ~772 MB download with the transformers.js backend so
 * both backends hit the browser's HTTP cache identically.
 */

import type {
  BackendConstructionOptions,
  InferenceBackend,
  Logits,
} from "./abstract.js";
import { modelForward, type ModelConfig, type ModelWeights } from "../inference/model.js";
import { MtPool } from "../inference/mt-pool.js";
import type { MultiThreadContext } from "../inference/expert.js";
import type { WorkerScratch } from "../inference/mt-expert.js";
import type { BlockWeights } from "../inference/block.js";
import { parseOnnxGraph, resolveTensorBytes } from "../model/onnx-reader.js";
import { fetchBytesCached } from "../model/opfs-fetch.js";
import { PII_WASM_BYTES, PII_WASM_MT_BYTES } from "./pii-wasm-inline.js";

/** Exports declared in `src/zig/wasm_exports.zig`. Keep in sync. */
export interface PiiWasmExports {
  readonly memory: WebAssembly.Memory;
  /** Idempotent. Explicit call is optional; `alloc`/`reset`/`heap_mark_now` lazy-init. */
  heap_init(): number;
  alloc(n: number): number;
  heap_mark_now(): void;
  reset(): void;
  heap_permanent(): number;
  heap_used(): number;
  echo(x: number): number;
  sum_i32(ptr: number, len: number): number;
  rms_norm(x: number, gamma: number, out: number, T: number, D: number, eps: number): void;
  // int4-block matmul variants: `w_zp` is `0` for symmetric decode (no zero-points),
  // or a pointer into WASM memory for asymmetric (ONNX MatMulNBits semantics).
  matmul_fp16_x_int4block(x: number, w_int4: number, w_scales: number, w_zp: number, bias: number, out: number, T: number, N: number, D: number): void;
  matmul_fp16_x_int4block_out_f32(x: number, w_int4: number, w_scales: number, w_zp: number, bias: number, out: number, T: number, N: number, D: number): void;
  matmul_f32_x_int4block(x: number, w_int4: number, w_scales: number, w_zp: number, bias: number, out: number, T: number, N: number, D: number): void;
  matmul_f32_x_int4block_out_f32(x: number, w_int4: number, w_scales: number, w_zp: number, bias: number, out: number, T: number, N: number, D: number): void;
  topk_partial_f32(x: number, out_idx: number, out_val: number, rows: number, cols: number, k: number): void;
  rope_apply(qk: number, cos: number, sin: number, T: number, H: number, head_dim: number): void;
  banded_attention(
    q: number, k: number, v: number, sinks: number, mask: number, out: number,
    T: number, H_q: number, H_kv: number, head_dim: number, window: number,
  ): void;
  banded_attention_partial(
    q: number, k: number, v: number, sinks: number, mask: number, out: number,
    T: number, H_q: number, H_kv: number, head_dim: number, window: number,
    h_q_start: number, h_q_count: number,
  ): void;
  scale_fp16_inplace(x: number, scale: number, n: number): void;
  add_fp16(a: number, b: number, out: number, n: number): void;
  gather_fp16(src: number, indices: number, dst: number, m: number, D: number): void;
  gather_f32(src: number, indices: number, dst: number, m: number, D: number): void;
  scatter_add_weighted_f32(
    target: number, values: number, indices: number, weights: number,
    m: number, D: number,
  ): void;
  scatter_add_weighted_f32_scalar(
    target: number, values: number, indices: number, weights: number,
    m: number, D: number,
  ): void;
  zero_f32(ptr: number, n: number): void;
  convert_fp16_to_f32(src: number, dst: number, n: number): void;
  cast_f32_to_fp16_scaled(src: number, dst: number, n: number, scale: number): void;
  softmax_f32(x: number, out: number, rows: number, cols: number): void;
  swiglu_clamp_f32(gate_up: number, out: number, T: number, D: number): void;
  embed_lookup_int4(
    embed_int4: number, embed_scales: number, embed_zp: number,
    ids: number, out: number, T: number, V: number, D: number,
  ): void;
}

/**
 * Dtype tag on each pinned tensor. The Zig kernels read raw bytes — this
 * is pure JS-side bookkeeping so callers can sanity-check what they're
 * holding.
 */
export enum WeightDType {
  F32 = 0,
  F16 = 1,
  I8 = 2,
  U8 = 3,
  I32 = 4,
}

export interface WeightTensorInfo {
  readonly name: string;
  readonly dtype: WeightDType;
  readonly shape: readonly number[];
  /** Offset (in bytes) into `memory.buffer` where the tensor's data lives. */
  readonly dataOffset: number;
  /** Size in bytes. */
  readonly dataSize: number;
}

/**
 * int4-block32 weight bundle matching our matmul kernel's calling
 * convention: a packed int4 tensor, fp16 per-block scales, uint4 per-block
 * zero-points, and a fp16 bias. Used for every MatMulNBits-shaped weight
 * in the model (attention projections, router, classifier).
 */
export interface Int4BlockWeight {
  int4: WeightTensorInfo;
  scales: WeightTensorInfo;
  zp: WeightTensorInfo;
  bias: WeightTensorInfo;
}

/**
 * Detect whether SharedArrayBuffer is usable in the current
 * environment. Node has it unconditionally; browsers require the page
 * to be served with `Cross-Origin-Opener-Policy: same-origin` and
 * `Cross-Origin-Embedder-Policy: require-corp` (cross-origin
 * isolation) before SAB is exposed.
 */
export function sharedMemorySupported(): boolean {
  if (typeof SharedArrayBuffer === "undefined") return false;
  // crossOriginIsolated is undefined in Node and true/false in browsers.
  const coi = (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated;
  return coi !== false;
}

/**
 * Load `pii.wasm` and return its typed exports plus a readiness check.
 * If `url` is `null` or undefined, uses the inlined bytes baked into
 * the bundle (`PII_WASM_BYTES`) — this is the default and avoids the
 * extra HTTP request for a ~3 KB file.
 */
export async function loadPiiWasm(url?: string | URL | null): Promise<PiiWasmExports> {
  let instance: WebAssembly.Instance;
  if (url == null) {
    const result = await WebAssembly.instantiate(PII_WASM_BYTES as BufferSource, {});
    instance = result.instance;
  } else {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(
        `loadPiiWasm: fetch ${url} → ${response.status} ${response.statusText}`,
      );
    }
    const result = await WebAssembly.instantiateStreaming(response, {});
    instance = result.instance;
  }
  const exports = instance.exports as unknown as PiiWasmExports;

  const required: readonly (keyof PiiWasmExports)[] = [
    "memory",
    "alloc",
    "heap_mark_now",
    "reset",
    "heap_permanent",
    "heap_used",
    "echo",
    "sum_i32",
    "rms_norm",
    "matmul_fp16_x_int4block",
    "matmul_fp16_x_int4block_out_f32",
    "matmul_f32_x_int4block",
    "matmul_f32_x_int4block_out_f32",
    "topk_partial_f32",
    "rope_apply",
    "banded_attention",
    "banded_attention_partial",
    "scale_fp16_inplace",
    "add_fp16",
    "gather_fp16",
    "gather_f32",
    "scatter_add_weighted_f32",
    "scatter_add_weighted_f32_scalar",
    "zero_f32",
    "convert_fp16_to_f32",
    "cast_f32_to_fp16_scaled",
    "softmax_f32",
    "swiglu_clamp_f32",
    "embed_lookup_int4",
  ];
  for (const name of required) {
    if (!(name in exports)) {
      throw new Error(`loadPiiWasm: missing export "${String(name)}"`);
    }
  }

  return exports;
}

/**
 * Load `pii-mt.wasm` — same kernels as `pii.wasm` but built with the
 * WASM atomics + bulk_memory features and an imported shared memory.
 * The returned exports use `memory.buffer` as a SharedArrayBuffer, which
 * means the same `WebAssembly.Memory` can be passed to Worker threads
 * and they all see the same address space. Single-threaded callers can
 * use this build identically to `loadPiiWasm` — atomics are opt-in per
 * kernel call, not required for correctness.
 *
 * Throws if SAB isn't available in this environment (no
 * cross-origin-isolated context in browsers, etc.). Caller should
 * feature-check via `sharedMemorySupported()` first.
 */
export async function loadPiiWasmShared(
  sharedMemory?: WebAssembly.Memory,
): Promise<PiiWasmExports> {
  if (!sharedMemorySupported()) {
    throw new Error(
      "loadPiiWasmShared: SharedArrayBuffer is not available in this environment " +
        "(in browsers, the page must be served with " +
        "Cross-Origin-Opener-Policy: same-origin and " +
        "Cross-Origin-Embedder-Policy: require-corp)",
    );
  }
  // 64 pages = 4 MB initial; max 32768 pages = 2 GB (matches the WASM
  // build's --max-memory). Caller can pre-create a memory of any
  // larger initial size and pass it in (handy for sharing across
  // workers — main creates it once, workers receive the same handle).
  const memory = sharedMemory ?? new WebAssembly.Memory({
    initial: 64,
    maximum: 32768,
    shared: true,
  });
  if (!(memory.buffer instanceof SharedArrayBuffer)) {
    throw new Error("loadPiiWasmShared: provided memory is not SAB-backed");
  }
  const result = await WebAssembly.instantiate(
    PII_WASM_MT_BYTES as BufferSource,
    { env: { memory } },
  );
  // The mt-WASM imports memory rather than exporting it; expose the
  // imported memory via the same `exports.memory` shape so callers
  // can use the returned object identically to `loadPiiWasm`.
  const exports = {
    ...(result.instance.exports as object),
    memory,
  } as unknown as PiiWasmExports;
  return exports;
}

// ---------- dtype conversion ----------
//
// Weights in `model_q4f16.onnx` are a mix of: fp16 (biases, norms, scales),
// f32 (router scales, router bias, attention sinks), uint8 (packed int4
// quant + zp). Kernels consume fp16 throughout the activation path
// (matching ORT Web's q4f16 numerics) — fp16 weights pass through as-is;
// only f32 router scales/bias need a down-convert at load time.

const _cvBuf = new ArrayBuffer(4);
const _cvU32 = new Uint32Array(_cvBuf);
const _cvF32 = new Float32Array(_cvBuf);

function f32ToFp16Bits(f: number): number {
  if (f === 0) return 0;
  _cvF32[0] = f;
  const u32 = _cvU32[0]!;
  const sign = (u32 >>> 16) & 0x8000;
  const exp32 = (u32 >>> 23) & 0xff;
  const mant23 = u32 & 0x7fffff;
  if (exp32 === 0xff) {
    return (sign | 0x7c00 | (mant23 ? 0x200 : 0)) & 0xffff;
  }
  let exp16 = exp32 - 127 + 15;
  if (exp16 >= 0x1f) return (sign | 0x7c00) & 0xffff;
  if (exp16 <= 0) {
    if (exp16 < -10) return sign;
    const shift = 14 - exp16;
    const mant24 = mant23 | 0x800000;
    return (sign | ((mant24 + (1 << (shift - 1))) >>> shift)) & 0xffff;
  }
  // Normal, RNE
  const lsb = (mant23 >>> 13) & 1;
  let m10 = (mant23 + 0xfff + lsb) >>> 13;
  if (m10 >= 0x400) {
    m10 = 0;
    exp16 += 1;
    if (exp16 >= 0x1f) return (sign | 0x7c00) & 0xffff;
  }
  return (sign | (exp16 << 10) | m10) & 0xffff;
}

function f32BufferToFp16(src: Uint8Array): Uint8Array {
  const n = src.byteLength / 4;
  const dst = new Uint8Array(n * 2);
  const sv = new DataView(src.buffer, src.byteOffset, src.byteLength);
  const dv = new DataView(dst.buffer);
  for (let i = 0; i < n; i++) {
    _cvU32[0] = sv.getUint32(i * 4, true);
    dv.setUint16(i * 2, f32ToFp16Bits(_cvF32[0]!), true);
  }
  return dst;
}

// ---------- ONNX → WASM memory ----------

function pinBytes(
  wasm: PiiWasmExports,
  name: string,
  dtype: WeightDType,
  shape: readonly number[],
  bytes: Uint8Array,
): WeightTensorInfo {
  const ptr = wasm.alloc(bytes.byteLength);
  if (ptr === 0) {
    throw new Error(`loadOnnxWeights: WASM OOM allocating ${bytes.byteLength} bytes for "${name}"`);
  }
  new Uint8Array(wasm.memory.buffer, ptr, bytes.byteLength).set(bytes);
  return { name, dtype, shape: [...shape], dataOffset: ptr, dataSize: bytes.byteLength };
}

/**
 * Fetch `onnx/model_q4f16.onnx` + `.onnx_data` from `modelSource`, parse
 * the graph, preprocess each tensor into the dtype/layout our kernels
 * expect, and pin them all below the reset mark in WASM memory. Returns
 * a name → info map keyed by our canonical (dotted) names.
 */
export async function loadOnnxWeights(
  wasm: PiiWasmExports,
  modelSource: string,
): Promise<Map<string, WeightTensorInfo>> {
  const base = modelSource.endsWith("/") ? modelSource : `${modelSource}/`;
  const graphUrl = `${base}onnx/model_q4f16.onnx`;
  const dataUrl = `${base}onnx/model_q4f16.onnx_data`;

  const [graphBuf, dataBuf] = await Promise.all([
    fetchBytesCached(graphUrl),
    fetchBytesCached(dataUrl),
  ]);
  const graph = parseOnnxGraph(new Uint8Array(graphBuf));
  const extData = new Uint8Array(dataBuf);

  const bytesOf = (name: string): Uint8Array => {
    const t = graph.get(name);
    if (!t) throw new Error(`loadOnnxWeights: missing ONNX tensor "${name}"`);
    return resolveTensorBytes(t, extData);
  };
  const shapeOf = (name: string): readonly number[] => {
    const t = graph.get(name);
    if (!t) throw new Error(`loadOnnxWeights: missing ONNX tensor "${name}"`);
    return t.shape;
  };

  const out = new Map<string, WeightTensorInfo>();
  const put = (key: string, info: WeightTensorInfo): void => { out.set(key, info); };

  // Embed table — int4 + fp16 scales + uint4 zp, no preprocessing.
  put("embed.int4",   pinBytes(wasm, "embed.int4",   WeightDType.U8,  shapeOf("model_embed_tokens_weight_quant"),  bytesOf("model_embed_tokens_weight_quant")));
  put("embed.scales", pinBytes(wasm, "embed.scales", WeightDType.F16, shapeOf("model_embed_tokens_weight_scales"), bytesOf("model_embed_tokens_weight_scales")));
  put("embed.zp",     pinBytes(wasm, "embed.zp",     WeightDType.U8,  shapeOf("model_embed_tokens_weight_zp"),     bytesOf("model_embed_tokens_weight_zp")));

  // Final norm — fp16 [D], pass-through.
  {
    const name = "model.layers.8.final_norm_layernorm.weight";
    put("final_norm", pinBytes(wasm, "final_norm", WeightDType.F16, shapeOf(name), bytesOf(name)));
  }

  // Classifier — int4 + fp16 scales + uint4 zp. ONNX has no bias on
  // `score`; synthesise a zero fp16 [num_classes] so the kernel's bias
  // load doesn't dereference 0.
  {
    const quantShape = shapeOf("model_score_MatMul_weight_quant");
    put("score.int4",   pinBytes(wasm, "score.int4",   WeightDType.U8,  quantShape,                                      bytesOf("model_score_MatMul_weight_quant")));
    put("score.scales", pinBytes(wasm, "score.scales", WeightDType.F16, shapeOf("model_score_MatMul_weight_scales"),     bytesOf("model_score_MatMul_weight_scales")));
    put("score.zp",     pinBytes(wasm, "score.zp",     WeightDType.U8,  shapeOf("model_score_MatMul_weight_zp"),         bytesOf("model_score_MatMul_weight_zp")));
    const numClasses = quantShape[0]!;
    put("score.bias",   pinBytes(wasm, "score.bias",   WeightDType.F16, [numClasses], new Uint8Array(numClasses * 2)));
  }

  // Count transformer layers (layer 8 carries only the final norm).
  let numLayers = 0;
  for (let L = 0; L < 64; L++) {
    if (graph.has(`model.layers.${L}.input_layernorm.weight`)) numLayers = L + 1;
  }
  if (numLayers === 0) throw new Error("loadOnnxWeights: no transformer layers found");

  for (let L = 0; L < numLayers; L++) {
    // Layer norms (fp16 pass-through).
    for (const n of ["input_layernorm", "post_attention_layernorm"] as const) {
      const oname = `model.layers.${L}.${n}.weight`;
      put(`layers.${L}.${n}`, pinBytes(wasm, `layers.${L}.${n}`, WeightDType.F16, shapeOf(oname), bytesOf(oname)));
    }

    // Attention: Q/K/V/O — int4 + fp16 scales + uint4 zp + fp16 bias pass-through.
    for (const proj of ["q_proj", "k_proj", "v_proj", "o_proj"] as const) {
      const qBase = `model_layers_${L}_attn_${proj}_MatMul`;
      const k = `layers.${L}.attn.${proj}`;
      put(`${k}.int4`,   pinBytes(wasm, `${k}.int4`,   WeightDType.U8,  shapeOf(`${qBase}_weight_quant`),  bytesOf(`${qBase}_weight_quant`)));
      put(`${k}.scales`, pinBytes(wasm, `${k}.scales`, WeightDType.F16, shapeOf(`${qBase}_weight_scales`), bytesOf(`${qBase}_weight_scales`)));
      put(`${k}.zp`,     pinBytes(wasm, `${k}.zp`,     WeightDType.U8,  shapeOf(`${qBase}_weight_zp`),     bytesOf(`${qBase}_weight_zp`)));
      const biasName = `model.layers.${L}.attn.${proj}.Add.bias`;
      put(`${k}.bias`,   pinBytes(wasm, `${k}.bias`,   WeightDType.F16, shapeOf(biasName), bytesOf(biasName)));
    }

    // Attention sinks — stays as f32. ONNX stores [1, H_q, 1, 1]; flatten to [H_q].
    {
      const sName = `model.layers.${L}.attn.sinks`;
      const flatLen = shapeOf(sName).reduce((a, b) => a * b, 1);
      put(`layers.${L}.attn.sinks`, pinBytes(wasm, `layers.${L}.attn.sinks`, WeightDType.F32, [flatLen], bytesOf(sName)));
    }

    // Router — int4 + (f32→fp16) scales + uint4 zp + (f32→fp16) bias.
    {
      const rBase = `/model/layers_${L}/moe/router/MatMul`;
      const k = `layers.${L}.router`;
      put(`${k}.int4`,   pinBytes(wasm, `${k}.int4`,   WeightDType.U8,  shapeOf(`${rBase}_weight_fp32_quant`),                bytesOf(`${rBase}_weight_fp32_quant`)));
      put(`${k}.scales`, pinBytes(wasm, `${k}.scales`, WeightDType.F16, shapeOf(`${rBase}_weight_fp32_scales`),               f32BufferToFp16(bytesOf(`${rBase}_weight_fp32_scales`))));
      put(`${k}.zp`,     pinBytes(wasm, `${k}.zp`,     WeightDType.U8,  shapeOf(`${rBase}_weight_fp32_zp`),                   bytesOf(`${rBase}_weight_fp32_zp`)));
      const biasName = `/model/layers.${L}/moe/router/Add.bias_fp32`;
      put(`${k}.bias`,   pinBytes(wasm, `${k}.bias`,   WeightDType.F16, shapeOf(biasName),                                    f32BufferToFp16(bytesOf(biasName))));
    }

    // Experts — ONNX stores uint4 values centered on 8 (modal byte is
    // 0x88, i.e. nibbles (8,8) = zero in the dequant space). No zp
    // tensor ships with the graph; we synthesise one full of 0x88 so
    // the kernel computes `(q - 8) * scale` per block and recovers the
    // signed-centered weights.
    for (const [onnxProj, ourKey] of [
      ["gate_up_proj", "gate_up"] as const,
      ["down_proj",    "down"]    as const,
    ]) {
      const eBase = `model_layers_${L}_moe_experts_${onnxProj}`;
      const k = `layers.${L}.experts.${ourKey}`;
      const quantShape = shapeOf(`${eBase}_weight_quant`);           // [E, N, D/2]
      const scalesShape = shapeOf(`${eBase}_weight_scales`);         // [E, N, nBlocks]
      const nBlocks = scalesShape[2]!;
      const zpPerRow = (nBlocks + 1) >>> 1;
      const E = quantShape[0]!;
      const Nrows = quantShape[1]!;
      const zp = new Uint8Array(E * Nrows * zpPerRow);
      zp.fill(0x88);
      put(`${k}.int4`,   pinBytes(wasm, `${k}.int4`,   WeightDType.U8,  quantShape,                    bytesOf(`${eBase}_weight_quant`)));
      put(`${k}.scales`, pinBytes(wasm, `${k}.scales`, WeightDType.F16, scalesShape,                   bytesOf(`${eBase}_weight_scales`)));
      put(`${k}.zp`,     pinBytes(wasm, `${k}.zp`,     WeightDType.U8,  [E, Nrows, zpPerRow],          zp));
      const biasName = `model.layers.${L}.moe.experts.${onnxProj}.bias`;
      put(`${k}.bias`,   pinBytes(wasm, `${k}.bias`,   WeightDType.F16, shapeOf(biasName),             bytesOf(biasName)));
    }
  }

  // Everything past this mark is scratch; `reset()` will only rewind to here.
  wasm.heap_mark_now();
  return out;
}

const WARMUP_T = 16;

/**
 * Privacy-filter architecture constants baked into the Stage-1 backend.
 * Matches `openai/privacy-filter/config.json`. NUM_LAYERS and
 * numExperts/numExpertsInBlob are auto-detected from the loaded blob
 * so test fixtures with truncated layers/experts work through the same
 * code path as production.
 */
const PF_CONFIG: Omit<ModelConfig, "vocabSize" | "numExperts" | "numExpertsInBlob"> = {
  hiddenSize: 640,
  numHeads: 14,
  numKvHeads: 2,
  headDim: 64,
  slidingWindow: 128,
  intermediateSize: 640,
  numExpertsPerTok: 4,
  rmsNormEps: 1e-5,
  numClasses: 33,
  rope: {
    headDim: 64,
    theta: 150000.0,
    factor: 32.0,
    originalMaxPositionEmbeddings: 4096,
    betaFast: 32.0,
    betaSlow: 1.0,
    truncate: false,
  },
};

export interface WasmBackendOptions extends BackendConstructionOptions {
  /**
   * Override the URL the backend fetches `pii.wasm` from. Defaults to
   * the bytes inlined into the JS bundle, which is the right choice for
   * every deployment so far. Only set this if you want to host an
   * alternative .wasm build separately.
   */
  wasmModuleUrl?: string | URL;
  /**
   * Multi-thread mode.
   *   "auto" (default) — use multi-threaded WASM if SharedArrayBuffer
   *     is available; otherwise fall back to single-thread. In
   *     browsers SAB requires cross-origin isolation (COOP/COEP
   *     response headers); in Node it's always available.
   *   "off" — always single-thread, no worker pool.
   *   "force" — fail loudly if SAB isn't available, surfacing the
   *     COOP/COEP misconfiguration instead of silently degrading.
   */
  multiThread?: "auto" | "off" | "force";
  /**
   * Number of worker threads when multi-thread is enabled. Defaults to
   * `navigator.hardwareConcurrency` clamped to [2, 4]. Higher counts
   * have diminishing returns past the model's natural parallelism
   * (T_chunk per worker shrinks).
   */
  numThreads?: number;
}

function weightByName(map: ReadonlyMap<string, WeightTensorInfo>, name: string): WeightTensorInfo {
  const t = map.get(name);
  if (!t) throw new Error(`WasmBackend: missing weight "${name}"`);
  return t;
}

function detectNumLayers(map: ReadonlyMap<string, WeightTensorInfo>): number {
  let max = -1;
  for (const name of map.keys()) {
    const m = /^layers\.(\d+)\.input_layernorm$/.exec(name);
    if (m) max = Math.max(max, Number.parseInt(m[1]!, 10));
  }
  if (max < 0) throw new Error("WasmBackend: no transformer layers found in weight map");
  return max + 1;
}

function buildModelWeights(
  map: ReadonlyMap<string, WeightTensorInfo>,
  numLayers: number,
): ModelWeights {
  const g = (n: string): WeightTensorInfo => weightByName(map, n);
  const i4 = (prefix: string) => ({
    int4:   g(`${prefix}.int4`),
    scales: g(`${prefix}.scales`),
    zp:     g(`${prefix}.zp`),
    bias:   g(`${prefix}.bias`),
  });
  const blocks: BlockWeights[] = [];
  for (let L = 0; L < numLayers; L++) {
    blocks.push({
      inputLayernorm: g(`layers.${L}.input_layernorm`),
      postAttentionLayernorm: g(`layers.${L}.post_attention_layernorm`),
      attn: {
        qProj: i4(`layers.${L}.attn.q_proj`),
        kProj: i4(`layers.${L}.attn.k_proj`),
        vProj: i4(`layers.${L}.attn.v_proj`),
        oProj: i4(`layers.${L}.attn.o_proj`),
        sinks: g(`layers.${L}.attn.sinks`),
      },
      router: i4(`layers.${L}.router`),
      experts: {
        gateUpInt4:   g(`layers.${L}.experts.gate_up.int4`),
        gateUpScales: g(`layers.${L}.experts.gate_up.scales`),
        gateUpZp:     g(`layers.${L}.experts.gate_up.zp`),
        gateUpBias:   g(`layers.${L}.experts.gate_up.bias`),
        downInt4:     g(`layers.${L}.experts.down.int4`),
        downScales:   g(`layers.${L}.experts.down.scales`),
        downZp:       g(`layers.${L}.experts.down.zp`),
        downBias:     g(`layers.${L}.experts.down.bias`),
      },
    });
  }
  return {
    embedInt4:   g("embed.int4"),
    embedScales: g("embed.scales"),
    embedZp:     g("embed.zp"),
    blocks,
    finalLayernorm: g("final_norm"),
    classifier: i4("score"),
  };
}

/**
 * Per-worker scratch capacity, sized for `MAX_T` tokens at 4
 * experts-per-token (the model's K). Each worker can therefore handle
 * up to `MAX_T / numThreads` token rows in its slice.
 */
const MAX_T = 2048;
const MAX_M_PER_WORKER = MAX_T;

export class WasmBackend implements InferenceBackend {
  readonly name = "wasm" as const;
  private wasm: PiiWasmExports | null = null;
  private weightMap: Map<string, WeightTensorInfo> | null = null;
  private modelWeights: ModelWeights | null = null;
  private numExpertsInBlob = 0;
  /** Multi-thread mode plumbing. Null when running single-threaded. */
  private mtPool: MtPool | null = null;
  private mtScratch: WorkerScratch[] | null = null;
  private mtAccPtr = 0;
  private readonly opts: WasmBackendOptions;

  constructor(opts: WasmBackendOptions) {
    this.opts = opts;
  }

  /** Reflects the actual mode after `warmup()`. */
  get threadingMode(): "single" | "multi" {
    return this.mtPool ? "multi" : "single";
  }

  async warmup(): Promise<void> {
    const mtRequest = this.opts.multiThread ?? "auto";
    const supported = sharedMemorySupported();
    const useMt = mtRequest === "force" || (mtRequest === "auto" && supported);
    if (mtRequest === "force" && !supported) {
      throw new Error(
        "WasmBackend: multiThread:'force' requested but SharedArrayBuffer is not " +
          "available (in browsers, the page must be served with " +
          "Cross-Origin-Opener-Policy: same-origin and " +
          "Cross-Origin-Embedder-Policy: require-corp)",
      );
    }

    if (useMt) {
      // Pre-create a SAB-backed memory big enough for ~770 MB weights
      // plus per-forward scratch. Overshoot defensively; the host only
      // commits pages on first touch.
      const memory = new WebAssembly.Memory({
        initial: 24576, // 1.5 GB
        maximum: 32768, // 2 GB hard cap (WASM32 limit)
        shared: true,
      });
      this.wasm = await loadPiiWasmShared(memory);
    } else {
      this.wasm = await loadPiiWasm(this.opts.wasmModuleUrl);
    }

    const echoed = this.wasm.echo(42);
    if (echoed !== 42) {
      throw new Error(`WasmBackend: echo round-trip returned ${echoed}, expected 42`);
    }

    this.weightMap = await loadOnnxWeights(this.wasm, this.opts.bundle.modelSource);
    const numLayers = detectNumLayers(this.weightMap);
    this.modelWeights = buildModelWeights(this.weightMap, numLayers);
    this.numExpertsInBlob = this.modelWeights.blocks[0]!.router.int4.shape[0]!;

    if (useMt) {
      const D = PF_CONFIG.hiddenSize;
      const dff = PF_CONFIG.intermediateSize;
      // 6-thread default — sweep on M-series shows the sweet spot at 6
      // (T=32: 4t=93ms, 6t=74ms; T=128: 4t=315ms, 6t=258ms). 8 threads
      // regresses, likely from E-core contention competing with the
      // main thread. Cap at hardwareConcurrency-1 so the main thread
      // always has at least one core.
      const cores = (typeof navigator !== "undefined" ? navigator.hardwareConcurrency : 4) || 4;
      const desired = this.opts.numThreads ??
        Math.max(2, Math.min(6, cores - 1));

      // Allocate per-worker scratch + the f32 MoE accumulator BEFORE
      // marking the heap, so `reset()` after each forward keeps them.
      this.mtScratch = [];
      const M = MAX_M_PER_WORKER;
      for (let w = 0; w < desired; w++) {
        const xGatheredPtr = this.wasm.alloc(M * D * 4);
        const gateUpPtr   = this.wasm.alloc(M * 2 * dff * 4);
        const gluPtr      = this.wasm.alloc(M * dff * 4);
        const outF32Ptr   = this.wasm.alloc(M * D * 4);
        const tokIdxPtr   = this.wasm.alloc(M * 4);
        const weightsPtr  = this.wasm.alloc(M * 4);
        if (!xGatheredPtr || !gateUpPtr || !gluPtr || !outF32Ptr || !tokIdxPtr || !weightsPtr) {
          throw new Error(`WasmBackend: OOM allocating worker scratch (worker ${w})`);
        }
        this.mtScratch.push({ xGatheredPtr, gateUpPtr, gluPtr, outF32Ptr, tokIdxPtr, weightsPtr });
      }
      // Shared MoE accumulator across all workers. Tokens are
      // partitioned by index range so writes are disjoint.
      this.mtAccPtr = this.wasm.alloc(MAX_T * D * 4);
      if (!this.mtAccPtr) throw new Error("WasmBackend: OOM allocating MoE accumulator");
      // 4-byte slot used as the cross-thread WASM-memory fence
      // target. Atomics on a different SAB (the pool's signal buffer)
      // don't synchronize plain WASM-memory accesses; this slot
      // gives the pool an atomic location *inside* the memory SAB
      // it can release/acquire on.
      const fencePtr = this.wasm.alloc(4);
      if (!fencePtr) throw new Error("WasmBackend: OOM allocating fence slot");
      // Per-worker shadow stacks. Each WebAssembly.Instance has its
      // own `__stack_pointer` global, but the stack memory those
      // pointers reference lives in the shared linear memory — so
      // without per-instance stack regions every worker pushes
      // locals to the same address and corrupts each other's frames.
      // 256 KB per worker is comfortable for our deepest call chain.
      const STACK_BYTES = 256 * 1024;
      const stackTops: number[] = [];
      for (let w = 0; w < desired; w++) {
        const base = this.wasm.alloc(STACK_BYTES);
        if (!base) throw new Error(`WasmBackend: OOM allocating worker ${w} stack`);
        // Stack grows downward from the top.
        stackTops.push(base + STACK_BYTES);
      }
      // Re-mark the heap so subsequent `reset()` calls (one per
      // forward) preserve the worker scratch + accumulator + fence
      // + stacks.
      this.wasm.heap_mark_now();

      this.mtPool = new MtPool(this.wasm.memory, desired);
      this.mtPool.setMemoryFenceSlot(fencePtr);
      this.mtPool.setWorkerStackTops(stackTops);
      await this.mtPool.warmup();
    }

    // Prefill: one dummy forward so V8 JITs everything + heap reaches
    // steady state. Done after pool init so the MT path is also warmed.
    const warmupTokens = new Int32Array(WARMUP_T);
    const dummyMask = new Uint8Array(WARMUP_T).fill(1);
    await this.forward(warmupTokens, dummyMask);
  }

  async forward(
    tokenIds: Int32Array,
    attentionMask: Uint8Array,
  ): Promise<Logits> {
    if (!this.wasm || !this.modelWeights) {
      throw new Error("WasmBackend: call warmup() before forward()");
    }
    const wasm = this.wasm;
    const T = tokenIds.length;
    if (attentionMask.length !== T) {
      throw new Error(
        `WasmBackend.forward: attentionMask length ${attentionMask.length} does not match tokenIds length ${T}`,
      );
    }
    const vocabSize = this.modelWeights.embedInt4.shape[0]!;

    const idsBytes = T * 4;
    const logitsBytes = T * PF_CONFIG.numClasses * 2;
    const idsPtr = wasm.alloc(idsBytes);
    const logitsPtr = wasm.alloc(logitsBytes);
    if (idsPtr === 0 || logitsPtr === 0) {
      throw new Error("WasmBackend.forward: alloc OOM for inputs/outputs");
    }
    new Int32Array(wasm.memory.buffer, idsPtr, T).set(tokenIds);

    // If every mask byte is 1, skip the alloc/copy and pass maskPtr=0
    // so banded_attention takes its faster mask-free path.
    let maskPtr = 0;
    for (let i = 0; i < T; i++) {
      if (attentionMask[i] !== 1) {
        const p = wasm.alloc(T);
        if (p === 0) throw new Error("WasmBackend.forward: alloc OOM for mask");
        new Uint8Array(wasm.memory.buffer, p, T).set(attentionMask);
        maskPtr = p;
        break;
      }
    }

    const mt: MultiThreadContext | undefined = (this.mtPool && this.mtScratch)
      ? { pool: this.mtPool, workerScratch: this.mtScratch, accPtr: this.mtAccPtr }
      : undefined;
    await modelForward(wasm, idsPtr, logitsPtr, this.modelWeights, {
      ...PF_CONFIG,
      vocabSize,
      numExperts: this.numExpertsInBlob,
      numExpertsInBlob: this.numExpertsInBlob,
    }, T, maskPtr, mt);

    // Upcast fp16 → f32 for the backend contract.
    const fp16View = new Uint16Array(wasm.memory.buffer, logitsPtr, T * PF_CONFIG.numClasses);
    const f32 = new Float32Array(fp16View.length);
    for (let i = 0; i < fp16View.length; i++) {
      const h = fp16View[i]!;
      const sign = (h & 0x8000) << 16;
      const exp = (h >> 10) & 0x1f;
      const mant = h & 0x3ff;
      if (exp === 0) {
        if (mant === 0) { f32[i] = sign ? -0 : 0; continue; }
        let m = mant, e = 1;
        while ((m & 0x400) === 0) { m <<= 1; e--; }
        m &= 0x3ff;
        _cvU32[0] = (sign | ((e + 112) << 23) | (m << 13)) >>> 0;
      } else if (exp === 0x1f) {
        _cvU32[0] = (sign | 0x7f800000 | (mant << 13)) >>> 0;
      } else {
        _cvU32[0] = (sign | ((exp + 112) << 23) | (mant << 13)) >>> 0;
      }
      f32[i] = _cvF32[0]!;
    }

    wasm.reset();

    return {
      data: f32,
      sequenceLength: T,
      numClasses: PF_CONFIG.numClasses,
    };
  }

  dispose(): void {
    this.mtPool?.dispose();
    this.mtPool = null;
    this.mtScratch = null;
    this.wasm?.reset();
    this.wasm = null;
    this.weightMap = null;
    this.modelWeights = null;
  }
}
