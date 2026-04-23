/**
 * Stage 1 backend: custom Zig+WASM inference engine.
 *
 * This file is the JS side of the bridge; the Zig side lives at
 * `src/zig/wasm_exports.zig` and compiles to `dist/pii.wasm` via
 * `npm run build:zig`.
 *
 * Status: Phase A (scaffolding) — module loads, ABI round-trips, heap
 * init works. `forward()` throws until kernels land (Phases C/D).
 *
 * See `docs/roadmap.md` → Stage 1 for the build order.
 */

import type {
  BackendConstructionOptions,
  InferenceBackend,
  Logits,
} from "./abstract.js";
import { modelForward, type ModelConfig, type ModelWeights } from "../inference/model.js";
import type { BlockWeights } from "../inference/block.js";
import { PII_WASM_BYTES } from "./pii-wasm-inline.js";

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
  weights_load(ptr: number, size: number): number;
  weights_count(): number;
  weights_dtype(idx: number): number;
  weights_ndim(idx: number): number;
  weights_shape(idx: number, out: number): number;
  weights_data_ptr(idx: number): number;
  weights_data_size(idx: number): number;
  weights_name(idx: number, out: number): number;
  rms_norm(x: number, gamma: number, out: number, T: number, D: number, eps: number): void;
  matmul_bf16(x: number, w: number, bias: number, out: number, T: number, N: number, D: number): void;
  matmul_bf16_out_f32(x: number, w: number, bias: number, out: number, T: number, N: number, D: number): void;
  // int4-block matmul variants: `w_zp` is `0` for symmetric decode (no zero-points),
  // or a pointer into WASM memory for asymmetric (ONNX MatMulNBits semantics).
  matmul_bf16_x_int4block(x: number, w_int4: number, w_scales: number, w_zp: number, bias: number, out: number, T: number, N: number, D: number): void;
  matmul_bf16_x_int4block_out_f32(x: number, w_int4: number, w_scales: number, w_zp: number, bias: number, out: number, T: number, N: number, D: number): void;
  matmul_f32_x_int4block_out_f32(x: number, w_int4: number, w_scales: number, w_zp: number, bias: number, out: number, T: number, N: number, D: number): void;
  topk_partial_f32(x: number, out_idx: number, out_val: number, rows: number, cols: number, k: number): void;
  rope_apply(qk: number, cos: number, sin: number, T: number, H: number, head_dim: number): void;
  banded_attention(
    q: number, k: number, v: number, sinks: number, mask: number, out: number,
    T: number, H_q: number, H_kv: number, head_dim: number, window: number,
  ): void;
  scale_bf16_inplace(x: number, scale: number, n: number): void;
  add_bf16(a: number, b: number, out: number, n: number): void;
  gather_bf16(src: number, indices: number, dst: number, m: number, D: number): void;
  scatter_add_weighted_f32(
    target: number, values: number, indices: number, weights: number,
    m: number, D: number,
  ): void;
  zero_f32(ptr: number, n: number): void;
  cast_f32_to_bf16_scaled(src: number, dst: number, n: number, scale: number): void;
  softmax_f32(x: number, out: number, rows: number, cols: number): void;
  swiglu_clamp_f32(gate_up: number, out: number, T: number, D: number): void;
  embed_lookup(embed: number, ids: number, out: number, T: number, V: number, D: number): void;
  embed_lookup_int4(
    embed_int4: number, embed_scales: number, embed_zp: number,
    ids: number, out: number, T: number, V: number, D: number,
  ): void;
}

/** Dtype codes emitted by `scripts/convert_weights.py`. */
export enum WeightDType {
  F32 = 0,
  F16 = 1,
  BF16 = 2,
  I8 = 3,
  U8 = 4,
  I32 = 5,
  /**
   * Signed symmetric int4, blockwise along the last dim (block=32),
   * per-block fp16 scale, no zero-point. Layout per tensor:
   *   [N, D/2] packed u8 (low nibble = even d index, high = odd)
   *   then [N, D/32] fp16 scales.
   */
  I4_BLOCK32_SYM = 6,
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
    "weights_load",
    "weights_count",
    "weights_dtype",
    "weights_ndim",
    "weights_shape",
    "weights_data_ptr",
    "weights_data_size",
    "weights_name",
    "rms_norm",
    "matmul_bf16",
    "matmul_bf16_out_f32",
    "matmul_bf16_x_int4block",
    "matmul_bf16_x_int4block_out_f32",
    "matmul_f32_x_int4block_out_f32",
    "topk_partial_f32",
    "rope_apply",
    "banded_attention",
    "scale_bf16_inplace",
    "add_bf16",
    "gather_bf16",
    "scatter_add_weighted_f32",
    "zero_f32",
    "cast_f32_to_bf16_scaled",
    "softmax_f32",
    "swiglu_clamp_f32",
    "embed_lookup",
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
 * Fetch a `pii-weights.bin` blob, optionally verify its sha256, copy
 * it into WASM linear memory, parse it, and pin it below the bump
 * mark so subsequent `reset()` calls preserve it. Returns the tensor
 * table for JS-side inspection; the data lives in WASM memory and is
 * referenced by `info.dataOffset`.
 *
 * `expectedSha256` is optional but strongly recommended for
 * production — a truncated or corrupted weight blob would otherwise
 * silently produce garbage logits.
 */
export async function loadWeights(
  wasm: PiiWasmExports,
  url: string | URL,
  expectedSha256?: string,
): Promise<readonly WeightTensorInfo[]> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`loadWeights: fetch ${url} → ${response.status} ${response.statusText}`);
  }
  const buf = await response.arrayBuffer();

  if (expectedSha256 !== undefined) {
    const digest = await crypto.subtle.digest("SHA-256", buf);
    const got = [...new Uint8Array(digest)]
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
    if (got !== expectedSha256.toLowerCase()) {
      throw new Error(`loadWeights: sha256 mismatch — got ${got}, expected ${expectedSha256}`);
    }
  }

  const size = buf.byteLength;
  const ptr = wasm.alloc(size);
  if (ptr === 0) {
    throw new Error(`loadWeights: WASM OOM allocating ${size} bytes`);
  }
  new Uint8Array(wasm.memory.buffer, ptr, size).set(new Uint8Array(buf));

  const rc = wasm.weights_load(ptr, size);
  if (rc !== 0) {
    throw new Error(`loadWeights: parser returned error ${rc}`);
  }

  // Pin the blob + any internal Zig-side bookkeeping below the reset
  // line. Everything allocated after this call is scratch that goes
  // away on `reset()`.
  wasm.heap_mark_now();

  const n = wasm.weights_count();
  const shapeBuf = wasm.alloc(16);
  const nameBuf = wasm.alloc(64);
  const decoder = new TextDecoder();

  const out: WeightTensorInfo[] = [];
  for (let i = 0; i < n; i++) {
    wasm.weights_shape(i, shapeBuf);
    const shapeU32 = new Uint32Array(wasm.memory.buffer, shapeBuf, 4);
    const ndim = wasm.weights_ndim(i);
    const shape = Array.from(shapeU32.slice(0, ndim));

    const nameLen = wasm.weights_name(i, nameBuf);
    const nameBytes = new Uint8Array(wasm.memory.buffer, nameBuf, nameLen);
    const name = decoder.decode(nameBytes);

    out.push({
      name,
      dtype: wasm.weights_dtype(i) as WeightDType,
      shape,
      dataOffset: wasm.weights_data_ptr(i),
      dataSize: wasm.weights_data_size(i),
    });
  }

  // Drop the scratch we just used. The mark stays where it was — at
  // the top of the weight blob — so the weights survive.
  wasm.reset();
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
  /** URL or path to the `pii-weights.bin` blob produced by `scripts/convert_weights.py`. */
  weightsUrl: string | URL;
  /** Optional sha256 for integrity check. */
  weightsSha256?: string;
  /**
   * Override the URL the backend fetches `pii.wasm` from. Defaults to
   * a path relative to the bundled JS (`./pii.wasm`), which works when
   * the bundle and the .wasm live in the same directory. Node tests
   * that import directly from source use a different relative path;
   * provide this option to pin the URL explicitly.
   */
  wasmModuleUrl?: string | URL;
}

function weightByName(map: ReadonlyMap<string, WeightTensorInfo>, name: string): WeightTensorInfo {
  const t = map.get(name);
  if (!t) throw new Error(`WasmBackend: missing weight "${name}"`);
  return t;
}

function detectNumLayers(map: ReadonlyMap<string, WeightTensorInfo>): number {
  let max = -1;
  for (const name of map.keys()) {
    const m = /^model\.layers\.(\d+)\./.exec(name);
    if (m) max = Math.max(max, Number.parseInt(m[1]!, 10));
  }
  if (max < 0) throw new Error("WasmBackend: no model.layers.* tensors in blob");
  return max + 1;
}

function buildModelWeights(
  map: ReadonlyMap<string, WeightTensorInfo>,
  numLayers: number,
): ModelWeights {
  const blocks: BlockWeights[] = [];
  for (let L = 0; L < numLayers; L++) {
    const p = `model.layers.${L}.`;
    blocks.push({
      inputLayernorm: weightByName(map, `${p}input_layernorm.weight`),
      postAttentionLayernorm: weightByName(map, `${p}post_attention_layernorm.weight`),
      attn: {
        qProj: weightByName(map, `${p}self_attn.q_proj.weight`),
        qProjBias: weightByName(map, `${p}self_attn.q_proj.bias`),
        kProj: weightByName(map, `${p}self_attn.k_proj.weight`),
        kProjBias: weightByName(map, `${p}self_attn.k_proj.bias`),
        vProj: weightByName(map, `${p}self_attn.v_proj.weight`),
        vProjBias: weightByName(map, `${p}self_attn.v_proj.bias`),
        oProj: weightByName(map, `${p}self_attn.o_proj.weight`),
        oProjBias: weightByName(map, `${p}self_attn.o_proj.bias`),
        sinks: weightByName(map, `${p}self_attn.sinks`),
      },
      routerW: weightByName(map, `${p}mlp.router.weight`),
      routerB: weightByName(map, `${p}mlp.router.bias`),
      experts: {
        gateUp: weightByName(map, `${p}mlp.experts.gate_up_proj`),
        gateUpBias: weightByName(map, `${p}mlp.experts.gate_up_proj_bias`),
        down: weightByName(map, `${p}mlp.experts.down_proj`),
        downBias: weightByName(map, `${p}mlp.experts.down_proj_bias`),
      },
    });
  }
  return {
    embedTokens: weightByName(map, "model.embed_tokens.weight"),
    blocks,
    finalLayernorm: weightByName(map, "model.norm.weight"),
    classifierW: weightByName(map, "score.weight"),
    classifierB: weightByName(map, "score.bias"),
  };
}

export class WasmBackend implements InferenceBackend {
  readonly name = "wasm" as const;
  private wasm: PiiWasmExports | null = null;
  private weightMap: Map<string, WeightTensorInfo> | null = null;
  private modelWeights: ModelWeights | null = null;
  private numExpertsInBlob = 0;
  private readonly opts: WasmBackendOptions;

  constructor(opts: WasmBackendOptions) {
    this.opts = opts;
  }

  async warmup(): Promise<void> {
    // If the caller provides an explicit URL, fetch from there (useful
    // if they want to host a different .wasm build). Otherwise use the
    // bytes baked into the bundle — saves one round trip and avoids
    // the `new URL(..., import.meta.url)` resolution quirk.
    this.wasm = await loadPiiWasm(this.opts.wasmModuleUrl);

    const echoed = this.wasm.echo(42);
    if (echoed !== 42) {
      throw new Error(`WasmBackend: echo round-trip returned ${echoed}, expected 42`);
    }

    const tensors = await loadWeights(this.wasm, this.opts.weightsUrl, this.opts.weightsSha256);
    this.weightMap = new Map(tensors.map((t) => [t.name, t]));
    const numLayers = detectNumLayers(this.weightMap);
    this.modelWeights = buildModelWeights(this.weightMap, numLayers);
    // numExperts (the router's output size) comes from the router weight's
    // first dim. Truncated test blobs have < 128 experts; full blobs 128.
    this.numExpertsInBlob = this.modelWeights.blocks[0]!.routerW.shape[0]!;

    // Prefill: run one dummy forward so V8 JITs every hot-path kernel
    // and the bump heap hits its steady-state size. Amortizes that
    // cost out of the user's first real request. T chosen small enough
    // to keep warmup fast but large enough that the TR-tiled matmul
    // path (TR=4) is actually exercised.
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
    const vocabSize = this.modelWeights.embedTokens.shape[0]!;

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

    modelForward(wasm, idsPtr, logitsPtr, this.modelWeights, {
      ...PF_CONFIG,
      vocabSize,
      numExperts: this.numExpertsInBlob,
      numExpertsInBlob: this.numExpertsInBlob,
    }, T, maskPtr);

    // Upcast bf16 → f32 for the backend contract.
    const bf16View = new Uint16Array(wasm.memory.buffer, logitsPtr, T * PF_CONFIG.numClasses);
    const f32 = new Float32Array(bf16View.length);
    const tmp = new ArrayBuffer(4);
    const tmpU32 = new Uint32Array(tmp);
    const tmpF32 = new Float32Array(tmp);
    for (let i = 0; i < bf16View.length; i++) {
      tmpU32[0] = bf16View[i]! << 16;
      f32[i] = tmpF32[0]!;
    }

    wasm.reset();

    return {
      data: f32,
      sequenceLength: T,
      numClasses: PF_CONFIG.numClasses,
    };
  }

  dispose(): void {
    this.wasm?.reset();
    this.wasm = null;
    this.weightMap = null;
    this.modelWeights = null;
  }
}
