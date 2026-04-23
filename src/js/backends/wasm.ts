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

/** Exports declared in `src/zig/wasm_exports.zig`. Keep in sync. */
interface PiiWasmExports {
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
  matmul_bf16_x_int4block(x: number, w_int4: number, w_scales: number, bias: number, out: number, T: number, N: number, D: number): void;
  matmul_bf16_x_int4block_out_f32(x: number, w_int4: number, w_scales: number, bias: number, out: number, T: number, N: number, D: number): void;
  matmul_f32_x_int4block_out_f32(x: number, w_int4: number, w_scales: number, bias: number, out: number, T: number, N: number, D: number): void;
  topk_partial_f32(x: number, out_idx: number, out_val: number, rows: number, cols: number, k: number): void;
  rope_apply(qk: number, cos: number, sin: number, T: number, H: number, head_dim: number): void;
  banded_attention(
    q: number, k: number, v: number, sinks: number, out: number,
    T: number, H_q: number, H_kv: number, head_dim: number, window: number,
  ): void;
  scale_bf16_inplace(x: number, scale: number, n: number): void;
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
 * Load `pii.wasm` from the given URL and return its typed exports
 * plus a readiness check. Exported (not just used internally) so the
 * Stage-1 kernel unit tests can instantiate the module directly
 * without going through the full backend.
 */
export async function loadPiiWasm(url: string | URL): Promise<PiiWasmExports> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `loadPiiWasm: fetch ${url} → ${response.status} ${response.statusText}`,
    );
  }
  const { instance } = await WebAssembly.instantiateStreaming(response, {});
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
    "gather_bf16",
    "scatter_add_weighted_f32",
    "zero_f32",
    "cast_f32_to_bf16_scaled",
    "softmax_f32",
    "swiglu_clamp_f32",
    "embed_lookup",
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

const DEFAULT_WASM_URL = new URL("../../../dist/pii.wasm", import.meta.url);

export class WasmBackend implements InferenceBackend {
  readonly name = "wasm" as const;
  private wasm: PiiWasmExports | null = null;
  private readonly opts: BackendConstructionOptions;

  constructor(opts: BackendConstructionOptions) {
    this.opts = opts;
  }

  async warmup(): Promise<void> {
    this.wasm = await loadPiiWasm(DEFAULT_WASM_URL);

    // Plumbing smoke-test: confirm ABI round-trip works before any real
    // compute. If this ever fires, the module we loaded doesn't match
    // the interface in `wasm_exports.zig`.
    const echoed = this.wasm.echo(42);
    if (echoed !== 42) {
      throw new Error(`WasmBackend: echo round-trip returned ${echoed}, expected 42`);
    }
  }

  async forward(
    _tokenIds: Int32Array,
    _attentionMask: Uint8Array,
  ): Promise<Logits> {
    // Phase A: no kernels yet. Weight loading is Phase B; RMSNorm + int4
    // matmul are Phase C; attention + MoE are Phase D. Until those land
    // this backend exists only so the build pipeline + JS bridge are
    // exercised end-to-end.
    throw new Error(
      "WasmBackend.forward is not implemented yet — Stage 1 kernels are pending. "
        + "Use the default `backend: \"auto\"` (transformers.js) until Stage 1 ships.",
    );
  }

  dispose(): void {
    // WebAssembly.Memory is GC'd once the instance is unreferenced; no
    // explicit free is required. Reset the bump heap to drop per-call
    // scratch.
    this.wasm?.reset();
    this.wasm = null;
  }
}
