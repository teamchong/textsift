/**
 * Attention forward composition.
 *
 * Wires the Phase-D kernels together into one attention block call:
 *
 *   x  →  q_proj / k_proj / v_proj       (fp16 matmul × 3)
 *      →  rope_apply on Q, K              (interleaved pair rotation)
 *      →  scale Q and K by head_dim^-0.25 (one fp16 round each)
 *      →  banded_attention                (scores + sink + softmax + AV)
 *      →  o_proj                          (fp16 matmul)
 *      →  out
 *
 * All scratch (Q, K, V, attention output) comes from the WASM bump
 * allocator. Caller is expected to `reset()` afterwards to drop it.
 */

import type { Int4BlockWeight, PiiWasmExports, WeightTensorInfo } from "../backends/wasm.js";

export interface AttentionWeights {
  qProj: Int4BlockWeight;
  kProj: Int4BlockWeight;
  vProj: Int4BlockWeight;
  oProj: Int4BlockWeight;
  /** f32 [H_q] (upstream keeps sinks in fp32; ONNX stores as [1,H_q,1,1]). */
  sinks: WeightTensorInfo;
}

export interface AttentionConfig {
  hiddenSize: number;     // d_model (= 640)
  numHeads: number;       // H_q (= 14)
  numKvHeads: number;     // H_kv (= 2)
  headDim: number;        // (= 64)
  slidingWindow: number;  // (= 128)
}

export interface AttentionTables {
  /** fp16 cos table in WASM memory, shape [seqLen, head_dim/2]. */
  ropeCosPtr: number;
  /** fp16 sin table in WASM memory, shape [seqLen, head_dim/2]. */
  ropeSinPtr: number;
}

export function attentionForward(
  wasm: PiiWasmExports,
  hiddenPtr: number,
  outputPtr: number,
  weights: AttentionWeights,
  config: AttentionConfig,
  tables: AttentionTables,
  T: number,
  /**
   * Pointer to a `u8 [T]` mask in WASM memory (1 = valid token, 0 = padding).
   * Pass `0` to treat all keys as valid.
   */
  maskPtr: number = 0,
): void {
  const Hq = config.numHeads;
  const Hkv = config.numKvHeads;
  const hd = config.headDim;
  const D = config.hiddenSize;
  if (Hq % Hkv !== 0) {
    throw new Error(`attentionForward: numHeads (${Hq}) must be divisible by numKvHeads (${Hkv})`);
  }

  // Scratch: Q [T, Hq*hd], K [T, Hkv*hd], V [T, Hkv*hd], attn_out [T, Hq*hd], all fp16.
  const qPtr = wasm.alloc(T * Hq * hd * 2);
  const kPtr = wasm.alloc(T * Hkv * hd * 2);
  const vPtr = wasm.alloc(T * Hkv * hd * 2);
  const attnOutPtr = wasm.alloc(T * Hq * hd * 2);
  if (qPtr === 0 || kPtr === 0 || vPtr === 0 || attnOutPtr === 0) {
    throw new Error("attentionForward: scratch alloc OOM");
  }

  // Q/K/V projections (fp16 x × int4 W → fp16 out, asymmetric uint4 with per-block ZP).
  wasm.matmul_fp16_x_int4block(
    hiddenPtr,
    weights.qProj.int4.dataOffset, weights.qProj.scales.dataOffset, weights.qProj.zp.dataOffset,
    weights.qProj.bias.dataOffset, qPtr, T, Hq * hd, D,
  );
  wasm.matmul_fp16_x_int4block(
    hiddenPtr,
    weights.kProj.int4.dataOffset, weights.kProj.scales.dataOffset, weights.kProj.zp.dataOffset,
    weights.kProj.bias.dataOffset, kPtr, T, Hkv * hd, D,
  );
  wasm.matmul_fp16_x_int4block(
    hiddenPtr,
    weights.vProj.int4.dataOffset, weights.vProj.scales.dataOffset, weights.vProj.zp.dataOffset,
    weights.vProj.bias.dataOffset, vPtr, T, Hkv * hd, D,
  );

  // RoPE on Q and K (interleaved pairs). cos/sin already have yarn's
  // attention_scaling folded in (≈ 1.3466 for factor=32). The ONNX
  // export path omits the `q * head_dim^-0.25` scale the upstream
  // Python code applies; Q·K goes straight into softmax without it,
  // and we match that.
  wasm.rope_apply(qPtr, tables.ropeCosPtr, tables.ropeSinPtr, T, Hq, hd);
  wasm.rope_apply(kPtr, tables.ropeCosPtr, tables.ropeSinPtr, T, Hkv, hd);

  // Banded attention (scores + sink + softmax + AV combine).
  wasm.banded_attention(
    qPtr, kPtr, vPtr, weights.sinks.dataOffset, maskPtr, attnOutPtr,
    T, Hq, Hkv, hd, config.slidingWindow,
  );

  // O projection: [T, Hq*hd] @ [D, Hq*hd]^T + [D] → [T, D]  (int4 W, asymmetric).
  wasm.matmul_fp16_x_int4block(
    attnOutPtr,
    weights.oProj.int4.dataOffset, weights.oProj.scales.dataOffset, weights.oProj.zp.dataOffset,
    weights.oProj.bias.dataOffset, outputPtr, T, D, Hq * hd,
  );
}
