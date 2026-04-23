/**
 * Attention forward composition.
 *
 * Wires the Phase-D kernels together into one attention block call:
 *
 *   x  →  q_proj / k_proj / v_proj       (bf16 matmul × 3)
 *      →  rope_apply on Q, K              (interleaved pair rotation)
 *      →  scale Q and K by head_dim^-0.25 (one bf16 round each)
 *      →  banded_attention                (scores + sink + softmax + AV)
 *      →  o_proj                          (bf16 matmul)
 *      →  out
 *
 * All scratch (Q, K, V, attention output) comes from the WASM bump
 * allocator. Caller is expected to `reset()` afterwards to drop it.
 */

import type { PiiWasmExports, WeightTensorInfo } from "../backends/wasm.js";

export interface AttentionWeights {
  qProj: WeightTensorInfo;
  qProjBias: WeightTensorInfo;
  kProj: WeightTensorInfo;
  kProjBias: WeightTensorInfo;
  vProj: WeightTensorInfo;
  vProjBias: WeightTensorInfo;
  oProj: WeightTensorInfo;
  oProjBias: WeightTensorInfo;
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
  /** bf16 cos table in WASM memory, shape [seqLen, head_dim/2]. */
  ropeCosPtr: number;
  /** bf16 sin table in WASM memory, shape [seqLen, head_dim/2]. */
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

  // Scratch: Q [T, Hq*hd], K [T, Hkv*hd], V [T, Hkv*hd], attn_out [T, Hq*hd], all bf16.
  const qPtr = wasm.alloc(T * Hq * hd * 2);
  const kPtr = wasm.alloc(T * Hkv * hd * 2);
  const vPtr = wasm.alloc(T * Hkv * hd * 2);
  const attnOutPtr = wasm.alloc(T * Hq * hd * 2);
  if (qPtr === 0 || kPtr === 0 || vPtr === 0 || attnOutPtr === 0) {
    throw new Error("attentionForward: scratch alloc OOM");
  }

  // Q/K/V projections.
  wasm.matmul_bf16(hiddenPtr, weights.qProj.dataOffset, weights.qProjBias.dataOffset, qPtr, T, Hq * hd, D);
  wasm.matmul_bf16(hiddenPtr, weights.kProj.dataOffset, weights.kProjBias.dataOffset, kPtr, T, Hkv * hd, D);
  wasm.matmul_bf16(hiddenPtr, weights.vProj.dataOffset, weights.vProjBias.dataOffset, vPtr, T, Hkv * hd, D);

  // RoPE on Q and K (interleaved pairs). cos/sin already have yarn's
  // attention_scaling folded in.
  wasm.rope_apply(qPtr, tables.ropeCosPtr, tables.ropeSinPtr, T, Hq, hd);
  wasm.rope_apply(kPtr, tables.ropeCosPtr, tables.ropeSinPtr, T, Hkv, hd);

  // Q/K scale by head_dim ** -0.25. Upstream:
  //     query_states = query_states * self.scaling
  //     key_states   = key_states   * self.scaling
  // applied AFTER rope, with scaling = head_dim ** -0.25.
  const qkScale = Math.pow(hd, -0.25);
  wasm.scale_bf16_inplace(qPtr, qkScale, T * Hq * hd);
  wasm.scale_bf16_inplace(kPtr, qkScale, T * Hkv * hd);

  // Banded attention (scores + sink + softmax + AV combine).
  wasm.banded_attention(
    qPtr, kPtr, vPtr, weights.sinks.dataOffset, maskPtr, attnOutPtr,
    T, Hq, Hkv, hd, config.slidingWindow,
  );

  // O projection: [T, Hq*hd] @ [D, Hq*hd]^T + [D] → [T, D]
  wasm.matmul_bf16(
    attnOutPtr, weights.oProj.dataOffset, weights.oProjBias.dataOffset,
    outputPtr, T, D, Hq * hd,
  );
}
