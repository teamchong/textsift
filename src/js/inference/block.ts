/**
 * Transformer block forward composition.
 *
 *   residual = h
 *   h = input_layernorm(h)
 *   h = self_attn(h, cos, sin)
 *   h = residual + h
 *
 *   residual = h
 *   h = post_attention_layernorm(h)
 *   h = mlp(h)                         # router + experts + scale-by-K
 *   h = residual + h
 *
 * All intermediate buffers are bump-alloc'd; caller resets the heap
 * between forward passes.
 */

import type { Int4BlockWeight, PiiWasmExports, WeightTensorInfo } from "../backends/wasm.js";
import { attentionForward, type AttentionWeights, type AttentionConfig, type AttentionTables } from "./attention.js";
import { expertDispatch, type ExpertWeights, type ExpertConfig, type MultiThreadContext } from "./expert.js";

export interface BlockWeights {
  inputLayernorm: WeightTensorInfo;           // fp16 [D]
  postAttentionLayernorm: WeightTensorInfo;   // fp16 [D]
  attn: AttentionWeights;
  /**
   * MoE router: int4-block W [num_experts, D] + f16 scales + uint4 ZP + fp16 bias.
   * Upstream does f32 matmul for routing stability; we consume fp16 input
   * and produce f32 logits via `matmul_fp16_x_int4block_out_f32`.
   */
  router: Int4BlockWeight;
  experts: ExpertWeights;
}

export interface BlockConfig extends AttentionConfig, ExpertConfig {
  rmsNormEps: number;        // (= 1e-5)
  numExpertsInBlob: number;  // E loaded into the blob (may be < numExperts)
}

/**
 * Run the router: matmul in fp32, topk, softmax over top-k, divide by K.
 * Writes to caller-owned `routingIdxPtr` (i32 [T, K]) and
 * `routingScoresPtr` (f32 [T, K]).
 */
export function routerForward(
  wasm: PiiWasmExports,
  hiddenF32Ptr: number,
  routingIdxPtr: number,
  routingScoresPtr: number,
  router: Int4BlockWeight,
  numExperts: number,
  K: number,
  T: number,
  D: number,
): void {
  // Logits in fp32 (upstream: `F.linear(hidden.float(), W.float(), b.float())`).
  const logitsPtr = wasm.alloc(T * numExperts * 4);
  wasm.matmul_f32_x_int4block_out_f32(
    hiddenF32Ptr,
    router.int4.dataOffset, router.scales.dataOffset, router.zp.dataOffset,
    router.bias.dataOffset, logitsPtr,
    T, numExperts, D,
  );
  const topValPtr = wasm.alloc(T * K * 4);
  wasm.topk_partial_f32(logitsPtr, routingIdxPtr, topValPtr, T, numExperts, K);
  wasm.softmax_f32(topValPtr, routingScoresPtr, T, K);
  const scoresView = new Float32Array(wasm.memory.buffer, routingScoresPtr, T * K);
  const invK = 1 / K;
  for (let i = 0; i < scoresView.length; i++) scoresView[i] = scoresView[i]! * invK;
}

/**
 * Full transformer block: norms → attention → residual → MoE → residual.
 * Sole production entry point called by `modelForward`.
 */
export async function blockForward(
  wasm: PiiWasmExports,
  inputPtr: number,
  outputPtr: number,
  weights: BlockWeights,
  tables: AttentionTables,
  config: BlockConfig,
  T: number,
  maskPtr: number = 0,
  mt?: MultiThreadContext,
): Promise<void> {
  const D = config.hiddenSize;
  const K = config.numExpertsPerTok;

  // Residual save 1.
  const residualPtr = wasm.alloc(T * D * 2);
  new Uint8Array(wasm.memory.buffer, residualPtr, T * D * 2).set(
    new Uint8Array(wasm.memory.buffer, inputPtr, T * D * 2),
  );

  const normed1Ptr = wasm.alloc(T * D * 2);
  wasm.rms_norm(inputPtr, weights.inputLayernorm.dataOffset, normed1Ptr, T, D, config.rmsNormEps);

  const attnOutPtr = wasm.alloc(T * D * 2);
  await attentionForward(wasm, normed1Ptr, attnOutPtr, weights.attn, config, tables, T, maskPtr, mt);

  const h1Ptr = wasm.alloc(T * D * 2);
  wasm.add_fp16(residualPtr, attnOutPtr, h1Ptr, T * D);

  const residual2Ptr = wasm.alloc(T * D * 2);
  new Uint8Array(wasm.memory.buffer, residual2Ptr, T * D * 2).set(
    new Uint8Array(wasm.memory.buffer, h1Ptr, T * D * 2),
  );

  const normed2Ptr = wasm.alloc(T * D * 2);
  wasm.rms_norm(h1Ptr, weights.postAttentionLayernorm.dataOffset, normed2Ptr, T, D, config.rmsNormEps);

  // Pre-widen normed2 once — used by both the router matmul and
  // every expert's gate_up, so a single conversion here avoids per-
  // caller duplication.
  const normed2F32Ptr = wasm.alloc(T * D * 4);
  wasm.convert_fp16_to_f32(normed2Ptr, normed2F32Ptr, T * D);

  // Router + expert dispatch (both read f32 hidden).
  const routingIdxPtr = wasm.alloc(T * K * 4);
  const routingScoresPtr = wasm.alloc(T * K * 4);
  routerForward(
    wasm, normed2F32Ptr, routingIdxPtr, routingScoresPtr,
    weights.router, config.numExperts, K, T, D,
  );

  const moeOutPtr = wasm.alloc(T * D * 2);
  await expertDispatch(
    wasm, normed2F32Ptr, moeOutPtr, routingIdxPtr, routingScoresPtr,
    weights.experts,
    { ...config, numExperts: config.numExpertsInBlob },
    T,
    mt,
  );

  wasm.add_fp16(residual2Ptr, moeOutPtr, outputPtr, T * D);
}
