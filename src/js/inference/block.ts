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

import type { PiiWasmExports, WeightTensorInfo } from "../backends/wasm.js";
import { attentionForward, type AttentionWeights, type AttentionConfig, type AttentionTables } from "./attention.js";
import { expertDispatch, type ExpertWeights, type ExpertConfig } from "./expert.js";

export interface BlockWeights {
  inputLayernorm: WeightTensorInfo;           // bf16 [D]
  postAttentionLayernorm: WeightTensorInfo;   // bf16 [D]
  attn: AttentionWeights;
  routerW: WeightTensorInfo;                  // bf16 [num_experts, D]
  routerB: WeightTensorInfo;                  // bf16 [num_experts]
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
  hiddenPtr: number,
  routingIdxPtr: number,
  routingScoresPtr: number,
  weightsW: WeightTensorInfo,
  weightsB: WeightTensorInfo,
  numExperts: number,
  K: number,
  T: number,
  D: number,
): void {
  // Logits in fp32 (upstream: `F.linear(hidden.float(), W.float(), b.float())`).
  const logitsPtr = wasm.alloc(T * numExperts * 4);
  wasm.matmul_bf16_out_f32(
    hiddenPtr, weightsW.dataOffset, weightsB.dataOffset, logitsPtr,
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
 * Transformer block with externally provided routing. Useful for tests
 * where we want to exercise the block body with synthetic routing (e.g.
 * when the blob carries only a subset of experts).
 */
export function blockForwardWithRouting(
  wasm: PiiWasmExports,
  inputPtr: number,
  outputPtr: number,
  routingIdxPtr: number,
  routingScoresPtr: number,
  weights: BlockWeights,
  tables: AttentionTables,
  config: BlockConfig,
  T: number,
): void {
  const D = config.hiddenSize;

  // Residual save 1.
  const residualPtr = wasm.alloc(T * D * 2);
  new Uint8Array(wasm.memory.buffer, residualPtr, T * D * 2).set(
    new Uint8Array(wasm.memory.buffer, inputPtr, T * D * 2),
  );

  const normed1Ptr = wasm.alloc(T * D * 2);
  wasm.rms_norm(inputPtr, weights.inputLayernorm.dataOffset, normed1Ptr, T, D, config.rmsNormEps);

  const attnOutPtr = wasm.alloc(T * D * 2);
  attentionForward(wasm, normed1Ptr, attnOutPtr, weights.attn, config, tables, T);

  const h1Ptr = wasm.alloc(T * D * 2);
  wasm.add_bf16(residualPtr, attnOutPtr, h1Ptr, T * D);

  const residual2Ptr = wasm.alloc(T * D * 2);
  new Uint8Array(wasm.memory.buffer, residual2Ptr, T * D * 2).set(
    new Uint8Array(wasm.memory.buffer, h1Ptr, T * D * 2),
  );

  const normed2Ptr = wasm.alloc(T * D * 2);
  wasm.rms_norm(h1Ptr, weights.postAttentionLayernorm.dataOffset, normed2Ptr, T, D, config.rmsNormEps);

  const moeOutPtr = wasm.alloc(T * D * 2);
  expertDispatch(
    wasm, normed2Ptr, moeOutPtr, routingIdxPtr, routingScoresPtr,
    weights.experts,
    { ...config, numExperts: config.numExpertsInBlob },
    T,
  );

  wasm.add_bf16(residual2Ptr, moeOutPtr, outputPtr, T * D);
}

/**
 * Full transformer block: runs its own router, then delegates to
 * `blockForwardWithRouting`. This is the production entry point used by
 * `modelForward`.
 */
export function blockForward(
  wasm: PiiWasmExports,
  inputPtr: number,
  outputPtr: number,
  weights: BlockWeights,
  tables: AttentionTables,
  config: BlockConfig,
  T: number,
): void {
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
  attentionForward(wasm, normed1Ptr, attnOutPtr, weights.attn, config, tables, T);

  const h1Ptr = wasm.alloc(T * D * 2);
  wasm.add_bf16(residualPtr, attnOutPtr, h1Ptr, T * D);

  const residual2Ptr = wasm.alloc(T * D * 2);
  new Uint8Array(wasm.memory.buffer, residual2Ptr, T * D * 2).set(
    new Uint8Array(wasm.memory.buffer, h1Ptr, T * D * 2),
  );

  const normed2Ptr = wasm.alloc(T * D * 2);
  wasm.rms_norm(h1Ptr, weights.postAttentionLayernorm.dataOffset, normed2Ptr, T, D, config.rmsNormEps);

  // Router + expert dispatch.
  const routingIdxPtr = wasm.alloc(T * K * 4);
  const routingScoresPtr = wasm.alloc(T * K * 4);
  routerForward(
    wasm, normed2Ptr, routingIdxPtr, routingScoresPtr,
    weights.routerW, weights.routerB, config.numExperts, K, T, D,
  );

  const moeOutPtr = wasm.alloc(T * D * 2);
  expertDispatch(
    wasm, normed2Ptr, moeOutPtr, routingIdxPtr, routingScoresPtr,
    weights.experts,
    { ...config, numExperts: config.numExpertsInBlob },
    T,
  );

  wasm.add_bf16(residual2Ptr, moeOutPtr, outputPtr, T * D);
}
