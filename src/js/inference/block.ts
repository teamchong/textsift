/**
 * Transformer block forward composition.
 *
 *   residual = input
 *   h_f32    = rms_norm(input) → f32           # fused norm + widen
 *   attn_out = self_attn(h_f32, cos, sin)
 *
 *   residual2, h_f32 = add+rmsnorm(residual, attn_out) → (fp16, f32)
 *                                              # fused add + norm + widen
 *   moe_out  = mlp(h_f32)                      # router + experts + scale-by-K
 *   output   = residual2 + moe_out
 *
 * All intermediate buffers are bump-alloc'd; caller resets the heap
 * between forward passes.
 */

import type { Int4BlockWeight, PiiWasmExports, WeightTensorInfo } from "../backends/wasm.js";
import { attentionForward, type AttentionWeights, type AttentionConfig, type AttentionTables } from "./attention.js";
import { expertDispatch, type ExpertWeights, type ExpertConfig, type MultiThreadContext } from "./expert.js";
import type { KernelCall, WorkerScript } from "./mt-pool.js";

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
 *
 * Under MT, the matmul (the dominant cost — `T*numExperts*D` ops, ~1 ms
 * per layer) is partitioned across workers along T. The T*K topk +
 * softmax + invK pass is small enough to stay single-threaded.
 */
export async function routerForward(
  wasm: PiiWasmExports,
  hiddenF32Ptr: number,
  routingIdxPtr: number,
  routingScoresPtr: number,
  router: Int4BlockWeight,
  numExperts: number,
  K: number,
  T: number,
  D: number,
  mt?: MultiThreadContext,
): Promise<void> {
  // Logits in fp32 (upstream: `F.linear(hidden.float(), W.float(), b.float())`).
  const logitsPtr = wasm.alloc(T * numExperts * 4);
  if (mt && T >= mt.pool.numThreads * 2) {
    const N = mt.pool.numThreads;
    const scripts: WorkerScript[] = [];
    for (let w = 0; w < N; w++) {
      const tStart = Math.floor((w * T) / N);
      const tEnd = Math.floor(((w + 1) * T) / N);
      const tCount = tEnd - tStart;
      const calls: KernelCall[] = [];
      if (tCount > 0) {
        calls.push({
          kernel: "matmul_f32_x_int4block_out_f32",
          args: [
            hiddenF32Ptr + tStart * D * 4,
            router.int4.dataOffset, router.scales.dataOffset, router.zp.dataOffset,
            router.bias.dataOffset,
            logitsPtr + tStart * numExperts * 4,
            tCount, numExperts, D,
          ],
        });
      } else {
        calls.push({ kernel: "echo", args: [0] });
      }
      scripts.push(calls);
    }
    await mt.pool.run(scripts);
  } else {
    wasm.matmul_f32_x_int4block_out_f32(
      hiddenF32Ptr,
      router.int4.dataOffset, router.scales.dataOffset, router.zp.dataOffset,
      router.bias.dataOffset, logitsPtr,
      T, numExperts, D,
    );
  }
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

  // Pre-attention norm + widen, fused. `inputPtr` is the residual stream
  // for the attention add — attention reads only the f32 normed copy, so
  // the residual lives on inside the caller's input buffer with no
  // explicit copy.
  const hiddenF32Ptr = wasm.alloc(T * D * 4);
  if (hiddenF32Ptr === 0) throw new Error("blockForward: hidden f32 alloc OOM");
  wasm.rms_norm_fp16_to_f32(
    inputPtr, weights.inputLayernorm.dataOffset, hiddenF32Ptr,
    T, D, config.rmsNormEps,
  );

  const attnOutPtr = wasm.alloc(T * D * 2);
  await attentionForward(wasm, hiddenF32Ptr, attnOutPtr, weights.attn, config, tables, T, maskPtr, mt);

  // Post-attention add+norm+widen, fused. Writes the new residual stream
  // (fp16) and the f32 input the router/experts consume in one pass per
  // row. `inputPtr` carries the pre-attention residual1 directly.
  const residual2Ptr = wasm.alloc(T * D * 2);
  const normed2F32Ptr = wasm.alloc(T * D * 4);
  if (residual2Ptr === 0 || normed2F32Ptr === 0) {
    throw new Error("blockForward: post-attention scratch alloc OOM");
  }

  if (mt && T >= mt.pool.numThreads * 2) {
    const N = mt.pool.numThreads;
    const scripts: WorkerScript[] = [];
    for (let w = 0; w < N; w++) {
      const tStart = Math.floor((w * T) / N);
      const tEnd = Math.floor(((w + 1) * T) / N);
      const tCount = tEnd - tStart;
      const calls: KernelCall[] = [];
      if (tCount > 0) {
        const off2 = tStart * D * 2;
        const off4 = tStart * D * 4;
        calls.push({
          kernel: "add_rmsnorm_fp16_to_f32",
          args: [
            inputPtr + off2, attnOutPtr + off2, weights.postAttentionLayernorm.dataOffset,
            residual2Ptr + off2, normed2F32Ptr + off4,
            tCount, D, config.rmsNormEps,
          ],
        });
      } else {
        calls.push({ kernel: "echo", args: [0] });
      }
      scripts.push(calls);
    }
    await mt.pool.run(scripts);
  } else {
    wasm.add_rmsnorm_fp16_to_f32(
      inputPtr, attnOutPtr, weights.postAttentionLayernorm.dataOffset,
      residual2Ptr, normed2F32Ptr,
      T, D, config.rmsNormEps,
    );
  }

  // Router + expert dispatch (both read f32 hidden).
  const routingIdxPtr = wasm.alloc(T * K * 4);
  const routingScoresPtr = wasm.alloc(T * K * 4);
  await routerForward(
    wasm, normed2F32Ptr, routingIdxPtr, routingScoresPtr,
    weights.router, config.numExperts, K, T, D,
    mt,
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
