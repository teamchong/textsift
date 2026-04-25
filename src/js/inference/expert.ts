/**
 * MoE expert dispatch (expert-major).
 *
 * Mirrors `OpenAIPrivacyFilterExperts.forward` exactly, with f32
 * intermediates throughout (`_keep_in_fp32_modules` indicates the
 * model is numerically sensitive to accumulation dtype here):
 *
 *   for each expert e with ≥1 token routed to it:
 *     x_e           = gather(hidden, token_idx_e)                 fp16
 *     gate_up       = matmul(x_e, gate_up_proj[e]) + bias_gu[e]   f32 ← fp16 x × int4 W
 *     glu, up       = apply_gate(gate_up)                         f32 × f32
 *     out           = matmul(glu, down_proj[e])   + bias_d[e]     f32 ← f32 x × int4 W
 *     acc[token_idx_e] += routing_scores_e * out                  f32 scatter
 *
 *   final = fp16(acc * num_experts_per_tok)
 *
 * Expert weights are stored in the blob as:
 *   gate_up_proj: int4-block32-sym, shape [E, 2*d_ff, d_model] (transposed
 *     from upstream's [E, d_model, 2*d_ff] so our x @ W.T matmul fits).
 *   down_proj:   int4-block32-sym, shape [E, d_model, d_ff] (transposed
 *     from upstream's [E, d_ff, d_model]).
 *   biases:       fp16 [E, N].
 *
 * The packed tensor blob lays out `[E*N, D/2]` int4 bytes first, then
 * `[E*N, D/32]` fp16 scales. Per-expert pointers are computed via
 * `expertSlicePointers` below.
 */

import type { TextsiftExports, WeightTensorInfo } from "../backends/wasm.js";
import type { MtPool } from "./mt-pool.js";
import {
  expertDispatchParallel,
  type WorkerScratch,
} from "./mt-expert.js";

export interface ExpertWeights {
  /** uint4 packed, shape [E, 2*d_ff, d_model/2]. */
  gateUpInt4: WeightTensorInfo;
  /** fp16 scales, shape [E, 2*d_ff, d_model/32]. */
  gateUpScales: WeightTensorInfo;
  /** uint4 zero-points packed 2/byte, shape [E, 2*d_ff, ceil(d_model/32/2)]. */
  gateUpZp: WeightTensorInfo;
  /** fp16 (or fp16) bias, shape [E, 2*d_ff]. */
  gateUpBias: WeightTensorInfo;
  /** uint4 packed, shape [E, d_model, d_ff/2]. */
  downInt4: WeightTensorInfo;
  /** fp16 scales, shape [E, d_model, d_ff/32]. */
  downScales: WeightTensorInfo;
  /** uint4 zero-points, shape [E, d_model, ceil(d_ff/32/2)]. */
  downZp: WeightTensorInfo;
  /** fp16 (or fp16) bias, shape [E, d_model]. */
  downBias: WeightTensorInfo;
}

export interface ExpertConfig {
  hiddenSize: number;          // d_model (= 640)
  intermediateSize: number;    // d_ff (= 640)
  numExperts: number;          // E total in the blob (may be < model.num_local_experts in tests)
  numExpertsPerTok: number;    // K top-k (= 4)
}

function expertSlicePointers(
  int4Tensor: WeightTensorInfo,
  scalesTensor: WeightTensorInfo,
  zpTensor: WeightTensorInfo,
  expertIdx: number,
  N: number,
  D: number,
): { int4Ptr: number; scalesPtr: number; zpPtr: number } {
  if (D % 32 !== 0) throw new Error(`D=${D} not divisible by 32`);
  const nBlocks = D >>> 5;
  const int4Stride = (N * D) >>> 1;
  const scalesStride = N * nBlocks * 2;
  const zpStride = N * ((nBlocks + 1) >>> 1);
  return {
    int4Ptr: int4Tensor.dataOffset + expertIdx * int4Stride,
    scalesPtr: scalesTensor.dataOffset + expertIdx * scalesStride,
    zpPtr: zpTensor.dataOffset + expertIdx * zpStride,
  };
}

function expertBiasPointer(tensor: WeightTensorInfo, expertIdx: number): number {
  // Bias is a 2-D fp16 tensor, [E, N]. Per-expert stride: N * 2 bytes.
  if (tensor.shape.length !== 2) {
    throw new Error(`expert bias expects 2D tensor, got shape ${tensor.shape}`);
  }
  const N = tensor.shape[1]!;
  return tensor.dataOffset + expertIdx * N * 2;
}

/**
 * Multi-thread plumbing the dispatcher accepts when running under
 * `MtPool`. When set, expert dispatch is partitioned across workers
 * by token slice; each worker runs its own expert-major loop on
 * disjoint output rows.
 */
export interface MultiThreadContext {
  pool: MtPool;
  workerScratch: readonly WorkerScratch[];
  /** Pre-allocated f32 [T, D] accumulator in shared memory. */
  accPtr: number;
}

export async function expertDispatch(
  wasm: TextsiftExports,
  /** f32 [T, D]; the caller has already widened hidden from fp16. */
  hiddenF32Ptr: number,
  outputPtr: number,
  routingIndicesPtr: number,
  routingScoresPtr: number,
  weights: ExpertWeights,
  config: ExpertConfig,
  T: number,
  mt?: MultiThreadContext,
): Promise<void> {
  if (mt) {
    return expertDispatchParallel(
      mt.pool, wasm, hiddenF32Ptr, outputPtr,
      routingIndicesPtr, routingScoresPtr,
      weights, config, T, mt.workerScratch, mt.accPtr,
    );
  }
  const { hiddenSize: D, intermediateSize: dff, numExperts: E, numExpertsPerTok: K } = config;

  // Build inverse routing: for each expert, list of (token, k-position) pairs.
  // Views are lifted BEFORE any alloc() call so they don't detach.
  const routingIndices = new Int32Array(wasm.memory.buffer, routingIndicesPtr, T * K);
  const routingScores = new Float32Array(wasm.memory.buffer, routingScoresPtr, T * K);
  const tokensPerExpert: number[][] = [];
  const kPosPerExpert: number[][] = [];
  const scoresPerExpert: number[][] = [];
  for (let e = 0; e < E; e++) {
    tokensPerExpert.push([]);
    kPosPerExpert.push([]);
    scoresPerExpert.push([]);
  }
  for (let t = 0; t < T; t++) {
    for (let k = 0; k < K; k++) {
      const e = routingIndices[t * K + k]!;
      if (e < 0 || e >= E) continue;
      tokensPerExpert[e]!.push(t);
      kPosPerExpert[e]!.push(k);
      scoresPerExpert[e]!.push(routingScores[t * K + k]!);
    }
  }

  // f32 accumulator across all tokens. Zero-init once.
  const accPtr = wasm.alloc(T * D * 4);
  if (accPtr === 0) throw new Error("expertDispatch: accumulator alloc OOM");
  wasm.zero_f32(accPtr, T * D);

  // Scratch sizing: max tokens any single expert could see is `T * K` (all
  // tokens × all k-positions). Allocate that upper bound once and reuse.
  const maxM = T * K;
  const xGatheredPtr = wasm.alloc(maxM * D * 4);           // f32 [m, D]
  const gateUpPtr = wasm.alloc(maxM * 2 * dff * 4);        // f32 [m, 2*dff]
  const gluPtr = wasm.alloc(maxM * dff * 4);               // f32 [m, dff]
  const outF32Ptr = wasm.alloc(maxM * D * 4);              // f32 [m, D]
  const tokIdxPtr = wasm.alloc(maxM * 4);                  // i32 [m]
  const weightsPtr = wasm.alloc(maxM * 4);                 // f32 [m]
  if (xGatheredPtr === 0 || gateUpPtr === 0 || gluPtr === 0 ||
      outF32Ptr === 0 || tokIdxPtr === 0 || weightsPtr === 0) {
    throw new Error("expertDispatch: scratch alloc OOM");
  }

  for (let e = 0; e < E; e++) {
    const m = tokensPerExpert[e]!.length;
    if (m === 0) continue;

    // Load token indices + weights into wasm memory for this expert.
    const idxView = new Int32Array(wasm.memory.buffer, tokIdxPtr, m);
    const wView = new Float32Array(wasm.memory.buffer, weightsPtr, m);
    for (let i = 0; i < m; i++) {
      idxView[i] = tokensPerExpert[e]![i]!;
      wView[i] = scoresPerExpert[e]![i]!;
    }

    // Hidden is already f32 (converted once per layer by the caller);
    // gather the rows this expert owns and feed straight into the
    // f32-input int4 matmul. No per-expert fp16 widening.
    wasm.gather_f32(hiddenF32Ptr, tokIdxPtr, xGatheredPtr, m, D);

    // gate_up = x @ W_gu^T + bias_gu   (f32 x × int4 W → f32)
    const guSlice = expertSlicePointers(
      weights.gateUpInt4, weights.gateUpScales, weights.gateUpZp, e, 2 * dff, D,
    );
    const guBiasPtr = expertBiasPointer(weights.gateUpBias, e);
    wasm.matmul_f32_x_int4block_out_f32(
      xGatheredPtr, guSlice.int4Ptr, guSlice.scalesPtr, guSlice.zpPtr, guBiasPtr,
      gateUpPtr, m, 2 * dff, D,
    );

    // SwiGLU-with-clamp: [m, 2*dff] → [m, dff]
    wasm.swiglu_clamp_f32(gateUpPtr, gluPtr, m, dff);

    // out = glu @ W_d^T + bias_d   (f32 x × int4 W → f32)
    const dSlice = expertSlicePointers(
      weights.downInt4, weights.downScales, weights.downZp, e, D, dff,
    );
    const dBiasPtr = expertBiasPointer(weights.downBias, e);
    wasm.matmul_f32_x_int4block_out_f32(
      gluPtr, dSlice.int4Ptr, dSlice.scalesPtr, dSlice.zpPtr, dBiasPtr,
      outF32Ptr, m, D, dff,
    );

    // acc[token_idx[i]] += weights[i] * out_f32[i]
    wasm.scatter_add_weighted_f32(accPtr, outF32Ptr, tokIdxPtr, weightsPtr, m, D);
  }

  // final output = fp16(acc * num_experts_per_tok)
  wasm.cast_f32_to_fp16_scaled(accPtr, outputPtr, T * D, K);
}
