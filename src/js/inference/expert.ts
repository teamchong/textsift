/**
 * MoE expert dispatch (expert-major).
 *
 * Mirrors `OpenAIPrivacyFilterExperts.forward` exactly, with f32
 * intermediates throughout (`_keep_in_fp32_modules` indicates the
 * model is numerically sensitive to accumulation dtype here):
 *
 *   for each expert e with ≥1 token routed to it:
 *     x_e           = gather(hidden, token_idx_e)                 bf16
 *     gate_up       = matmul(x_e, gate_up_proj[e]) + bias_gu[e]   f32 ← bf16 x × int4 W
 *     glu, up       = apply_gate(gate_up)                         f32 × f32
 *     out           = matmul(glu, down_proj[e])   + bias_d[e]     f32 ← f32 x × int4 W
 *     acc[token_idx_e] += routing_scores_e * out                  f32 scatter
 *
 *   final = bf16(acc * num_experts_per_tok)
 *
 * Expert weights are stored in the blob as:
 *   gate_up_proj: int4-block32-sym, shape [E, 2*d_ff, d_model] (transposed
 *     from upstream's [E, d_model, 2*d_ff] so our x @ W.T matmul fits).
 *   down_proj:   int4-block32-sym, shape [E, d_model, d_ff] (transposed
 *     from upstream's [E, d_ff, d_model]).
 *   biases:       bf16 [E, N].
 *
 * The packed tensor blob lays out `[E*N, D/2]` int4 bytes first, then
 * `[E*N, D/32]` fp16 scales. Per-expert pointers are computed via
 * `expertSlicePointers` below.
 */

import type { PiiWasmExports, WeightTensorInfo } from "../backends/wasm.js";

export interface ExpertWeights {
  /** int4-block32-sym, shape [E, 2*d_ff, d_model] (transposed). */
  gateUp: WeightTensorInfo;
  /** bf16, shape [E, 2*d_ff]. */
  gateUpBias: WeightTensorInfo;
  /** int4-block32-sym, shape [E, d_model, d_ff] (transposed). */
  down: WeightTensorInfo;
  /** bf16, shape [E, d_model]. */
  downBias: WeightTensorInfo;
}

export interface ExpertConfig {
  hiddenSize: number;          // d_model (= 640)
  intermediateSize: number;    // d_ff (= 640)
  numExperts: number;          // E total in the blob (may be < model.num_local_experts in tests)
  numExpertsPerTok: number;    // K top-k (= 4)
}

function expertSlicePointers(
  tensor: WeightTensorInfo, expertIdx: number,
): { int4Ptr: number; scalesPtr: number } {
  if (tensor.shape.length !== 3) {
    throw new Error(`expert slice expects 3D tensor, got shape ${tensor.shape}`);
  }
  const E = tensor.shape[0]!;
  const N = tensor.shape[1]!;
  const D = tensor.shape[2]!;
  if (D % 32 !== 0) throw new Error(`D=${D} not divisible by 32`);
  const int4Stride = (N * D) >>> 1;            // bytes per expert for int4 data
  const scalesStride = ((N * D) >>> 5) * 2;    // bytes per expert for fp16 scales
  const totalInt4 = E * int4Stride;
  return {
    int4Ptr: tensor.dataOffset + expertIdx * int4Stride,
    scalesPtr: tensor.dataOffset + totalInt4 + expertIdx * scalesStride,
  };
}

function expertBiasPointer(tensor: WeightTensorInfo, expertIdx: number): number {
  // Bias is a 2-D bf16 tensor, [E, N]. Per-expert stride: N * 2 bytes.
  if (tensor.shape.length !== 2) {
    throw new Error(`expert bias expects 2D tensor, got shape ${tensor.shape}`);
  }
  const N = tensor.shape[1]!;
  return tensor.dataOffset + expertIdx * N * 2;
}

export function expertDispatch(
  wasm: PiiWasmExports,
  hiddenPtr: number,
  outputPtr: number,
  routingIndicesPtr: number,
  routingScoresPtr: number,
  weights: ExpertWeights,
  config: ExpertConfig,
  T: number,
): void {
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
  const xGatheredPtr = wasm.alloc(maxM * D * 2);           // bf16 [m, D]
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

    // Gather hidden rows for this expert's tokens.
    wasm.gather_bf16(hiddenPtr, tokIdxPtr, xGatheredPtr, m, D);

    // gate_up = x @ W_gu^T + bias_gu   (bf16 x × int4 W → f32)
    const guSlice = expertSlicePointers(weights.gateUp, e);
    const guBiasPtr = expertBiasPointer(weights.gateUpBias, e);
    wasm.matmul_bf16_x_int4block_out_f32(
      xGatheredPtr, guSlice.int4Ptr, guSlice.scalesPtr, guBiasPtr,
      gateUpPtr, m, 2 * dff, D,
    );

    // SwiGLU-with-clamp: [m, 2*dff] → [m, dff]
    wasm.swiglu_clamp_f32(gateUpPtr, gluPtr, m, dff);

    // out = glu @ W_d^T + bias_d   (f32 x × int4 W → f32)
    const dSlice = expertSlicePointers(weights.down, e);
    const dBiasPtr = expertBiasPointer(weights.downBias, e);
    wasm.matmul_f32_x_int4block_out_f32(
      gluPtr, dSlice.int4Ptr, dSlice.scalesPtr, dBiasPtr,
      outF32Ptr, m, D, dff,
    );

    // acc[token_idx[i]] += weights[i] * out_f32[i]
    wasm.scatter_add_weighted_f32(accPtr, outF32Ptr, tokIdxPtr, weightsPtr, m, D);
  }

  // final output = bf16(acc * num_experts_per_tok)
  wasm.cast_f32_to_bf16_scaled(accPtr, outputPtr, T * D, K);
}
