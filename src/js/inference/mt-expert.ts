/**
 * Parallel MoE expert dispatch using the MtPool worker pool.
 *
 * Mirrors `expertDispatch` in `expert.ts` but partitions tokens
 * across workers. Each worker runs the standard expert-major
 * dispatch loop on its disjoint token slice, writing to non-
 * overlapping rows of the shared `acc` buffer. No atomics needed —
 * partition guarantees disjoint output regions.
 *
 * Per-worker scratch buffers (xGather, gateUp, glu, outF32, tokIdx,
 * weights) live in shared memory at base offsets pre-allocated by
 * the caller. Each worker only touches its assigned slot.
 */

import type { TextsiftExports, WeightTensorInfo } from "../backends/wasm.js";
import type { ExpertWeights, ExpertConfig } from "./expert.js";
import type { MtPool, KernelCall, WorkerScript } from "./mt-pool.js";

/**
 * Per-worker scratch slot pointers. Sized for `maxMperWorker` rows of
 * gathered work — i.e. `(T / numThreads) * numExpertsPerTok` upper
 * bound on rows any one worker sees.
 */
export interface WorkerScratch {
  xGatheredPtr: number;
  gateUpPtr: number;
  gluPtr: number;
  outF32Ptr: number;
  tokIdxPtr: number;
  weightsPtr: number;
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
  const N = tensor.shape[1]!;
  return tensor.dataOffset + expertIdx * N * 2;
}

/**
 * Build per-worker scripts for one expert dispatch invocation. Caller
 * supplies the full routing buffers; we partition tokens by index range
 * across the pool.
 */
export async function expertDispatchParallel(
  pool: MtPool,
  wasm: TextsiftExports,
  hiddenF32Ptr: number,
  outputPtr: number,
  routingIndicesPtr: number,
  routingScoresPtr: number,
  weights: ExpertWeights,
  config: ExpertConfig,
  T: number,
  workerScratch: readonly WorkerScratch[],
  accPtr: number,
): Promise<void> {
  const { hiddenSize: D, intermediateSize: dff, numExperts: E, numExpertsPerTok: K } = config;
  const N = pool.numThreads;
  if (workerScratch.length !== N) {
    throw new Error(`expertDispatchParallel: need ${N} scratch slots, got ${workerScratch.length}`);
  }

  // Lift routing views before any alloc-style activity.
  const routingIndices = new Int32Array(wasm.memory.buffer, routingIndicesPtr, T * K);
  const routingScores = new Float32Array(wasm.memory.buffer, routingScoresPtr, T * K);

  // Main zeros the entire accumulator once, then dispatches workers.
  // The pool's release fence on the WASM-memory SAB before postMessage
  // makes these zero stores visible to every worker before they start
  // their scatter sequence.
  wasm.zero_f32(accPtr, T * D);

  // Build per-worker schedules. Each worker handles a contiguous token
  // range and runs its own expert-major dispatch over that slice.
  const scripts: WorkerScript[] = [];
  for (let w = 0; w < N; w++) {
    const tStart = Math.floor((w * T) / N);
    const tEnd = Math.floor(((w + 1) * T) / N);
    const tCount = tEnd - tStart;
    const calls: KernelCall[] = [];

    if (tCount > 0) {
      // Per-worker scratch slot.
      const slot = workerScratch[w]!;

      // Invert routing for this slice (JS-side bookkeeping).
      const tokensPerExpert: number[][] = [];
      const scoresPerExpert: number[][] = [];
      for (let e = 0; e < E; e++) {
        tokensPerExpert.push([]);
        scoresPerExpert.push([]);
      }
      for (let t = tStart; t < tEnd; t++) {
        for (let k = 0; k < K; k++) {
          const e = routingIndices[t * K + k]!;
          if (e < 0 || e >= E) continue;
          tokensPerExpert[e]!.push(t);
          scoresPerExpert[e]!.push(routingScores[t * K + k]!);
        }
      }

      // For each expert with at least one token: gather → gate_up →
      // swiglu → down → scatter_add. Stage indices + weights in
      // per-worker scratch from the main thread. Explicit length on
      // the TypedArray view so we don't accidentally rely on the
      // length default (which behaves quirkily on SAB-backed buffers
      // in some runtimes).
      const stageCap = tCount * K;
      const idxArr = new Int32Array(wasm.memory.buffer, slot.tokIdxPtr, stageCap);
      const wArr = new Float32Array(wasm.memory.buffer, slot.weightsPtr, stageCap);
      let stageOffset = 0;
      const expertOffsets: { e: number; off: number; m: number }[] = [];

      for (let e = 0; e < E; e++) {
        const m = tokensPerExpert[e]!.length;
        if (m === 0) continue;
        for (let i = 0; i < m; i++) {
          idxArr[stageOffset + i] = tokensPerExpert[e]![i]!;
          wArr[stageOffset + i] = scoresPerExpert[e]![i]!;
        }
        expertOffsets.push({ e, off: stageOffset, m });
        stageOffset += m;
      }

      for (const { e, off, m } of expertOffsets) {
        const idxPtr = slot.tokIdxPtr + off * 4;
        const wPtr   = slot.weightsPtr + off * 4;
        const xGPtr  = slot.xGatheredPtr; // overwritten per expert
        const guPtr  = slot.gateUpPtr;
        const glPtr  = slot.gluPtr;
        const oPtr   = slot.outF32Ptr;

        const guSlice = expertSlicePointers(
          weights.gateUpInt4, weights.gateUpScales, weights.gateUpZp, e, 2 * dff, D,
        );
        const guBias = expertBiasPointer(weights.gateUpBias, e);
        const dSlice = expertSlicePointers(
          weights.downInt4, weights.downScales, weights.downZp, e, D, dff,
        );
        const dBias = expertBiasPointer(weights.downBias, e);

        calls.push(
          { kernel: "gather_f32", args: [hiddenF32Ptr, idxPtr, xGPtr, m, D] },
          { kernel: "matmul_f32_x_int4block_out_f32",
            args: [xGPtr, guSlice.int4Ptr, guSlice.scalesPtr, guSlice.zpPtr, guBias, guPtr, m, 2 * dff, D] },
          { kernel: "swiglu_clamp_f32", args: [guPtr, glPtr, m, dff] },
          { kernel: "matmul_f32_x_int4block_out_f32",
            args: [glPtr, dSlice.int4Ptr, dSlice.scalesPtr, dSlice.zpPtr, dBias, oPtr, m, D, dff] },
          { kernel: "scatter_add_weighted_f32", args: [accPtr, oPtr, idxPtr, wPtr, m, D] },
        );
      }

      // Worker writes its slice of the final fp16 moe_out — keeps
      // every read/write of acc inside one WASM instance, sequencing
      // the accumulate→cast pair via straight-line WASM execution
      // rather than relying on cross-instance shared-memory ordering.
      calls.push({
        kernel: "cast_f32_to_fp16_scaled",
        args: [accPtr + tStart * D * 4, outputPtr + tStart * D * 2, tCount * D, K],
      });
    }

    if (calls.length === 0) {
      // Worker has nothing to do — give it a no-op so the pool's
      // fixed-fanout Atomics counter gets its increment.
      calls.push({ kernel: "echo", args: [0] });
    }
    scripts.push(calls);
  }

  await pool.run(scripts);
}
