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
import type { KernelCall, WorkerScript } from "./mt-pool.js";
import type { MultiThreadContext } from "./expert.js";

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

export async function attentionForward(
  wasm: PiiWasmExports,
  /**
   * f32 [T, D]; the caller has already widened hidden via the fused
   * `rms_norm_fp16_to_f32` so the Q/K/V int4 matmuls can run directly
   * against an f32 input without an extra widening pass.
   */
  hiddenF32Ptr: number,
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
  mt?: MultiThreadContext,
): Promise<void> {
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
  const attnF32Ptr = wasm.alloc(T * Hq * hd * 4);
  if (qPtr === 0 || kPtr === 0 || vPtr === 0 || attnOutPtr === 0 || attnF32Ptr === 0) {
    throw new Error("attentionForward: scratch alloc OOM");
  }

  // Q/K/V projections. Single-threaded by default; with an MtPool the
  // T axis is partitioned across workers — each worker calls all three
  // matmuls on its T slice. Same kernel, just sliced X-input and out
  // pointers; no kernel-signature changes needed.
  if (mt && T >= mt.pool.numThreads * 2) {
    const N = mt.pool.numThreads;
    const scripts: WorkerScript[] = [];
    for (let w = 0; w < N; w++) {
      const tStart = Math.floor((w * T) / N);
      const tEnd = Math.floor(((w + 1) * T) / N);
      const tCount = tEnd - tStart;
      const calls: KernelCall[] = [];
      if (tCount > 0) {
        const xOff = hiddenF32Ptr + tStart * D * 4;
        const qOff = qPtr + tStart * Hq * hd * 2;
        const kOff = kPtr + tStart * Hkv * hd * 2;
        const vOff = vPtr + tStart * Hkv * hd * 2;
        calls.push(
          { kernel: "matmul_f32_x_int4block",
            args: [xOff, weights.qProj.int4.dataOffset, weights.qProj.scales.dataOffset, weights.qProj.zp.dataOffset, weights.qProj.bias.dataOffset, qOff, tCount, Hq * hd, D] },
          { kernel: "matmul_f32_x_int4block",
            args: [xOff, weights.kProj.int4.dataOffset, weights.kProj.scales.dataOffset, weights.kProj.zp.dataOffset, weights.kProj.bias.dataOffset, kOff, tCount, Hkv * hd, D] },
          { kernel: "matmul_f32_x_int4block",
            args: [xOff, weights.vProj.int4.dataOffset, weights.vProj.scales.dataOffset, weights.vProj.zp.dataOffset, weights.vProj.bias.dataOffset, vOff, tCount, Hkv * hd, D] },
          // cos/sin tables are [T_full, head_dim/2] fp16. RoPE reads
          // `cos_row = cos_ptr + t * (head_dim/2)` and the kernel's t
          // loops 0..tCount, so the cos/sin pointer must be advanced
          // by tStart rows or workers see global token 0's angles
          // instead of their slice's angles.
          { kernel: "rope_apply",
            args: [qOff, tables.ropeCosPtr + tStart * (hd / 2) * 2, tables.ropeSinPtr + tStart * (hd / 2) * 2, tCount, Hq, hd] },
          { kernel: "rope_apply",
            args: [kOff, tables.ropeCosPtr + tStart * (hd / 2) * 2, tables.ropeSinPtr + tStart * (hd / 2) * 2, tCount, Hkv, hd] },
        );
      } else {
        calls.push({ kernel: "echo", args: [0] });
      }
      scripts.push(calls);
    }
    await mt.pool.run(scripts);
  } else {
    wasm.matmul_f32_x_int4block(
      hiddenF32Ptr,
      weights.qProj.int4.dataOffset, weights.qProj.scales.dataOffset, weights.qProj.zp.dataOffset,
      weights.qProj.bias.dataOffset, qPtr, T, Hq * hd, D,
    );
    wasm.matmul_f32_x_int4block(
      hiddenF32Ptr,
      weights.kProj.int4.dataOffset, weights.kProj.scales.dataOffset, weights.kProj.zp.dataOffset,
      weights.kProj.bias.dataOffset, kPtr, T, Hkv * hd, D,
    );
    wasm.matmul_f32_x_int4block(
      hiddenF32Ptr,
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
  }

  // Banded attention. Sliding-window attention crosses arbitrary T
  // boundaries (window=128, T<=128 means every query attends to every
  // key) so a per-T worker split would force expensive cross-worker
  // key fetches. The natural parallelism axis is H_q — query heads
  // are independent and the K/V tensors are read-only across workers.
  // `banded_attention_partial` takes (h_q_start, h_q_count); we
  // partition by GQA group to keep each worker's K/V access slice
  // contiguous when the math allows.
  if (mt && Hq >= mt.pool.numThreads * 2) {
    const N = mt.pool.numThreads;
    const scripts: WorkerScript[] = [];
    for (let w = 0; w < N; w++) {
      const hStart = Math.floor((w * Hq) / N);
      const hEnd = Math.floor(((w + 1) * Hq) / N);
      const hCount = hEnd - hStart;
      const calls: KernelCall[] = [];
      if (hCount > 0) {
        calls.push({
          kernel: "banded_attention_partial",
          args: [qPtr, kPtr, vPtr, weights.sinks.dataOffset, maskPtr, attnOutPtr,
                 T, Hq, Hkv, hd, config.slidingWindow,
                 hStart, hCount],
        });
      } else {
        calls.push({ kernel: "echo", args: [0] });
      }
      scripts.push(calls);
    }
    await mt.pool.run(scripts);
  } else {
    wasm.banded_attention(
      qPtr, kPtr, vPtr, weights.sinks.dataOffset, maskPtr, attnOutPtr,
      T, Hq, Hkv, hd, config.slidingWindow,
    );
  }

  // O projection: pre-widen attn_out, then f32-input int4 matmul.
  // Same partition pattern as Q/K/V when MT is on.
  wasm.convert_fp16_to_f32(attnOutPtr, attnF32Ptr, T * Hq * hd);
  if (mt && T >= mt.pool.numThreads * 2) {
    const N = mt.pool.numThreads;
    const scripts: WorkerScript[] = [];
    for (let w = 0; w < N; w++) {
      const tStart = Math.floor((w * T) / N);
      const tEnd = Math.floor(((w + 1) * T) / N);
      const tCount = tEnd - tStart;
      const calls: KernelCall[] = [];
      if (tCount > 0) {
        const xOff = attnF32Ptr + tStart * Hq * hd * 4;
        const oOff = outputPtr + tStart * D * 2;
        calls.push({
          kernel: "matmul_f32_x_int4block",
          args: [xOff, weights.oProj.int4.dataOffset, weights.oProj.scales.dataOffset, weights.oProj.zp.dataOffset, weights.oProj.bias.dataOffset, oOff, tCount, D, Hq * hd],
        });
      } else {
        calls.push({ kernel: "echo", args: [0] });
      }
      scripts.push(calls);
    }
    await mt.pool.run(scripts);
  } else {
    wasm.matmul_f32_x_int4block(
      attnF32Ptr,
      weights.oProj.int4.dataOffset, weights.oProj.scales.dataOffset, weights.oProj.zp.dataOffset,
      weights.oProj.bias.dataOffset, outputPtr, T, D, Hq * hd,
    );
  }
}
