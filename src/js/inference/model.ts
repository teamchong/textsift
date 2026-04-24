/**
 * Full model forward composition.
 *
 *   h = embed_tokens(input_ids)
 *   for block in blocks:
 *     h = block_forward(h)
 *   h = final_layernorm(h)
 *   logits = classifier_head(h)
 *
 * Owns the ping-pong between two hidden-state buffers across blocks.
 * Uses the bump allocator for all scratch; caller resets after one
 * forward pass.
 */

import type { Int4BlockWeight, PiiWasmExports, WeightTensorInfo } from "../backends/wasm.js";
import { blockForward, type BlockWeights, type BlockConfig } from "./block.js";
import { buildRopeTables, type YarnRopeConfig } from "./rope.js";

export interface ModelWeights {
  /** Embed table: int4-block32 packed W [vocab, D] + fp16 scales + uint4 zp. */
  embedInt4: WeightTensorInfo;
  embedScales: WeightTensorInfo;
  embedZp: WeightTensorInfo;
  /** One `BlockWeights` per transformer layer. */
  blocks: BlockWeights[];
  /** `model.norm.weight`: fp16 [D]. */
  finalLayernorm: WeightTensorInfo;
  /**
   * Classifier head: int4-block32 W [num_classes, D] + fp16 scales + uint4 zp.
   * ONNX has no bias for `score`; loader synthesises a zero fp16 bias so the
   * kernel signature stays uniform.
   */
  classifier: Int4BlockWeight;
}

export interface ModelConfig extends BlockConfig {
  /** Effective vocab size reachable via `embedTokens` (= vocab in config, or truncated for tests). */
  vocabSize: number;
  /** Number of classes the classifier head produces (33 for this model). */
  numClasses: number;
  rope: YarnRopeConfig;
}

/**
 * Run one forward pass. `inputIdsPtr` points to an `i32 [T]` buffer of
 * token IDs. `logitsPtr` must be a `fp16 [T, num_classes]` buffer owned
 * by the caller. All other scratch is bump-alloc'd.
 */
export function modelForward(
  wasm: PiiWasmExports,
  inputIdsPtr: number,
  logitsPtr: number,
  weights: ModelWeights,
  config: ModelConfig,
  T: number,
  /** Optional `u8 [T]` mask pointer (1=valid, 0=padding). 0 = no mask. */
  maskPtr: number = 0,
): void {
  const D = config.hiddenSize;

  // Precompute RoPE tables once for this seq len. All blocks share them.
  const { cos, sin } = buildRopeTables(config.rope, T);
  const cosBytes = new Uint8Array(cos.buffer, cos.byteOffset, cos.byteLength);
  const sinBytes = new Uint8Array(sin.buffer, sin.byteOffset, sin.byteLength);
  const cosPtr = wasm.alloc(cosBytes.byteLength);
  new Uint8Array(wasm.memory.buffer, cosPtr, cosBytes.byteLength).set(cosBytes);
  const sinPtr = wasm.alloc(sinBytes.byteLength);
  new Uint8Array(wasm.memory.buffer, sinPtr, sinBytes.byteLength).set(sinBytes);
  const tables = { ropeCosPtr: cosPtr, ropeSinPtr: sinPtr };

  // Embedding → h0. Ping-pong between h0 and h1 across blocks.
  const h0Ptr = wasm.alloc(T * D * 2);
  const h1Ptr = wasm.alloc(T * D * 2);
  wasm.embed_lookup_int4(
    weights.embedInt4.dataOffset, weights.embedScales.dataOffset, weights.embedZp.dataOffset,
    inputIdsPtr, h0Ptr, T, config.vocabSize, D,
  );

  let srcPtr = h0Ptr;
  let dstPtr = h1Ptr;
  for (const blockWeights of weights.blocks) {
    blockForward(wasm, srcPtr, dstPtr, blockWeights, tables, config, T, maskPtr);
    const tmp = srcPtr;
    srcPtr = dstPtr;
    dstPtr = tmp;
  }

  // Final rmsnorm → overwrite `dstPtr` (was tmp scratch).
  wasm.rms_norm(srcPtr, weights.finalLayernorm.dataOffset, dstPtr, T, D, config.rmsNormEps);

  // Classifier head: [T, D] × [num_classes, D]^T + [num_classes] → [T, num_classes].
  // Pre-widen the final-norm output so the int4 matmul runs against f32.
  const classifierXPtr = wasm.alloc(T * D * 4);
  wasm.convert_fp16_to_f32(dstPtr, classifierXPtr, T * D);
  wasm.matmul_f32_x_int4block(
    classifierXPtr,
    weights.classifier.int4.dataOffset,
    weights.classifier.scales.dataOffset,
    weights.classifier.zp.dataOffset,
    weights.classifier.bias.dataOffset,
    logitsPtr, T, config.numClasses, D,
  );
}
