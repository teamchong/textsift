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

import type { PiiWasmExports, WeightTensorInfo } from "../backends/wasm.js";
import { blockForward, type BlockWeights, type BlockConfig } from "./block.js";
import { buildRopeTables, type YarnRopeConfig } from "./rope.js";

export interface ModelWeights {
  /** `model.embed_tokens.weight`: bf16 [vocab, D]. */
  embedTokens: WeightTensorInfo;
  /** One `BlockWeights` per transformer layer. */
  blocks: BlockWeights[];
  /** `model.norm.weight`: bf16 [D]. */
  finalLayernorm: WeightTensorInfo;
  /** `score.weight`: bf16 [num_classes, D]. */
  classifierW: WeightTensorInfo;
  /** `score.bias`: bf16 [num_classes]. */
  classifierB: WeightTensorInfo;
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
 * token IDs. `logitsPtr` must be a `bf16 [T, num_classes]` buffer owned
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
  wasm.embed_lookup(weights.embedTokens.dataOffset, inputIdsPtr, h0Ptr, T, config.vocabSize, D);

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
  wasm.matmul_bf16(
    dstPtr, weights.classifierW.dataOffset, weights.classifierB.dataOffset,
    logitsPtr, T, config.numClasses, D,
  );
}
