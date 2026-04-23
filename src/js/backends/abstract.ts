/**
 * Backend contract shared by WebGPU, WASM, and transformers.js paths.
 *
 * Each backend implementation owns:
 *   - Per-backend model loading (uploading weights to GPU buffers or
 *     instantiating a WASM module against a `SharedArrayBuffer`).
 *   - One `forward(tokenIds, attentionMask)` call per inference.
 *   - Lifecycle: `warmup()` after construction, `dispose()` on release.
 *
 * Backends do NOT see:
 *   - Chunking (done upstream in `privacy-filter.ts`).
 *   - Viterbi decoding (done in JS downstream of backends).
 *   - BIOES merging / redaction application (also JS downstream).
 *
 * This separation means all three backends produce identical logits
 * for identical inputs. That invariant is verified by the cross-backend
 * conformance test suite.
 */

import type { LoadedModelBundle } from "../model/loader.js";

/**
 * Output of one forward pass: a 2D logit array, shape
 * `[sequenceLength, numClasses]`, f32. numClasses is 33 for the
 * upstream model: 1 background + 8 spans × 4 BIOES tags.
 */
export interface Logits {
  /** Flat row-major buffer of length `sequenceLength * numClasses`. */
  data: Float32Array;
  sequenceLength: number;
  numClasses: number;
}

export interface InferenceBackend {
  /** A debug name used in progress events + error messages. */
  readonly name: "webgpu" | "wasm" | "transformers-js";

  /** Run the model once, compiling pipelines if not already done. */
  warmup(): Promise<void>;

  /**
   * Single forward pass for one chunk.
   * @param tokenIds  Int32Array of length N (tokenized input).
   * @param attentionMask Uint8Array of length N (1 = real token, 0 = padding).
   */
  forward(tokenIds: Int32Array, attentionMask: Uint8Array): Promise<Logits>;

  /** Release GPU buffers / WASM memory. Safe to call multiple times. */
  dispose(): void;
}

/** Common constructor argument for all backend implementations. */
export interface BackendConstructionOptions {
  bundle: LoadedModelBundle;
  quantization: "int4" | "int8" | "fp16";
}
