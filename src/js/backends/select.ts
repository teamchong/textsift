/**
 * Runtime backend selection.
 *
 * Stage 0 ships a single backend — transformers.js, running the quantized
 * ONNX model via ORT Web's WebGPU execution provider.  Stage 1 will add a
 * Zig+WASM path and this module picks between them.
 */

import type { InferenceBackend } from "./abstract.js";
import type { LoadedModelBundle } from "../model/loader.js";

export interface SelectOptions {
  quantization: "int4" | "int8" | "fp16";
  /** "auto" lets transformers.js pick the execution provider (WebGPU in browsers). */
  device: "auto" | "wasm" | "webgpu";
  bundle: LoadedModelBundle;
}

export async function selectBackend(opts: SelectOptions): Promise<InferenceBackend> {
  const { TransformersJsBackend } = await import("./transformers-js.js");
  return new TransformersJsBackend(opts);
}
