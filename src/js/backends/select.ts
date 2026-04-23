/**
 * Runtime backend selection.
 *
 * Stage 0 ships a single backend — transformers.js, running the quantized
 * ONNX model via ORT Web's WebGPU execution provider. Stage 1 adds a
 * Zig+WASM path; this module picks between them based on the
 * caller-declared backend.
 */

import type { InferenceBackend } from "./abstract.js";
import type { LoadedModelBundle } from "../model/loader.js";

export interface SelectOptions {
  quantization: "int4" | "int8" | "fp16";
  /** "auto" lets transformers.js pick the execution provider (WebGPU in browsers). */
  device: "auto" | "wasm" | "webgpu";
  bundle: LoadedModelBundle;
  /** Explicit backend selection. Defaults to the transformers.js path. */
  backend?: "auto" | "transformers-js" | "wasm";
  /** Required when `backend: "wasm"`. URL/path to the `pii-weights.bin` blob. */
  wasmWeightsUrl?: string | URL;
  /** Optional sha256 of the weight blob for integrity check. */
  wasmWeightsSha256?: string;
  /** Override for the `pii.wasm` module URL; defaults to a path sibling to the bundled JS. */
  wasmModuleUrl?: string | URL;
}

export async function selectBackend(opts: SelectOptions): Promise<InferenceBackend> {
  if (opts.backend === "wasm") {
    if (!opts.wasmWeightsUrl) {
      throw new Error(
        "selectBackend: backend=\"wasm\" requires `wasmWeightsUrl` pointing to a pii-weights.bin blob.",
      );
    }
    const { WasmBackend } = await import("./wasm.js");
    return new WasmBackend({
      ...opts,
      weightsUrl: opts.wasmWeightsUrl,
      weightsSha256: opts.wasmWeightsSha256,
      wasmModuleUrl: opts.wasmModuleUrl,
    });
  }
  const { TransformersJsBackend } = await import("./transformers-js.js");
  return new TransformersJsBackend(opts);
}
