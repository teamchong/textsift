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
  /** Override for the `pii.wasm` module URL. Defaults to the bytes inlined into the JS bundle. */
  wasmModuleUrl?: string | URL;
}

export async function selectBackend(opts: SelectOptions): Promise<InferenceBackend> {
  if (opts.backend === "wasm") {
    const { WasmBackend } = await import("./wasm.js");
    return new WasmBackend({
      bundle: opts.bundle,
      quantization: opts.quantization,
      device: opts.device,
      wasmModuleUrl: opts.wasmModuleUrl,
    });
  }
  const { TransformersJsBackend } = await import("./transformers-js.js");
  return new TransformersJsBackend(opts);
}
