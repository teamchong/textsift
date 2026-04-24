/**
 * Runtime backend selection.
 *
 * Stage 0 ships the transformers.js backend (ORT Web WebGPU EP). Stage 1
 * adds a custom Zig+WASM path. Stage 2 adds a custom WGSL path. All three
 * consume the same `onnx/model_q4f16.onnx` + `.onnx_data` and produce the
 * same logit shape; the selection is purely an execution-engine choice.
 */

import type { InferenceBackend } from "./abstract.js";
import type { LoadedModelBundle } from "../model/loader.js";

export interface SelectOptions {
  quantization: "int4" | "int8" | "fp16";
  /** "auto" lets transformers.js pick the execution provider (WebGPU in browsers). */
  device: "auto" | "wasm" | "webgpu";
  bundle: LoadedModelBundle;
  /** Explicit backend selection. Defaults to the transformers.js path. */
  backend?: "auto" | "transformers-js" | "wasm" | "webgpu";
  /** Override for the `pii.wasm` module URL. Defaults to the bytes inlined into the JS bundle. */
  wasmModuleUrl?: string | URL;
}

export async function selectBackend(opts: SelectOptions): Promise<InferenceBackend> {
  if (opts.backend === "webgpu") {
    const { WebGpuBackend } = await import("./webgpu.js");
    return new WebGpuBackend({
      bundle: opts.bundle,
      quantization: opts.quantization,
      device: opts.device,
    });
  }
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
