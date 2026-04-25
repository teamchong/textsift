/**
 * Backend selection for textsift-core.
 *
 * textsift-core ships only the custom backends — WebGPU (WGSL kernels)
 * and WASM (Zig + SIMD). The umbrella `textsift` package layers an
 * additional transformers.js backend on top by injecting its own
 * `InferenceBackend` directly via `PrivacyFilter.create()`'s `backend`
 * option (when set to a backend instance, the selector is bypassed).
 */

import type { InferenceBackend } from "./abstract.js";
import type { LoadedModelBundle } from "../model/loader.js";

export interface SelectOptions {
  quantization: "int4" | "int8" | "fp16";
  /** Execution device for the WebGPU backend. */
  device: "auto" | "wasm" | "webgpu";
  bundle: LoadedModelBundle;
  /**
   * Which backend to instantiate. textsift-core knows `"webgpu"` and
   * `"wasm"`. The umbrella package handles its own backends before
   * calling here.
   */
  backend: "wasm" | "webgpu";
  /** Override for the `textsift.wasm` module URL. Defaults to the bytes inlined into the JS bundle. */
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
  const { WasmBackend } = await import("./wasm.js");
  return new WasmBackend({
    bundle: opts.bundle,
    quantization: opts.quantization,
    device: opts.device,
    wasmModuleUrl: opts.wasmModuleUrl,
  });
}
