/**
 * Backend selection for textsift/browser.
 *
 * Two backends ship: WebGPU (WGSL kernels) and WASM (Zig + SIMD).
 * Callers can also inject a custom `InferenceBackend` via
 * `PrivacyFilter.create()`'s `backend` option (when set to an
 * instance, the selector is bypassed) — used by the bench for the
 * transformers.js comparator, which is not part of the package.
 */

import type { InferenceBackend } from "./abstract.js";
import type { LoadedModelBundle } from "../model/loader.js";

export interface SelectOptions {
  quantization: "int4" | "int8" | "fp16";
  /** Execution device for the WebGPU backend. */
  device: "auto" | "wasm" | "webgpu";
  bundle: LoadedModelBundle;
  /**
   * Which backend to instantiate. Caller-supplied custom backends
   * (e.g., the bench's transformers.js comparator) are wired through
   * `PrivacyFilter.create({ backend: <instance> })` directly and
   * never reach this selector.
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
