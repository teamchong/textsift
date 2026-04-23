/**
 * Runtime backend selection.
 *
 * Preference ladder:
 *   "auto"    → try WebGPU → fall back to WASM.
 *   "webgpu"  → WebGPU only; throw if unavailable.
 *   "wasm"    → WASM only, universal support.
 *
 * Backends are loaded lazily — importing `select.ts` doesn't pull in
 * WGSL or the Zig WASM unless that path is actually chosen.
 */

import type { InferenceBackend } from "./abstract.js";
import type { LoadedModelBundle } from "../model/loader.js";
import { PrivacyFilterError } from "../types.js";

export interface SelectOptions {
  preference: "auto" | "webgpu" | "wasm";
  quantization: "int4" | "int8" | "fp16";
  bundle: LoadedModelBundle;
}

export async function selectBackend(
  opts: SelectOptions,
): Promise<InferenceBackend> {
  if (opts.preference === "webgpu") {
    if (!(await isWebGPUViable())) {
      throw new PrivacyFilterError(
        "WebGPU backend was requested but `navigator.gpu` is unavailable or the adapter is missing required features.",
        "BACKEND_UNAVAILABLE",
      );
    }
    return createWebGPUBackend(opts);
  }

  if (opts.preference === "wasm") {
    return createWasmBackend(opts);
  }

  // auto
  if (await isWebGPUViable()) {
    try {
      return await createWebGPUBackend(opts);
    } catch (e) {
      // WebGPU reported as viable but pipeline build failed — fall back.
      console.warn("[pii-wasm] WebGPU init failed, falling back to WASM:", (e as Error).message);
    }
  }
  return createWasmBackend(opts);
}

async function isWebGPUViable(): Promise<boolean> {
  const gpu = (globalThis as { navigator?: { gpu?: unknown } }).navigator?.gpu;
  if (!gpu) return false;
  try {
    const adapter = await (gpu as { requestAdapter(): Promise<{ features: Set<string> } | null> })
      .requestAdapter();
    if (!adapter) return false;
    return adapter.features.has("shader-f16");
  } catch {
    return false;
  }
}

async function createWebGPUBackend(opts: SelectOptions): Promise<InferenceBackend> {
  const { WebGPUBackend } = await import("./webgpu.js");
  return new WebGPUBackend(opts);
}

async function createWasmBackend(opts: SelectOptions): Promise<InferenceBackend> {
  // Stage 0 routes through transformers.js while the custom Zig+WASM
  // path is under development. Once the Zig build is wired, this import
  // switches to ./wasm.js.
  const { TransformersJsBackend } = await import("./transformers-js.js");
  return new TransformersJsBackend(opts);
}
