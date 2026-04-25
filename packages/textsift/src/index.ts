/**
 * textsift — umbrella package.
 *
 * Re-exports everything from textsift-core and adds a transformers.js
 * fallback backend so `PrivacyFilter.create()` with no arguments works
 * even on browsers without WebGPU and Node runtimes that don't have
 * WASM SIMD.
 *
 *   const filter = await PrivacyFilter.create();
 *
 * In textsift-core that picks `webgpu` (browser) or `wasm` (everywhere
 * else). Here, the same call additionally falls through to
 * transformers.js when neither path is viable — matching the behaviour
 * of the pre-split package for backwards compat.
 *
 * Bundles ~228 KB gzipped (vs ~76 KB for textsift-core alone).
 * If you don't need the auto fallback, depend on `textsift-core`
 * directly.
 */

import {
  PrivacyFilter as CorePrivacyFilter,
  type CreateOptions,
  type BackendResolver,
} from "textsift-core";
import type { InferenceBackend } from "textsift-core";
import { TransformersJsBackend } from "./transformers-js-backend.js";

/**
 * Wrapper class that injects a transformers.js fallback into the
 * `"auto"` decision. All public methods (redact, detect, redactBatch,
 * dispose) come from the core class via prototype delegation —
 * `PrivacyFilter` inherits from `CorePrivacyFilter`.
 */
export class PrivacyFilter extends CorePrivacyFilter {
  /**
   * Factory: same signature as `textsift-core`'s, but the `"auto"`
   * backend resolves through transformers.js when WebGPU is
   * unavailable. Backwards-compatible with the pre-split package.
   */
  static override async create(opts: CreateOptions = {}): Promise<PrivacyFilter> {
    const resolver: BackendResolver = {
      async resolveAuto({ bundle, hasWebGPU, isNode, quantization }) {
        // In browsers with WebGPU, let core pick its native backend
        // (faster than transformers.js). In Node, also let core pick
        // (the custom WASM backend has SIMD, transformers.js doesn't).
        // We only kick in when neither works — typically browsers
        // without WebGPU, where the core would otherwise fall back to
        // single-thread WASM. transformers.js + ORT Web is roughly
        // comparable there and provides one less runtime surface.
        if (hasWebGPU) return null;
        if (isNode) return null;
        const backend = new TransformersJsBackend({
          bundle,
          quantization,
          device: "auto",
        });
        return backend as InferenceBackend;
      },
    };
    const instance = await CorePrivacyFilter.createWithResolver(opts, resolver);
    // Cast through unknown — we know the runtime instance is fine
    // because `createWithResolver` returns a CorePrivacyFilter that
    // we're presenting under the umbrella's class identity.
    return instance as unknown as PrivacyFilter;
  }
}

export { TransformersJsBackend } from "./transformers-js-backend.js";

// Re-exports from textsift-core. Same surface as if the consumer had
// installed textsift-core directly.
export {
  WasmBackend,
  WebGpuBackend,
  ModelLoader,
  Tokenizer,
  ALL_SPAN_LABELS,
  PrivacyFilterError,
  sharedMemorySupported,
  loadTextsift,
  loadTextsiftShared,
  getCachedModelInfo,
  clearCachedModel,
} from "textsift-core";

export type {
  CreateOptions,
  RedactOptions,
  RedactResult,
  DetectResult,
  DetectedSpan,
  SpanLabel,
  ProgressEvent,
  MarkerStrategy,
  PrivacyFilterErrorCode,
  InferenceBackend,
  Logits,
  BackendConstructionOptions,
  LoadedModelBundle,
  ModelConfig,
  ModelLoaderOptions,
  EncodeResult,
  TokenizerLoadOptions,
  CachedModelInfo,
  BackendResolver,
  DetectStreamHandle,
  DetectStreamOptions,
  RedactStreamHandle,
  RedactStreamOptions,
} from "textsift-core";
