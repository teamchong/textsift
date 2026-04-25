/**
 * textsift-core — PII detection + redaction for browser, Node, and edge
 * runtimes. Lean (~30 KB gzipped, no transformers.js dependency).
 *
 * Public API surface:
 *
 *   const filter = await PrivacyFilter.create();
 *   const result = await filter.redact(text);
 *   const spans  = await filter.detect(text);
 *   const batch  = await filter.redactBatch([a,b,c]);
 *   filter.dispose();
 *
 * Backends bundled here: WebGPU (custom WGSL kernels) + WASM (custom
 * Zig + SIMD). The umbrella `textsift` package adds an auto fallback
 * to transformers.js via `BackendResolver`.
 */

export { PrivacyFilter, type BackendResolver } from "./privacy-filter.js";
export type { DetectStreamSession, DetectStreamOptions } from "./inference/stream.js";

// Advanced: direct backend access for tests + power users who want to
// skip the tokenizer / chunking pipeline. Most consumers should use
// `PrivacyFilter.create({ backend: "wasm" | "webgpu" })` instead.
export {
  WasmBackend,
  sharedMemorySupported,
  loadTextsift,
  loadTextsiftShared,
} from "./backends/wasm.js";
export { WebGpuBackend } from "./backends/webgpu.js";
export type { InferenceBackend, Logits, BackendConstructionOptions } from "./backends/abstract.js";

export { ModelLoader } from "./model/loader.js";
export type { LoadedModelBundle, ModelConfig, ModelLoaderOptions } from "./model/loader.js";

// Tokenizer — public so callers can count tokens, build their own
// chunking strategies, or run text encoding without going through the
// full PrivacyFilter pipeline. textsift-core's only tokenizer is the
// native o200k-style BPE; no transformers.js dependency.
export { Tokenizer } from "./model/tokenizer.js";
export type { EncodeResult, TokenizerLoadOptions } from "./model/tokenizer.js";

// Storage management — the model is large (~770 MB) and lives in
// OPFS. Apps that surface storage usage to users (e.g. an "edit
// preferences" page) can call these directly.
export { getCachedModelInfo, clearCachedModel } from "./model/opfs-fetch.js";
export type { CachedModelInfo } from "./model/opfs-fetch.js";

export type {
  CreateOptions,
  RedactOptions,
  RedactResult,
  DetectResult,
  DetectedSpan,
  SpanLabel,
  ProgressEvent,
  MarkerStrategy,
} from "./types.js";

export { ALL_SPAN_LABELS, PrivacyFilterError } from "./types.js";
export type { PrivacyFilterErrorCode } from "./types.js";
