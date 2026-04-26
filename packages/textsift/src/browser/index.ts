/**
 * textsift/browser — PII detection + redaction for browser, Node, and edge
 * runtimes. ~76 KB gzipped, zero runtime dependencies.
 *
 * Public API surface:
 *
 *   const filter = await PrivacyFilter.create();
 *   const result = await filter.redact(text);
 *   const spans  = await filter.detect(text);
 *   const batch  = await filter.redactBatch([a,b,c]);
 *   filter.dispose();
 *
 * Backends: WebGPU (custom WGSL kernels) + WASM (custom Zig + SIMD).
 */

export { PrivacyFilter, type BackendResolver } from "./privacy-filter.js";
export type {
  DetectStreamHandle,
  DetectStreamOptions,
  RedactStreamHandle,
  RedactStreamOptions,
} from "./inference/stream.js";

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
// full PrivacyFilter pipeline. The only tokenizer is the native
// o200k-style BPE built into this package.
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
  Label,
  ProgressEvent,
  MarkerStrategy,
  Rule,
  RuleSeverity,
} from "./types.js";

// Built-in rule presets. `secretRules` is the curated array; the
// `presets: ["secrets"]` shorthand on `create()` resolves to this.
export { secretRules, RULE_PRESETS } from "./inference/rule-presets.js";
export type { RulePresetName } from "./inference/rule-presets.js";

// Built-in marker strategies. `markerPresets.faker()` swaps the
// default `[label]` markers for realistic-looking fake values.
export { markerPresets } from "./inference/marker-presets.js";
export type { MarkerPresetName } from "./inference/marker-presets.js";

export { ALL_SPAN_LABELS, PrivacyFilterError } from "./types.js";
export type { PrivacyFilterErrorCode } from "./types.js";
