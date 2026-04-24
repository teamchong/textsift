/**
 * @yourorg/pii-wasm — browser-native PII detection and redaction.
 *
 * Public API surface:
 *
 *   const filter = await PrivacyFilter.create();
 *   const result = await filter.redact(text);
 *   const spans  = await filter.detect(text);        // redaction skipped
 *   const batch  = await filter.redactBatch([a,b,c]);
 *   filter.dispose();
 *
 * Everything else (tokenization, chunking, Viterbi CRF, BIOES merging,
 * backend selection, OPFS caching) is internal and not exported from
 * the package root.
 */

export { PrivacyFilter } from "./privacy-filter.js";

// Advanced: direct backend access for tests + power users who want to
// skip the tokenizer / chunking pipeline. Most consumers should use
// `PrivacyFilter.create({ backend: "wasm", ... })` instead.
export { WasmBackend } from "./backends/wasm.js";
export { WebGpuBackend } from "./backends/webgpu.js";
export { ModelLoader } from "./model/loader.js";

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
