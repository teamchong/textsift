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
