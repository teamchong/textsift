/**
 * Public types for @yourorg/pii-wasm.
 *
 * The public API is intentionally narrow and hides the following
 * internal complexity:
 *
 *   - Model download + OPFS caching
 *   - Tokenization (SentencePiece / BPE)
 *   - Attention-mask construction + chunking for long inputs
 *   - Forward pass (WebGPU or WASM backend)
 *   - Viterbi CRF decoding with loaded calibration biases
 *   - BIOES tag merging to character-level spans
 *   - Replacement-text application preserving offsets
 *
 * Consumers of the library see only `PrivacyFilter.create()`, `redact()`,
 * `detect()`, and `redactBatch()` — plus a small set of options and result
 * objects defined in this file.
 */

/**
 * The 8 canonical PII span categories emitted by `openai/privacy-filter`.
 * Matches the BIOES tag prefix in the upstream model's output schema.
 */
export type SpanLabel =
  | "account_number"
  | "private_address"
  | "private_email"
  | "private_person"
  | "private_phone"
  | "private_url"
  | "private_date"
  | "secret";

/** All known span labels as a tuple. */
export const ALL_SPAN_LABELS: readonly SpanLabel[] = [
  "account_number",
  "private_address",
  "private_email",
  "private_person",
  "private_phone",
  "private_url",
  "private_date",
  "secret",
] as const;

/** A single detected PII span with character-level offsets into the input text. */
export interface DetectedSpan {
  /** Which PII category this span belongs to. */
  label: SpanLabel;
  /** Start offset in the original input text (UTF-16 code units). */
  start: number;
  /** End offset (exclusive) in the original input text. */
  end: number;
  /** Substring of the original text, i.e. `input.slice(start, end)`. */
  text: string;
  /** The marker string used in the redacted output for this span. */
  marker: string;
  /** Model's confidence for this span (0..1). */
  confidence: number;
}

/** A successful redaction, returned from `redact()`. */
export interface RedactResult {
  /** Original input text, unchanged. */
  input: string;
  /** Input text with every detected span replaced by its marker. */
  redactedText: string;
  /** All spans the model detected, sorted by `start`. */
  spans: readonly DetectedSpan[];
  /** Count of spans per category (absent categories omitted). */
  summary: Readonly<Partial<Record<SpanLabel, number>>>;
  /**
   * True iff at least one span was detected. Cheap to check before
   * deciding whether to surface a "your input contains PII" warning.
   */
  containsPii: boolean;
  /**
   * Informational warning, populated only when the input was chunked
   * (> `maxChunkTokens`) and a span may straddle a chunk boundary.
   * Most callers can ignore this.
   */
  warning?: string;
}

/** Options passed at factory time via `PrivacyFilter.create(opts?)`. */
export interface CreateOptions {
  /**
   * Where to fetch the model weights. Defaults to the HuggingFace Hub
   * URL for `openai/privacy-filter`. Enterprise / air-gapped deploys
   * point this at their own mirror (R2, S3, on-prem CDN).
   */
  modelSource?: string;

  /**
   * Backend selection. `"auto"` picks WebGPU when `navigator.gpu`
   * reports a compatible adapter, WASM otherwise.
   */
  backend?: "auto" | "webgpu" | "wasm";

  /**
   * Quantization level for the model weights. Defaults to `"int4"`.
   * `"fp16"` is provided for accuracy debugging and should rarely
   * be needed.
   */
  quantization?: "int4" | "int8" | "fp16";

  /**
   * Called during the (one-time, per-origin) model download + warmup.
   * Not called on subsequent calls that hit the OPFS cache.
   */
  onProgress?: (event: ProgressEvent) => void;

  /**
   * Passed through to `fetch()` for weight downloads. Lets callers
   * abort a long load with an `AbortSignal`.
   */
  signal?: AbortSignal;

  /**
   * Override the default marker strategy. Default marker for a span
   * labelled `private_person` is `"[private_person]"`. Return `null`
   * to emit the original text unchanged for a given category.
   */
  markers?: MarkerStrategy;

  /**
   * Per-category allow list. When provided, spans outside this set
   * are detected but left unredacted in the output. Default: all
   * categories redacted.
   */
  enabledCategories?: readonly SpanLabel[];

  /**
   * Maximum tokens per chunk when the input exceeds the model's
   * context window. Defaults to 2048 (well under the 128k max but
   * keeps per-chunk latency bounded).
   */
  maxChunkTokens?: number;
}

/** Per-call options passed to `redact()` / `detect()`. */
export interface RedactOptions {
  /** Override the session's marker strategy for this call only. */
  markers?: MarkerStrategy;
  /** Override the session's enabled-categories list for this call only. */
  enabledCategories?: readonly SpanLabel[];
  /** AbortSignal for cancelling a long inference. */
  signal?: AbortSignal;
}

/**
 * Marker strategy: either a map of label → string (static per-category),
 * or a function that's called for each detected span and can return any
 * string (including an indexed variant like `"[PERSON_0]"`, useful when
 * you want to re-insert the original PII in the response later).
 *
 * Return value semantics:
 *   - `string`                 → use that string as the marker.
 *   - `null`                   → do not redact this span; leave original text.
 *   - `undefined` (from map)   → fall back to the default `"[<label>]"`.
 */
export type MarkerStrategy =
  | Readonly<Partial<Record<SpanLabel, string | null>>>
  | ((span: DetectedSpan, index: number) => string | null);

/** Progress events emitted during model load. */
export type ProgressEvent =
  | { stage: "download"; loaded: number; total: number; url: string }
  | { stage: "cache-hit"; total: number }
  | { stage: "compile"; backend: "webgpu" | "wasm" }
  | { stage: "warmup" }
  | { stage: "ready" };

/**
 * Result of a purely-detection call (`detect()` — no redacted text built).
 * Useful when the caller wants to highlight spans in a UI rather than
 * produce a clean redacted string.
 */
export interface DetectResult {
  input: string;
  spans: readonly DetectedSpan[];
  summary: Readonly<Partial<Record<SpanLabel, number>>>;
  containsPii: boolean;
  warning?: string;
}

/**
 * Error thrown when inference cannot proceed — model load failure,
 * backend unavailable, input too large after chunking, etc.
 */
export class PrivacyFilterError extends Error {
  constructor(
    message: string,
    public readonly code: PrivacyFilterErrorCode,
    public override readonly cause?: Error,
  ) {
    super(message);
    this.name = "PrivacyFilterError";
  }
}

export type PrivacyFilterErrorCode =
  | "MODEL_DOWNLOAD_FAILED"
  | "BACKEND_UNAVAILABLE"
  | "CALIBRATION_INVALID"
  | "INPUT_TOO_LARGE"
  | "ABORTED"
  | "INTERNAL";
