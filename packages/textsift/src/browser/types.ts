/**
 * Public types for textsift.
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

/**
 * Span label is either one of the 8 model categories or a string the
 * caller has defined via a custom rule. The latter is opaque to
 * textsift — proxy code reads it and decides what to do.
 */
export type Label = SpanLabel | string;

/**
 * Severity tag attached to a custom rule. textsift doesn't act on
 * severity itself (it's not a guard, only a detector); proxy code
 * reads `severity` and decides whether to block, redact, log, or warn.
 * Borrowed from the codesift / linter-style severity vocabulary.
 */
export type RuleSeverity = "block" | "warn" | "track";

/** A single detected span with character-level offsets into the input text. */
export interface DetectedSpan {
  /**
   * Which category this span belongs to. For model spans this is one
   * of the 8 `SpanLabel` values; for custom-rule spans this is the
   * label the rule defined (any string).
   */
  label: Label;
  /**
   * Where the span came from. `"model"` when produced by
   * openai/privacy-filter; `"rule"` when produced by a custom rule.
   * Proxy code can use `source === "rule"` to apply rule-specific
   * policy (block / warn / track) per `severity`.
   */
  source: "model" | "rule";
  /**
   * Severity, populated only when `source === "rule"`. Carries the
   * rule's declared severity through the pipeline so callers don't
   * need to look it up by id.
   */
  severity?: RuleSeverity;
  /** Start offset in the original input text (UTF-16 code units). */
  start: number;
  /** End offset (exclusive) in the original input text. */
  end: number;
  /** Substring of the original text, i.e. `input.slice(start, end)`. */
  text: string;
  /** The marker string used in the redacted output for this span. */
  marker: string;
  /** Model's confidence for this span (0..1); `1.0` for rule spans (deterministic). */
  confidence: number;
}

/**
 * Custom detection rule. Rules run alongside the model's detection
 * pass; matches are merged into the same `DetectedSpan[]` output with
 * `source: "rule"` and the rule's `label`/`severity`.
 *
 * Two match shapes — pick whichever fits:
 *
 *   { label, pattern: RegExp }                 — every regex match becomes a span
 *   { label, match: (text) => Array<...> }    — arbitrary code returns spans
 *
 * Use cases beyond PII:
 *   - "block on `eval()` in code"
 *   - "warn on lazy-LLM phrases (TODO markers, hedging language)"
 *   - "track API keys / JWTs the model wasn't trained on"
 *   - "block prompts that contain prior conversation IDs"
 *
 * Rules don't replace the model — they augment it. Both run; results
 * are merged with overlap dedup (see `mergeRuleSpans`).
 */
export type Rule =
  | {
      /** Span label produced when this rule matches. Any string. */
      label: string;
      /** Severity for the proxy's block/warn/track decision. Default: "warn". */
      severity?: RuleSeverity;
      /** Replacement marker for `redact()`. Default: `[label]`. */
      marker?: string;
      /**
       * Regex matched against the full input text. Must be global
       * (`/g`) so `matchAll()` walks every match. Capture groups are
       * not used — the whole match is the span.
       */
      pattern: RegExp;
    }
  | {
      label: string;
      severity?: RuleSeverity;
      marker?: string;
      /**
       * Function that returns `{ start, end }` ranges for every
       * match in `text`. Useful when matching needs context outside
       * a single regex (e.g. proximity rules, "git checkout without
       * a fingerprint reference nearby").
       */
      match: (text: string) => Array<{ start: number; end: number }>;
    };

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
   * Backend selection.
   *   - `"auto"` (default): WebGPU in browsers when an adapter with
   *     `shader-f16` is available, falling back to the WASM backend
   *     everywhere else (Node, edge runtimes, browsers without WebGPU).
   *   - `"wasm"`: custom Zig+SIMD WASM backend. Multi-threaded when the
   *     page is cross-origin-isolated (COOP/COEP headers), single-thread
   *     otherwise.
   *   - `"webgpu"`: custom WGSL backend. Requires `shader-f16`; throws
   *     if the adapter can't enable it.
   */
  backend?: "auto" | "wasm" | "webgpu";

  /**
   * Override the URL of the `textsift.wasm` module. Defaults to a sibling of
   * the bundled JS (`./textsift.wasm`). Required when hosting the .wasm at
   * a different path than the bundled entry point.
   */
  wasmModuleUrl?: string;

  /**
   * Quantization level for the model weights. Defaults to `"int8"`, which
   * maps to `onnx/model_q4f16.onnx` (~772 MB) — the smallest ONNX export
   * in openai/privacy-filter that runs in a browser without OOM.  `"int4"`
   * maps to the same file.  `"fp16"` maps to `onnx/model_fp16.onnx` (~2 GB)
   * for maximum accuracy; only use it if memory permits.
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

  /**
   * Custom detection rules. Run alongside model detection; matches
   * merge into the same `DetectedSpan[]` with `source: "rule"`. See
   * the `Rule` type for the two supported shapes (regex vs function).
   *
   * Rules apply on every `detect()` / `redact()` call unless
   * overridden per-call via `RedactOptions.rules`.
   */
  rules?: readonly Rule[];

  /**
   * Built-in rule presets to enable. Currently supported:
   *   - `"secrets"` — high-precision detectors for credentials/API keys
   *     not in the model's training set (JWT, GitHub PAT, AWS, Slack,
   *     OpenAI/Anthropic/Google keys, Stripe, npm, PEM private keys).
   *
   * Presets and `rules` compose — both run, both contribute spans.
   */
  presets?: readonly string[];
}

/**
 * Per-column classification result from `classifyColumns()`. One
 * entry per column in the input table.
 */
export interface ColumnClassification {
  /** Column index in the input rows (0-based). */
  index: number;
  /** Header name, populated when `headers` or `headerRow` is provided. */
  header?: string;
  /**
   * Most frequently detected category in the sampled cells, or `null`
   * if no spans were detected at all. For mixed-content columns this
   * is a best-effort majority pick; the full distribution is in
   * `labelCounts`.
   */
  label: SpanLabel | string | null;
  /**
   * Fraction (0..1) of sampled cells that contained at least one
   * span of the chosen `label`. A confidence of 0.6 means 60% of
   * sampled cells matched. `0` when `label === null`.
   */
  confidence: number;
  /** Number of cells sampled (excludes empty / null cells). */
  samples: number;
  /** Full distribution of detected labels across the sampled cells. */
  labelCounts: Readonly<Record<string, number>>;
}

/** How `redactTable()` handles cells in PII-classified columns. */
export type RedactTableMode = "redact" | "synth" | "drop_column";

/** Options for `classifyColumns()`. */
export interface ClassifyTableOptions {
  /**
   * Max number of cells to sample per column. Default: 50. Sampling
   * is deterministic — first N non-empty cells starting from the top
   * (post-header). For wider audits, raise this; for fast first-pass
   * checks, lower it.
   */
  sampleSize?: number;
  /**
   * Set to `true` if the first row of `rows` contains column names.
   * Headers come back populated and that row is excluded from
   * sampling. Default: `false`.
   */
  headerRow?: boolean;
  /**
   * Explicit header names. Takes precedence over `headerRow`. Length
   * must match the number of columns in the first data row.
   */
  headers?: readonly string[];
  /** Custom-rule list applied to each sampled cell (same as `detect()`). */
  rules?: readonly Rule[];
  /** Built-in rule presets to enable for sampling. */
  presets?: readonly string[];
  /** AbortSignal for cancelling a long classify run. */
  signal?: AbortSignal;
}

/** Options for `redactTable()`. */
export interface RedactTableOptions extends ClassifyTableOptions {
  /**
   * How to handle PII-classified columns:
   *   - `"redact"` (default) — run `redact()` per cell to replace
   *     spans with markers. Uses the filter's `markers` strategy.
   *   - `"synth"` — same as redact but with `markerPresets.faker()`
   *     applied per-cell, even if the filter wasn't constructed
   *     with faker markers. Useful when most calls want plain
   *     redaction but tabular dumps want synthetic-looking data.
   *   - `"drop_column"` — omit the entire column from the output.
   *     Output rows have fewer columns than input rows.
   */
  mode?: RedactTableMode;
  /**
   * Skip the classify step by passing pre-computed classifications.
   * Useful when you've already classified once (for audit) and want
   * to redact without re-sampling.
   */
  classifications?: readonly ColumnClassification[];
  /** Per-cell marker strategy override (same shape as `RedactOptions.markers`). */
  markers?: MarkerStrategy;
  /** Per-call category allow-list (same as `RedactOptions.enabledCategories`). */
  enabledCategories?: readonly SpanLabel[];
}

/** Per-call options passed to `redact()` / `detect()`. */
export interface RedactOptions {
  /** Override the session's marker strategy for this call only. */
  markers?: MarkerStrategy;
  /** Override the session's enabled-categories list for this call only. */
  enabledCategories?: readonly SpanLabel[];
  /**
   * Override or extend the session's custom-rule list for this call
   * only. If set, replaces the session-level rules entirely (no
   * merging). To extend, splice them at the call site.
   */
  rules?: readonly Rule[];
  /**
   * Override the session's preset list for this call only. Same
   * semantics as the create-time option.
   */
  presets?: readonly string[];
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
