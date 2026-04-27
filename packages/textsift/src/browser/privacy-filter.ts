/**
 * `PrivacyFilter` — the clean API surface textsift/browser exports.
 *
 * All internal machinery lives behind this class:
 *   - Model weights: downloaded + cached by ModelLoader / OPFS.
 *   - Tokenizer: native o200k-style BPE, no third-party runtime dep.
 *   - Backend: WebGPU (custom WGSL) or WASM (custom Zig + SIMD128).
 *   - Viterbi CRF decoder: constructed from the calibration artifact
 *     shipped alongside the weights.
 *   - Chunking: inputs over `maxChunkTokens` are split with a sliding
 *     window and re-merged at span level.
 *   - Redaction applicator: character-level placement with marker
 *     strategy resolution.
 *   - Rule engine: regex / match-fn rules merged with model spans.
 */

import {
  ALL_SPAN_LABELS,
  type ClassifyTableOptions,
  type ColumnClassification,
  type CreateOptions,
  type DetectResult,
  type DetectedSpan,
  PrivacyFilterError,
  type RedactOptions,
  type RedactResult,
  type RedactTableOptions,
  type Rule,
  type SpanLabel,
} from "./types.js";
import type { InferenceBackend } from "./backends/abstract.js";
import { selectBackend } from "./backends/select.js";
import { ModelLoader, type LoadedModelBundle } from "./model/loader.js";
import { Tokenizer } from "./model/tokenizer.js";
import { ViterbiDecoder } from "./inference/viterbi.js";
import { loadCalibration } from "./model/calibration.js";
import { chunkInput, mergeChunkResults, type Chunk } from "./inference/chunking.js";
import { bioesToSpans } from "./inference/spans.js";
import { applyRedaction } from "./inference/redact.js";
import { runRules, mergeRuleSpans } from "./inference/rules.js";
import { classifyColumns as classifyColumnsImpl, redactTable as redactTableImpl } from "./inference/table.js";
import { resolvePresets } from "./inference/rule-presets.js";
import {
  streamDetect,
  streamRedact,
  type DetectStreamHandle,
  type DetectStreamOptions,
  type RedactStreamHandle,
  type RedactStreamOptions,
} from "./inference/stream.js";

/** Internal ready state. */
type FilterState =
  | { kind: "uninitialised" }
  | { kind: "loading"; promise: Promise<void> }
  | {
      kind: "ready";
      backend: InferenceBackend;
      tokenizer: Tokenizer;
      viterbi: ViterbiDecoder;
    }
  | { kind: "disposed" };

/**
 * Per-create extension hook for callers that want to override the
 * `"auto"` backend decision (e.g., the bench injects a custom
 * transformers.js comparator). Most callers never set this — they
 * let `PrivacyFilter.create()` pick WebGPU when available, WASM
 * otherwise.
 */
export interface BackendResolver {
  /**
   * Called when `opts.backend === "auto"`. Receives the loaded bundle
   * and the runtime capability flags. Return:
   *   - an `InferenceBackend` instance to use it directly, OR
   *   - `null` to fall back to the built-in choice
   *     (`"webgpu"` if `hasWebGPU`, else `"wasm"`).
   */
  resolveAuto(args: {
    bundle: LoadedModelBundle;
    hasWebGPU: boolean;
    isNode: boolean;
  }): Promise<InferenceBackend | null>;
}

export class PrivacyFilter {
  private state: FilterState = { kind: "uninitialised" };
  private queue: Promise<unknown> = Promise.resolve();
  protected readonly opts: CreateOptions;
  protected readonly backendResolver: BackendResolver | null;

  /** Private constructor; use `PrivacyFilter.create()`. */
  protected constructor(opts: CreateOptions, backendResolver: BackendResolver | null) {
    this.opts = opts;
    this.backendResolver = backendResolver;
  }

  /**
   * Factory: downloads the model (or hits OPFS cache), selects a backend,
   * and warms up the inference pipeline. Idempotent across concurrent
   * `create()` calls with the same options.
   */
  static async create(opts: CreateOptions = {}): Promise<PrivacyFilter> {
    const instance = new PrivacyFilter(opts, null);
    await instance.ensureReady();
    return instance;
  }

  /**
   * Advanced factory: injects a custom backend resolver (see
   * `BackendResolver`). Used by the bench to force a comparator
   * backend; not part of the everyday API.
   */
  static async createWithResolver(
    opts: CreateOptions,
    resolver: BackendResolver,
  ): Promise<PrivacyFilter> {
    const instance = new PrivacyFilter(opts, resolver);
    await instance.ensureReady();
    return instance;
  }

  /**
   * Redact PII in `input`. Two call shapes, picked by the type of
   * `input`:
   *
   *   // (1) Batch — pass a string, await the result.
   *   const result = await filter.redact("Hi John, my email is x@y.com");
   *   result.redactedText;
   *
   *   // (2) Streaming — pass an AsyncIterable<string>. Returns sync.
   *   //     Three surfaces on the handle: textStream (yields safe-to
   *   //     -emit redacted text fragments), spanStream (yields stable
   *   //     spans), result (resolves to the full RedactResult).
   *   const handle = filter.redact(llmStream);
   *   for await (const piece of handle.textStream) downstream.write(piece);
   *
   * Cost: batch is one detection + one redaction pass. Streaming is
   * O(N) over the input stream — same trailing-window logic as
   * detect(stream), with text held back inside the safety margin
   * until the trailing edge advances past it.
   */
  redact(text: string, opts?: RedactOptions): Promise<RedactResult>;
  redact(input: AsyncIterable<string>, opts?: RedactStreamOptions): RedactStreamHandle;
  redact(
    input: string | AsyncIterable<string>,
    opts: RedactOptions | RedactStreamOptions = {},
  ): Promise<RedactResult> | RedactStreamHandle {
    if (typeof input === "string") {
      return this.redactOne(input, opts as RedactOptions);
    }
    const merged: RedactStreamOptions = {
      enabledCategories:
        (opts as RedactStreamOptions).enabledCategories ?? this.opts.enabledCategories,
      markers: (opts as RedactStreamOptions).markers ?? this.opts.markers,
      rules: resolveRules(
        (opts as RedactStreamOptions).rules ?? this.opts.rules,
        (opts as RedactStreamOptions).presets ?? this.opts.presets,
      ),
      windowTokens: (opts as RedactStreamOptions).windowTokens,
      safetyMarginTokens: (opts as RedactStreamOptions).safetyMarginTokens,
      signal: (opts as RedactStreamOptions).signal,
    };
    return streamRedact(
      this.ensureReady().then((r) => ({
        backend: r.backend,
        tokenizer: r.tokenizer,
        viterbi: r.viterbi,
      })),
      input,
      merged,
    );
  }

  private async redactOne(text: string, callOpts: RedactOptions): Promise<RedactResult> {
    const ready = await this.ensureReady();
    return this.enqueue(async () => {
      const spans = await this.runDetection(ready, text, callOpts);
      const enabled = callOpts.enabledCategories ?? this.opts.enabledCategories ?? ALL_SPAN_LABELS;
      const enabledSet = new Set<SpanLabel>(enabled);
      const markers = callOpts.markers ?? this.opts.markers;
      const { redactedText, applied } = applyRedaction(text, spans, enabledSet, markers);
      return {
        input: text,
        redactedText,
        spans: applied,
        summary: buildSummary(applied),
        containsPii: applied.length > 0,
      };
    });
  }

  /**
   * Detect PII in `input`. Two call shapes, picked by the type of
   * `input`:
   *
   *   // (1) Batch — pass a string, await the result.
   *   const result = await filter.detect("Hi John, my email is x@y.com");
   *   result.spans;          // DetectedSpan[]
   *
   *   // (2) Streaming — pass an AsyncIterable<string> (e.g. an LLM
   *   //     output stream). Returns sync; iterate spans as they
   *   //     become detectable, OR await `.result` for the final
   *   //     DetectResult shape, OR both.
   *   const handle = filter.detect(llmStream);
   *   for await (const span of handle.spanStream) {
   *     if (span.label === "secret") abort();
   *   }
   *   const final = await handle.result;
   *
   * Cost: batch is O(T) inference. Streaming is O(N) total over a
   * stream of N tokens — each chunk arrival runs inference on a
   * trailing window, not the whole buffer.
   */
  detect(text: string, opts?: RedactOptions): Promise<DetectResult>;
  detect(input: AsyncIterable<string>, opts?: DetectStreamOptions): DetectStreamHandle;
  detect(
    input: string | AsyncIterable<string>,
    opts: RedactOptions | DetectStreamOptions = {},
  ): Promise<DetectResult> | DetectStreamHandle {
    if (typeof input === "string") {
      return this.detectBatch(input, opts as RedactOptions);
    }
    const merged: DetectStreamOptions = {
      enabledCategories:
        (opts as DetectStreamOptions).enabledCategories ?? this.opts.enabledCategories,
      rules: resolveRules(
        (opts as DetectStreamOptions).rules ?? this.opts.rules,
        (opts as DetectStreamOptions).presets ?? this.opts.presets,
      ),
      windowTokens: (opts as DetectStreamOptions).windowTokens,
      safetyMarginTokens: (opts as DetectStreamOptions).safetyMarginTokens,
      signal: (opts as DetectStreamOptions).signal,
    };
    return streamDetect(
      this.ensureReady().then((r) => ({
        backend: r.backend,
        tokenizer: r.tokenizer,
        viterbi: r.viterbi,
      })),
      input,
      merged,
    );
  }

  private async detectBatch(text: string, callOpts: RedactOptions): Promise<DetectResult> {
    const ready = await this.ensureReady();
    return this.enqueue(async () => {
      const spans = await this.runDetection(ready, text, callOpts);
      return {
        input: text,
        spans,
        summary: buildSummary(spans),
        containsPii: spans.length > 0,
      };
    });
  }

  /**
   * Batch form: processes inputs serially in-order (the backend is
   * single-threaded within this instance; for real parallelism create
   * multiple `PrivacyFilter` instances).
   */
  async redactBatch(inputs: readonly string[], opts?: RedactOptions): Promise<RedactResult[]> {
    const results: RedactResult[] = [];
    for (const input of inputs) {
      results.push(await this.redact(input, opts));
    }
    return results;
  }

  /**
   * Classify each column of a tabular dataset by sampling cells and
   * running per-cell `detect()`. Returns one entry per column with
   * the most-frequent detected label, a confidence (fraction of
   * sampled cells matching that label), and the full label
   * distribution.
   *
   * Use case: GDPR right-to-be-forgotten audits, pre-export data
   * checks, knowing which columns of a 50-column CSV need encryption
   * before shipping to a vendor.
   *
   * ```ts
   * const cols = await filter.classifyColumns([
   *   ["alice@example.com", "Alice Carter", "100"],
   *   ["bob@example.com",   "Bob Davis",    "250"],
   * ], { sampleSize: 50, headers: ["email", "name", "amount"] });
   * // → [
   * //     { index: 0, header: "email",  label: "private_email",  confidence: 1.0, ... },
   * //     { index: 1, header: "name",   label: "private_person", confidence: 1.0, ... },
   * //     { index: 2, header: "amount", label: null,             confidence: 0,   ... },
   * //   ]
   * ```
   */
  async classifyColumns(
    rows: readonly (readonly string[])[],
    opts: ClassifyTableOptions = {},
  ): Promise<ColumnClassification[]> {
    return classifyColumnsImpl(
      (text, callOpts) => this.detect(text, callOpts) as Promise<DetectResult>,
      rows,
      opts,
    );
  }

  /**
   * Redact a tabular dataset. Classifies columns first (or uses the
   * `classifications` you pass to skip that step), then per-cell
   * applies one of three modes to PII columns:
   *
   *   - `"redact"` (default) — replace spans with markers using the
   *     filter's `markers` strategy, per-cell.
   *   - `"synth"` — same but with `markerPresets.faker()` applied
   *     across the table (so consistency holds across rows).
   *   - `"drop_column"` — omit the column entirely from output.
   *
   * Output rows are fresh `string[]` arrays. Non-PII columns pass
   * through unchanged. If `headerRow` was set, the header row is
   * emitted first in the output.
   *
   * ```ts
   * const clean = await filter.redactTable(rows, {
   *   headerRow: true,
   *   mode: "synth",   // realistic fakes for downstream test fixtures
   * });
   * ```
   */
  async redactTable(
    rows: readonly (readonly string[])[],
    opts: RedactTableOptions = {},
  ): Promise<string[][]> {
    return redactTableImpl(
      (text, callOpts) => this.detect(text, callOpts) as Promise<DetectResult>,
      (text, callOpts) => this.redact(text, callOpts) as Promise<RedactResult>,
      this.opts.markers,
      rows,
      opts,
    );
  }


  /**
   * Release GPU / WASM resources. Further calls throw. Safe to call
   * multiple times.
   */
  dispose(): void {
    if (this.state.kind === "ready") {
      this.state.backend.dispose();
    }
    this.state = { kind: "disposed" };
  }

  /**
   * Symbol.dispose support — lets callers use the TS 5.2+ `using`
   * declaration for automatic cleanup at scope exit:
   *
   * ```ts
   * using filter = await PrivacyFilter.create();
   * await filter.redact(text);
   * // filter.dispose() called automatically when this scope ends
   * ```
   */
  [Symbol.dispose](): void {
    this.dispose();
  }

  // --------------------------------------------------------------
  // Internals
  // --------------------------------------------------------------

  private async ensureReady(): Promise<
    Extract<FilterState, { kind: "ready" }>
  > {
    if (this.state.kind === "ready") return this.state;
    if (this.state.kind === "disposed") {
      throw new PrivacyFilterError("filter was disposed", "INTERNAL");
    }
    if (this.state.kind === "loading") {
      await this.state.promise;
      const after = this.state as FilterState;
      if (after.kind !== "ready") {
        throw new PrivacyFilterError("filter failed to initialise", "INTERNAL");
      }
      return after;
    }

    const loadPromise = this.doLoad();
    this.state = { kind: "loading", promise: loadPromise };
    await loadPromise;
    const after = this.state as FilterState;
    if (after.kind !== "ready") {
      throw new PrivacyFilterError("filter failed to initialise", "INTERNAL");
    }
    return after;
  }

  private async doLoad(): Promise<void> {
    const progress = this.opts.onProgress;

    try {
      const loader = new ModelLoader({
        source: this.opts.modelSource ?? DEFAULT_MODEL_SOURCE,
        signal: this.opts.signal,
        onProgress: progress,
      });
      const bundle = await loader.load();
      const calibration = loadCalibration(bundle.calibrationJson);
      const tokenizer = await Tokenizer.fromBundle(bundle, {
        signal: this.opts.signal,
        onProgress: progress,
      });

      // Backend selection: explicit `backend` wins; `"auto"` (or
      // unspecified) picks the fastest path available in this
      // environment. WebGPU when an adapter with `shader-f16` is
      // available; otherwise the WASM path. The umbrella `textsift`
      // package can override the auto fallback via `backendResolver`.
      const requested = this.opts.backend ?? "auto";
      const isNode =
        typeof process !== "undefined" &&
        !!(process as { versions?: { node?: string } }).versions?.node;
      const hasWebGPU =
        !isNode &&
        typeof navigator !== "undefined" &&
        !!(navigator as { gpu?: unknown }).gpu;

      let backend: InferenceBackend | null = null;

      if (requested === "auto" && this.backendResolver) {
        backend = await this.backendResolver.resolveAuto({
          bundle,
          hasWebGPU,
          isNode,
        });
      }

      if (!backend) {
        const chosen: "webgpu" | "wasm" =
          requested === "webgpu" ? "webgpu"
          : requested === "wasm" ? "wasm"
          : hasWebGPU ? "webgpu"
          : "wasm";
        progress?.({ stage: "compile", backend: chosen });
        backend = await selectBackend({
          device: hasWebGPU ? "webgpu" : "auto",
          bundle,
          backend: chosen,
          wasmModuleUrl: this.opts.wasmModuleUrl,
        });
      } else {
        progress?.({ stage: "compile", backend: "wasm" });
      }

      progress?.({ stage: "warmup" });
      await backend.warmup();

      const viterbi = new ViterbiDecoder(calibration, bundle.labelSet);

      this.state = { kind: "ready", backend, tokenizer, viterbi };
      progress?.({ stage: "ready" });
    } catch (e) {
      if (e instanceof PrivacyFilterError) throw e;
      throw new PrivacyFilterError(
        `failed to initialise PrivacyFilter: ${(e as Error).message}`,
        "MODEL_DOWNLOAD_FAILED",
        e as Error,
      );
    }
  }

  private async runDetection(
    ready: Extract<FilterState, { kind: "ready" }>,
    text: string,
    opts: RedactOptions,
  ): Promise<DetectedSpan[]> {
    const maxChunkTokens = this.opts.maxChunkTokens ?? DEFAULT_MAX_CHUNK_TOKENS;
    const chunks: Chunk[] = chunkInput(text, ready.tokenizer, { maxChunkTokens });

    const perChunkSpans: DetectedSpan[][] = [];
    for (const chunk of chunks) {
      opts.signal?.throwIfAborted();
      const logits = await ready.backend.forward(chunk.tokenIds, chunk.attentionMask);
      const tags = ready.viterbi.decode(logits, chunk.tokenIds.length);
      const spans = bioesToSpans(tags, chunk, ready.tokenizer);
      perChunkSpans.push(spans);
    }

    const modelSpans = mergeChunkResults(perChunkSpans, chunks);
    const rules = resolveRules(
      opts.rules ?? this.opts.rules,
      opts.presets ?? this.opts.presets,
    );
    const allSpans = rules.length === 0
      ? modelSpans
      : mergeRuleSpans(modelSpans, runRules(text, rules));

    // Confidence threshold: drop low-confidence model spans. Rule
    // spans always have confidence: 1.0 so they pass any threshold.
    // Per-call option wins over the create-time default; both default
    // to 0 (keep every span).
    const threshold = opts.minConfidence ?? this.opts.minConfidence ?? 0;
    if (threshold <= 0) return allSpans;
    return allSpans.filter((s) => s.confidence >= threshold);
  }

  private enqueue<T>(fn: () => Promise<T>): Promise<T> {
    const next = this.queue.then(fn, fn);
    this.queue = next.catch(() => undefined);
    return next;
  }
}

/**
 * Compose `presets` and `rules` into a single rule list. Both
 * sources contribute; presets resolve to a fixed rule set defined
 * in `rule-presets.ts`. Returns `[]` when neither is set so callers
 * can short-circuit.
 */
function resolveRules(
  rules: readonly Rule[] | undefined,
  presets: readonly string[] | undefined,
): readonly Rule[] {
  const presetRules = presets && presets.length > 0 ? resolvePresets(presets) : [];
  const userRules = rules ?? [];
  if (presetRules.length === 0) return userRules;
  if (userRules.length === 0) return presetRules;
  return [...presetRules, ...userRules];
}

function buildSummary(
  spans: readonly DetectedSpan[],
): Partial<Record<SpanLabel, number>> {
  const out: Record<string, number> = {};
  for (const span of spans) {
    out[span.label] = (out[span.label] ?? 0) + 1;
  }
  return out as Partial<Record<SpanLabel, number>>;
}

const DEFAULT_MODEL_SOURCE =
  "https://huggingface.co/openai/privacy-filter/resolve/main/";
const DEFAULT_MAX_CHUNK_TOKENS = 2048;
