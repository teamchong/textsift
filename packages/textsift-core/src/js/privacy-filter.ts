/**
 * `PrivacyFilter` — the clean API surface textsift-core exports.
 *
 * All internal machinery lives behind this class:
 *   - Model weights: downloaded + cached by ModelLoader / OPFS.
 *   - Tokenizer: native o200k-style BPE, no transformers.js dependency.
 *   - Backend: WebGPU or WASM (textsift-core); the umbrella `textsift`
 *     package can inject a transformers.js backend via the `backend`
 *     option when `"auto"` falls through.
 *   - Viterbi CRF decoder: constructed from the calibration artifact
 *     shipped alongside the weights.
 *   - Chunking: inputs over `maxChunkTokens` are split with a sliding
 *     window and re-merged at span level.
 *   - Redaction applicator: character-level placement with marker
 *     strategy resolution.
 */

import {
  ALL_SPAN_LABELS,
  type CreateOptions,
  type DetectResult,
  type DetectedSpan,
  PrivacyFilterError,
  type RedactOptions,
  type RedactResult,
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
 * Per-create extension hook used by the umbrella `textsift` package to
 * resolve the `"auto"` backend to a transformers.js implementation
 * when WebGPU and SIMD-capable WASM are both unavailable. Most callers
 * never set this — they let textsift-core's defaults pick.
 */
export interface BackendResolver {
  /**
   * Called when `opts.backend === "auto"`. Receives the loaded bundle
   * and the runtime capability flags textsift-core has already
   * detected. Return:
   *   - an `InferenceBackend` instance to use it directly, OR
   *   - `null` to let textsift-core fall back to its built-in choice
   *     (`"webgpu"` if `hasWebGPU`, else `"wasm"`).
   */
  resolveAuto(args: {
    bundle: LoadedModelBundle;
    hasWebGPU: boolean;
    isNode: boolean;
    quantization: "int4" | "int8" | "fp16";
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
   * Advanced factory used by the umbrella `textsift` package to inject
   * a transformers.js fallback into the `"auto"` decision. Public-ish
   * but documented as advanced — the umbrella is the only intended
   * caller.
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
      rules: (opts as RedactStreamOptions).rules ?? this.opts.rules,
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
      rules: (opts as DetectStreamOptions).rules ?? this.opts.rules,
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
   * Release GPU / WASM resources. Further calls throw. Safe to call
   * multiple times.
   */
  dispose(): void {
    if (this.state.kind === "ready") {
      this.state.backend.dispose();
    }
    this.state = { kind: "disposed" };
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
      const quantization = this.opts.quantization ?? "int8";

      let backend: InferenceBackend | null = null;

      if (requested === "auto" && this.backendResolver) {
        backend = await this.backendResolver.resolveAuto({
          bundle,
          hasWebGPU,
          isNode,
          quantization,
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
          quantization,
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
    const rules = opts.rules ?? this.opts.rules;
    if (!rules || rules.length === 0) return modelSpans;
    const ruleSpans = runRules(text, rules);
    return mergeRuleSpans(modelSpans, ruleSpans);
  }

  private enqueue<T>(fn: () => Promise<T>): Promise<T> {
    const next = this.queue.then(fn, fn);
    this.queue = next.catch(() => undefined);
    return next;
  }
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
