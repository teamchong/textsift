/**
 * `PrivacyFilter` — the clean API surface the library exports.
 *
 * All internal machinery lives behind this class:
 *   - Model weights: downloaded + cached by ModelLoader.
 *   - Tokenizer: SentencePiece/BPE wrapper, shared across calls.
 *   - Backend: WebGPU or WASM, chosen once in `create()`.
 *   - Viterbi CRF decoder: constructed from the calibration artifact
 *     shipped alongside the weights.
 *   - Chunking: inputs over `maxChunkTokens` are split with a sliding
 *     window and re-merged at span level.
 *   - Redaction applicator: character-level placement with marker
 *     strategy resolution.
 *
 * A consumer calls `create()` once and then `redact()` / `detect()` /
 * `redactBatch()` many times. The class is reusable across calls and
 * thread-safe within a single runtime (it queues concurrent calls).
 */

import {
  ALL_SPAN_LABELS,
  type CreateOptions,
  type DetectResult,
  type DetectedSpan,
  type MarkerStrategy,
  PrivacyFilterError,
  type RedactOptions,
  type RedactResult,
  type SpanLabel,
} from "./types.js";
import type { InferenceBackend } from "./backends/abstract.js";
import { selectBackend } from "./backends/select.js";
import { ModelLoader } from "./model/loader.js";
import { Tokenizer } from "./model/tokenizer.js";
import { ViterbiDecoder } from "./inference/viterbi.js";
import { loadCalibration } from "./model/calibration.js";
import { chunkInput, mergeChunkResults, type Chunk } from "./inference/chunking.js";
import { bioesToSpans } from "./inference/spans.js";
import { applyRedaction } from "./inference/redact.js";

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

export class PrivacyFilter {
  private state: FilterState = { kind: "uninitialised" };
  private queue: Promise<unknown> = Promise.resolve();
  private readonly opts: CreateOptions;

  /** Private constructor; use `PrivacyFilter.create()`. */
  private constructor(opts: CreateOptions) {
    this.opts = opts;
  }

  /**
   * Factory: downloads the model (or hits OPFS cache), selects a backend,
   * and warms up the inference pipeline. Idempotent across concurrent
   * `create()` calls with the same options.
   */
  static async create(opts: CreateOptions = {}): Promise<PrivacyFilter> {
    const instance = new PrivacyFilter(opts);
    await instance.ensureReady();
    return instance;
  }

  /**
   * Redact PII in `text` and return both the redacted string and the
   * span metadata. Safe to call concurrently (calls are queued).
   */
  async redact(text: string, callOpts: RedactOptions = {}): Promise<RedactResult> {
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

  /** Detect-only variant: same inference, no redacted-string construction. */
  async detect(text: string, callOpts: RedactOptions = {}): Promise<DetectResult> {
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
      const tokenizer = await Tokenizer.fromBundle(bundle);

      // Backend selection: explicit `backend` wins; "auto" (or
      // unspecified) picks the fastest path available in this
      // environment. Browser: transformers.js on WebGPU when the
      // adapter supports it, otherwise our WASM path. Node: our WASM
      // path — onnxruntime-node's CPU EP has no kernel for
      // `GatherBlockQuantized` / `MatMulNBits`, which this model
      // requires, so tjs can't run at all in Node for it.
      const requested = this.opts.backend ?? "auto";
      const wantsStage1 = requested === "wasm";
      const wantsStage2 = requested === "webgpu";
      // Node 21+ defines `navigator` globally, so `typeof navigator` isn't
      // the right environment check any more. `process.versions.node`
      // is — it's only set when running on the Node runtime (absent in
      // browsers, Bun, Deno, and workers).
      const isNode =
        typeof process !== "undefined" &&
        !!(process as { versions?: { node?: string } }).versions?.node;
      const hasWebGPU =
        !isNode &&
        typeof navigator !== "undefined" &&
        !!(navigator as { gpu?: unknown }).gpu;
      const chosen: "transformers-js" | "wasm" | "webgpu" =
        wantsStage2 ? "webgpu"
        : wantsStage1 ? "wasm"
        : isNode ? "wasm"
        : "transformers-js";
      const device: "auto" | "wasm" | "webgpu" = hasWebGPU ? "webgpu" : "auto";
      const compileBackend: "webgpu" | "wasm" =
        chosen === "webgpu" || (chosen === "transformers-js" && device === "webgpu") ? "webgpu" : "wasm";
      progress?.({ stage: "compile", backend: compileBackend });
      const backend = await selectBackend({
        quantization: this.opts.quantization ?? "int8",
        device,
        bundle,
        backend: chosen,
        wasmModuleUrl: this.opts.wasmModuleUrl,
      });

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

    return mergeChunkResults(perChunkSpans, chunks);
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
  const out: Partial<Record<SpanLabel, number>> = {};
  for (const span of spans) {
    out[span.label] = (out[span.label] ?? 0) + 1;
  }
  return out;
}

const DEFAULT_MODEL_SOURCE =
  "https://huggingface.co/openai/privacy-filter/resolve/main/";
const DEFAULT_MAX_CHUNK_TOKENS = 2048;
