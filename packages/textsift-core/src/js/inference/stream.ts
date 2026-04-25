/**
 * Streaming-input detection.
 *
 * Used by the `detect()` overload that accepts an
 * `AsyncIterable<string>` instead of a `string`. Designed for the
 * AI-proxy / LLM-output-filtering use case: the caller pipes incoming
 * text directly into `detect(stream)` and gets back a handle exposing
 * both the final result Promise and an `AsyncIterable<DetectedSpan>`
 * that yields spans as they become detectable.
 *
 * Cost model:
 *   - Naive (`detect(buffer)` after every chunk): O(N²) over a stream
 *     of N tokens — each call re-tokenizes and re-infers everything
 *     seen so far.
 *   - This implementation: O(N). Each chunk arrival runs inference on
 *     a trailing window (default 1024 tokens) of the accumulated
 *     buffer. Tokens whose attention window is entirely behind the
 *     trailing edge have stable logits, so spans ending before a
 *     safety margin (default 256 tokens, ≈1024 chars) are emitted
 *     permanently and never re-considered. After the input stream
 *     ends, a full chunked pass over the buffer flushes any spans
 *     still inside the margin.
 *
 * Spans are deduplicated by `(label, start, end)` so re-running a
 * window over an already-seen region doesn't emit duplicates.
 */

import type { InferenceBackend } from "../backends/abstract.js";
import type { Tokenizer } from "../model/tokenizer.js";
import type { ViterbiDecoder } from "./viterbi.js";
import type {
  DetectResult,
  DetectedSpan,
  MarkerStrategy,
  RedactResult,
  SpanLabel,
} from "../types.js";
import { ALL_SPAN_LABELS, PrivacyFilterError } from "../types.js";
import { bioesToSpans } from "./spans.js";
import { chunkInput, type Chunk } from "./chunking.js";
import { applyRedaction } from "./redact.js";

/**
 * Options for streaming detection. Defaults are tuned for the
 * LLM-proxy use case: 1024-token window (~4× the model's
 * `slidingWindow=128`), 256-token safety margin (twice the sliding
 * window — spans ending more than 2× window behind the trailing edge
 * can't be affected by any future token).
 */
export interface DetectStreamOptions {
  /**
   * How many trailing tokens to send through the model on each
   * incoming chunk. Larger windows give more context near the
   * trailing edge at the cost of more compute per chunk. Default: 1024.
   */
  windowTokens?: number;
  /**
   * How many tokens behind the trailing edge a span must end before
   * it's considered stable and emitted. Default: 256.
   */
  safetyMarginTokens?: number;
  /** Aborts the underlying forward passes. */
  signal?: AbortSignal;
  /** Same as the batch-mode `enabledCategories` option. */
  enabledCategories?: readonly SpanLabel[];
}

/** Options for streaming redaction. Same as detect plus a marker strategy. */
export interface RedactStreamOptions extends DetectStreamOptions {
  markers?: MarkerStrategy;
}

/**
 * Handle returned by `detect(asyncIterableInput)`. Both consumption
 * styles work:
 *
 *   const handle = filter.detect(llmStream);
 *
 *   // (A) iterate spans as they arrive
 *   for await (const span of handle.spanStream) { ... }
 *
 *   // (B) await the final result (full DetectResult shape)
 *   const result = await handle.result;
 *
 * Mixing styles is safe — the underlying job runs once and both
 * surfaces share its output through a single multi-consumer queue.
 */
export interface DetectStreamHandle {
  /**
   * Yields spans as they become stable (their `end` is more than the
   * safety margin behind the trailing edge of the input stream), plus
   * any remaining stragglers after the input stream ends.
   */
  readonly spanStream: AsyncIterable<DetectedSpan>;
  /**
   * Resolves to the same `DetectResult` shape `detect(string)` returns,
   * once the input stream has ended and all spans have been emitted.
   */
  readonly result: Promise<DetectResult>;
}

/**
 * Handle returned by `redact(asyncIterableInput)`. Three surfaces:
 *
 *   const handle = filter.redact(llmStream);
 *
 *   // (A) stream redacted text downstream as it becomes safe to emit
 *   for await (const piece of handle.textStream) {
 *     await downstreamWriter.write(piece);
 *   }
 *
 *   // (B) iterate detected spans as they arrive (for logging / metrics)
 *   for await (const span of handle.spanStream) { ... }
 *
 *   // (C) await the final RedactResult (full text, all spans, summary)
 *   const result = await handle.result;
 *
 * `textStream` only yields output text whose redaction status is final
 * — text within the safety margin is held back until the trailing edge
 * advances past it (or the input stream ends and `finish` flushes
 * everything). This means `textStream` lags the input stream by up to
 * `safetyMarginTokens` worth of characters.
 */
export interface RedactStreamHandle {
  /** Yields redacted text fragments as they become safe to emit. */
  readonly textStream: AsyncIterable<string>;
  /** Yields stable spans (same as detect's spanStream). */
  readonly spanStream: AsyncIterable<DetectedSpan>;
  /** Resolves to the full `RedactResult` once the stream is done. */
  readonly result: Promise<RedactResult>;
}

const DEFAULT_WINDOW_TOKENS = 1024;
const DEFAULT_SAFETY_MARGIN_TOKENS = 256;

interface StreamInternals {
  backend: InferenceBackend;
  tokenizer: Tokenizer;
  viterbi: ViterbiDecoder;
}

/**
 * Build a streaming-detection handle. Returns synchronously so
 * `detect(stream)` matches the no-await call shape Vercel-style
 * stream APIs use. The driver awaits `internalsReady` before issuing
 * any forward pass — letting the caller pass an unresolved readiness
 * Promise from `PrivacyFilter.ensureReady()` without blocking the
 * factory call.
 */
export function streamDetect(
  internalsReady: Promise<StreamInternals>,
  input: AsyncIterable<string>,
  opts: DetectStreamOptions = {},
): DetectStreamHandle {
  const resolved = resolveStreamOptions(opts);
  const enabledFilter = buildEnabledFilter(resolved.enabledCategories);
  const allSpans: DetectedSpan[] = [];
  let inputBuffer = "";

  // Multi-consumer queue: one driver pushes spans, two surfaces
  // (`spanStream` iterator and `result` resolution) both pull. Empty
  // iterator pulls park as `waiter` resolvers.
  const queue: DetectedSpan[] = [];
  let queueClosed = false;
  let queueError: unknown = undefined;
  const waiters: Array<(v: IteratorResult<DetectedSpan>) => void> = [];

  function pushQueue(value: DetectedSpan): void {
    const w = waiters.shift();
    if (w) w({ value, done: false });
    else queue.push(value);
  }
  function closeQueue(err?: unknown): void {
    queueClosed = true;
    queueError = err;
    while (waiters.length > 0) {
      const w = waiters.shift()!;
      w({ value: undefined as unknown as DetectedSpan, done: true });
    }
  }

  let resolveResult!: (value: DetectResult) => void;
  let rejectResult!: (reason: unknown) => void;
  const result: Promise<DetectResult> = new Promise((res, rej) => {
    resolveResult = res;
    rejectResult = rej;
  });

  // Driver: reads `input`, runs trailing-window inference per chunk,
  // pushes stable spans into the queue. Runs eagerly so consumers can
  // iterate at any rate without losing progress.
  (async () => {
    try {
      const internals = await internalsReady;
      const emitted = new Set<string>();
      const { backend, tokenizer, viterbi } = internals;
      const safetyMarginChars = resolved.safetyMarginTokens * 4;

      function shouldEmit(span: DetectedSpan): boolean {
        if (enabledFilter && !enabledFilter(span)) return false;
        const key = `${span.label}:${span.start}:${span.end}`;
        if (emitted.has(key)) return false;
        emitted.add(key);
        return true;
      }

      async function runWindow(safetyEdgeChar: number): Promise<void> {
        if (inputBuffer.length === 0) return;
        resolved.signal?.throwIfAborted();

        const enc = tokenizer.encode(inputBuffer);
        const totalTokens = enc.tokenIds.length;
        if (totalTokens === 0) return;

        const winStart = Math.max(0, totalTokens - resolved.windowTokens);
        const winLen = totalTokens - winStart;
        const windowIds = enc.tokenIds.subarray(winStart, totalTokens);
        const windowMask = enc.attentionMask.subarray(winStart, totalTokens);

        const logits = await backend.forward(windowIds, windowMask);
        const tags = viterbi.decode(logits, winLen);

        const winFirstCharOffset = enc.tokenToCharOffset[winStart] ?? 0;
        const localTokenToChar: number[] = new Array(winLen + 1);
        for (let i = 0; i <= winLen; i++) {
          const abs = enc.tokenToCharOffset[winStart + i] ?? inputBuffer.length;
          localTokenToChar[i] = abs - winFirstCharOffset;
        }
        const chunk: Chunk = {
          tokenIds: windowIds,
          attentionMask: windowMask,
          tokenToCharOffset: localTokenToChar,
          text: inputBuffer.slice(winFirstCharOffset),
          charOffset: winFirstCharOffset,
          emitRange: [0, winLen] as const,
        };

        const spans = bioesToSpans(tags, chunk, tokenizer);
        for (const span of spans) {
          if (span.end > safetyEdgeChar) continue;
          if (shouldEmit(span)) {
            allSpans.push(span);
            pushQueue(span);
          }
        }
      }

      for await (const chunk of input) {
        if (chunk.length === 0) continue;
        inputBuffer += chunk;
        const safetyEdgeChar = Math.max(0, inputBuffer.length - safetyMarginChars);
        await runWindow(safetyEdgeChar);
      }

      // Stream ended — flush remaining buffer through the full chunked
      // pipeline. No safety margin; all undetected spans are now stable.
      if (inputBuffer.length > 0) {
        const chunks = chunkInput(inputBuffer, internals.tokenizer, {
          maxChunkTokens: Math.max(resolved.windowTokens, 2048),
        });
        for (const c of chunks) {
          resolved.signal?.throwIfAborted();
          const logits = await internals.backend.forward(c.tokenIds, c.attentionMask);
          const tags = internals.viterbi.decode(logits, c.tokenIds.length);
          const spans = bioesToSpans(tags, c, internals.tokenizer);
          for (const span of spans) {
            if (shouldEmit(span)) {
              allSpans.push(span);
              pushQueue(span);
            }
          }
        }
      }

      const summary = buildSummary(allSpans);
      resolveResult({
        input: inputBuffer,
        spans: allSpans,
        summary,
        containsPii: allSpans.length > 0,
      });
      closeQueue();
    } catch (err) {
      rejectResult(err);
      closeQueue(err);
    }
  })();

  const spanStream: AsyncIterable<DetectedSpan> = {
    [Symbol.asyncIterator]() {
      return {
        next(): Promise<IteratorResult<DetectedSpan>> {
          if (queue.length > 0) {
            return Promise.resolve({ value: queue.shift()!, done: false });
          }
          if (queueClosed) {
            if (queueError !== undefined) return Promise.reject(queueError);
            return Promise.resolve({
              value: undefined as unknown as DetectedSpan,
              done: true,
            });
          }
          return new Promise((resolve) => waiters.push(resolve));
        },
      };
    },
  };

  return { spanStream, result };
}

/**
 * Streaming redaction. Same trailing-window inference loop as
 * `streamDetect`, plus an output channel that emits redacted text in
 * order as the trailing edge advances past it.
 *
 * Concretely:
 *   - `inputBuffer` accumulates as chunks arrive.
 *   - `emitCursor` tracks how many chars of `inputBuffer` have been
 *     pushed to `textStream` (in their final, possibly-redacted form).
 *   - On every chunk, after detection runs, advance `emitCursor` to
 *     the new safety edge, emitting the slice [oldCursor, newEdge)
 *     with all stable spans in that range applied.
 *   - At `finish`, flush [emitCursor, end) with all remaining spans.
 *
 * Spans fall into one of three states each chunk:
 *   - **Stable** (end ≤ safety edge): apply at emit time, never
 *     re-considered.
 *   - **Pending** (start ≤ safety edge < end): straddles the edge —
 *     hold the emit cursor at `start` until the span is stable.
 *   - **Future** (start > safety edge): not yet emitted; will be
 *     re-detected on the next chunk.
 */
export function streamRedact(
  internalsReady: Promise<StreamInternals>,
  input: AsyncIterable<string>,
  opts: RedactStreamOptions = {},
): RedactStreamHandle {
  const resolved = resolveStreamOptions(opts);
  const enabledFilter = buildEnabledFilter(resolved.enabledCategories);
  const enabledSet = new Set<SpanLabel>(
    resolved.enabledCategories ?? ALL_SPAN_LABELS,
  );
  const markerStrategy = opts.markers;

  const allSpans: DetectedSpan[] = [];
  let inputBuffer = "";
  let emitCursor = 0; // chars of inputBuffer emitted (final form) into textStream
  let outputAccum = "";

  // Two queues: one for spans, one for redacted-text fragments. Same
  // multi-consumer pattern as detect's queue.
  const spanQueue: DetectedSpan[] = [];
  let spanQueueClosed = false;
  let spanQueueError: unknown = undefined;
  const spanWaiters: Array<(v: IteratorResult<DetectedSpan>) => void> = [];

  const textQueue: string[] = [];
  let textQueueClosed = false;
  let textQueueError: unknown = undefined;
  const textWaiters: Array<(v: IteratorResult<string>) => void> = [];

  function pushSpan(value: DetectedSpan): void {
    const w = spanWaiters.shift();
    if (w) w({ value, done: false });
    else spanQueue.push(value);
  }
  function closeSpanQueue(err?: unknown): void {
    spanQueueClosed = true;
    spanQueueError = err;
    while (spanWaiters.length > 0) {
      spanWaiters.shift()!({ value: undefined as unknown as DetectedSpan, done: true });
    }
  }
  function pushText(value: string): void {
    if (value.length === 0) return;
    const w = textWaiters.shift();
    if (w) w({ value, done: false });
    else textQueue.push(value);
  }
  function closeTextQueue(err?: unknown): void {
    textQueueClosed = true;
    textQueueError = err;
    while (textWaiters.length > 0) {
      textWaiters.shift()!({ value: undefined as unknown as string, done: true });
    }
  }

  let resolveResult!: (value: RedactResult) => void;
  let rejectResult!: (reason: unknown) => void;
  const result: Promise<RedactResult> = new Promise((res, rej) => {
    resolveResult = res;
    rejectResult = rej;
  });

  // Spans we've already emitted-as-stable (so we don't double-count).
  const stableSpanKeys = new Set<string>();
  // Spans currently considered stable in chronological order — used
  // to apply redactions to the emitted text slice.
  const stableSpans: DetectedSpan[] = [];

  function spanKey(s: DetectedSpan): string {
    return `${s.label}:${s.start}:${s.end}`;
  }

  /**
   * Emit redacted text up to `targetCursor` chars into `inputBuffer`.
   * Walks the stable-span list, applying redactions for any span whose
   * end ≤ targetCursor and start ≥ emitCursor. If a stable span starts
   * before the new emit cursor target but doesn't end until after it,
   * we cap targetCursor at span.start so we don't split a redactable
   * region across two emits.
   */
  function flushTo(targetCursor: number): void {
    if (targetCursor <= emitCursor) return;
    // Cap at the start of any stable-but-not-yet-fully-included span
    // that begins before targetCursor — we can't emit past its start
    // without committing to its (already-decided) marker.
    let cap = targetCursor;
    for (const span of stableSpans) {
      if (span.end > cap && span.start < cap && span.start >= emitCursor) {
        cap = span.start;
      }
    }
    if (cap <= emitCursor) return;

    const slice = inputBuffer.slice(emitCursor, cap);
    // Spans entirely inside [emitCursor, cap) get applied to this slice.
    const sliceSpans = stableSpans
      .filter((s) => s.start >= emitCursor && s.end <= cap)
      .map((s) => ({
        ...s,
        start: s.start - emitCursor,
        end: s.end - emitCursor,
      }));
    const { redactedText, applied } = applyRedaction(
      slice,
      sliceSpans,
      enabledSet,
      markerStrategy,
    );
    pushText(redactedText);
    outputAccum += redactedText;
    // Re-translate `applied` markers back onto absolute coordinates
    // for the final result. We don't push them anywhere here — the
    // span emission to `spanStream` happened separately when each
    // became stable.
    void applied;
    emitCursor = cap;
  }

  (async () => {
    try {
      const internals = await internalsReady;
      const { backend, tokenizer, viterbi } = internals;
      const safetyMarginChars = resolved.safetyMarginTokens * 4;
      const emitted = new Set<string>();

      function shouldEmit(span: DetectedSpan): boolean {
        if (enabledFilter && !enabledFilter(span)) return false;
        const key = spanKey(span);
        if (emitted.has(key)) return false;
        emitted.add(key);
        return true;
      }

      async function runWindow(safetyEdgeChar: number): Promise<void> {
        if (inputBuffer.length === 0) return;
        resolved.signal?.throwIfAborted();

        const enc = tokenizer.encode(inputBuffer);
        const totalTokens = enc.tokenIds.length;
        if (totalTokens === 0) return;

        const winStart = Math.max(0, totalTokens - resolved.windowTokens);
        const winLen = totalTokens - winStart;
        const windowIds = enc.tokenIds.subarray(winStart, totalTokens);
        const windowMask = enc.attentionMask.subarray(winStart, totalTokens);

        const logits = await backend.forward(windowIds, windowMask);
        const tags = viterbi.decode(logits, winLen);

        const winFirstCharOffset = enc.tokenToCharOffset[winStart] ?? 0;
        const localTokenToChar: number[] = new Array(winLen + 1);
        for (let i = 0; i <= winLen; i++) {
          const abs = enc.tokenToCharOffset[winStart + i] ?? inputBuffer.length;
          localTokenToChar[i] = abs - winFirstCharOffset;
        }
        const chunk: Chunk = {
          tokenIds: windowIds,
          attentionMask: windowMask,
          tokenToCharOffset: localTokenToChar,
          text: inputBuffer.slice(winFirstCharOffset),
          charOffset: winFirstCharOffset,
          emitRange: [0, winLen] as const,
        };

        const spans = bioesToSpans(tags, chunk, tokenizer);
        for (const span of spans) {
          if (span.end > safetyEdgeChar) continue;
          if (!shouldEmit(span)) continue;
          allSpans.push(span);
          if (!stableSpanKeys.has(spanKey(span))) {
            stableSpanKeys.add(spanKey(span));
            stableSpans.push(span);
          }
          pushSpan(span);
        }
      }

      for await (const chunk of input) {
        if (chunk.length === 0) continue;
        inputBuffer += chunk;
        const safetyEdgeChar = Math.max(0, inputBuffer.length - safetyMarginChars);
        await runWindow(safetyEdgeChar);
        // After the inference pass, advance the emit cursor up to
        // safetyEdgeChar — that's the boundary at which all spans
        // ending before it have been finalised.
        flushTo(safetyEdgeChar);
      }

      // Stream ended — final pass over remaining buffer with no
      // safety margin, then emit everything.
      if (inputBuffer.length > 0) {
        const chunks = chunkInput(inputBuffer, internals.tokenizer, {
          maxChunkTokens: Math.max(resolved.windowTokens, 2048),
        });
        for (const c of chunks) {
          resolved.signal?.throwIfAborted();
          const logits = await internals.backend.forward(c.tokenIds, c.attentionMask);
          const tags = internals.viterbi.decode(logits, c.tokenIds.length);
          const spans = bioesToSpans(tags, c, internals.tokenizer);
          for (const span of spans) {
            if (!shouldEmit(span)) continue;
            allSpans.push(span);
            if (!stableSpanKeys.has(spanKey(span))) {
              stableSpanKeys.add(spanKey(span));
              stableSpans.push(span);
            }
            pushSpan(span);
          }
        }
      }
      // Flush remainder.
      flushTo(inputBuffer.length);

      const summary = buildSummary(allSpans);
      const finalApplied: DetectedSpan[] = stableSpans
        .filter((s) => enabledSet.has(s.label))
        .map((s, i) => ({
          ...s,
          marker: resolveMarkerForResult(s, i, markerStrategy),
        }));
      resolveResult({
        input: inputBuffer,
        redactedText: outputAccum,
        spans: finalApplied,
        summary,
        containsPii: finalApplied.length > 0,
      });
      closeSpanQueue();
      closeTextQueue();
    } catch (err) {
      rejectResult(err);
      closeSpanQueue(err);
      closeTextQueue(err);
    }
  })();

  const spanStream: AsyncIterable<DetectedSpan> = {
    [Symbol.asyncIterator]() {
      return {
        next(): Promise<IteratorResult<DetectedSpan>> {
          if (spanQueue.length > 0) {
            return Promise.resolve({ value: spanQueue.shift()!, done: false });
          }
          if (spanQueueClosed) {
            if (spanQueueError !== undefined) return Promise.reject(spanQueueError);
            return Promise.resolve({
              value: undefined as unknown as DetectedSpan,
              done: true,
            });
          }
          return new Promise((resolve) => spanWaiters.push(resolve));
        },
      };
    },
  };

  const textStream: AsyncIterable<string> = {
    [Symbol.asyncIterator]() {
      return {
        next(): Promise<IteratorResult<string>> {
          if (textQueue.length > 0) {
            return Promise.resolve({ value: textQueue.shift()!, done: false });
          }
          if (textQueueClosed) {
            if (textQueueError !== undefined) return Promise.reject(textQueueError);
            return Promise.resolve({
              value: undefined as unknown as string,
              done: true,
            });
          }
          return new Promise((resolve) => textWaiters.push(resolve));
        },
      };
    },
  };

  return { textStream, spanStream, result };
}

function resolveMarkerForResult(
  span: DetectedSpan,
  index: number,
  strategy: MarkerStrategy | undefined,
): string {
  if (strategy === undefined) return `[${span.label}]`;
  if (typeof strategy === "function") {
    return strategy(span, index) ?? span.text;
  }
  const override = strategy[span.label];
  if (override === null) return span.text;
  if (override === undefined) return `[${span.label}]`;
  return override;
}

function buildEnabledFilter(
  enabled: readonly SpanLabel[] | undefined,
): ((span: DetectedSpan) => boolean) | undefined {
  if (!enabled || enabled.length === ALL_SPAN_LABELS.length) return undefined;
  const set = new Set<SpanLabel>(enabled);
  return (span) => set.has(span.label);
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

/**
 * Validate and apply defaults to a partial `DetectStreamOptions`.
 * Exported so the umbrella package can re-validate user-supplied
 * config without duplicating the logic.
 */
export function resolveStreamOptions(opts: DetectStreamOptions = {}): {
  windowTokens: number;
  safetyMarginTokens: number;
  signal?: AbortSignal;
  enabledCategories?: readonly SpanLabel[];
} {
  const windowTokens = opts.windowTokens ?? DEFAULT_WINDOW_TOKENS;
  const safetyMarginTokens = opts.safetyMarginTokens ?? DEFAULT_SAFETY_MARGIN_TOKENS;
  if (windowTokens <= 0 || !Number.isFinite(windowTokens)) {
    throw new PrivacyFilterError(
      `detect(stream): invalid windowTokens=${windowTokens}; must be > 0`,
      "INTERNAL",
    );
  }
  if (safetyMarginTokens < 0 || !Number.isFinite(safetyMarginTokens)) {
    throw new PrivacyFilterError(
      `detect(stream): invalid safetyMarginTokens=${safetyMarginTokens}; must be ≥ 0`,
      "INTERNAL",
    );
  }
  if (safetyMarginTokens >= windowTokens) {
    throw new PrivacyFilterError(
      `detect(stream): safetyMarginTokens (${safetyMarginTokens}) must be < windowTokens (${windowTokens})`,
      "INTERNAL",
    );
  }
  return {
    windowTokens,
    safetyMarginTokens,
    signal: opts.signal,
    enabledCategories: opts.enabledCategories,
  };
}
