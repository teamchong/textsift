/**
 * Incremental detection over a growing input buffer.
 *
 * Designed for the LLM-proxy use case where text arrives in chunks and
 * the caller wants spans as they become detectable, not after the full
 * stream has been received.
 *
 * Cost model:
 *   - Naive (`detect(buffer)` every chunk): O(N²) over a stream of N
 *     tokens — each call re-tokenizes and re-infers everything seen so
 *     far.
 *   - This API: O(N). Each `append()` runs inference on a trailing
 *     window of `WINDOW_TOKENS` tokens (default 1024). Tokens whose
 *     attention window is entirely behind the trailing edge have
 *     stable logits, so spans ending before a `SAFETY_MARGIN_TOKENS`
 *     buffer (default 256) are emitted permanently and never
 *     re-considered. `finish()` runs a full pass over whatever's
 *     left in the buffer, with no margin restriction.
 *
 * Spans are deduplicated by `(label, start, end)` so re-running a
 * window over an already-seen region doesn't emit duplicates.
 */

import type { InferenceBackend } from "../backends/abstract.js";
import type { Tokenizer } from "../model/tokenizer.js";
import type { ViterbiDecoder } from "./viterbi.js";
import type { DetectedSpan } from "../types.js";
import { PrivacyFilterError } from "../types.js";
import { bioesToSpans } from "./spans.js";
import { chunkInput, type Chunk } from "./chunking.js";

/**
 * Options for `PrivacyFilter.startStream()`. Defaults are tuned for
 * the LLM-proxy use case: 1024-token window (~4× the model's
 * `slidingWindow=128`, plenty of context above and below any new
 * token), 256-token safety margin (twice the sliding window — spans
 * ending more than 2× window behind the trailing edge can't be
 * affected by any future token).
 */
export interface DetectStreamOptions {
  /**
   * How many trailing tokens to send through the model on each
   * `append()` pass. Larger windows give more context (better
   * accuracy near the trailing edge) at the cost of more compute per
   * append. Default: 1024.
   */
  windowTokens?: number;
  /**
   * How many tokens behind the trailing edge a span must end before
   * it's considered stable and emitted. Set high enough that further
   * appends can't change predictions for tokens at or before
   * `currentEnd - safetyMarginTokens`. Default: 256.
   */
  safetyMarginTokens?: number;
  /** Aborts the underlying forward pass. */
  signal?: AbortSignal;
}

const DEFAULT_WINDOW_TOKENS = 1024;
const DEFAULT_SAFETY_MARGIN_TOKENS = 256;

/**
 * Stateful streaming session. Returned by `PrivacyFilter.startStream()`.
 */
export interface DetectStreamSession {
  /**
   * Append text to the stream. Returns an async iterable that yields
   * any newly-stable spans. The iterable completes when this append's
   * inference pass has finished. Subsequent appends keep the buffer.
   *
   * Spans whose `end` is before `currentTextLength - safetyMargin (in
   * chars)` are stable and yielded; spans within the margin wait for
   * a future append (or the final `finish()`) before being emitted.
   */
  append(text: string): AsyncIterable<DetectedSpan>;

  /**
   * Flush the entire remaining buffer through the full chunked
   * pipeline. Yields any spans not yet emitted (including those that
   * were within the safety margin and have now "settled" because
   * there's no more text to come).
   */
  finish(): AsyncIterable<DetectedSpan>;

  /** Current accumulated input text. */
  readonly text: string;
}

interface SessionInternals {
  backend: InferenceBackend;
  tokenizer: Tokenizer;
  viterbi: ViterbiDecoder;
  options: Required<Omit<DetectStreamOptions, "signal">> & { signal?: AbortSignal };
}

export function createDetectStream(
  internals: SessionInternals,
  enabledFilter?: (span: DetectedSpan) => boolean,
): DetectStreamSession {
  const { backend, tokenizer, viterbi, options } = internals;
  const windowTokens = options.windowTokens;
  const safetyMarginTokens = options.safetyMarginTokens;

  let buffer = "";
  // (label, start, end) → true. Spans we've already yielded.
  const emitted = new Set<string>();

  function spanKey(s: DetectedSpan): string {
    return `${s.label}:${s.start}:${s.end}`;
  }

  function shouldEmit(span: DetectedSpan): boolean {
    if (enabledFilter && !enabledFilter(span)) return false;
    const key = spanKey(span);
    if (emitted.has(key)) return false;
    emitted.add(key);
    return true;
  }

  /**
   * Run the model on a trailing-window slice of the current buffer
   * and yield the spans that fall on/before `safetyEdgeChar`.
   */
  async function* runWindow(
    safetyEdgeChar: number,
  ): AsyncGenerator<DetectedSpan> {
    if (buffer.length === 0) return;
    options.signal?.throwIfAborted();

    const enc = tokenizer.encode(buffer);
    const totalTokens = enc.tokenIds.length;
    if (totalTokens === 0) return;

    // Pick a trailing window. If the buffer fits, use it whole.
    const winStart = Math.max(0, totalTokens - windowTokens);
    const winLen = totalTokens - winStart;

    const windowIds = enc.tokenIds.subarray(winStart, totalTokens);
    const windowMask = enc.attentionMask.subarray(winStart, totalTokens);

    const logits = await backend.forward(windowIds, windowMask);
    const tags = viterbi.decode(logits, winLen);

    // Build a Chunk-shaped object so bioesToSpans can do its
    // standard offset accounting. tokenToCharOffset is the slice of
    // the global one shifted to chunk-local indices.
    const winFirstCharOffset = enc.tokenToCharOffset[winStart] ?? 0;
    const localTokenToChar: number[] = new Array(winLen + 1);
    for (let i = 0; i <= winLen; i++) {
      const abs = enc.tokenToCharOffset[winStart + i] ?? buffer.length;
      localTokenToChar[i] = abs - winFirstCharOffset;
    }
    const chunk: Chunk = {
      tokenIds: windowIds,
      attentionMask: windowMask,
      tokenToCharOffset: localTokenToChar,
      text: buffer.slice(winFirstCharOffset),
      charOffset: winFirstCharOffset,
      emitRange: [0, winLen] as const,
    };

    const spans = bioesToSpans(tags, chunk, tokenizer);
    for (const span of spans) {
      if (span.end > safetyEdgeChar) continue; // not stable yet
      if (shouldEmit(span)) yield span;
    }
  }

  return {
    get text() {
      return buffer;
    },

    async *append(text: string) {
      if (text.length === 0) return;
      buffer += text;

      // Translate the safety margin from tokens to a character-edge.
      // Cheap heuristic: assume 1 token ≈ 4 chars. Off-by-some doesn't
      // matter — the margin only needs to be loose-but-sufficient.
      const safetyMarginChars = safetyMarginTokens * 4;
      const safetyEdgeChar = Math.max(0, buffer.length - safetyMarginChars);

      for await (const span of runWindow(safetyEdgeChar)) {
        yield span;
      }
    },

    async *finish() {
      if (buffer.length === 0) return;
      // Final pass: full chunked pipeline (handles arbitrarily long
      // buffers correctly), no safety-margin restriction.
      const chunks = chunkInput(buffer, tokenizer, {
        maxChunkTokens: Math.max(windowTokens, 2048),
      });
      for (const chunk of chunks) {
        options.signal?.throwIfAborted();
        const logits = await backend.forward(chunk.tokenIds, chunk.attentionMask);
        const tags = viterbi.decode(logits, chunk.tokenIds.length);
        const spans = bioesToSpans(tags, chunk, tokenizer);
        for (const span of spans) {
          if (shouldEmit(span)) yield span;
        }
      }
    },
  };
}

/**
 * Type-guard helper exported for users who construct a
 * `DetectStreamOptions` from untyped sources (e.g. a config file).
 * Validates and applies defaults; throws on bad input.
 */
export function resolveStreamOptions(
  opts: DetectStreamOptions = {},
): Required<Omit<DetectStreamOptions, "signal">> & { signal?: AbortSignal } {
  const windowTokens = opts.windowTokens ?? DEFAULT_WINDOW_TOKENS;
  const safetyMarginTokens = opts.safetyMarginTokens ?? DEFAULT_SAFETY_MARGIN_TOKENS;
  if (windowTokens <= 0 || !Number.isFinite(windowTokens)) {
    throw new PrivacyFilterError(
      `DetectStream: invalid windowTokens=${windowTokens}; must be > 0`,
      "INTERNAL",
    );
  }
  if (safetyMarginTokens < 0 || !Number.isFinite(safetyMarginTokens)) {
    throw new PrivacyFilterError(
      `DetectStream: invalid safetyMarginTokens=${safetyMarginTokens}; must be ≥ 0`,
      "INTERNAL",
    );
  }
  if (safetyMarginTokens >= windowTokens) {
    throw new PrivacyFilterError(
      `DetectStream: safetyMarginTokens (${safetyMarginTokens}) must be < windowTokens (${windowTokens})`,
      "INTERNAL",
    );
  }
  return {
    windowTokens,
    safetyMarginTokens,
    signal: opts.signal,
  };
}
