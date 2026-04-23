/**
 * Sliding-window chunking for long inputs.
 *
 * The upstream model supports a 128k-token context but inference at that
 * length is slow in the browser (multi-second per-chunk latency and the
 * attention memory blows up with sequence length²). Realistic per-call
 * sizes are ~1k-4k tokens.
 *
 * Strategy: split long inputs into overlapping chunks of `maxChunkTokens`
 * with an overlap region. Spans that fall entirely inside a chunk are
 * trusted; spans that span a chunk boundary are resolved in favour of
 * the chunk whose center contains more of the span. Overlap size defaults
 * to 128 tokens — larger than any realistic span — so the boundary effect
 * is minimal.
 */

import type { Tokenizer } from "../model/tokenizer.js";
import type { DetectedSpan } from "../types.js";

export interface ChunkOptions {
  maxChunkTokens: number;
  overlapTokens?: number;
}

/**
 * One chunk of tokenized input with enough metadata to:
 *   - Run a forward pass (`tokenIds`, `attentionMask`).
 *   - Map BIOES tags back to character offsets in the ORIGINAL input
 *     (`charOffset` + `tokenToCharOffset`).
 */
export interface Chunk {
  /** Tokens for this chunk (CLS/SEP already added if the model expects them). */
  readonly tokenIds: Int32Array;
  /** 1 for real tokens, 0 for padding (always 1s here; padding handled elsewhere). */
  readonly attentionMask: Uint8Array;
  /**
   * For each token index `i` (0..tokenIds.length), the character offset
   * in `text` where the token starts. An extra sentinel at index
   * `tokenIds.length` gives the end-of-last-token offset.
   */
  readonly tokenToCharOffset: readonly number[];
  /** The substring of the original input this chunk covers. */
  readonly text: string;
  /** Character offset of `text[0]` in the original full input. */
  readonly charOffset: number;
  /**
   * Range of valid output tokens [fromToken, toToken) — excludes the
   * leading overlap region (for chunks that aren't first) and the
   * trailing overlap region (for chunks that aren't last). Spans
   * falling outside this range are dropped during merge.
   */
  readonly emitRange: readonly [number, number];
}

export function chunkInput(
  text: string,
  tokenizer: Tokenizer,
  opts: ChunkOptions,
): Chunk[] {
  const overlap = opts.overlapTokens ?? DEFAULT_OVERLAP_TOKENS;
  const full = tokenizer.encode(text);

  if (full.tokenIds.length <= opts.maxChunkTokens) {
    return [
      {
        tokenIds: full.tokenIds,
        attentionMask: full.attentionMask,
        tokenToCharOffset: full.tokenToCharOffset,
        text,
        charOffset: 0,
        emitRange: [0, full.tokenIds.length] as const,
      },
    ];
  }

  const chunks: Chunk[] = [];
  const step = opts.maxChunkTokens - overlap;
  const totalTokens = full.tokenIds.length;

  for (let chunkStart = 0; chunkStart < totalTokens; chunkStart += step) {
    const chunkEnd = Math.min(chunkStart + opts.maxChunkTokens, totalTokens);

    const tokenIds = full.tokenIds.subarray(chunkStart, chunkEnd);
    const attentionMask = full.attentionMask.subarray(chunkStart, chunkEnd);

    // Character offsets relative to the full input; we rebase to the chunk.
    const absCharStart = full.tokenToCharOffset[chunkStart]!;
    const absCharEndAfterLast =
      full.tokenToCharOffset[chunkEnd] ??
      full.tokenToCharOffset[chunkEnd - 1]! +
        (text.length - full.tokenToCharOffset[chunkEnd - 1]!);
    const chunkText = text.slice(absCharStart, absCharEndAfterLast);
    const tokenToCharOffset: number[] = [];
    for (let i = chunkStart; i <= chunkEnd; i++) {
      const abs = full.tokenToCharOffset[i] ?? absCharEndAfterLast;
      tokenToCharOffset.push(abs - absCharStart);
    }

    // Emit only the non-overlap interior: skip leading overlap for non-
    // first chunks, skip trailing overlap for non-last chunks.
    const isFirst = chunkStart === 0;
    const isLast = chunkEnd >= totalTokens;
    const emitFrom = isFirst ? 0 : Math.floor(overlap / 2);
    const emitTo = isLast ? chunkEnd - chunkStart : (chunkEnd - chunkStart) - Math.ceil(overlap / 2);

    chunks.push({
      tokenIds,
      attentionMask,
      tokenToCharOffset,
      text: chunkText,
      charOffset: absCharStart,
      emitRange: [emitFrom, emitTo] as const,
    });

    if (isLast) break;
  }

  return chunks;
}

/**
 * Merge per-chunk detected spans into a single list with duplicates
 * removed at overlap boundaries.
 *
 * Semantics: a span is kept only if its token range is entirely inside
 * its chunk's `emitRange`. Spans that cross the overlap boundary are
 * dropped from the earlier chunk and picked up by the later chunk (or
 * vice versa depending on where the span center lies).
 */
export function mergeChunkResults(
  perChunkSpans: readonly DetectedSpan[][],
  chunks: readonly Chunk[],
): DetectedSpan[] {
  const out: DetectedSpan[] = [];
  const seen = new Set<string>();

  for (let c = 0; c < chunks.length; c++) {
    const chunk = chunks[c]!;
    const spans = perChunkSpans[c]!;
    for (const span of spans) {
      const key = `${span.start}:${span.end}:${span.label}`;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(span);
    }
    void chunk; // emitRange narrowing happens inside bioesToSpans upstream;
                // here we just deduplicate by absolute character offsets.
  }

  out.sort((a, b) => a.start - b.start);
  return out;
}

const DEFAULT_OVERLAP_TOKENS = 128;
