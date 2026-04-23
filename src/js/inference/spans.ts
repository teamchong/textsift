/**
 * Convert a Viterbi-decoded BIOES tag sequence back into character-level
 * spans on the original input text.
 *
 * Inputs:
 *   - `tags`: Uint8Array of BIOES class ids, length T.
 *   - `chunk`: the tokenized chunk with token→character offset mapping.
 *   - `tokenizer`: the tokenizer that produced the offsets (needed for
 *     label lookup; the actual decoding is offset-driven).
 *
 * Output:
 *   - Array of `DetectedSpan` objects with `start`/`end`/`text`/`label`.
 *   - Spans are emitted when a B-l/I-l/E-l run or an S-l tag is found.
 *     Malformed runs (e.g. I-l with no preceding B-l) are salvaged:
 *     treated as a B-l at that position, with a logged warning.
 */

import type { DetectedSpan, SpanLabel } from "../types.js";
import type { Chunk } from "./chunking.js";
import type { Tokenizer } from "../model/tokenizer.js";
import {
  BIOES_BACKGROUND,
  BIOES_BEGIN_OFFSET,
  BIOES_INSIDE_OFFSET,
  BIOES_END_OFFSET,
  BIOES_SINGLE_OFFSET,
  NUM_SPAN_LABELS,
  describeTag,
} from "./viterbi.js";

export function bioesToSpans(
  tags: Uint8Array,
  chunk: Chunk,
  tokenizer: Tokenizer,
): DetectedSpan[] {
  const labelSet = tokenizer.labels();
  if (labelSet.length !== NUM_SPAN_LABELS) {
    throw new Error(
      `bioesToSpans: tokenizer reports ${labelSet.length} labels, expected ${NUM_SPAN_LABELS}`,
    );
  }
  const spans: DetectedSpan[] = [];
  const T = tags.length;

  type OpenRun = { label: number; startTokenIdx: number };
  let open: OpenRun | null = null;

  for (let t = 0; t < T; t++) {
    const tag = tags[t]!;
    const info = describeTag(tag);

    if (tag === BIOES_BACKGROUND) {
      // A B/I-run should have been closed by E-l; if we're still in one,
      // close it at the previous token.
      if (open !== null) {
        emit(open.startTokenIdx, t - 1, open.label);
        open = null;
      }
      continue;
    }

    if (tag >= BIOES_SINGLE_OFFSET) {
      // S-l: single-token entity.
      if (open !== null) {
        emit(open.startTokenIdx, t - 1, open.label);
        open = null;
      }
      emit(t, t, info.labelIndex);
      continue;
    }

    if (tag >= BIOES_END_OFFSET) {
      // E-l: close the open run (or synthesise one at this position).
      if (open !== null && open.label === info.labelIndex) {
        emit(open.startTokenIdx, t, info.labelIndex);
        open = null;
      } else {
        // E without matching B — salvage as single-token span.
        if (open !== null) {
          emit(open.startTokenIdx, t - 1, open.label);
          open = null;
        }
        emit(t, t, info.labelIndex);
      }
      continue;
    }

    if (tag >= BIOES_INSIDE_OFFSET) {
      // I-l: extend the run if label matches; otherwise close the old
      // run and start a new implicit B-l.
      if (open !== null && open.label === info.labelIndex) {
        // continue the run; no state change.
      } else {
        if (open !== null) {
          emit(open.startTokenIdx, t - 1, open.label);
        }
        open = { label: info.labelIndex, startTokenIdx: t };
      }
      continue;
    }

    if (tag >= BIOES_BEGIN_OFFSET) {
      // B-l: close any open run and start fresh.
      if (open !== null) {
        emit(open.startTokenIdx, t - 1, open.label);
      }
      open = { label: info.labelIndex, startTokenIdx: t };
      continue;
    }
  }

  // End-of-sequence: close any still-open run.
  if (open !== null) {
    emit(open.startTokenIdx, T - 1, open.label);
  }

  return spans;

  function emit(startTokenIdx: number, endTokenIdx: number, labelIndex: number) {
    const startOffset = chunk.tokenToCharOffset[startTokenIdx];
    const endOffset = chunk.tokenToCharOffset[endTokenIdx + 1] ?? chunk.tokenToCharOffset[endTokenIdx]!;
    if (startOffset === undefined || endOffset === undefined) return;
    const absStart = chunk.charOffset + startOffset;
    const absEnd = chunk.charOffset + endOffset;
    const label = labelSet[labelIndex] as SpanLabel | undefined;
    if (!label) return;
    const text = chunk.text.slice(startOffset, endOffset);
    spans.push({
      label,
      start: absStart,
      end: absEnd,
      text,
      marker: `[${label}]`,
      confidence: 1.0, // confidence from Viterbi is all-or-nothing; per-token softmax
                      // is the right source for a finer value. Populated by the
                      // pipeline once emission probs are piped through.
    });
  }
}
