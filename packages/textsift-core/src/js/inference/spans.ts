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
import { NUM_SPAN_LABELS, describeTag } from "./viterbi.js";

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

    switch (info.prefix) {
      case "O": {
        // A B/I-run should have been closed by E-l; if we're still in one,
        // close it at the previous token.
        if (open !== null) {
          emit(open.startTokenIdx, t - 1, open.label);
          open = null;
        }
        break;
      }
      case "S": {
        // S-l: single-token entity.
        if (open !== null) {
          emit(open.startTokenIdx, t - 1, open.label);
          open = null;
        }
        emit(t, t, info.labelIndex);
        break;
      }
      case "E": {
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
        break;
      }
      case "I": {
        // I-l: extend the run if label matches; otherwise close the old
        // run and start a new implicit B-l.
        if (open === null || open.label !== info.labelIndex) {
          if (open !== null) {
            emit(open.startTokenIdx, t - 1, open.label);
          }
          open = { label: info.labelIndex, startTokenIdx: t };
        }
        break;
      }
      case "B": {
        // B-l: close any open run and start fresh.
        if (open !== null) {
          emit(open.startTokenIdx, t - 1, open.label);
        }
        open = { label: info.labelIndex, startTokenIdx: t };
        break;
      }
    }
  }

  // End-of-sequence: close any still-open run.
  if (open !== null) {
    emit(open.startTokenIdx, T - 1, open.label);
  }

  return spans;

  function emit(startTokenIdx: number, endTokenIdx: number, labelIndex: number) {
    const rawStartOffset = chunk.tokenToCharOffset[startTokenIdx];
    const rawEndOffset =
      chunk.tokenToCharOffset[endTokenIdx + 1] ?? chunk.tokenToCharOffset[endTokenIdx]!;
    if (rawStartOffset === undefined || rawEndOffset === undefined) return;
    const label = labelSet[labelIndex] as SpanLabel | undefined;
    if (!label) return;

    // Match upstream opf's `trim_char_spans_whitespace`: BPE tokens on
    // ByteLevel BPE often include a leading space that should not bleed
    // into the emitted PII span. Trim symmetrically at both ends.
    let startOffset = rawStartOffset;
    let endOffset = rawEndOffset;
    while (startOffset < endOffset && isWhitespace(chunk.text.charCodeAt(startOffset))) {
      startOffset++;
    }
    while (endOffset > startOffset && isWhitespace(chunk.text.charCodeAt(endOffset - 1))) {
      endOffset--;
    }
    if (endOffset <= startOffset) return;

    spans.push({
      label,
      source: "model",
      start: chunk.charOffset + startOffset,
      end: chunk.charOffset + endOffset,
      text: chunk.text.slice(startOffset, endOffset),
      marker: `[${label}]`,
      confidence: 1.0, // confidence from Viterbi is all-or-nothing; per-token softmax
                      // is the right source for a finer value. Populated by the
                      // pipeline once emission probs are piped through.
    });
  }
}

function isWhitespace(code: number): boolean {
  // Match JavaScript `String.prototype.trim` whitespace semantics, which
  // align with Python `str.isspace` for the characters relevant to English
  // model outputs (space, tab, newline, CR, form feed, NBSP, etc.).
  return (
    code === 0x20 ||
    code === 0x09 ||
    code === 0x0a ||
    code === 0x0b ||
    code === 0x0c ||
    code === 0x0d ||
    code === 0xa0
  );
}
