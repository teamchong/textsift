/**
 * Viterbi decoder for BIOES-tagged token classification.
 *
 * Inputs:
 *   - Per-token emission logits shape `[T, C]` where C = 33:
 *     class 0       = background "O"
 *     classes 1..8  = B-<label>
 *     classes 9..16 = I-<label>
 *     classes 17..24 = E-<label>
 *     classes 25..32 = S-<label>
 *
 * Algorithm:
 *   - Classic Viterbi: for every token t and every tag j, compute the
 *     best cumulative score of a tag sequence ending in j at time t.
 *   - Transition score = calibration bias indexed by (tag_from, tag_to).
 *     Most transitions are illegal (forbidden) and get `-Infinity`, e.g.
 *     "I-person" may only follow "B-person" or "I-person" of the same
 *     label. Six bias values from the calibration JSON parameterise the
 *     *allowed* transitions.
 *
 * Output:
 *   - A length-T array of integer tag ids in `[0, 32]`. The downstream
 *     `bioesToSpans` collapses this into span tuples.
 */

import type { ViterbiCalibration } from "../model/calibration.js";

/** BIOES tag family. */
export const BIOES_BACKGROUND = 0;
export const BIOES_BEGIN_OFFSET = 1;   // B-span_0 = 1, B-span_7 = 8
export const BIOES_INSIDE_OFFSET = 9;  // I-span_0 = 9, I-span_7 = 16
export const BIOES_END_OFFSET = 17;    // E-span_0 = 17, E-span_7 = 24
export const BIOES_SINGLE_OFFSET = 25; // S-span_0 = 25, S-span_7 = 32

export const NUM_BIOES_CLASSES = 33;
export const NUM_SPAN_LABELS = 8;

const NEG_INF = -1e30;

/** Logit tensor shape. */
export interface LogitGrid {
  readonly data: Float32Array;       // length T * C
  readonly sequenceLength: number;   // T
  readonly numClasses: number;       // C (must equal NUM_BIOES_CLASSES)
}

export class ViterbiDecoder {
  private readonly transitions: Float32Array;
  private readonly startScores: Float32Array;
  private readonly labelSet: readonly string[];

  constructor(calibration: ViterbiCalibration, labelSet: readonly string[]) {
    if (labelSet.length !== NUM_SPAN_LABELS) {
      throw new Error(
        `ViterbiDecoder: expected ${NUM_SPAN_LABELS} labels, got ${labelSet.length}`,
      );
    }
    this.labelSet = labelSet;
    this.transitions = buildTransitionMatrix(calibration);
    this.startScores = buildStartScores();
  }

  /** Expose the label lookup so downstream span-builders don't need it separately. */
  getLabelSet(): readonly string[] {
    return this.labelSet;
  }

  /**
   * Decode the most-likely BIOES tag sequence from per-token logits.
   * `sequenceLength` defaults to `logits.sequenceLength` but callers
   * can pass a smaller value when the input was padded.
   */
  decode(logits: LogitGrid, sequenceLength?: number): Uint8Array {
    if (logits.numClasses !== NUM_BIOES_CLASSES) {
      throw new Error(
        `ViterbiDecoder: expected ${NUM_BIOES_CLASSES} classes, got ${logits.numClasses}`,
      );
    }
    const T = sequenceLength ?? logits.sequenceLength;
    if (T <= 0) return new Uint8Array(0);
    if (T > logits.sequenceLength) {
      throw new Error(
        `ViterbiDecoder: sequenceLength ${T} exceeds logits.sequenceLength ${logits.sequenceLength}`,
      );
    }

    const C = NUM_BIOES_CLASSES;
    const alpha = new Float32Array(T * C);    // best score ending at (t, j)
    const back = new Uint8Array(T * C);       // argmax predecessor

    // t = 0 — initialise with start scores plus emissions.
    for (let j = 0; j < C; j++) {
      alpha[j] = this.startScores[j]! + logits.data[j]!;
      back[j] = 0xFF; // sentinel: no predecessor
    }

    // t = 1..T-1.
    for (let t = 1; t < T; t++) {
      const emissionBase = t * C;
      const prevBase = (t - 1) * C;
      const currBase = t * C;
      for (let j = 0; j < C; j++) {
        let best = NEG_INF;
        let argmax = 0;
        const transCol = this.transitions.subarray(j * C, (j + 1) * C);
        for (let i = 0; i < C; i++) {
          const s = alpha[prevBase + i]! + transCol[i]!;
          if (s > best) {
            best = s;
            argmax = i;
          }
        }
        alpha[currBase + j] = best + logits.data[emissionBase + j]!;
        back[currBase + j] = argmax;
      }
    }

    // Final backtrace.
    const tags = new Uint8Array(T);
    let bestFinal = NEG_INF;
    let finalTag = 0;
    const lastBase = (T - 1) * C;
    for (let j = 0; j < C; j++) {
      if (alpha[lastBase + j]! > bestFinal) {
        bestFinal = alpha[lastBase + j]!;
        finalTag = j;
      }
    }
    tags[T - 1] = finalTag;
    for (let t = T - 1; t > 0; t--) {
      tags[t - 1] = back[t * C + tags[t]!]!;
    }
    return tags;
  }
}

// -----------------------------------------------------------------
// Transition matrix construction
// -----------------------------------------------------------------

/**
 * Build the [C, C] transition score matrix in row-major layout where
 * `trans[to * C + from]` is the score added when transitioning from
 * `from` to `to`. Illegal transitions get NEG_INF; legal ones get the
 * corresponding calibration bias.
 */
function buildTransitionMatrix(cal: ViterbiCalibration): Float32Array {
  const C = NUM_BIOES_CLASSES;
  const t = new Float32Array(C * C).fill(NEG_INF);

  // Helper: mark `from → to` legal with score s.
  const allow = (from: number, to: number, score: number) => {
    t[to * C + from] = score;
  };

  // O → O.
  allow(BIOES_BACKGROUND, BIOES_BACKGROUND, cal.backgroundStay);

  for (let lbl = 0; lbl < NUM_SPAN_LABELS; lbl++) {
    const B = BIOES_BEGIN_OFFSET + lbl;
    const I = BIOES_INSIDE_OFFSET + lbl;
    const E = BIOES_END_OFFSET + lbl;
    const S = BIOES_SINGLE_OFFSET + lbl;

    // O → B-l and O → S-l: start of a new entity.
    allow(BIOES_BACKGROUND, B, cal.backgroundToStart);
    allow(BIOES_BACKGROUND, S, cal.backgroundToStart);

    // B-l → I-l (continue) and B-l → E-l (single-inside → end).
    allow(B, I, cal.insideToContinue);
    allow(B, E, cal.insideToEnd);

    // I-l → I-l (continue) and I-l → E-l (end entity).
    allow(I, I, cal.insideToContinue);
    allow(I, E, cal.insideToEnd);

    // E-l → O (back to background).
    allow(E, BIOES_BACKGROUND, cal.endToBackground);
    // S-l → O (single-token entity, then background).
    allow(S, BIOES_BACKGROUND, cal.endToBackground);

    // E-l → B-l' and E-l → S-l' and S-l → B-l' and S-l → S-l' — entity
    // immediately followed by another entity. Allowed across labels.
    for (let lbl2 = 0; lbl2 < NUM_SPAN_LABELS; lbl2++) {
      const B2 = BIOES_BEGIN_OFFSET + lbl2;
      const S2 = BIOES_SINGLE_OFFSET + lbl2;
      allow(E, B2, cal.endToStart);
      allow(E, S2, cal.endToStart);
      allow(S, B2, cal.endToStart);
      allow(S, S2, cal.endToStart);
    }
  }

  return t;
}

function buildStartScores(): Float32Array {
  // Legal starts: O, B-*, S-*. Illegal: I-*, E-*.
  const C = NUM_BIOES_CLASSES;
  const s = new Float32Array(C).fill(NEG_INF);
  s[BIOES_BACKGROUND] = 0;
  for (let lbl = 0; lbl < NUM_SPAN_LABELS; lbl++) {
    s[BIOES_BEGIN_OFFSET + lbl] = 0;
    s[BIOES_SINGLE_OFFSET + lbl] = 0;
  }
  return s;
}

/** For tests and debugging: categorise a tag id. */
export function describeTag(tag: number): { prefix: "O" | "B" | "I" | "E" | "S"; labelIndex: number } {
  if (tag === BIOES_BACKGROUND) return { prefix: "O", labelIndex: -1 };
  if (tag >= BIOES_BEGIN_OFFSET && tag < BIOES_INSIDE_OFFSET) {
    return { prefix: "B", labelIndex: tag - BIOES_BEGIN_OFFSET };
  }
  if (tag >= BIOES_INSIDE_OFFSET && tag < BIOES_END_OFFSET) {
    return { prefix: "I", labelIndex: tag - BIOES_INSIDE_OFFSET };
  }
  if (tag >= BIOES_END_OFFSET && tag < BIOES_SINGLE_OFFSET) {
    return { prefix: "E", labelIndex: tag - BIOES_END_OFFSET };
  }
  return { prefix: "S", labelIndex: tag - BIOES_SINGLE_OFFSET };
}
