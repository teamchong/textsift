/**
 * Tokenizer wrapper — converts text to token ids + character offset
 * tables, shared across all backends.
 *
 * The upstream privacy-filter uses tiktoken-style BPE with offset
 * tracking (token index → character index). We rely on the pre-built
 * tokenizer artifact (`tokenizer.json`) shipped with the model and
 * parse it with the `@huggingface/transformers` tokenizer utility at
 * runtime — no C extension, no WASM dependency, ~15 KB of code.
 */

import type { LoadedModelBundle } from "./loader.js";
import type { SpanLabel } from "../types.js";

export interface EncodeResult {
  /** One int32 per token (vocab id). */
  tokenIds: Int32Array;
  /** One byte per token: 1 = real, 0 = padding (always 1 from `encode`). */
  attentionMask: Uint8Array;
  /**
   * Character offset in the input text for each token. Length is
   * `tokenIds.length + 1` — the extra entry gives the end-of-last-token
   * offset (= text length when no truncation).
   */
  tokenToCharOffset: readonly number[];
}

export class Tokenizer {
  private readonly impl: unknown;
  private readonly labelList: readonly SpanLabel[];

  private constructor(impl: unknown, labels: readonly SpanLabel[]) {
    this.impl = impl;
    this.labelList = labels;
  }

  static async fromBundle(bundle: LoadedModelBundle): Promise<Tokenizer> {
    // Wire in the upstream tokenizer implementation once the model loader
    // is producing a real bundle. Construction from `tokenizerJson` +
    // `tokenizerConfig` is straightforward via `@huggingface/transformers`
    // but kept out of this file so the unit-testable logic (Viterbi,
    // chunking, redact) isn't coupled to that dependency.
    void bundle;
    throw new Error(
      "Tokenizer.fromBundle pending implementation — blocked on ModelLoader returning a real bundle. "
        + "See `src/js/model/tokenizer.ts` for the shape the downstream code expects.",
    );
  }

  labels(): readonly SpanLabel[] {
    return this.labelList;
  }

  encode(_text: string): EncodeResult {
    void this.impl;
    throw new Error("Tokenizer.encode pending implementation.");
  }
}
