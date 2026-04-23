/**
 * Tokenizer wrapper around `@huggingface/transformers`' `AutoTokenizer`.
 *
 * Gives the rest of the library three things:
 *   - `tokenIds` + `attentionMask` ready to hand to the model.
 *   - `tokenToCharOffset` — UTF-16 offset in the original input for each
 *     token (with an end sentinel), so `bioesToSpans` can map tag runs
 *     back to character spans without a second lookup pass.
 *   - `decodedMismatch` — informational flag raised when the tokenizer
 *     round-trip doesn't reproduce the input byte-for-byte (rare with
 *     the ByteLevel BPE upstream ships; matches upstream opf's warning
 *     semantics).
 *
 * transformers.js 4.2.0 does not expose character offsets via its public
 * `encode()` / `_call()` API (the token-classification pipeline carries
 * an explicit TODO to add them). We reconstruct offsets here via the
 * GPT-2 byte-level reverse map: each token string is a sequence of
 * printable code points that map 1:1 back to input bytes, so per-token
 * byte lengths are exact, and a byte→UTF-16 table built from the input
 * completes the mapping. This matches upstream opf's tiktoken-based
 * approach (`opf/_core/spans.py::token_char_ranges_for_text`).
 */

import { AutoTokenizer } from "@huggingface/transformers";
import type { LoadedModelBundle } from "./loader.js";
import type { SpanLabel } from "../types.js";

/** Result of tokenizing one input string. */
export interface EncodeResult {
  /** One int32 per token (vocab id). */
  tokenIds: Int32Array;
  /** One byte per token: 1 = real, 0 = padding (always 1 from `encode`). */
  attentionMask: Uint8Array;
  /**
   * UTF-16 character offset in the input text for each token. Length is
   * `tokenIds.length + 1` — the extra entry gives the end-of-last-token
   * offset (= `input.length` when no truncation).
   */
  tokenToCharOffset: readonly number[];
  /**
   * True if the per-token byte reconstruction didn't match the input
   * UTF-8 byte-for-byte. When true, offsets are best-effort; callers
   * may want to surface a warning.
   */
  decodedMismatch: boolean;
}

type PretrainedTokenizer = Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>;

export class Tokenizer {
  private readonly impl: PretrainedTokenizer;
  private readonly labelList: readonly SpanLabel[];

  private constructor(impl: PretrainedTokenizer, labels: readonly SpanLabel[]) {
    this.impl = impl;
    this.labelList = labels;
  }

  static async fromBundle(bundle: LoadedModelBundle): Promise<Tokenizer> {
    const impl = await AutoTokenizer.from_pretrained(bundle.modelId);
    return new Tokenizer(impl, bundle.labelSet as readonly SpanLabel[]);
  }

  labels(): readonly SpanLabel[] {
    return this.labelList;
  }

  encode(text: string): EncodeResult {
    const ids = this.impl.encode(text, { add_special_tokens: false });
    const tokens = this.impl.tokenize(text, { add_special_tokens: false });
    if (ids.length !== tokens.length) {
      throw new Error(
        `Tokenizer.encode: ids/tokens length mismatch (${ids.length} vs ${tokens.length})`,
      );
    }

    const n = ids.length;
    const tokenIds = new Int32Array(n);
    const attentionMask = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      tokenIds[i] = ids[i]!;
      attentionMask[i] = 1;
    }

    // Cumulative UTF-8 byte length per token.
    const byteBoundaries = new Int32Array(n + 1);
    let bc = 0;
    let mismatch = false;
    for (let i = 0; i < n; i++) {
      const tok = tokens[i]!;
      const tokByteLen = byteLevelTokenByteLength(tok);
      if (tokByteLen < 0) {
        mismatch = true;
        break;
      }
      bc += tokByteLen;
      byteBoundaries[i + 1] = bc;
    }

    // Byte offset (UTF-8) for each UTF-16 code unit of the input, plus
    // a sentinel at text.length. Used to flip token byte boundaries
    // into UTF-16 character offsets.
    const utf16ToByte = buildUtf16ToByte(text);
    const inputByteLen = utf16ToByte[text.length]!;
    if (!mismatch && bc !== inputByteLen) {
      mismatch = true;
    }

    // byte → utf16 reverse lookup; monotonic, so a dual-cursor walk
    // flips all n+1 boundaries in O(n + text.length).
    const tokenToCharOffset: number[] = new Array(n + 1);
    let u16 = 0;
    for (let i = 0; i <= n; i++) {
      const targetByte = mismatch ? Math.min(byteBoundaries[i]!, inputByteLen) : byteBoundaries[i]!;
      while (u16 < text.length && utf16ToByte[u16]! < targetByte) u16++;
      tokenToCharOffset[i] = u16;
    }

    return {
      tokenIds,
      attentionMask,
      tokenToCharOffset,
      decodedMismatch: mismatch,
    };
  }
}

// --------------------------------------------------------------
// GPT-2 ByteLevel helpers
// --------------------------------------------------------------

/**
 * Reverse of GPT-2's `bytes_to_unicode` table. Maps each printable
 * code point used in ByteLevel-encoded tokens back to its original
 * byte value (0–255). Missing entries signal a non-ByteLevel token
 * (e.g. an added special token), which we treat as `decodedMismatch`.
 */
const BYTE_LEVEL_CHAR_TO_BYTE: ReadonlyMap<number, number> = buildGpt2ReverseMap();

function buildGpt2ReverseMap(): Map<number, number> {
  const bs: number[] = [];
  for (let i = "!".charCodeAt(0); i <= "~".charCodeAt(0); i++) bs.push(i);
  for (let i = "¡".charCodeAt(0); i <= "¬".charCodeAt(0); i++) bs.push(i);
  for (let i = "®".charCodeAt(0); i <= "ÿ".charCodeAt(0); i++) bs.push(i);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const map = new Map<number, number>();
  for (let i = 0; i < bs.length; i++) {
    map.set(cs[i]!, bs[i]!);
  }
  return map;
}

function byteLevelTokenByteLength(token: string): number {
  let len = 0;
  for (let i = 0; i < token.length; i++) {
    const cp = token.charCodeAt(i);
    if (!BYTE_LEVEL_CHAR_TO_BYTE.has(cp)) return -1;
    len++;
  }
  return len;
}

/**
 * UTF-8 byte offset at each UTF-16 code-unit position of `text`. Length
 * is `text.length + 1`; the last entry is the total UTF-8 byte length.
 * Surrogate pairs get the byte position at both the high and low
 * surrogate (they encode a single UTF-8 4-byte sequence).
 */
function buildUtf16ToByte(text: string): Int32Array {
  const out = new Int32Array(text.length + 1);
  let byteCursor = 0;
  let i = 0;
  while (i < text.length) {
    out[i] = byteCursor;
    const code = text.charCodeAt(i);
    if (code < 0x80) {
      byteCursor += 1;
      i += 1;
    } else if (code < 0x800) {
      byteCursor += 2;
      i += 1;
    } else if (code >= 0xd800 && code <= 0xdbff && i + 1 < text.length) {
      out[i + 1] = byteCursor;
      byteCursor += 4;
      i += 2;
    } else {
      byteCursor += 3;
      i += 1;
    }
  }
  out[text.length] = byteCursor;
  return out;
}
