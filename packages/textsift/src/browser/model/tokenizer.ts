/**
 * Native o200k-style BPE tokenizer for `openai/privacy-filter`.
 *
 * Self-contained — no `@huggingface/transformers` dependency. Loads
 * `tokenizer.json` from the model source, parses the BPE vocab +
 * merges, and exposes `encode(text)` returning the same shape the
 * rest of the library expects (`tokenIds`, `attentionMask`,
 * `tokenToCharOffset`).
 *
 * Three pieces:
 *   1. Pretokenizer — a Sequence of (Split with Unicode regex,
 *      Isolated behavior) followed by (ByteLevel, use_regex=false).
 *      We expand the regex's `(?i:...)` inline case-insensitive groups
 *      to plain alternatives so it parses as a standard JS regex.
 *      Behavior=Isolated means matched portions and gaps both become
 *      separate pieces; we handle that by walking matchAll() and
 *      emitting gap text as its own piece between matches.
 *   2. Byte-level encoding — UTF-8 bytes mapped through the GPT-2
 *      `bytes_to_unicode` table to produce printable chars (one char
 *      per byte). Same map every BPE-with-byte-level model uses.
 *   3. BPE merges — for each pretoken (now a string of byte-mapped
 *      chars), iteratively merge adjacent symbol pairs in priority
 *      order until no learned merge applies. Final symbols → vocab
 *      ids.
 *
 * Special tokens (the 21 added tokens with ids ≥ 199998 like
 * `<|startoftext|>` and `<|endoftext|>`) are matched up-front by a
 * literal-string regex so they're never split by the BPE.
 */

import type { LoadedModelBundle } from "./loader.js";
import type { ProgressEvent, SpanLabel } from "../types.js";
import { PrivacyFilterError } from "../types.js";

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
   * UTF-8 byte-for-byte. Always false for valid UTF-8 input — kept on
   * the interface so older call sites that surface it don't need
   * changes.
   */
  decodedMismatch: boolean;
}

/** Options for `Tokenizer.fromBundle`. */
export interface TokenizerLoadOptions {
  signal?: AbortSignal;
  onProgress?: (event: ProgressEvent) => void;
}

interface TokenizerJson {
  model: {
    type: "BPE";
    vocab: Record<string, number>;
    merges: Array<[string, string]>;
  };
  added_tokens: Array<{
    id: number;
    content: string;
    special: boolean;
  }>;
}

export class Tokenizer {
  private readonly vocab: ReadonlyMap<string, number>;
  private readonly idToToken: readonly string[];
  /** merge pair → rank (lower = applied first). */
  private readonly mergeRanks: ReadonlyMap<string, number>;
  /**
   * Map from byte (0..255) to the printable code point GPT-2's
   * `bytes_to_unicode` assigns. Used during byte-level encoding.
   */
  private readonly byteToChar: readonly number[];
  /**
   * Reverse of `byteToChar`: code point → byte. Used during decoding
   * and during the per-token byte-length calculation that drives
   * character offsets.
   */
  private readonly charToByte: ReadonlyMap<number, number>;
  private readonly specialTokens: ReadonlyMap<string, number>;
  /**
   * Regex matching any special token literal. Used to split the input
   * before BPE so special tokens are never broken up.
   */
  private readonly specialTokenSplitRegex: RegExp | null;
  private readonly pretokenizerRegex: RegExp;
  private readonly labelList: readonly SpanLabel[];
  /**
   * Cache of pretoken (byte-mapped string) → array of token ids.
   * Same pretoken appears many times in real text; the BPE inner loop
   * is the dominant cost so caching pays.
   */
  private readonly bpeCache: Map<string, number[]> = new Map();

  private constructor(args: {
    vocab: ReadonlyMap<string, number>;
    idToToken: readonly string[];
    mergeRanks: ReadonlyMap<string, number>;
    byteToChar: readonly number[];
    charToByte: ReadonlyMap<number, number>;
    specialTokens: ReadonlyMap<string, number>;
    specialTokenSplitRegex: RegExp | null;
    pretokenizerRegex: RegExp;
    labels: readonly SpanLabel[];
  }) {
    this.vocab = args.vocab;
    this.idToToken = args.idToToken;
    this.mergeRanks = args.mergeRanks;
    this.byteToChar = args.byteToChar;
    this.charToByte = args.charToByte;
    this.specialTokens = args.specialTokens;
    this.specialTokenSplitRegex = args.specialTokenSplitRegex;
    this.pretokenizerRegex = args.pretokenizerRegex;
    this.labelList = args.labels;
  }

  /**
   * Load `tokenizer.json` from the model source and build a Tokenizer.
   * Single fetch, ~27 MB for o200k-style — cached by the browser HTTP
   * cache after first load.
   */
  static async fromBundle(
    bundle: LoadedModelBundle,
    opts: TokenizerLoadOptions = {},
  ): Promise<Tokenizer> {
    const url = `${bundle.modelSource}tokenizer.json`;
    const res = await fetch(url, { signal: opts.signal });
    if (!res.ok) {
      throw new PrivacyFilterError(
        `tokenizer fetch failed: ${url} → ${res.status} ${res.statusText}`,
        "MODEL_DOWNLOAD_FAILED",
      );
    }
    const text = await res.text();
    opts.onProgress?.({
      stage: "download",
      loaded: text.length,
      total: Number(res.headers.get("content-length")) || text.length,
      url,
    });

    let json: TokenizerJson;
    try {
      json = JSON.parse(text) as TokenizerJson;
    } catch (e) {
      throw new PrivacyFilterError(
        `tokenizer.json parse error: ${(e as Error).message}`,
        "MODEL_DOWNLOAD_FAILED",
        e as Error,
      );
    }
    if (json.model?.type !== "BPE") {
      throw new PrivacyFilterError(
        `tokenizer.json model.type=${JSON.stringify(json.model?.type)}; expected "BPE"`,
        "MODEL_DOWNLOAD_FAILED",
      );
    }

    // Build vocab map + reverse list.
    const vocab = new Map<string, number>(Object.entries(json.model.vocab));
    let maxId = -1;
    for (const id of vocab.values()) if (id > maxId) maxId = id;
    for (const t of json.added_tokens ?? []) {
      vocab.set(t.content, t.id);
      if (t.id > maxId) maxId = t.id;
    }
    const idToToken: string[] = new Array(maxId + 1);
    for (const [tok, id] of vocab) idToToken[id] = tok;

    // Merge ranks. Each merges entry is `[a, b]`; the merge applies to
    // adjacent symbols `a` and `b`, producing `a + b`. Earlier entries
    // have higher priority (lower rank).
    const mergeRanks = new Map<string, number>();
    const merges = json.model.merges;
    for (let i = 0; i < merges.length; i++) {
      const m = merges[i];
      if (!m) continue;
      mergeRanks.set(m[0] + " " + m[1], i);
    }

    // Byte-level mapping (GPT-2's bytes_to_unicode).
    const byteToChar = buildByteToChar();
    const charToByte = new Map<number, number>();
    for (let b = 0; b < 256; b++) charToByte.set(byteToChar[b]!, b);

    // Special-token literal-match regex.
    const specialTokens = new Map<string, number>();
    const specialLiterals: string[] = [];
    for (const t of json.added_tokens ?? []) {
      if (t.special) {
        specialTokens.set(t.content, t.id);
        specialLiterals.push(escapeRegex(t.content));
      }
    }
    // Sort longest-first so longer specials match before shorter
    // prefixes ever could.
    specialLiterals.sort((a, b) => b.length - a.length);
    const specialTokenSplitRegex = specialLiterals.length
      ? new RegExp(specialLiterals.join("|"), "g")
      : null;

    // Pretokenizer regex. Source pattern uses `(?i:...)` inline case-
    // insensitive groups for English contractions; we expand them to
    // plain alternatives so the regex parses as standard JS.
    const pretokenizerRegex = buildPretokenizerRegex();

    return new Tokenizer({
      vocab,
      idToToken,
      mergeRanks,
      byteToChar,
      charToByte,
      specialTokens,
      specialTokenSplitRegex,
      pretokenizerRegex,
      labels: bundle.labelSet as readonly SpanLabel[],
    });
  }

  labels(): readonly SpanLabel[] {
    return this.labelList;
  }

  /** Public accessor for vocab size — only used by tests. */
  vocabSize(): number {
    return this.idToToken.length;
  }

  /**
   * Encode `text` to token ids + character offsets. Matches the
   * `add_special_tokens=False` behavior of HF tokenizers.
   */
  encode(text: string): EncodeResult {
    const tokenIds: number[] = [];
    const tokenStarts: number[] = [];
    const tokenEnds: number[] = [];

    // Phase 1: split around special tokens. Each non-special slice is
    // passed through the regular BPE pipeline; special tokens are
    // emitted as a single token with their pre-assigned id.
    const slices = this.splitOnSpecialTokens(text);
    for (const slice of slices) {
      if (slice.special) {
        const id = this.specialTokens.get(slice.text);
        if (id === undefined) {
          throw new PrivacyFilterError(
            `Tokenizer.encode: special token mapping missing for ${JSON.stringify(slice.text)}`,
            "INTERNAL",
          );
        }
        tokenIds.push(id);
        tokenStarts.push(slice.start);
        tokenEnds.push(slice.start + slice.text.length);
        continue;
      }

      // Phase 2: pretokenize the non-special slice.
      const pretokens = this.pretokenize(slice.text, slice.start);
      for (const p of pretokens) {
        // Phase 3: byte-level encode the pretoken text.
        const mapped = this.byteLevelEncode(p.text);
        // Phase 4: BPE merge → token strings → ids.
        const ids = this.bpe(mapped);
        // Phase 5: distribute char offsets across the BPE outputs.
        const offsets = this.distributeOffsets(p.text, ids, p.start);
        for (let i = 0; i < ids.length; i++) {
          tokenIds.push(ids[i]!);
          tokenStarts.push(offsets[i]!.start);
          tokenEnds.push(offsets[i]!.end);
        }
      }
    }

    const n = tokenIds.length;
    const idsArr = new Int32Array(n);
    const mask = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      idsArr[i] = tokenIds[i]!;
      mask[i] = 1;
    }

    // tokenToCharOffset has n+1 entries: start of each token plus the
    // end of the last (= text.length when no truncation).
    const offsets: number[] = new Array(n + 1);
    for (let i = 0; i < n; i++) offsets[i] = tokenStarts[i]!;
    offsets[n] = n === 0 ? 0 : tokenEnds[n - 1]!;

    return {
      tokenIds: idsArr,
      attentionMask: mask,
      tokenToCharOffset: offsets,
      decodedMismatch: false,
    };
  }

  /** Decode an array of token ids back to a string. */
  decode(ids: readonly number[] | Int32Array | Uint32Array): string {
    const bytes: number[] = [];
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i]!;
      const tok = this.idToToken[id];
      if (tok === undefined) continue;
      // Special tokens decode to their literal content. Their characters
      // aren't byte-level-mapped, so a byte-table lookup would fail.
      if (this.specialTokens.has(tok)) {
        const enc = new TextEncoder().encode(tok);
        for (let j = 0; j < enc.length; j++) bytes.push(enc[j]!);
        continue;
      }
      for (let j = 0; j < tok.length; j++) {
        const cp = tok.charCodeAt(j);
        const b = this.charToByte.get(cp);
        if (b === undefined) {
          // Surrogate pair or unmapped char → fall back to UTF-8 encode.
          // Defensive; trained vocab should not contain these.
          const enc = new TextEncoder().encode(tok[j] ?? "");
          for (let k = 0; k < enc.length; k++) bytes.push(enc[k]!);
          continue;
        }
        bytes.push(b);
      }
    }
    return new TextDecoder("utf-8").decode(new Uint8Array(bytes));
  }

  // --------------------------------------------------------------
  // Internals
  // --------------------------------------------------------------

  private splitOnSpecialTokens(
    text: string,
  ): Array<{ text: string; start: number; special: boolean }> {
    if (!this.specialTokenSplitRegex) {
      return [{ text, start: 0, special: false }];
    }
    const out: Array<{ text: string; start: number; special: boolean }> = [];
    this.specialTokenSplitRegex.lastIndex = 0;
    let cursor = 0;
    let m: RegExpExecArray | null;
    while ((m = this.specialTokenSplitRegex.exec(text)) !== null) {
      const idx = m.index;
      if (idx > cursor) {
        out.push({ text: text.slice(cursor, idx), start: cursor, special: false });
      }
      out.push({ text: m[0], start: idx, special: true });
      cursor = idx + m[0].length;
    }
    if (cursor < text.length) {
      out.push({ text: text.slice(cursor), start: cursor, special: false });
    }
    return out;
  }

  private pretokenize(
    text: string,
    baseOffset: number,
  ): Array<{ text: string; start: number }> {
    // HF tokenizers' Split + behavior=Isolated emits matched substrings
    // AND gaps as separate pieces, in order. matchAll() walks matches;
    // we stitch in the gaps.
    const out: Array<{ text: string; start: number }> = [];
    this.pretokenizerRegex.lastIndex = 0;
    let cursor = 0;
    let m: RegExpExecArray | null;
    while ((m = this.pretokenizerRegex.exec(text)) !== null) {
      if (m[0].length === 0) {
        // Zero-width match would loop forever — advance cursor manually.
        this.pretokenizerRegex.lastIndex++;
        continue;
      }
      const idx = m.index;
      if (idx > cursor) {
        out.push({
          text: text.slice(cursor, idx),
          start: baseOffset + cursor,
        });
      }
      out.push({ text: m[0], start: baseOffset + idx });
      cursor = idx + m[0].length;
    }
    if (cursor < text.length) {
      out.push({
        text: text.slice(cursor),
        start: baseOffset + cursor,
      });
    }
    return out;
  }

  private byteLevelEncode(text: string): string {
    // UTF-8 encode the pretoken, then map each byte through byteToChar.
    const enc = new TextEncoder().encode(text);
    let out = "";
    for (let i = 0; i < enc.length; i++) {
      out += String.fromCharCode(this.byteToChar[enc[i]!]!);
    }
    return out;
  }

  /**
   * Run the BPE merge loop on a byte-mapped pretoken string. Returns
   * an array of vocab ids.
   */
  private bpe(mapped: string): number[] {
    if (mapped.length === 0) return [];
    const cached = this.bpeCache.get(mapped);
    if (cached) return cached;

    let symbols: string[] = mapped.split("");
    while (symbols.length > 1) {
      let bestIdx = -1;
      let bestRank = Number.MAX_SAFE_INTEGER;
      for (let i = 0; i < symbols.length - 1; i++) {
        const r = this.mergeRanks.get(symbols[i]! + " " + symbols[i + 1]!);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          bestIdx = i;
        }
      }
      if (bestIdx < 0) break;
      // Merge all instances of the chosen pair in one pass.
      const a = symbols[bestIdx]!;
      const b = symbols[bestIdx + 1]!;
      const merged: string[] = [];
      let i = 0;
      while (i < symbols.length) {
        if (i < symbols.length - 1 && symbols[i] === a && symbols[i + 1] === b) {
          merged.push(a + b);
          i += 2;
        } else {
          merged.push(symbols[i]!);
          i++;
        }
      }
      symbols = merged;
    }

    const ids: number[] = new Array(symbols.length);
    for (let i = 0; i < symbols.length; i++) {
      const id = this.vocab.get(symbols[i]!);
      if (id === undefined) {
        throw new PrivacyFilterError(
          `Tokenizer.bpe: symbol ${JSON.stringify(symbols[i])} not in vocab — corrupt tokenizer.json?`,
          "INTERNAL",
        );
      }
      ids[i] = id;
    }
    this.bpeCache.set(mapped, ids);
    return ids;
  }

  /**
   * Distribute the pretoken's character span across its BPE token
   * outputs. For each token, we know its byte length (each char in
   * a byte-mapped token corresponds to exactly one source byte).
   * Cumulative byte boundaries → utf16 offsets via a per-pretoken
   * byte → utf16 table.
   */
  private distributeOffsets(
    pretokenText: string,
    ids: readonly number[],
    pretokenStart: number,
  ): Array<{ start: number; end: number }> {
    const utf16ToByte = buildUtf16ToByte(pretokenText);
    const totalBytes = utf16ToByte[pretokenText.length]!;

    const byteBoundaries: number[] = new Array(ids.length + 1);
    byteBoundaries[0] = 0;
    let bc = 0;
    for (let i = 0; i < ids.length; i++) {
      const tok = this.idToToken[ids[i]!];
      let byteLen = 0;
      if (tok !== undefined) {
        for (let j = 0; j < tok.length; j++) {
          byteLen++; // each char is one source byte in the byte-level scheme
        }
      }
      bc += byteLen;
      byteBoundaries[i + 1] = bc;
    }

    if (bc !== totalBytes) {
      for (let i = 0; i < byteBoundaries.length; i++) {
        if (byteBoundaries[i]! > totalBytes) byteBoundaries[i] = totalBytes;
      }
    }

    const out: Array<{ start: number; end: number }> = new Array(ids.length);
    const charOffsets: number[] = new Array(byteBoundaries.length);
    let u16 = 0;
    for (let i = 0; i < byteBoundaries.length; i++) {
      const target = byteBoundaries[i]!;
      while (u16 < pretokenText.length && utf16ToByte[u16]! < target) u16++;
      charOffsets[i] = u16;
    }
    for (let i = 0; i < ids.length; i++) {
      out[i] = {
        start: pretokenStart + charOffsets[i]!,
        end: pretokenStart + charOffsets[i + 1]!,
      };
    }
    return out;
  }
}

// --------------------------------------------------------------
// Static helpers
// --------------------------------------------------------------

/**
 * GPT-2's `bytes_to_unicode`: each byte 0..255 → a printable Unicode
 * code point. Bytes whose code point is already printable map to
 * themselves; the rest map into the 256+n range.
 */
function buildByteToChar(): number[] {
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
  const out: number[] = new Array(256);
  for (let i = 0; i < bs.length; i++) {
    out[bs[i]!] = cs[i]!;
  }
  return out;
}

/**
 * Build the pretokenizer regex. Source pattern from tokenizer.json:
 *
 *   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
 *   |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
 *   |\p{N}{1,3}
 *   | ?[^\s\p{L}\p{N}]+[\r\n/]*
 *   |\s*[\r\n]+
 *   |\s+(?!\S)
 *   |\s+
 *
 * The `(?i:...)` inline case-insensitive groups are expanded to
 * explicit alternatives so the regex parses as standard JS without
 * needing the regexp-modifiers proposal.
 */
function buildPretokenizerRegex(): RegExp {
  const C = caseInsensitiveContractions();
  const src =
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+" + C +
    "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*" + C +
    "|\\p{N}{1,3}" +
    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*" +
    "|\\s*[\\r\\n]+" +
    "|\\s+(?!\\S)" +
    "|\\s+";
  return new RegExp(src, "gu");
}

function caseInsensitiveContractions(): string {
  const cases = (s: string): string[] => {
    if (s.length === 0) return [""];
    const c = s[0]!;
    const tail = cases(s.slice(1));
    const lo = c.toLowerCase();
    const up = c.toUpperCase();
    const variants = lo === up ? [c] : [lo, up];
    const out: string[] = [];
    for (const v of variants) for (const t of tail) out.push(v + t);
    return out;
  };
  const all: string[] = [];
  for (const base of ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]) {
    for (const v of cases(base)) all.push(v);
  }
  return "(?:" + all.join("|") + ")?";
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
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
