/**
 * Minimal ONNX graph reader.
 *
 * ONNX files are protobuf. The full schema is large but we only need a
 * tiny subset — the initializer list + external_data references — so
 * we hand-roll a protobuf decoder rather than pull in protobufjs (~500
 * KB min). The decoder handles just the wire types that appear in
 * `openai/privacy-filter`'s model_q4f16.onnx (varint + length-delimited).
 *
 * What we extract per tensor:
 *   name, shape, dtype, extOffset, extLength
 * `extOffset`/`extLength` are byte ranges into the external data file
 * (e.g. `model_q4f16.onnx_data`) where the raw tensor bytes live.
 */

/** ONNX dtype codes (subset we care about). */
export enum OnnxDType {
  FLOAT = 1,
  UINT8 = 2,
  INT8 = 3,
  UINT16 = 4,
  INT16 = 5,
  INT32 = 6,
  INT64 = 7,
  STRING = 8,
  BOOL = 9,
  FLOAT16 = 10,
  DOUBLE = 11,
  UINT32 = 12,
  UINT64 = 13,
  BFLOAT16 = 16,
  INT4 = 22,
  UINT4 = 21,
}

/**
 * A tensor entry from the ONNX graph. Bytes live either INLINE (stored
 * in the .onnx file itself via `raw_data` / typed data arrays) or
 * EXTERNAL (referenced by file + offset + length in a sidecar data
 * file, typical for `onnx/*.onnx_data`).
 */
export interface OnnxTensorRef {
  name: string;
  shape: readonly number[];
  dtype: OnnxDType;
  inline: Uint8Array | null;
  /** Only set when `inline === null`. */
  extLocation: string;
  extOffset: number;
  extLength: number;
}

/** Decoded bytes for a tensor, regardless of where they were stored. */
export interface OnnxTensorBytes {
  name: string;
  shape: readonly number[];
  dtype: OnnxDType;
  bytes: Uint8Array;
}

// ---------- protobuf primitives ----------

class Reader {
  pos = 0;
  constructor(readonly bytes: Uint8Array) {}
  get eof(): boolean {
    return this.pos >= this.bytes.length;
  }
  /** Read a varint as a regular number. Safe for values ≤ 2^53. */
  varint(): number {
    let result = 0;
    let shift = 0;
    while (this.pos < this.bytes.length) {
      const b = this.bytes[this.pos++]!;
      result += (b & 0x7f) * Math.pow(2, shift);
      if ((b & 0x80) === 0) return result;
      shift += 7;
      if (shift > 56) throw new Error("varint too long");
    }
    throw new Error("varint truncated");
  }
  /** Skip a field by its wire type. */
  skip(wire: number): void {
    switch (wire) {
      case 0: this.varint(); return;                      // varint
      case 1: this.pos += 8; return;                      // 64-bit
      case 2: {                                            // length-delimited
        const len = this.varint();
        this.pos += len;
        return;
      }
      case 5: this.pos += 4; return;                      // 32-bit
      default: throw new Error(`unknown wire type ${wire}`);
    }
  }
  /** Read a length-delimited sub-message and return a sub-reader for it. */
  sub(): Reader {
    const len = this.varint();
    const sub = new Reader(this.bytes.subarray(this.pos, this.pos + len));
    this.pos += len;
    return sub;
  }
  /** Read a length-delimited string. */
  str(): string {
    const len = this.varint();
    const s = utf8Decode(this.bytes.subarray(this.pos, this.pos + len));
    this.pos += len;
    return s;
  }
}

const TD = new TextDecoder("utf-8", { fatal: false });
function utf8Decode(b: Uint8Array): string {
  return TD.decode(b);
}

// ---------- ONNX field numbers (subset) ----------

// ModelProto
const F_MODEL_GRAPH = 7;

// GraphProto
const F_GRAPH_INITIALIZER = 5;

// TensorProto
const F_TENSOR_DIMS = 1;              // repeated int64
const F_TENSOR_DATA_TYPE = 2;         // int32
const F_TENSOR_FLOAT_DATA = 4;        // repeated float, packed (FLOAT)
const F_TENSOR_INT32_DATA = 5;        // repeated int32, packed (INT32/INT16/INT8/UINT16/UINT8/BOOL/FLOAT16/BFLOAT16)
const F_TENSOR_INT64_DATA = 7;        // repeated int64 (INT64)
const F_TENSOR_NAME = 8;              // string
const F_TENSOR_RAW_DATA = 9;          // bytes — typed raw
const F_TENSOR_DOUBLE_DATA = 10;      // repeated double (DOUBLE)
const F_TENSOR_UINT64_DATA = 11;      // repeated uint64 (UINT64/UINT32)
const F_TENSOR_EXTERNAL_DATA = 13;    // repeated StringStringEntryProto
const F_TENSOR_DATA_LOCATION = 14;    // int32 (0 = DEFAULT, 1 = EXTERNAL)

// StringStringEntryProto
const F_SSEP_KEY = 1;
const F_SSEP_VALUE = 2;

// ---------- decoding ----------

function readStringStringEntry(r: Reader): { key: string; value: string } {
  let key = "";
  let value = "";
  while (!r.eof) {
    const tag = r.varint();
    const field = tag >>> 3;
    const wire = tag & 7;
    if (field === F_SSEP_KEY && wire === 2) key = r.str();
    else if (field === F_SSEP_VALUE && wire === 2) value = r.str();
    else r.skip(wire);
  }
  return { key, value };
}

function readTensorProto(r: Reader): OnnxTensorRef | null {
  let name = "";
  let dtype = 0;
  const dims: number[] = [];
  let extOffset = 0;
  let extLength = 0;
  let extLocation = "";
  let hasExternal = false;
  let inlineBytes: Uint8Array | null = null;
  while (!r.eof) {
    const tag = r.varint();
    const field = tag >>> 3;
    const wire = tag & 7;
    if (field === F_TENSOR_NAME && wire === 2) {
      name = r.str();
    } else if (field === F_TENSOR_DATA_TYPE && wire === 0) {
      dtype = r.varint();
    } else if (field === F_TENSOR_DIMS && wire === 0) {
      dims.push(r.varint());
    } else if (field === F_TENSOR_DIMS && wire === 2) {
      const sub = r.sub();
      while (!sub.eof) dims.push(sub.varint());
    } else if (field === F_TENSOR_EXTERNAL_DATA && wire === 2) {
      const entry = readStringStringEntry(r.sub());
      if (entry.key === "location") extLocation = entry.value;
      else if (entry.key === "offset") extOffset = Number(entry.value);
      else if (entry.key === "length") extLength = Number(entry.value);
      hasExternal = true;
    } else if (field === F_TENSOR_DATA_LOCATION && wire === 0) {
      if (r.varint() === 1) hasExternal = true;
    } else if (field === F_TENSOR_RAW_DATA && wire === 2) {
      const len = r.varint();
      inlineBytes = r.bytes.slice(r.pos, r.pos + len);
      r.pos += len;
    } else if (field === F_TENSOR_FLOAT_DATA && wire === 2) {
      // Packed float[] stored little-endian — already the raw-byte
      // layout our kernels consume.
      const len = r.varint();
      inlineBytes = r.bytes.slice(r.pos, r.pos + len);
      r.pos += len;
    } else {
      r.skip(wire);
    }
  }
  if (!name) return null;
  if (!hasExternal && !inlineBytes) return null;
  return {
    name,
    shape: dims,
    dtype: dtype as OnnxDType,
    inline: hasExternal ? null : inlineBytes,
    extLocation, extOffset, extLength,
  };
}

function readGraph(r: Reader, out: Map<string, OnnxTensorRef>): void {
  while (!r.eof) {
    const tag = r.varint();
    const field = tag >>> 3;
    const wire = tag & 7;
    if (field === F_GRAPH_INITIALIZER && wire === 2) {
      const sub = r.sub();
      const t = readTensorProto(sub);
      if (t) out.set(t.name, t);
    } else {
      r.skip(wire);
    }
  }
}

export function parseOnnxGraph(bytes: Uint8Array): Map<string, OnnxTensorRef> {
  const out = new Map<string, OnnxTensorRef>();
  const r = new Reader(bytes);
  while (!r.eof) {
    const tag = r.varint();
    const field = tag >>> 3;
    const wire = tag & 7;
    if (field === F_MODEL_GRAPH && wire === 2) {
      readGraph(r.sub(), out);
    } else {
      r.skip(wire);
    }
  }
  return out;
}

/**
 * Resolve a tensor's raw bytes given the parsed graph and the contents
 * of the external data file (for tensors that store `extLocation`).
 * For inline tensors, returns the bytes directly. For external tensors,
 * returns a slice of `extData`.
 */
export function resolveTensorBytes(
  tensor: OnnxTensorRef,
  extData: Uint8Array | null,
): Uint8Array {
  if (tensor.inline) return tensor.inline;
  if (!extData) {
    throw new Error(`tensor "${tensor.name}" is external but no extData was provided`);
  }
  return extData.subarray(tensor.extOffset, tensor.extOffset + tensor.extLength);
}

