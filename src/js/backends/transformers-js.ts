/**
 * Stage-0 baseline backend: `@huggingface/transformers` (transformers.js).
 *
 * Loads `openai/privacy-filter` via the low-level `AutoModelForTokenClassification`
 * and runs one forward pass per chunk, returning raw per-token logits.
 *
 * We bypass the `pipeline()` helper because it runs its own argmax /
 * BIOES fusion and only surfaces decoded `entity` strings — we need the
 * raw `[T, 33]` tensor so the same Viterbi decoder and span merger
 * drive every backend (transformers.js today, Zig+WASM later).
 *
 * Migration path: once `./wasm.js` (Zig+WASM) ships, `backends/select.ts`
 * routes around this backend; it stays as the portability fallback.
 */

import {
  AutoModelForTokenClassification,
  Tensor,
} from "@huggingface/transformers";
import type {
  BackendConstructionOptions,
  InferenceBackend,
  Logits,
} from "./abstract.js";

type PretrainedModel = Awaited<
  ReturnType<typeof AutoModelForTokenClassification.from_pretrained>
>;

interface ModelOutput {
  readonly logits: {
    readonly dims: readonly number[];
    readonly data: ArrayLike<number> & { BYTES_PER_ELEMENT?: number };
    readonly type: string;
  };
}

type TransformersDType = "fp16" | "q4f16";

/**
 * Map our `quantization` option onto a transformers.js `dtype` value that
 * ORT Web can dispatch against the ONNX exports in openai/privacy-filter.
 *
 * Available exports on the hub:
 *   model.onnx            fp32 (~5.2 GB across three shards) — too large for browsers
 *   model_fp16.onnx       fp16 weights (~2 GB) — OOM in browser even with WebGPU
 *   model_q4f16.onnx      block-int4 weights + fp16 activations (~772 MB)
 *   model_q4.onnx         block-int4 (~875 MB)
 *   model_quantized.onnx  block-int4 (~1.5 GB, "q8" suffix alias)
 *
 * model_q4f16.onnx is the smallest export that fits in a browser context.
 * It uses MatMulNBits and GatherBlockQuantized; both are implemented as WebGPU
 * contrib ops in the asyncify WASM bundle and require device: "webgpu" —
 * the WASM CPU path has no implementation for either op.
 *
 * "fp16" requests use model_fp16.onnx for maximum accuracy if the caller
 * explicitly opts in despite the size; it also requires device: "webgpu"
 * because the fp32 ONNX parse buffer (~2 GB) OOMs the WASM heap.
 */
function dtypeFor(q: "int4" | "int8" | "fp16"): TransformersDType {
  if (q === "fp16") return "fp16";
  return "q4f16";
}

export class TransformersJsBackend implements InferenceBackend {
  readonly name = "transformers-js" as const;
  private model: PretrainedModel | null = null;
  private readonly opts: BackendConstructionOptions;

  constructor(opts: BackendConstructionOptions) {
    this.opts = opts;
  }

  async warmup(): Promise<void> {
    const dtype = dtypeFor(this.opts.quantization);
    this.model = await AutoModelForTokenClassification.from_pretrained(
      this.opts.bundle.modelId,
      { dtype, device: this.opts.device },
    );
  }

  async forward(
    tokenIds: Int32Array,
    attentionMask: Uint8Array,
  ): Promise<Logits> {
    const model = this.model;
    if (!model) {
      throw new Error("TransformersJsBackend.forward called before warmup()");
    }
    if (tokenIds.length !== attentionMask.length) {
      throw new Error(
        `tokenIds / attentionMask length mismatch: ${tokenIds.length} vs ${attentionMask.length}`,
      );
    }
    const n = tokenIds.length;

    const ids = new BigInt64Array(n);
    const mask = new BigInt64Array(n);
    for (let i = 0; i < n; i++) {
      ids[i] = BigInt(tokenIds[i]!);
      mask[i] = BigInt(attentionMask[i]!);
    }
    const input_ids = new Tensor("int64", ids, [1, n]);
    const attention_mask = new Tensor("int64", mask, [1, n]);

    const output = (await (model as unknown as (inputs: Record<string, Tensor>) => Promise<ModelOutput>)(
      { input_ids, attention_mask },
    )) as ModelOutput;

    const dims = output.logits.dims;
    if (dims.length !== 3 || dims[0] !== 1 || dims[1] !== n) {
      throw new Error(
        `unexpected logits shape: ${JSON.stringify(dims)} (expected [1, ${n}, 33])`,
      );
    }
    const numClasses = dims[2]!;
    const data = toFloat32(output.logits.data, output.logits.type, n * numClasses);

    return {
      data,
      sequenceLength: n,
      numClasses,
    };
  }

  dispose(): void {
    const m = this.model as unknown as { dispose?: () => Promise<void> | void } | null;
    m?.dispose?.();
    this.model = null;
  }
}

function toFloat32(
  raw: ArrayLike<number>,
  type: string,
  expectedLength: number,
): Float32Array {
  if (raw instanceof Float32Array) {
    return raw.length === expectedLength ? raw : raw.slice(0, expectedLength);
  }
  if (type === "float16" && raw instanceof Uint16Array) {
    return float16ToFloat32(raw, expectedLength);
  }
  const out = new Float32Array(expectedLength);
  const n = Math.min(raw.length, expectedLength);
  for (let i = 0; i < n; i++) out[i] = raw[i]!;
  return out;
}

/** Convert IEEE 754 half-precision (stored as uint16) to float32. */
function float16ToFloat32(src: Uint16Array, expectedLength: number): Float32Array {
  const out = new Float32Array(expectedLength);
  const n = Math.min(src.length, expectedLength);
  for (let i = 0; i < n; i++) {
    const h = src[i]!;
    const sign = (h & 0x8000) >> 15;
    const exp = (h & 0x7c00) >> 10;
    const frac = h & 0x03ff;
    let v: number;
    if (exp === 0) {
      v = frac === 0 ? 0 : Math.pow(2, -14) * (frac / 1024);
    } else if (exp === 0x1f) {
      v = frac === 0 ? Infinity : NaN;
    } else {
      v = Math.pow(2, exp - 15) * (1 + frac / 1024);
    }
    out[i] = sign ? -v : v;
  }
  return out;
}
