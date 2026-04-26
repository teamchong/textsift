// Bench-only helper: a transformers.js-backed InferenceBackend used
// as the baseline in `bench.spec.ts`. NOT shipped in the textsift
// package — `textsift` doesn't depend on transformers.js. Lives here
// as plain ESM JS so bench.html can import it via importmap without
// a separate compile step.
//
// Loads `openai/privacy-filter` via AutoModelForTokenClassification
// and runs one forward per chunk, returning raw logits in the same
// shape our other backends produce. Bypasses the high-level
// `pipeline()` so the same Viterbi decoder drives every backend.

import {
  AutoModelForTokenClassification,
  Tensor,
} from "@huggingface/transformers";

function dtypeFor(q) {
  if (q === "fp16") return "fp16";
  return "q4f16";
}

export class TransformersJsBackend {
  constructor(opts) {
    this.name = "transformers-js";
    this.model = null;
    this.opts = opts;
  }

  async warmup() {
    const dtype = dtypeFor(this.opts.quantization);
    this.model = await AutoModelForTokenClassification.from_pretrained(
      this.opts.bundle.modelId,
      { dtype, device: this.opts.device },
    );
  }

  async forward(tokenIds, attentionMask) {
    if (!this.model) {
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
      ids[i] = BigInt(tokenIds[i]);
      mask[i] = BigInt(attentionMask[i]);
    }
    const input_ids = new Tensor("int64", ids, [1, n]);
    const attention_mask = new Tensor("int64", mask, [1, n]);

    const output = await this.model({ input_ids, attention_mask });
    const dims = output.logits.dims;
    if (dims.length !== 3 || dims[0] !== 1 || dims[1] !== n) {
      throw new Error(
        `unexpected logits shape: ${JSON.stringify(dims)} (expected [1, ${n}, 33])`,
      );
    }
    const numClasses = dims[2];
    const data = toFloat32(output.logits.data, output.logits.type, n * numClasses);

    return { data, sequenceLength: n, numClasses };
  }

  dispose() {
    this.model?.dispose?.();
    this.model = null;
  }
}

function toFloat32(raw, type, expectedLength) {
  if (raw instanceof Float32Array) {
    return raw.length === expectedLength ? raw : raw.slice(0, expectedLength);
  }
  if (type === "float16" && raw instanceof Uint16Array) {
    return float16ToFloat32(raw, expectedLength);
  }
  const out = new Float32Array(expectedLength);
  const n = Math.min(raw.length, expectedLength);
  for (let i = 0; i < n; i++) out[i] = raw[i];
  return out;
}

function float16ToFloat32(src, expectedLength) {
  const out = new Float32Array(expectedLength);
  const n = Math.min(src.length, expectedLength);
  for (let i = 0; i < n; i++) {
    const h = src[i];
    const sign = (h & 0x8000) >> 15;
    const exp = (h & 0x7c00) >> 10;
    const frac = h & 0x03ff;
    let v;
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
