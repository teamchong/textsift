/**
 * YARN-scaled RoPE tables.
 *
 * Mirrors `_compute_yarn_parameters` and the `OpenAIPrivacyFilterRotaryEmbedding`
 * forward pass from `transformers/models/openai_privacy_filter/modeling_...py`.
 * Produces byte-identical cos/sin tables to the reference (up to f32 rounding).
 */

export interface YarnRopeConfig {
  /** head_dim (= 64 for openai/privacy-filter). */
  headDim: number;
  /** rope_theta (= 150000 upstream). */
  theta: number;
  /** `factor` — post-yarn / pre-yarn context ratio (= 32 upstream). */
  factor: number;
  /** Max positions seen during pretraining (= 4096 upstream). */
  originalMaxPositionEmbeddings: number;
  /** Extrapolation upper bound (= 32 upstream). */
  betaFast: number;
  /** Interpolation lower bound (= 1 upstream). */
  betaSlow: number;
  /** `truncate` flag on the rope_parameters dict (= false upstream). */
  truncate: boolean;
}

/** YARN inverse-frequency table of shape `[head_dim / 2]`. */
export function computeYarnInvFreq(cfg: YarnRopeConfig): Float32Array {
  const { headDim, theta, factor, originalMaxPositionEmbeddings, betaFast, betaSlow, truncate } = cfg;
  const halfDim = headDim / 2;

  const invFreqExtrapolation = new Float32Array(halfDim);
  const invFreqInterpolation = new Float32Array(halfDim);
  for (let i = 0; i < halfDim; i++) {
    const posFreq = Math.pow(theta, (2 * i) / headDim);
    invFreqExtrapolation[i] = 1 / posFreq;
    invFreqInterpolation[i] = 1 / (factor * posFreq);
  }

  // Inverse dimension formula — finds the dim index at which a full
  // wavelength reaches `numRotations` of rotation over originalMax.
  const findCorrectionDim = (numRotations: number): number =>
    (headDim * Math.log(originalMaxPositionEmbeddings / (numRotations * 2 * Math.PI))) /
    (2 * Math.log(theta));

  let low = findCorrectionDim(betaFast);
  let high = findCorrectionDim(betaSlow);
  if (truncate) {
    low = Math.floor(low);
    high = Math.ceil(high);
  }
  low = Math.max(low, 0);
  high = Math.min(high, headDim - 1);
  const rampMin = low;
  const rampMax = high === low ? low + 0.001 : high;

  const invFreq = new Float32Array(halfDim);
  for (let i = 0; i < halfDim; i++) {
    const linear = (i - rampMin) / (rampMax - rampMin);
    const ramp = Math.min(Math.max(linear, 0), 1);
    const extrapFactor = 1 - ramp;
    invFreq[i] =
      invFreqInterpolation[i]! * (1 - extrapFactor) +
      invFreqExtrapolation[i]! * extrapFactor;
  }
  return invFreq;
}

/**
 * Attention scaling factor = `0.1 * log(factor) + 1` for factor > 1, else 1.
 * Baked into the cos/sin tables so the apply kernel sees pre-scaled values.
 */
export function getYarnAttentionScaling(factor: number): number {
  if (factor <= 1) return 1.0;
  return 0.1 * Math.log(factor) + 1.0;
}

/** f32 → bf16 with round-to-nearest-ties-to-even. Matches PyTorch's cast. */
function f32ToBf16(f: number): number {
  const buf = new ArrayBuffer(4);
  new DataView(buf).setFloat32(0, f, true);
  const bits = new DataView(buf).getUint32(0, true);
  if ((bits & 0x7f800000) === 0x7f800000 && (bits & 0x007fffff) !== 0) {
    // NaN: preserve, set top mantissa bit.
    return ((bits | 0x00400000) >>> 16) & 0xffff;
  }
  const rounded = (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 0;
  return (rounded >>> 16) & 0xffff;
}

/**
 * Build cos/sin tables for positions `[0, seqLen)` as bf16. PyTorch's
 * `OpenAIPrivacyFilterRotaryEmbedding.forward` computes in f32 and
 * downcasts to the caller's dtype (bf16 in Phase D); the apply path
 * reads cos/sin as bf16. Storing bf16 here guarantees byte-identical
 * tables vs the PyTorch reference.
 */
export function computeRopeTables(
  invFreq: Float32Array,
  seqLen: number,
  attentionScaling: number,
): { cos: Uint16Array; sin: Uint16Array } {
  const halfDim = invFreq.length;
  const cos = new Uint16Array(seqLen * halfDim);
  const sin = new Uint16Array(seqLen * halfDim);
  for (let t = 0; t < seqLen; t++) {
    for (let i = 0; i < halfDim; i++) {
      const angle = t * invFreq[i]!;
      cos[t * halfDim + i] = f32ToBf16(Math.cos(angle) * attentionScaling);
      sin[t * halfDim + i] = f32ToBf16(Math.sin(angle) * attentionScaling);
    }
  }
  return { cos, sin };
}

/** Convenience: config → tables, matching PyTorch's behaviour. */
export function buildRopeTables(
  cfg: YarnRopeConfig,
  seqLen: number,
): { cos: Uint16Array; sin: Uint16Array; attentionScaling: number } {
  const invFreq = computeYarnInvFreq(cfg);
  const attentionScaling = getYarnAttentionScaling(cfg.factor);
  const { cos, sin } = computeRopeTables(invFreq, seqLen, attentionScaling);
  return { cos, sin, attentionScaling };
}
