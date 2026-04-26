// End-to-end native forward pass using synthetic weights.
//
// Uses the encoder-batched API: every `forward()` call issues
//   beginEncoder → ~140 enqueueDispatch → submitAndReadback
// which collapses to one wgpuQueueSubmit + one mapAsync per forward.
// Match what the browser WebGpuBackend.forward does.
//
// Weights are synthetic (seeded random bytes at production
// dimensions). Output logits are garbage but execution time is
// meaningful and directly comparable to the real model.

import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const NATIVE_PATH = resolve(HERE, "../../packages/textsift/dist/textsift-native.node");
const native = createRequire(import.meta.url)(NATIVE_PATH);

export const PF = Object.freeze({
  hiddenSize: 640,
  numHeads: 14,
  numKvHeads: 2,
  headDim: 64,
  slidingWindow: 128,
  intermediateSize: 640,
  numExpertsPerTok: 4,
  numExperts: 128,
  rmsNormEps: 1e-5,
  numLayers: 8,
  numClasses: 33,
  vocabSize: 200000,
});

function makeRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}
function rngBytes(seed, n) {
  const out = new Uint8Array(n);
  const rng = makeRng(seed);
  for (let i = 0; i < n; i++) out[i] = Math.floor(rng() * 256);
  return out;
}
function dimsBuf(values) {
  const buf = new ArrayBuffer(Math.max(16, ((values.length * 4 + 15) >> 4) << 4));
  const v = new DataView(buf);
  for (let i = 0; i < values.length; i++) {
    if (values[i].t === "u32") v.setUint32(i * 4, values[i].v, true);
    else v.setFloat32(i * 4, values[i].v, true);
  }
  return new Uint8Array(buf);
}
const u32 = (v) => ({ t: "u32", v });
const f32 = (v) => ({ t: "f32", v });

export class NativeForward {
  constructor(opts = {}) {
    this.cfg = { ...PF, ...opts };
    this.backend = native.createBackend();
    this.weights = new Map();
    this.scratch = null;
    this.maxT = 0;
    this._uploadWeights();
  }

  _addBuf(name, bytes) {
    const padded = (bytes.byteLength + 3) & ~3;
    let buf;
    if (padded === bytes.byteLength) {
      buf = bytes;
    } else {
      buf = new Uint8Array(padded);
      buf.set(bytes);
    }
    const ptr = native.createBuffer(this.backend, buf);
    this.weights.set(name, { ptr, size: padded });
  }

  _uploadWeights() {
    const cfg = this.cfg;
    const D = cfg.hiddenSize;
    const Hq = cfg.numHeads, Hkv = cfg.numKvHeads, hd = cfg.headDim;
    const dff = cfg.intermediateSize;
    const E = cfg.numExperts;
    const V = cfg.vocabSize;
    const F16 = 2, F32 = 4;
    const BLOCK = 32;
    let s = 100;

    const embedBlocks = D / BLOCK;
    const embedZpPerRow = (embedBlocks + 1) >>> 1;
    this._addBuf("embed.int4", rngBytes(s++, V * D / 2));
    this._addBuf("embed.scales", rngBytes(s++, V * embedBlocks * F16));
    this._addBuf("embed.zp", rngBytes(s++, V * embedZpPerRow));

    for (let L = 0; L < cfg.numLayers; L++) {
      const tag = (k) => `layers.${L}.${k}`;
      this._addBuf(tag("input_layernorm"), rngBytes(s++, D * F16));
      this._addBuf(tag("post_attention_layernorm"), rngBytes(s++, D * F16));

      const projDims = {
        q_proj: { N: Hq * hd, K: D },
        k_proj: { N: Hkv * hd, K: D },
        v_proj: { N: Hkv * hd, K: D },
        o_proj: { N: D, K: Hq * hd },
      };
      for (const [proj, { N, K }] of Object.entries(projDims)) {
        const blocks = K / BLOCK;
        const zpPerRow = (blocks + 1) >>> 1;
        this._addBuf(tag(`attn.${proj}.int4`), rngBytes(s++, N * K / 2));
        this._addBuf(tag(`attn.${proj}.scales`), rngBytes(s++, N * blocks * F16));
        this._addBuf(tag(`attn.${proj}.zp`), rngBytes(s++, N * zpPerRow));
        this._addBuf(tag(`attn.${proj}.bias`), rngBytes(s++, N * F16));
      }

      this._addBuf(tag("attn.sinks"), rngBytes(s++, Hq * F32));

      const routerBlocks = D / BLOCK;
      const routerZp = (routerBlocks + 1) >>> 1;
      this._addBuf(tag("router.int4"), rngBytes(s++, E * D / 2));
      this._addBuf(tag("router.scales"), rngBytes(s++, E * routerBlocks * F16));
      this._addBuf(tag("router.zp"), rngBytes(s++, E * routerZp));
      this._addBuf(tag("router.bias"), rngBytes(s++, E * F16));

      const gateUpBlocks = D / BLOCK;
      const gateUpZp = (gateUpBlocks + 1) >>> 1;
      this._addBuf(tag("experts.gate_up.int4"), rngBytes(s++, E * (2 * dff) * D / 2));
      this._addBuf(tag("experts.gate_up.scales"), rngBytes(s++, E * (2 * dff) * gateUpBlocks * F16));
      const guzp = new Uint8Array(E * (2 * dff) * gateUpZp);
      guzp.fill(0x88);
      this._addBuf(tag("experts.gate_up.zp"), guzp);
      this._addBuf(tag("experts.gate_up.bias"), rngBytes(s++, E * (2 * dff) * F16));

      const downBlocks = dff / BLOCK;
      const downZp = (downBlocks + 1) >>> 1;
      this._addBuf(tag("experts.down.int4"), rngBytes(s++, E * D * dff / 2));
      this._addBuf(tag("experts.down.scales"), rngBytes(s++, E * D * downBlocks * F16));
      const dzp = new Uint8Array(E * D * downZp);
      dzp.fill(0x88);
      this._addBuf(tag("experts.down.zp"), dzp);
      this._addBuf(tag("experts.down.bias"), rngBytes(s++, E * D * F16));
    }

    this._addBuf("final_norm", rngBytes(s++, D * F16));

    const scoreBlocks = D / BLOCK;
    const scoreZp = (scoreBlocks + 1) >>> 1;
    this._addBuf("score.int4", rngBytes(s++, cfg.numClasses * D / 2));
    this._addBuf("score.scales", rngBytes(s++, cfg.numClasses * scoreBlocks * F16));
    this._addBuf("score.zp", rngBytes(s++, cfg.numClasses * scoreZp));
    this._addBuf("score.bias", new Uint8Array(cfg.numClasses * F16));

    let totalBytes = 0;
    for (const v of this.weights.values()) totalBytes += v.size;
    console.log(
      `[NativeForward] weights uploaded: ${this.weights.size} tensors, ` +
        `${(totalBytes / 1024 / 1024).toFixed(1)} MB`,
    );
  }

  ensureScratch(maxT) {
    if (maxT <= this.maxT && this.scratch) return;
    if (this.scratch) {
      for (const ptr of Object.values(this.scratch)) native.releaseBuffer(ptr);
    }
    this.maxT = maxT;
    const cfg = this.cfg;
    const D = cfg.hiddenSize;
    const Hq = cfg.numHeads, Hkv = cfg.numKvHeads, hd = cfg.headDim;
    const dff = cfg.intermediateSize, K = cfg.numExpertsPerTok;
    const E = cfg.numExperts, NC = cfg.numClasses;
    const F16 = 2, F32 = 4;
    const T = maxT;
    const empty = (bytes) => native.createEmptyBuffer(this.backend, Math.max(16, (bytes + 3) & ~3));
    this.scratch = {
      idsBuf: empty(T * 4),
      maskBuf: empty(T),
      cosBuf: empty(T * (hd / 2) * F16),
      sinBuf: empty(T * (hd / 2) * F16),
      h0: empty(T * D * F16),
      h1: empty(T * D * F16),
      normed1: empty(T * D * F16),
      hiddenF32: empty(T * D * F32),
      qBuf: empty(T * Hq * hd * F16),
      kBuf: empty(T * Hkv * hd * F16),
      vBuf: empty(T * Hkv * hd * F16),
      attnOut: empty(T * Hq * hd * F16),
      oOut: empty(T * D * F16),
      routerLogits: empty(T * E * F32),
      routingIdx: empty(T * K * 4),
      routingScores: empty(T * K * F32),
      acc: empty(T * D * F32),
      gateUp: empty(T * K * (2 * dff) * F32),
      glu: empty(T * K * dff * F32),
      moeOut: empty(T * D * F16),
      logitsOut: empty(T * NC * F32),
    };
  }

  _ropeTables(T) {
    const cfg = this.cfg;
    const half = cfg.headDim / 2;
    const cos = new Float32Array(T * half);
    const sin = new Float32Array(T * half);
    const theta = 150000.0;
    for (let t = 0; t < T; t++) {
      for (let p = 0; p < half; p++) {
        const angle = t / Math.pow(theta, (2 * p) / cfg.headDim);
        cos[t * half + p] = Math.cos(angle);
        sin[t * half + p] = Math.sin(angle);
      }
    }
    return { cos, sin };
  }

  _packF16(arr) {
    const out = new Uint16Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      const buf = new Float32Array([v]);
      const u = new Uint32Array(buf.buffer)[0];
      const sign = (u >>> 16) & 0x8000;
      const exp = (u >>> 23) & 0xff;
      const mant = u & 0x7fffff;
      if (exp === 0xff) { out[i] = sign | 0x7c00 | (mant ? 0x200 : 0); continue; }
      if (exp === 0) { out[i] = sign; continue; }
      let e = exp - 127 + 15;
      if (e >= 0x1f) { out[i] = sign | 0x7c00; continue; }
      if (e <= 0) { out[i] = sign; continue; }
      const m = mant >>> 13;
      const r = (mant >>> 12) & 1;
      const sticky = (mant & 0xfff) !== 0;
      let h = sign | (e << 10) | m;
      if (r && (sticky || (m & 1))) h++;
      out[i] = h & 0xffff;
    }
    return new Uint8Array(out.buffer);
  }

  forward(tokenIds, attentionMask) {
    const cfg = this.cfg;
    const T = tokenIds.length;
    if (T > this.maxT) this.ensureScratch(T);
    const D = cfg.hiddenSize;
    const Hq = cfg.numHeads, Hkv = cfg.numKvHeads, hd = cfg.headDim;
    const dff = cfg.intermediateSize;
    const Kp = cfg.numExpertsPerTok;
    const E = cfg.numExperts;
    const F16 = 2, F32 = 4;
    const s = this.scratch;
    const wbuf = (name) => {
      const w = this.weights.get(name);
      return { bufPtr: w.ptr, byteLen: w.size };
    };

    // Upload tokenIds, mask, cos/sin tables (3 small queue.writeBuffer calls).
    const tokenBytes = new Uint8Array(tokenIds.buffer, tokenIds.byteOffset, T * 4);
    native.writeBuffer(this.backend, s.idsBuf, 0, tokenBytes);
    const maskPadded = new Uint8Array(Math.max(4, (T + 3) & ~3));
    maskPadded.set(attentionMask);
    native.writeBuffer(this.backend, s.maskBuf, 0, maskPadded);

    const { cos, sin } = this._ropeTables(T);
    const cosBytes = this._packF16(cos);
    const sinBytes = this._packF16(sin);
    const cosPad = new Uint8Array(Math.max(4, (cosBytes.byteLength + 3) & ~3)); cosPad.set(cosBytes);
    const sinPad = new Uint8Array(Math.max(4, (sinBytes.byteLength + 3) & ~3)); sinPad.set(sinBytes);
    native.writeBuffer(this.backend, s.cosBuf, 0, cosPad);
    native.writeBuffer(this.backend, s.sinBuf, 0, sinPad);

    let useMask = 0;
    for (let i = 0; i < T; i++) { if (attentionMask[i] !== 1) { useMask = 1; break; } }

    const enc = native.beginEncoder(this.backend);
    const enq = (name, uniform, extras, inputs, outBuf, outBinding, outBytes, dispatch, initial) => {
      const out = { binding: outBinding, bufPtr: outBuf, byteLen: outBytes };
      if (initial) out.initial = initial;
      native.enqueueDispatch(enc, name, uniform, extras, inputs, out, dispatch);
    };

    // Embed → h0
    enq(
      "embed_lookup_int4",
      dimsBuf([u32(T), u32(cfg.vocabSize), u32(D), u32(0)]),
      [],
      [
        { binding: 1, ...wbuf("embed.int4") },
        { binding: 2, ...wbuf("embed.scales") },
        { binding: 3, ...wbuf("embed.zp") },
        { binding: 4, bufPtr: s.idsBuf, byteLen: T * 4 },
      ],
      s.h0, 5, T * D * F16,
      [Math.ceil(T * D / 64), 1, 1],
    );

    let hCur = s.h0, hAlt = s.h1;
    for (let L = 0; L < cfg.numLayers; L++) {
      const tag = (k) => `layers.${L}.${k}`;

      // input_rmsnorm → normed1
      enq(
        "rms_norm",
        dimsBuf([u32(T), u32(D), f32(cfg.rmsNormEps), u32(0)]),
        [],
        [
          { binding: 1, bufPtr: hCur, byteLen: T * D * F16 },
          { binding: 2, ...wbuf(tag("input_layernorm")) },
        ],
        s.normed1, 3, T * D * F16,
        [T, 1, 1],
      );

      // QKV projections + O projection
      const matmul = (weightsBase, xPtr, yPtr, N, K) => {
        enq(
          "matmul_int4_fp16_f16",
          dimsBuf([u32(T), u32(N), u32(K), u32(0)]),
          [],
          [
            { binding: 1, bufPtr: xPtr, byteLen: T * K * F16 },
            { binding: 2, ...wbuf(`${weightsBase}.int4`) },
            { binding: 3, ...wbuf(`${weightsBase}.scales`) },
            { binding: 4, ...wbuf(`${weightsBase}.zp`) },
            { binding: 5, ...wbuf(`${weightsBase}.bias`) },
          ],
          yPtr, 6, T * N * F16,
          [Math.ceil(N / 64), Math.ceil(T / 4), 1],
        );
      };
      matmul(tag("attn.q_proj"), s.normed1, s.qBuf, Hq * hd, D);
      matmul(tag("attn.k_proj"), s.normed1, s.kBuf, Hkv * hd, D);
      matmul(tag("attn.v_proj"), s.normed1, s.vBuf, Hkv * hd, D);

      // RoPE in-place: qkBuf at binding 1 (read_write), cos/sin at 2/3.
      // The encoder API writes the output to qkBuf directly, so we don't
      // need to pass an initial — the buffer already holds the QKV output.
      const rope = (qkPtr, heads) => {
        const half = hd / 2;
        enq(
          "rope_apply",
          dimsBuf([u32(T), u32(heads), u32(hd), u32(0)]),
          [],
          [
            { binding: 2, bufPtr: s.cosBuf, byteLen: T * half * F16 },
            { binding: 3, bufPtr: s.sinBuf, byteLen: T * half * F16 },
          ],
          qkPtr, 1, T * heads * hd * F16,
          [Math.ceil(T * heads * half / 64), 1, 1],
        );
      };
      rope(s.qBuf, Hq);
      rope(s.kBuf, Hkv);

      // Banded attention
      enq(
        "banded_attention",
        dimsBuf([
          u32(T), u32(Hq), u32(Hkv), u32(hd),
          u32(cfg.slidingWindow), u32(useMask), u32(0), u32(0),
        ]),
        [],
        [
          { binding: 1, bufPtr: s.qBuf, byteLen: T * Hq * hd * F16 },
          { binding: 2, bufPtr: s.kBuf, byteLen: T * Hkv * hd * F16 },
          { binding: 3, bufPtr: s.vBuf, byteLen: T * Hkv * hd * F16 },
          { binding: 4, ...wbuf(tag("attn.sinks")) },
          { binding: 5, bufPtr: s.maskBuf, byteLen: maskPadded.byteLength },
        ],
        s.attnOut, 6, T * Hq * hd * F16,
        [T * Hq, 1, 1],
      );

      // O projection
      matmul(tag("attn.o_proj"), s.attnOut, s.oOut, D, Hq * hd);

      // sum residual: hCur + oOut → normed1 (residual stream feeding next add)
      enq(
        "add_fp16",
        dimsBuf([u32(T * D), u32(0), u32(0), u32(0)]),
        [],
        [
          { binding: 1, bufPtr: hCur, byteLen: T * D * F16 },
          { binding: 2, bufPtr: s.oOut, byteLen: T * D * F16 },
        ],
        s.normed1, 3, T * D * F16,
        [Math.ceil(T * D / 64), 1, 1],
      );

      // post_attn_layernorm(normed1) → moeOut (reused as scratch fp16)
      enq(
        "rms_norm",
        dimsBuf([u32(T), u32(D), f32(cfg.rmsNormEps), u32(0)]),
        [],
        [
          { binding: 1, bufPtr: s.normed1, byteLen: T * D * F16 },
          { binding: 2, ...wbuf(tag("post_attention_layernorm")) },
        ],
        s.moeOut, 3, T * D * F16,
        [T, 1, 1],
      );

      // widen → hiddenF32
      enq(
        "cast_fp16_to_f32",
        dimsBuf([u32(T * D), u32(0), u32(0), u32(0)]),
        [],
        [
          { binding: 1, bufPtr: s.moeOut, byteLen: T * D * F16 },
        ],
        s.hiddenF32, 2, T * D * F32,
        [Math.ceil(T * D / 64), 1, 1],
      );

      // Router matmul
      enq(
        "matmul_int4_f32_f32",
        dimsBuf([u32(T), u32(E), u32(D), u32(0)]),
        [],
        [
          { binding: 1, bufPtr: s.hiddenF32, byteLen: T * D * F32 },
          { binding: 2, ...wbuf(tag("router.int4")) },
          { binding: 3, ...wbuf(tag("router.scales")) },
          { binding: 4, ...wbuf(tag("router.zp")) },
          { binding: 5, ...wbuf(tag("router.bias")) },
        ],
        s.routerLogits, 6, T * E * F32,
        [Math.ceil(E / 64), Math.ceil(T / 4), 1],
      );

      // router_topk: writes routingIdx (binding 2) AND routingScores (binding 3).
      // The encoder API exposes one output per dispatch; routingIdx and
      // routingScores are bound as read_write storage in the kernel, so
      // both get written. We pass routingScores as the "primary output"
      // (any of the writable bindings works — Zig binds them by index)
      // and pass routingIdx as a writable storage input.
      enq(
        "router_topk",
        dimsBuf([u32(T), u32(E), u32(Kp), u32(0)]),
        [],
        [
          { binding: 1, bufPtr: s.routerLogits, byteLen: T * E * F32 },
          { binding: 2, bufPtr: s.routingIdx, byteLen: T * Kp * 4 },
        ],
        s.routingScores, 3, T * Kp * F32,
        [Math.ceil(T / 64), 1, 1],
      );

      // qmoe_gate_up
      enq(
        "qmoe_gate_up",
        dimsBuf([
          u32(T), u32(Kp), u32(2 * dff), u32(D),
          u32(0), u32(0), u32(0), u32(0),
        ]),
        [],
        [
          { binding: 1, bufPtr: s.hiddenF32, byteLen: T * D * F32 },
          { binding: 2, bufPtr: s.routingIdx, byteLen: T * Kp * 4 },
          { binding: 3, ...wbuf(tag("experts.gate_up.int4")) },
          { binding: 4, ...wbuf(tag("experts.gate_up.scales")) },
          { binding: 5, ...wbuf(tag("experts.gate_up.zp")) },
          { binding: 6, ...wbuf(tag("experts.gate_up.bias")) },
        ],
        s.gateUp, 7, T * Kp * (2 * dff) * F32,
        [T * Kp, Math.ceil((2 * dff) / 64), 1],
      );

      // swiglu_clamp
      enq(
        "swiglu_clamp",
        dimsBuf([u32(T * Kp), u32(dff), u32(0), u32(0)]),
        [],
        [
          { binding: 1, bufPtr: s.gateUp, byteLen: T * Kp * (2 * dff) * F32 },
        ],
        s.glu, 2, T * Kp * dff * F32,
        [Math.ceil(T * Kp * dff / 64), 1, 1],
      );

      // qmoe_down_scatter (atomic accumulate; pre-zero acc each forward)
      const accInit = new Uint8Array(T * D * F32);
      enq(
        "qmoe_down_scatter",
        dimsBuf([
          u32(T), u32(Kp), u32(D), u32(dff),
          u32(0), u32(0), u32(0), u32(0),
        ]),
        [],
        [
          { binding: 1, bufPtr: s.glu, byteLen: T * Kp * dff * F32 },
          { binding: 2, bufPtr: s.routingIdx, byteLen: T * Kp * 4 },
          { binding: 3, bufPtr: s.routingScores, byteLen: T * Kp * F32 },
          { binding: 4, ...wbuf(tag("experts.down.int4")) },
          { binding: 5, ...wbuf(tag("experts.down.scales")) },
          { binding: 6, ...wbuf(tag("experts.down.zp")) },
          { binding: 7, ...wbuf(tag("experts.down.bias")) },
        ],
        s.acc, 8, T * D * F32,
        [T * Kp, Math.ceil(D / 64), 1],
        accInit,
      );

      // cast_f32_to_fp16_scaled: acc * Kp → moeOut (fp16)
      const scaleBuf = (() => {
        const b = new ArrayBuffer(16);
        new Float32Array(b, 0, 1).set([Kp]);
        return new Uint8Array(b);
      })();
      enq(
        "cast_f32_to_fp16_scaled",
        dimsBuf([u32(T * D), u32(0), u32(0), u32(0)]),
        [{ binding: 1, bytes: scaleBuf }],
        [
          { binding: 2, bufPtr: s.acc, byteLen: T * D * F32 },
        ],
        s.moeOut, 3, T * D * F16,
        [Math.ceil(T * D / 64), 1, 1],
      );

      // residual: normed1 + moeOut → hAlt
      enq(
        "add_fp16",
        dimsBuf([u32(T * D), u32(0), u32(0), u32(0)]),
        [],
        [
          { binding: 1, bufPtr: s.normed1, byteLen: T * D * F16 },
          { binding: 2, bufPtr: s.moeOut, byteLen: T * D * F16 },
        ],
        hAlt, 3, T * D * F16,
        [Math.ceil(T * D / 64), 1, 1],
      );

      const tmp = hCur; hCur = hAlt; hAlt = tmp;
    }

    // Final rmsnorm + cast + classifier head
    enq(
      "rms_norm",
      dimsBuf([u32(T), u32(D), f32(cfg.rmsNormEps), u32(0)]),
      [],
      [
        { binding: 1, bufPtr: hCur, byteLen: T * D * F16 },
        { binding: 2, ...wbuf("final_norm") },
      ],
      s.normed1, 3, T * D * F16,
      [T, 1, 1],
    );

    enq(
      "cast_fp16_to_f32",
      dimsBuf([u32(T * D), u32(0), u32(0), u32(0)]),
      [],
      [
        { binding: 1, bufPtr: s.normed1, byteLen: T * D * F16 },
      ],
      s.hiddenF32, 2, T * D * F32,
      [Math.ceil(T * D / 64), 1, 1],
    );

    enq(
      "matmul_int4_f32_f32",
      dimsBuf([u32(T), u32(cfg.numClasses), u32(D), u32(0)]),
      [],
      [
        { binding: 1, bufPtr: s.hiddenF32, byteLen: T * D * F32 },
        { binding: 2, ...wbuf("score.int4") },
        { binding: 3, ...wbuf("score.scales") },
        { binding: 4, ...wbuf("score.zp") },
        { binding: 5, ...wbuf("score.bias") },
      ],
      s.logitsOut, 6, T * cfg.numClasses * F32,
      [Math.ceil(cfg.numClasses / 64), Math.ceil(T / 4), 1],
    );

    // Submit + readback final logits.
    const out = native.submitAndReadback(enc, s.logitsOut, 0, T * cfg.numClasses * 4);
    return new Float32Array(out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength));
  }

  dispose() {
    for (const v of this.weights.values()) native.releaseBuffer(v.ptr);
    if (this.scratch) {
      for (const ptr of Object.values(this.scratch)) native.releaseBuffer(ptr);
    }
    native.destroyBackend(this.backend);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const fwd = new NativeForward();
  const T = parseInt(process.env.T ?? "32", 10);
  fwd.ensureScratch(T);
  console.log(`[NativeForward] scratch allocated for T=${T}`);
  const ids = new Int32Array(T);
  for (let i = 0; i < ids.length; i++) ids[i] = (i * 7919) % PF.vocabSize;
  const mask = new Uint8Array(T).fill(1);

  for (let i = 0; i < 3; i++) fwd.forward(ids, mask);

  const N = 10;
  const samples = [];
  for (let i = 0; i < N; i++) {
    const t0 = performance.now();
    const logits = fwd.forward(ids, mask);
    samples.push(performance.now() - t0);
    if (i === 0) console.log(`[NativeForward] logits shape=(${T}, ${PF.numClasses}) len=${logits.length}`);
  }
  samples.sort((a, b) => a - b);
  console.log(
    `[NativeForward] forward latency T=${T}: ` +
      `median=${samples[Math.floor(N/2)].toFixed(1)}ms, ` +
      `min=${samples[0].toFixed(1)}ms, ` +
      `over ${N} iters`,
  );
  fwd.dispose();
  console.log("[NativeForward] disposed cleanly");
}
