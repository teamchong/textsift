// End-to-end Metal-direct forward pass using synthetic weights.
//
// Mirrors tests/native/forward.js exactly but uses the Metal-direct
// backend (hand-written MSL via Obj-C bridge). Same dispatches, same
// dispatch shapes, same shared scratch buffers — the only difference
// is which API actually compiles and dispatches the kernels.
//
// Use this to bench Metal-direct vs wgpu-native vs the browser path
// without any other variables changing.

import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { PF } from "./forward.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const NATIVE_PATH = resolve(HERE, "../../packages/textsift/dist/textsift-native.node");
const native = createRequire(import.meta.url)(NATIVE_PATH);

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
const f32v = (v) => ({ t: "f32", v });

export class MetalForward {
  constructor(opts = {}) {
    this.cfg = { ...PF, ...opts };
    this.backend = native.metalCreateBackend();
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
    const ptr = native.metalCreateBuffer(this.backend, buf);
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
      const guzp = new Uint8Array(E * (2 * dff) * gateUpZp); guzp.fill(0x88);
      this._addBuf(tag("experts.gate_up.zp"), guzp);
      this._addBuf(tag("experts.gate_up.bias"), rngBytes(s++, E * (2 * dff) * F16));

      const downBlocks = dff / BLOCK;
      const downZp = (downBlocks + 1) >>> 1;
      this._addBuf(tag("experts.down.int4"), rngBytes(s++, E * D * dff / 2));
      this._addBuf(tag("experts.down.scales"), rngBytes(s++, E * D * downBlocks * F16));
      const dzp = new Uint8Array(E * D * downZp); dzp.fill(0x88);
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
      `[MetalForward] weights uploaded: ${this.weights.size} tensors, ` +
        `${(totalBytes / 1024 / 1024).toFixed(1)} MB`,
    );
  }

  ensureScratch(maxT) {
    if (maxT <= this.maxT && this.scratch) return;
    if (this.scratch) {
      for (const ptr of Object.values(this.scratch)) native.metalReleaseBuffer(ptr);
    }
    this.maxT = maxT;
    const cfg = this.cfg;
    const D = cfg.hiddenSize;
    const Hq = cfg.numHeads, Hkv = cfg.numKvHeads, hd = cfg.headDim;
    const dff = cfg.intermediateSize, K = cfg.numExpertsPerTok;
    const E = cfg.numExperts, NC = cfg.numClasses;
    const F16 = 2, F32 = 4;
    const T = maxT;
    const empty = (bytes) => native.metalCreateEmptyBuffer(this.backend, Math.max(16, (bytes + 3) & ~3));
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
    const wbuf = (name) => this.weights.get(name).ptr;

    // Upload tokenIds, mask, cos/sin tables (small writes, before the encoder).
    const tokenBytes = new Uint8Array(tokenIds.buffer, tokenIds.byteOffset, T * 4);
    native.metalWriteBuffer(s.idsBuf, 0, tokenBytes);
    const maskPadded = new Uint8Array(Math.max(4, (T + 3) & ~3));
    maskPadded.set(attentionMask);
    native.metalWriteBuffer(s.maskBuf, 0, maskPadded);

    const { cos, sin } = this._ropeTables(T);
    const cosBytes = this._packF16(cos);
    const sinBytes = this._packF16(sin);
    const cosPad = new Uint8Array(Math.max(4, (cosBytes.byteLength + 3) & ~3)); cosPad.set(cosBytes);
    const sinPad = new Uint8Array(Math.max(4, (sinBytes.byteLength + 3) & ~3)); sinPad.set(sinBytes);
    native.metalWriteBuffer(s.cosBuf, 0, cosPad);
    native.metalWriteBuffer(s.sinBuf, 0, sinPad);

    let useMask = 0;
    for (let i = 0; i < T; i++) { if (attentionMask[i] !== 1) { useMask = 1; break; } }

    // Pre-zero the MoE accumulator each forward (down_scatter atomic-adds).
    const accInit = new Uint8Array(T * D * F32);
    native.metalWriteBuffer(s.acc, 0, accInit);

    this._tEncStart = performance.now();
    const enc = native.metalBeginEncoder(this.backend);

    // Metal binding helper: uniform always at index 0, then bufs by index.
    const dispatch = (name, uniform, bufBindings, grid, tg) => {
      const bindings = [{ index: 0, bytes: uniform }];
      for (const b of bufBindings) bindings.push(b);
      native.metalEnqueueDispatch(enc, name, bindings, grid, tg);
    };

    // Embed → h0
    dispatch(
      "embed_lookup_int4",
      dimsBuf([u32(T), u32(cfg.vocabSize), u32(D), u32(0)]),
      [
        { index: 1, bufPtr: wbuf("embed.int4") },
        { index: 2, bufPtr: wbuf("embed.scales") },
        { index: 3, bufPtr: wbuf("embed.zp") },
        { index: 4, bufPtr: s.idsBuf },
        { index: 5, bufPtr: s.h0 },
      ],
      [Math.ceil(T * D / 64), 1, 1], [64, 1, 1],
    );

    let hCur = s.h0, hAlt = s.h1;
    for (let L = 0; L < cfg.numLayers; L++) {
      const tag = (k) => `layers.${L}.${k}`;

      // input_rmsnorm → normed1
      dispatch(
        "rms_norm",
        dimsBuf([u32(T), u32(D), f32v(cfg.rmsNormEps), u32(0)]),
        [
          { index: 1, bufPtr: hCur },
          { index: 2, bufPtr: wbuf(tag("input_layernorm")) },
          { index: 3, bufPtr: s.normed1 },
        ],
        [T, 1, 1], [64, 1, 1],
      );

      const matmul = (weightsBase, xPtr, yPtr, N, K) => {
        dispatch(
          "matmul_int4_fp16_f16",
          dimsBuf([u32(T), u32(N), u32(K), u32(0)]),
          [
            { index: 1, bufPtr: xPtr },
            { index: 2, bufPtr: wbuf(`${weightsBase}.int4`) },
            { index: 3, bufPtr: wbuf(`${weightsBase}.scales`) },
            { index: 4, bufPtr: wbuf(`${weightsBase}.zp`) },
            { index: 5, bufPtr: wbuf(`${weightsBase}.bias`) },
            { index: 6, bufPtr: yPtr },
          ],
          [Math.ceil(N / 64), Math.ceil(T / 4), 1], [64, 1, 1],
        );
      };
      matmul(tag("attn.q_proj"), s.normed1, s.qBuf, Hq * hd, D);
      matmul(tag("attn.k_proj"), s.normed1, s.kBuf, Hkv * hd, D);
      matmul(tag("attn.v_proj"), s.normed1, s.vBuf, Hkv * hd, D);

      const rope = (qkPtr, heads) => {
        const half = hd / 2;
        dispatch(
          "rope_apply",
          dimsBuf([u32(T), u32(heads), u32(hd), u32(0)]),
          [
            { index: 1, bufPtr: qkPtr },
            { index: 2, bufPtr: s.cosBuf },
            { index: 3, bufPtr: s.sinBuf },
          ],
          [Math.ceil(T * heads * half / 64), 1, 1], [64, 1, 1],
        );
      };
      rope(s.qBuf, Hq);
      rope(s.kBuf, Hkv);

      // Banded attention
      dispatch(
        "banded_attention",
        dimsBuf([
          u32(T), u32(Hq), u32(Hkv), u32(hd),
          u32(cfg.slidingWindow), u32(useMask), u32(0), u32(0),
        ]),
        [
          { index: 1, bufPtr: s.qBuf },
          { index: 2, bufPtr: s.kBuf },
          { index: 3, bufPtr: s.vBuf },
          { index: 4, bufPtr: wbuf(tag("attn.sinks")) },
          { index: 5, bufPtr: s.maskBuf },
          { index: 6, bufPtr: s.attnOut },
        ],
        [T * Hq, 1, 1], [64, 1, 1],
      );

      // O projection
      matmul(tag("attn.o_proj"), s.attnOut, s.oOut, D, Hq * hd);

      // hCur + oOut → normed1
      dispatch(
        "add_fp16",
        dimsBuf([u32(T * D), u32(0), u32(0), u32(0)]),
        [
          { index: 1, bufPtr: hCur },
          { index: 2, bufPtr: s.oOut },
          { index: 3, bufPtr: s.normed1 },
        ],
        [Math.ceil(T * D / 64), 1, 1], [64, 1, 1],
      );

      // post_attn_layernorm(normed1) → moeOut
      dispatch(
        "rms_norm",
        dimsBuf([u32(T), u32(D), f32v(cfg.rmsNormEps), u32(0)]),
        [
          { index: 1, bufPtr: s.normed1 },
          { index: 2, bufPtr: wbuf(tag("post_attention_layernorm")) },
          { index: 3, bufPtr: s.moeOut },
        ],
        [T, 1, 1], [64, 1, 1],
      );

      // widen → hiddenF32
      dispatch(
        "cast_fp16_to_f32",
        dimsBuf([u32(T * D), u32(0), u32(0), u32(0)]),
        [
          { index: 1, bufPtr: s.moeOut },
          { index: 2, bufPtr: s.hiddenF32 },
        ],
        [Math.ceil(T * D / 64), 1, 1], [64, 1, 1],
      );

      // Router matmul
      dispatch(
        "matmul_int4_f32_f32",
        dimsBuf([u32(T), u32(E), u32(D), u32(0)]),
        [
          { index: 1, bufPtr: s.hiddenF32 },
          { index: 2, bufPtr: wbuf(tag("router.int4")) },
          { index: 3, bufPtr: wbuf(tag("router.scales")) },
          { index: 4, bufPtr: wbuf(tag("router.zp")) },
          { index: 5, bufPtr: wbuf(tag("router.bias")) },
          { index: 6, bufPtr: s.routerLogits },
        ],
        [Math.ceil(E / 64), Math.ceil(T / 4), 1], [64, 1, 1],
      );

      // router_topk
      dispatch(
        "router_topk",
        dimsBuf([u32(T), u32(E), u32(Kp), u32(0)]),
        [
          { index: 1, bufPtr: s.routerLogits },
          { index: 2, bufPtr: s.routingIdx },
          { index: 3, bufPtr: s.routingScores },
        ],
        [Math.ceil(T / 64), 1, 1], [64, 1, 1],
      );

      // qmoe_gate_up
      dispatch(
        "qmoe_gate_up",
        dimsBuf([
          u32(T), u32(Kp), u32(2 * dff), u32(D),
          u32(0), u32(0), u32(0), u32(0),
        ]),
        [
          { index: 1, bufPtr: s.hiddenF32 },
          { index: 2, bufPtr: s.routingIdx },
          { index: 3, bufPtr: wbuf(tag("experts.gate_up.int4")) },
          { index: 4, bufPtr: wbuf(tag("experts.gate_up.scales")) },
          { index: 5, bufPtr: wbuf(tag("experts.gate_up.zp")) },
          { index: 6, bufPtr: wbuf(tag("experts.gate_up.bias")) },
          { index: 7, bufPtr: s.gateUp },
        ],
        [T * Kp, Math.ceil((2 * dff) / 64), 1], [64, 1, 1],
      );

      // swiglu_clamp
      dispatch(
        "swiglu_clamp",
        dimsBuf([u32(T * Kp), u32(dff), u32(0), u32(0)]),
        [
          { index: 1, bufPtr: s.gateUp },
          { index: 2, bufPtr: s.glu },
        ],
        [Math.ceil(T * Kp * dff / 64), 1, 1], [64, 1, 1],
      );

      // qmoe_down_scatter (s.acc was zeroed before the encoder)
      dispatch(
        "qmoe_down_scatter",
        dimsBuf([
          u32(T), u32(Kp), u32(D), u32(dff),
          u32(0), u32(0), u32(0), u32(0),
        ]),
        [
          { index: 1, bufPtr: s.glu },
          { index: 2, bufPtr: s.routingIdx },
          { index: 3, bufPtr: s.routingScores },
          { index: 4, bufPtr: wbuf(tag("experts.down.int4")) },
          { index: 5, bufPtr: wbuf(tag("experts.down.scales")) },
          { index: 6, bufPtr: wbuf(tag("experts.down.zp")) },
          { index: 7, bufPtr: wbuf(tag("experts.down.bias")) },
          { index: 8, bufPtr: s.acc },
        ],
        [T * Kp, Math.ceil(D / 64), 1], [64, 1, 1],
      );

      // cast_f32_to_fp16_scaled: acc * Kp → moeOut (fp16)
      const scaleBuf = (() => {
        const b = new ArrayBuffer(16);
        new Float32Array(b, 0, 1).set([Kp]);
        return new Uint8Array(b);
      })();
      dispatch(
        "cast_f32_to_fp16_scaled",
        dimsBuf([u32(T * D), u32(0), u32(0), u32(0)]),
        [
          { index: 1, bytes: scaleBuf },
          { index: 2, bufPtr: s.acc },
          { index: 3, bufPtr: s.moeOut },
        ],
        [Math.ceil(T * D / 64), 1, 1], [64, 1, 1],
      );

      // residual: normed1 + moeOut → hAlt
      dispatch(
        "add_fp16",
        dimsBuf([u32(T * D), u32(0), u32(0), u32(0)]),
        [
          { index: 1, bufPtr: s.normed1 },
          { index: 2, bufPtr: s.moeOut },
          { index: 3, bufPtr: hAlt },
        ],
        [Math.ceil(T * D / 64), 1, 1], [64, 1, 1],
      );

      const tmp = hCur; hCur = hAlt; hAlt = tmp;
    }

    // Final rmsnorm + cast + classifier head
    dispatch(
      "rms_norm",
      dimsBuf([u32(T), u32(D), f32v(cfg.rmsNormEps), u32(0)]),
      [
        { index: 1, bufPtr: hCur },
        { index: 2, bufPtr: wbuf("final_norm") },
        { index: 3, bufPtr: s.normed1 },
      ],
      [T, 1, 1], [64, 1, 1],
    );

    dispatch(
      "cast_fp16_to_f32",
      dimsBuf([u32(T * D), u32(0), u32(0), u32(0)]),
      [
        { index: 1, bufPtr: s.normed1 },
        { index: 2, bufPtr: s.hiddenF32 },
      ],
      [Math.ceil(T * D / 64), 1, 1], [64, 1, 1],
    );

    dispatch(
      "matmul_int4_f32_f32",
      dimsBuf([u32(T), u32(cfg.numClasses), u32(D), u32(0)]),
      [
        { index: 1, bufPtr: s.hiddenF32 },
        { index: 2, bufPtr: wbuf("score.int4") },
        { index: 3, bufPtr: wbuf("score.scales") },
        { index: 4, bufPtr: wbuf("score.zp") },
        { index: 5, bufPtr: wbuf("score.bias") },
        { index: 6, bufPtr: s.logitsOut },
      ],
      [Math.ceil(cfg.numClasses / 64), Math.ceil(T / 4), 1], [64, 1, 1],
    );

    const tEncEnd = performance.now();
    const out = native.metalSubmitAndReadback(enc, s.logitsOut, 0, T * cfg.numClasses * 4);
    const tSubEnd = performance.now();
    if (process.env.PROFILE) {
      this._lastEncodeMs = tEncEnd - this._tEncStart;
      this._lastSubmitMs = tSubEnd - tEncEnd;
    }
    return new Float32Array(out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength));
  }

  dispose() {
    for (const v of this.weights.values()) native.metalReleaseBuffer(v.ptr);
    if (this.scratch) {
      for (const ptr of Object.values(this.scratch)) native.metalReleaseBuffer(ptr);
    }
    native.metalDestroyBackend(this.backend);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const fwd = new MetalForward();
  const T = parseInt(process.env.T ?? "32", 10);
  fwd.ensureScratch(T);
  console.log(`[MetalForward] scratch allocated for T=${T}`);
  const ids = new Int32Array(T);
  for (let i = 0; i < ids.length; i++) ids[i] = (i * 7919) % PF.vocabSize;
  const mask = new Uint8Array(T).fill(1);

  for (let i = 0; i < 3; i++) fwd.forward(ids, mask);

  const N = 10;
  const samples = [];
  const encs = [], subs = [];
  for (let i = 0; i < N; i++) {
    const t0 = performance.now();
    const logits = fwd.forward(ids, mask);
    samples.push(performance.now() - t0);
    if (process.env.PROFILE) {
      encs.push(fwd._lastEncodeMs);
      subs.push(fwd._lastSubmitMs);
    }
    if (i === 0) console.log(`[MetalForward] logits shape=(${T}, ${PF.numClasses}) len=${logits.length}`);
  }
  samples.sort((a, b) => a - b);
  console.log(
    `[MetalForward] forward latency T=${T}: ` +
      `median=${samples[Math.floor(N/2)].toFixed(1)}ms, ` +
      `min=${samples[0].toFixed(1)}ms, ` +
      `over ${N} iters`,
  );
  if (process.env.PROFILE) {
    encs.sort((a, b) => a - b);
    subs.sort((a, b) => a - b);
    console.log(
      `  phase breakdown: encode median=${encs[Math.floor(N/2)].toFixed(1)}ms, ` +
        `submit+readback median=${subs[Math.floor(N/2)].toFixed(1)}ms`,
    );
  }
  fwd.dispose();
  console.log("[MetalForward] disposed cleanly");
}
