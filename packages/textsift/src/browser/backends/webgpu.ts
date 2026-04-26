/**
 * Stage 2 backend: WGSL compute shaders on WebGPU.
 *
 * Uploads the same `onnx/model_q4f16.onnx` + `.onnx_data` the other two
 * backends consume and runs the full forward on the GPU. No per-kernel
 * round-trip through JS — input token IDs + attention mask go in, logits
 * come out; everything in between lives in storage buffers on the device.
 *
 * Requires `shader-f16`. `warmup()` throws if the adapter can't enable
 * it; callers (selectBackend) should fall back to the WASM path.
 *
 * Kernel set mirrors the WASM backend's forward pipeline:
 *   embed_lookup_int4 · rms_norm · matmul_int4 (f32→f16 and f32→f32)
 *   · rope_apply (interleaved pairs, native f16 arithmetic) · banded_attention
 *   · add_fp16 · cast_fp16_to_f32 · cast_f32_to_fp16_scaled
 *   · zero_f32 · router_topk · swiglu_clamp
 *   · qmoe_gate_up · qmoe_down_scatter (atomic f32 CAS scatter-add)
 *
 * MoE expert dispatch is token-major: one workgroup per (token, k_pick)
 * runs its own expert-sliced matmul and scatter-adds (weight × out) into
 * the per-token accumulator via a u32 CAS atomic-add on reinterpreted
 * f32 bits. This avoids the GPU-to-CPU readback the expert-major WASM
 * path uses to group by expert.
 */

import type {
  BackendConstructionOptions,
  InferenceBackend,
  Logits,
} from "./abstract.js";
import {
  INT4_ACCESS_WGSL,
  MATMUL_INT4_F32_F32_WGSL,
  MATMUL_INT4_FP16_F16_WGSL,
  EMBED_LOOKUP_INT4_WGSL,
  RMS_NORM_WGSL,
  ADD_RMSNORM_FP16_TO_F32_WGSL,
  CAST_FP16_TO_F32_WGSL,
  CAST_F32_TO_FP16_SCALED_WGSL,
  ADD_FP16_WGSL,
  ZERO_F32_WGSL,
  ROPE_APPLY_WGSL,
  BANDED_ATTENTION_WGSL,
  ROUTER_TOPK_WGSL,
  SWIGLU_CLAMP_WGSL,
  QMOE_GATE_UP_WGSL,
  QMOE_DOWN_SCATTER_WGSL,
} from "./shaders.js";
import { parseOnnxGraph, resolveTensorBytes } from "../model/onnx-reader.js";
import { fetchBytesCached } from "../model/opfs-fetch.js";
import { buildRopeTables } from "../inference/rope.js";

// ---------- tensor record ----------

/**
 * A GPU-resident tensor. `buffer` is a storage buffer owned by the
 * backend; `byteOffset`/`byteSize` let us point at sub-ranges of larger
 * packed allocations (e.g. per-expert slices of the expert blob).
 */
export interface GpuTensor {
  readonly name: string;
  readonly buffer: GPUBuffer;
  readonly byteOffset: number;
  readonly byteSize: number;
  readonly shape: readonly number[];
}

// ---------- shared int4 helpers (WGSL) ----------

/**
 * Byte/nibble accessors reused by every int4 kernel. Each kernel inlines
 * this preamble so we don't need an import graph in WGSL. `array<u32>`
 * is the canonical storage container for packed uint8/uint4 data —
 * storage buffers can't be declared as array<u8> directly.
 */


// ---------- WGSL: int4 matmul, f32 x → f32 out ----------

/**
 * int4-block32 matmul: f32 x [T, K] × packed-uint4 W [N, K] → f32 y [T, N].
 * Dequant: `(q_nibble - zp_nibble) * scale_block`. Used for router matmul
 * and classifier head (upstream keeps router and head in fp32 accum).
 *
 * Weight layout matches WASM byte-for-byte:
 *   `w_int4`: [N, K/2] u8 — byte j of row n packs `(w[n, 2j])` low nibble,
 *             `(w[n, 2j+1])` high nibble.
 *   `w_scales`: [N, K/32] f16.
 *   `w_zp`: [N, ceil(K/32/2)] u8 — uint4 zero-points, 2/byte.
 *   `bias`: [N] f16.
 */


// ---------- WGSL: int4 matmul, fp16 x → f16 out ----------

/**
 * fp16 x variant of the int4 matmul: reads X as f16 directly, widens
 * per-load to f32 for the MAC chain. Used for Q/K/V/O projections to
 * skip the explicit pre-widen cast dispatch — the activation traffic
 * is halved (f16 vs f32) which offsets the extra per-load widen work.
 */


// ---------- WGSL: GatherBlockQuantized embedding ----------

/**
 * Embedding table lookup with inline int4 dequant. Matches ONNX
 * GatherBlockQuantized semantics (block_size=32, uint4 weights, uint4
 * zero-points, fp16 scales). Output is fp16 so downstream rmsnorm +
 * matmul see their usual input dtype.
 *
 * OOB ids zero-fill, same as WASM `embed_lookup_int4`.
 */


// ---------- WGSL: RMSNorm ----------

/**
 * y[t, d] = x[t, d] * gamma[d] / sqrt(mean_d(x[t, :]^2) + eps)
 *
 * One workgroup per row (T rows). 64 threads per workgroup, each walks
 * a strided slice of the hidden axis. Sum-of-squares reduction via
 * workgroup memory tree (6 rounds for 64 threads). All accumulation in
 * f32 — 640-wide fp16 would lose bits.
 */


// ---------- WGSL: fused (a+b) → fp16 sum + RMSNorm → f32 ----------

/**
 * Mirror of the WASM `add_rmsnorm_fp16_to_f32` kernel: combines the
 * post-attention residual add, layernorm, and fp16→f32 widen into one
 * dispatch per layer. Saves two GPU compute barriers per layer plus
 * the intermediate fp16 normed buffer write.
 *
 * Per workgroup: one row of T. 64 threads cooperatively walk D, each
 * thread accumulating sumsq for its strided slice and writing the
 * fp16 sum (= a+b) as it goes. After the workgroup-shared sumsq
 * reduce, threads re-read fp16 sum, apply gamma * inv_rms, and write
 * the f32 norm output.
 */


// ---------- WGSL: dtype casts ----------





// ---------- WGSL: elementwise add, zero ----------



/**
 * Zeros the MoE accumulator before each block's scatter pass. Declared
 * as plain u32 here; the scatter kernel rebinds the same buffer as
 * array<atomic<u32>>. WebGPU permits different shader views of the same
 * buffer as long as usage flags (STORAGE) cover both.
 */


// ---------- WGSL: RoPE apply (interleaved) ----------

/**
 * In-place rotary position embedding apply. Interleaved-pair layout:
 * for each (t, h, p), rotate (qk[t, h, 2p], qk[t, h, 2p+1]) by cos/sin.
 * cos/sin are precomputed by JS as fp16 with YARN's attention_scaling
 * folded in, matching the WASM path.
 *
 * Three-rounding semantics: `a * c - b * s` uses WGSL f16 arithmetic.
 * Per the WGSL spec, each f16 op produces an f16 result, so the mul
 * rounds, then the subtract rounds — three roundings per output,
 * matching PyTorch eager fp16.
 */


// ---------- WGSL: banded GQA attention with sinks ----------

/**
 * Sliding-window attention with attention sinks, fp16 QKV in, fp16 out.
 * One workgroup per (t_query, h_query) — 64 threads per workgroup.
 * Threads share score/softmax scratch via workgroup memory.
 *
 * Flow per (t_q, h_q):
 *   1. Each thread computes a strided subset of scores (Q·K dot).
 *   2. Workgroup reduces `row_max` across scores ∪ {sink}.
 *   3. Threads exp(score - max), workgroup reduces `sum`.
 *   4. Normalize softmax (divide by sum).
 *   5. Each thread owns one head_dim lane (head_dim == 64 == wg size),
 *      accumulates softmax_i · V[t_k_i, h_kv, lane] across keys in
 *      window.
 *   6. Thread writes its lane to out[t_q, h_q, lane].
 *
 * `sink` contributes to the softmax denominator but not to the AV
 * combine — the 0-dim "sink key" with no value vector.
 */


// ---------- WGSL: router top-K + softmax / K ----------

/**
 * One thread per T row. Scans the full expert-logits row, maintains a
 * K-slot insertion-sorted top-K buffer in registers, then softmax over
 * the K values and divides by K — same post-processing as the WASM
 * block.routerForward. K is passed as a uniform for clarity but is
 * always 4 for this model.
 *
 * K is bounded by a small compile-time constant (MAX_K = 8) so the
 * top-K arrays fit in registers. For 128 experts × 8 slots = 1024
 * compares per thread, comfortably within a single dispatch.
 */


// ---------- WGSL: SwiGLU with clamp (packed gate/up) ----------

/**
 * Packed SwiGLU matching upstream's `swiglu(packed=True)` and ORT
 * QMoE's `swiglu_fusion=1`:
 *   gate = gate_up[..., 0::2].clamp(max=7)
 *   up   = gate_up[..., 1::2].clamp(-7, 7)
 *   glu  = gate * sigmoid(gate * 1.702)
 *   out  = (up + 1.0) * glu
 *
 * All compute in f32 — upstream's expert forward runs `.float()` the
 * entire expert block.
 */


// ---------- WGSL: QMoE gate_up (per-expert int4 matmul) ----------

/**
 * MoE gate_up_proj: for each (tk_pair = t * K + k_pick), select expert
 * via `routing_idx[tk_pair]`, run the int4 matmul slice of
 * `gate_up[expert_idx]` against `x[t, :]`. Output shape: [T*K, 2*dff].
 *
 * Weight layout per expert:
 *   int4:   [2*dff, D/2] bytes at offset `e * 2*dff * (D/2)` inside
 *           the flat packed buffer.
 *   scales: [2*dff, D/32] f16 at offset `e * 2*dff * (D/32)` elements.
 *   zp:     [2*dff, ceil((D/32)/2)] bytes at offset
 *           `e * 2*dff * zp_per_row`.
 *   bias:   [2*dff] f16 at offset `e * 2*dff` elements.
 *
 * One thread per output element (tk_pair, n-in-2*dff). Each thread
 * reads `routing_idx[tk_pair]` once and does the usual int4 MAC chain.
 */


// ---------- WGSL: QMoE down + atomic scatter-add ----------

/**
 * MoE down_proj: for each (tk_pair, output col in D), compute the int4
 * matmul slice of `down[expert_idx]` against the corresponding row of
 * `glu[tk_pair, :]`, then CAS-atomically add
 * `routing_scores[tk_pair] * out` into `acc[token_idx(tk), out_col]`.
 *
 * Atomic-add on f32 storage: buffer declared as `array<atomic<u32>>`;
 * the loop reads the current u32, bit-casts to f32, adds, bit-casts
 * back, tries `atomicCompareExchangeWeak`, retries on failure.
 */


// ---------- backend options ----------

export interface WebGpuBackendOptions extends BackendConstructionOptions {}

// ---------- pipeline bundle ----------

interface Pipelines {
  embed: GPUComputePipeline;
  rmsNorm: GPUComputePipeline;
  addRmsNormToF32: GPUComputePipeline;
  matmulF32F32: GPUComputePipeline;
  matmulFp16F16: GPUComputePipeline;
  castFp16ToF32: GPUComputePipeline;
  castF32ToFp16Scaled: GPUComputePipeline;
  addFp16: GPUComputePipeline;
  zero: GPUComputePipeline;
  rope: GPUComputePipeline;
  banded: GPUComputePipeline;
  routerTopk: GPUComputePipeline;
  swiglu: GPUComputePipeline;
  qmoeGateUp: GPUComputePipeline;
  qmoeDownScatter: GPUComputePipeline;
}

interface BindGroupLayouts {
  embed: GPUBindGroupLayout;
  rmsNorm: GPUBindGroupLayout;
  addRmsNormToF32: GPUBindGroupLayout;
  matmul: GPUBindGroupLayout;
  cast1to1: GPUBindGroupLayout;       // single in → single out (cast_fp16_to_f32)
  castScaled: GPUBindGroupLayout;     // dims + scale + src + dst
  add: GPUBindGroupLayout;
  zero: GPUBindGroupLayout;
  rope: GPUBindGroupLayout;
  banded: GPUBindGroupLayout;
  routerTopk: GPUBindGroupLayout;
  swiglu: GPUBindGroupLayout;
  qmoeGateUp: GPUBindGroupLayout;
  qmoeDown: GPUBindGroupLayout;
}

// ---------- scratch buffer plan ----------

interface Scratch {
  allocatedT: number;
  idsBuf: GPUBuffer;
  maskBuf: GPUBuffer;
  cosBuf: GPUBuffer;
  sinBuf: GPUBuffer;
  h0: GPUBuffer;
  h1: GPUBuffer;
  normed1: GPUBuffer;
  normed2: GPUBuffer;
  hiddenF32: GPUBuffer;
  qBuf: GPUBuffer;
  kBuf: GPUBuffer;
  vBuf: GPUBuffer;
  attnOut: GPUBuffer;
  oOut: GPUBuffer;
  routerLogits: GPUBuffer;
  routingIdx: GPUBuffer;
  routingScores: GPUBuffer;
  acc: GPUBuffer;
  gateUp: GPUBuffer;
  glu: GPUBuffer;
  moeOut: GPUBuffer;
  logitsOut: GPUBuffer;
  logitsReadback: GPUBuffer;
}

// ---------- model config (duplicated from wasm path) ----------

const PF_CONFIG = {
  hiddenSize: 640,
  numHeads: 14,
  numKvHeads: 2,
  headDim: 64,
  slidingWindow: 128,
  intermediateSize: 640,
  numExpertsPerTok: 4,
  rmsNormEps: 1e-5,
  numClasses: 33,
  rope: {
    headDim: 64,
    theta: 150000.0,
    factor: 32.0,
    originalMaxPositionEmbeddings: 4096,
    betaFast: 32.0,
    betaSlow: 1.0,
    truncate: false,
  },
} as const;

// ---------- backend ----------

export interface WebGpuWarmupTimings {
  adapterMs: number;
  deviceMs: number;
  onnxFetchMs: number;
  onnxParseMs: number;
  weightUploadMs: number;
  pipelineCompileMs: number;
  totalMs: number;
}

export class WebGpuBackend implements InferenceBackend {
  readonly name = "webgpu" as const;
  private device: GPUDevice | null = null;
  private weights: Map<string, GpuTensor> | null = null;
  private pipelines: Pipelines | null = null;
  private bgls: BindGroupLayouts | null = null;
  private scratch: Scratch | null = null;
  private numLayers = 0;
  private numExperts = 0;
  private vocabSize = 0;
  /** Populated by `warmup()`. `null` until warmup completes. */
  warmupTimings: WebGpuWarmupTimings | null = null;
  private readonly opts: WebGpuBackendOptions;

  constructor(opts: WebGpuBackendOptions) {
    this.opts = opts;
  }

  async warmup(): Promise<void> {
    if (typeof navigator === "undefined" || !navigator.gpu) {
      throw new Error("WebGpuBackend: navigator.gpu not available");
    }
    const tStart = performance.now();
    const tA0 = performance.now();
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) throw new Error("WebGpuBackend: no GPUAdapter");
    if (!adapter.features.has("shader-f16")) {
      throw new Error(
        "WebGpuBackend: adapter lacks shader-f16; caller should fall back to the wasm backend",
      );
    }
    const adapterMs = performance.now() - tA0;

    const tD0 = performance.now();
    const device = await adapter.requestDevice({
      requiredFeatures: ["shader-f16"],
      requiredLimits: {
        // Experts: 128 * 1280 * 320 bytes = 52 MiB per layer just for
        // gate_up quant. Max storage buffer size needs to fit that.
        maxStorageBufferBindingSize: Math.min(
          adapter.limits.maxStorageBufferBindingSize,
          1024 * 1024 * 1024,
        ),
        maxBufferSize: Math.min(adapter.limits.maxBufferSize, 1024 * 1024 * 1024),
        maxStorageBuffersPerShaderStage: Math.min(
          adapter.limits.maxStorageBuffersPerShaderStage,
          10,
        ),
      },
    });
    const deviceMs = performance.now() - tD0;
    // Surface any WebGPU validation or OOM errors as console errors so the
    // caller's diagnostics pick them up — WGSL is tricky enough that we
    // want every uncapturedevent visible.
    device.addEventListener("uncapturederror", (ev) => {
      const anyEv = ev as unknown as { error: { message: string } };
      console.error("WebGPU error: " + anyEv.error.message);
    });
    this.device = device;

    // Pipeline compile + weight upload are independent and both slow.
    // Overlap them: BGLs + pipeline compile run in parallel with the
    // ONNX fetch + parse + GPU buffer upload chain.
    this.bgls = createBindGroupLayouts(device);
    const tP0 = performance.now();
    const pipelinesPromise = createPipelines(device, this.bgls).then((p) => {
      const pipelineCompileMs = performance.now() - tP0;
      return { pipelines: p, pipelineCompileMs };
    });
    const weightsBreakdown = await loadOnnxWeightsGpuTimed(device, this.opts.bundle.modelSource);
    this.weights = weightsBreakdown.weights;
    const { pipelines, pipelineCompileMs } = await pipelinesPromise;
    this.pipelines = pipelines;

    this.numLayers = detectNumLayers(this.weights);
    this.vocabSize = this.weights.get("embed.int4")!.shape[0]!;
    const routerInt4 = this.weights.get("layers.0.router.int4");
    if (!routerInt4) throw new Error("WebGpuBackend: missing router weights");
    this.numExperts = routerInt4.shape[0]!;

    this.warmupTimings = {
      adapterMs,
      deviceMs,
      onnxFetchMs: weightsBreakdown.fetchMs,
      onnxParseMs: weightsBreakdown.parseMs,
      weightUploadMs: weightsBreakdown.uploadMs,
      pipelineCompileMs,
      totalMs: performance.now() - tStart,
    };
  }

  async forward(
    tokenIds: Int32Array,
    attentionMask: Uint8Array,
  ): Promise<Logits> {
    if (!this.device || !this.weights || !this.pipelines || !this.bgls) {
      throw new Error("WebGpuBackend.forward: call warmup() first");
    }
    const T = tokenIds.length;
    if (attentionMask.length !== T) {
      throw new Error(
        `WebGpuBackend.forward: attentionMask length ${attentionMask.length} does not match tokenIds length ${T}`,
      );
    }
    const device = this.device;
    this.ensureScratch(T);
    const scratch = this.scratch!;

    // Upload token ids + mask (pad mask to 4-byte alignment).
    device.queue.writeBuffer(
      scratch.idsBuf, 0,
      tokenIds.buffer as ArrayBuffer, tokenIds.byteOffset, T * 4,
    );
    const maskPadded = padTo4(attentionMask);
    device.queue.writeBuffer(
      scratch.maskBuf, 0,
      maskPadded.buffer as ArrayBuffer, maskPadded.byteOffset, maskPadded.byteLength,
    );
    let useMask = 0;
    for (let i = 0; i < T; i++) {
      if (attentionMask[i] !== 1) { useMask = 1; break; }
    }

    // Build RoPE tables for this sequence length and upload.
    const { cos, sin } = buildRopeTables(PF_CONFIG.rope, T);
    const cosBytes = new Uint8Array(cos.buffer, cos.byteOffset, cos.byteLength);
    const sinBytes = new Uint8Array(sin.buffer, sin.byteOffset, sin.byteLength);
    const cosPadded = padTo4(cosBytes);
    const sinPadded = padTo4(sinBytes);
    device.queue.writeBuffer(
      scratch.cosBuf, 0,
      cosPadded.buffer as ArrayBuffer, cosPadded.byteOffset, cosPadded.byteLength,
    );
    device.queue.writeBuffer(
      scratch.sinBuf, 0,
      sinPadded.buffer as ArrayBuffer, sinPadded.byteOffset, sinPadded.byteLength,
    );

    const D = PF_CONFIG.hiddenSize;
    const Hq = PF_CONFIG.numHeads;
    const Hkv = PF_CONFIG.numKvHeads;
    const hd = PF_CONFIG.headDim;
    const dff = PF_CONFIG.intermediateSize;
    const Kpick = PF_CONFIG.numExpertsPerTok;
    const E = this.numExperts;

    // Session-lived uniform buffers created per dispatch. We hold them in
    // this array so they survive until queue.submit() returns.
    const ownedResources: { destroy(): void }[] = [];
    const u = (bytes: ArrayBuffer): GPUBuffer => {
      const buf = device.createBuffer({
        size: Math.max(16, bytes.byteLength),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buf, 0, bytes);
      ownedResources.push(buf);
      return buf;
    };
    const dims4 = (a: number, b: number, c: number, d: number): ArrayBuffer =>
      new Uint32Array([a, b, c, d]).buffer as ArrayBuffer;
    const dimsEps = (T_: number, D_: number, eps: number): ArrayBuffer => {
      const buf = new ArrayBuffer(16);
      new Uint32Array(buf, 0, 2).set([T_, D_]);
      new Float32Array(buf, 8, 1).set([eps]);
      return buf;
    };
    const dimsScale = (scale: number): ArrayBuffer => {
      const buf = new ArrayBuffer(16);
      new Float32Array(buf, 0, 1).set([scale]);
      return buf;
    };
    const dims8 = (a: number, b: number, c: number, d: number,
                   e: number, f: number, g: number, h: number): ArrayBuffer =>
      new Uint32Array([a, b, c, d, e, f, g, h]).buffer as ArrayBuffer;

    const encoder = device.createCommandEncoder({ label: "pii.forward" });
    const pass = encoder.beginComputePass({ label: "pii.forward.pass" });

    const dispatchThreads = (
      pipeline: GPUComputePipeline,
      bindGroup: GPUBindGroup,
      totalThreads: number,
    ): void => {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.max(1, Math.ceil(totalThreads / 64)));
    };
    const dispatchGroups = (
      pipeline: GPUComputePipeline,
      bindGroup: GPUBindGroup,
      numGroups: number,
    ): void => {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.max(1, numGroups));
    };
    const dispatchGroups2D = (
      pipeline: GPUComputePipeline,
      bindGroup: GPUBindGroup,
      groupsX: number,
      groupsY: number,
    ): void => {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.max(1, groupsX), Math.max(1, groupsY));
    };
    const dispatch = dispatchThreads;

    // Embed lookup → h0.
    {
      const dimsBuf = u(dims4(T, this.vocabSize, D, 0));
      const embedInt4 = this.weights.get("embed.int4")!;
      const embedScales = this.weights.get("embed.scales")!;
      const embedZp = this.weights.get("embed.zp")!;
      const bg = device.createBindGroup({
        layout: this.bgls.embed,
        entries: [
          { binding: 0, resource: { buffer: dimsBuf } },
          resBinding(1, embedInt4),
          resBinding(2, embedScales),
          resBinding(3, embedZp),
          { binding: 4, resource: { buffer: scratch.idsBuf } },
          { binding: 5, resource: { buffer: scratch.h0 } },
        ],
      });
      dispatch(this.pipelines.embed, bg, T * D);
    }

    let hCur = scratch.h0;
    let hAlt = scratch.h1;
    for (let L = 0; L < this.numLayers; L++) {
      const w = (suffix: string) => this.weights!.get(`layers.${L}.${suffix}`)!;

      // input_layernorm(hCur) → normed1.
      dispatchGroups(
        this.pipelines.rmsNorm,
        device.createBindGroup({
          layout: this.bgls.rmsNorm,
          entries: [
            { binding: 0, resource: { buffer: u(dimsEps(T, D, PF_CONFIG.rmsNormEps)) } },
            { binding: 1, resource: { buffer: hCur } },
            resBinding(2, w("input_layernorm")),
            { binding: 3, resource: { buffer: scratch.normed1 } },
          ],
        }),
        T,
      );

      // Q/K/V projections read fp16 normed1 directly via the fp16-in
      // matmul variant. Tiled dispatch: each workgroup writes a
      // 4 (T) × 64 (N) tile, amortizing the int4 weight decode across
      // four T rows.
      const runMatmulFp16F16 = (
        weightsBase: string,
        xBuf: GPUBuffer,
        yBuf: GPUBuffer,
        N: number, K: number,
      ): void => {
        const wq = this.weights!.get(`${weightsBase}.int4`)!;
        const ws = this.weights!.get(`${weightsBase}.scales`)!;
        const wz = this.weights!.get(`${weightsBase}.zp`)!;
        const wb = this.weights!.get(`${weightsBase}.bias`)!;
        const bg = device.createBindGroup({
          layout: this.bgls!.matmul,
          entries: [
            { binding: 0, resource: { buffer: u(dims4(T, N, K, 0)) } },
            { binding: 1, resource: { buffer: xBuf } },
            resBinding(2, wq),
            resBinding(3, ws),
            resBinding(4, wz),
            resBinding(5, wb),
            { binding: 6, resource: { buffer: yBuf } },
          ],
        });
        const xGroups = Math.ceil(N / 64);
        const yGroups = Math.ceil(T / 4);  // matches TR=4 in MATMUL_INT4_FP16_F16_WGSL
        dispatchGroups2D(this.pipelines!.matmulFp16F16, bg, xGroups, yGroups);
      };
      runMatmulFp16F16(`layers.${L}.attn.q_proj`, scratch.normed1, scratch.qBuf, Hq * hd, D);
      runMatmulFp16F16(`layers.${L}.attn.k_proj`, scratch.normed1, scratch.kBuf, Hkv * hd, D);
      runMatmulFp16F16(`layers.${L}.attn.v_proj`, scratch.normed1, scratch.vBuf, Hkv * hd, D);

      // RoPE on Q and K.
      const runRope = (qkBuf: GPUBuffer, heads: number): void => {
        const half = hd / 2;
        const bg = device.createBindGroup({
          layout: this.bgls!.rope,
          entries: [
            { binding: 0, resource: { buffer: u(dims4(T, heads, hd, 0)) } },
            { binding: 1, resource: { buffer: qkBuf } },
            { binding: 2, resource: { buffer: scratch.cosBuf } },
            { binding: 3, resource: { buffer: scratch.sinBuf } },
          ],
        });
        dispatch(this.pipelines!.rope, bg, T * heads * half);
      };
      runRope(scratch.qBuf, Hq);
      runRope(scratch.kBuf, Hkv);

      // Banded attention: one workgroup per (t_q, h_q), 64 threads each.
      dispatchGroups(
        this.pipelines.banded,
        device.createBindGroup({
          layout: this.bgls.banded,
          entries: [
            { binding: 0, resource: { buffer: u(dims8(T, Hq, Hkv, hd, PF_CONFIG.slidingWindow, useMask, 0, 0)) } },
            { binding: 1, resource: { buffer: scratch.qBuf } },
            { binding: 2, resource: { buffer: scratch.kBuf } },
            { binding: 3, resource: { buffer: scratch.vBuf } },
            resBinding(4, w("attn.sinks")),
            { binding: 5, resource: { buffer: scratch.maskBuf } },
            { binding: 6, resource: { buffer: scratch.attnOut } },
          ],
        }),
        T * Hq,
      );

      // O projection reads fp16 attnOut directly (same fusion as Q/K/V).
      runMatmulFp16F16(`layers.${L}.attn.o_proj`, scratch.attnOut, scratch.oOut, D, Hq * hd);

      // Fused: (hCur + oOut) → h1 (fp16, written to normed1 buffer)
      // and rms_norm(h1) widened to f32 → hiddenF32. One dispatch
      // replaces add_fp16 + rms_norm + cast_fp16_to_f32 (3 dispatches).
      // The post-attention norm input (`normed1` here) doubles as the
      // residual stream into the final block add.
      dispatchGroups(
        this.pipelines.addRmsNormToF32,
        device.createBindGroup({
          layout: this.bgls.addRmsNormToF32,
          entries: [
            { binding: 0, resource: { buffer: u(dimsEps(T, D, PF_CONFIG.rmsNormEps)) } },
            { binding: 1, resource: { buffer: hCur } },
            { binding: 2, resource: { buffer: scratch.oOut } },
            resBinding(3, w("post_attention_layernorm")),
            { binding: 4, resource: { buffer: scratch.normed1 } },   // residual2 (fp16 sum)
            { binding: 5, resource: { buffer: scratch.hiddenF32 } }, // f32 normed
          ],
        }),
        T,
      );

      // Router: f32-in, f32-out int4 matmul.
      {
        const wq = this.weights.get(`layers.${L}.router.int4`)!;
        const ws = this.weights.get(`layers.${L}.router.scales`)!;
        const wz = this.weights.get(`layers.${L}.router.zp`)!;
        const wb = this.weights.get(`layers.${L}.router.bias`)!;
        dispatch(
          this.pipelines.matmulF32F32,
          device.createBindGroup({
            layout: this.bgls.matmul,
            entries: [
              { binding: 0, resource: { buffer: u(dims4(T, E, D, 0)) } },
              { binding: 1, resource: { buffer: scratch.hiddenF32 } },
              resBinding(2, wq),
              resBinding(3, ws),
              resBinding(4, wz),
              resBinding(5, wb),
              { binding: 6, resource: { buffer: scratch.routerLogits } },
            ],
          }),
          T * E,
        );
      }

      // Router top-K softmax / K.
      dispatch(
        this.pipelines.routerTopk,
        device.createBindGroup({
          layout: this.bgls.routerTopk,
          entries: [
            { binding: 0, resource: { buffer: u(dims4(T, E, Kpick, 0)) } },
            { binding: 1, resource: { buffer: scratch.routerLogits } },
            { binding: 2, resource: { buffer: scratch.routingIdx } },
            { binding: 3, resource: { buffer: scratch.routingScores } },
          ],
        }),
        T,
      );

      // Zero the MoE accumulator.
      dispatch(
        this.pipelines.zero,
        device.createBindGroup({
          layout: this.bgls.zero,
          entries: [
            { binding: 0, resource: { buffer: u(dims4(T * D, 0, 0, 0)) } },
            { binding: 1, resource: { buffer: scratch.acc } },
          ],
        }),
        T * D,
      );

      // QMoE gate_up: f32 hidden × int4 W[expert] → f32 [T*K, 2*dff].
      {
        const wq = this.weights.get(`layers.${L}.experts.gate_up.int4`)!;
        const ws = this.weights.get(`layers.${L}.experts.gate_up.scales`)!;
        const wz = this.weights.get(`layers.${L}.experts.gate_up.zp`)!;
        const wb = this.weights.get(`layers.${L}.experts.gate_up.bias`)!;
        dispatchGroups2D(
          this.pipelines.qmoeGateUp,
          device.createBindGroup({
            layout: this.bgls.qmoeGateUp,
            entries: [
              { binding: 0, resource: { buffer: u(dims8(T, Kpick, 2 * dff, D, 0, 0, 0, 0)) } },
              { binding: 1, resource: { buffer: scratch.hiddenF32 } },
              { binding: 2, resource: { buffer: scratch.routingIdx } },
              resBinding(3, wq),
              resBinding(4, ws),
              resBinding(5, wz),
              resBinding(6, wb),
              { binding: 7, resource: { buffer: scratch.gateUp } },
            ],
          }),
          T * Kpick,
          Math.ceil((2 * dff) / 64),
        );
      }

      // SwiGLU-with-clamp: [T*K, 2*dff] → [T*K, dff].
      dispatch(
        this.pipelines.swiglu,
        device.createBindGroup({
          layout: this.bgls.swiglu,
          entries: [
            { binding: 0, resource: { buffer: u(dims4(T * Kpick, dff, 0, 0)) } },
            { binding: 1, resource: { buffer: scratch.gateUp } },
            { binding: 2, resource: { buffer: scratch.glu } },
          ],
        }),
        T * Kpick * dff,
      );

      // QMoE down_proj + atomic scatter-add into acc. Tiled like
      // qmoe_gate_up: workgroup_id.x = (token, k_pick), .y = N-tile,
      // with cooperative glu-tile load shared across all 64 threads
      // in the workgroup.
      {
        const wq = this.weights.get(`layers.${L}.experts.down.int4`)!;
        const ws = this.weights.get(`layers.${L}.experts.down.scales`)!;
        const wz = this.weights.get(`layers.${L}.experts.down.zp`)!;
        const wb = this.weights.get(`layers.${L}.experts.down.bias`)!;
        dispatchGroups2D(
          this.pipelines.qmoeDownScatter,
          device.createBindGroup({
            layout: this.bgls.qmoeDown,
            entries: [
              { binding: 0, resource: { buffer: u(dims8(T, Kpick, D, dff, 0, 0, 0, 0)) } },
              { binding: 1, resource: { buffer: scratch.glu } },
              { binding: 2, resource: { buffer: scratch.routingIdx } },
              { binding: 3, resource: { buffer: scratch.routingScores } },
              resBinding(4, wq),
              resBinding(5, ws),
              resBinding(6, wz),
              resBinding(7, wb),
              { binding: 8, resource: { buffer: scratch.acc } },
            ],
          }),
          T * Kpick,
          Math.ceil(D / 64),
        );
      }

      // acc × K → moe_out (fp16). Matches the WASM final multiply by
      // num_experts_per_tok (K cancels with the /K embedded in
      // routing_scores so the net combine is sum_k softmax_k * out_k).
      dispatch(
        this.pipelines.castF32ToFp16Scaled,
        device.createBindGroup({
          layout: this.bgls.castScaled,
          entries: [
            { binding: 0, resource: { buffer: u(dims4(T * D, 0, 0, 0)) } },
            { binding: 1, resource: { buffer: u(dimsScale(Kpick)) } },
            { binding: 2, resource: { buffer: scratch.acc } },
            { binding: 3, resource: { buffer: scratch.moeOut } },
          ],
        }),
        T * D,
      );

      // Residual: h1 + moe_out → hAlt.
      dispatch(
        this.pipelines.addFp16,
        device.createBindGroup({
          layout: this.bgls.add,
          entries: [
            { binding: 0, resource: { buffer: u(dims4(T * D, 0, 0, 0)) } },
            { binding: 1, resource: { buffer: scratch.normed1 } },  // h1
            { binding: 2, resource: { buffer: scratch.moeOut } },
            { binding: 3, resource: { buffer: hAlt } },
          ],
        }),
        T * D,
      );

      // Ping-pong.
      const tmp = hCur;
      hCur = hAlt;
      hAlt = tmp;
    }

    // Final rmsnorm on hCur → normed1.
    dispatchGroups(
      this.pipelines.rmsNorm,
      device.createBindGroup({
        layout: this.bgls.rmsNorm,
        entries: [
          { binding: 0, resource: { buffer: u(dimsEps(T, D, PF_CONFIG.rmsNormEps)) } },
          { binding: 1, resource: { buffer: hCur } },
          resBinding(2, this.weights.get("final_norm")!),
          { binding: 3, resource: { buffer: scratch.normed1 } },
        ],
      }),
      T,
    );

    // Widen final norm → hiddenF32.
    dispatch(
      this.pipelines.castFp16ToF32,
      device.createBindGroup({
        layout: this.bgls.cast1to1,
        entries: [
          { binding: 0, resource: { buffer: u(dims4(T * D, 0, 0, 0)) } },
          { binding: 1, resource: { buffer: scratch.normed1 } },
          { binding: 2, resource: { buffer: scratch.hiddenF32 } },
        ],
      }),
      T * D,
    );

    // Classifier head: f32 hidden × int4 W → f32 logits.
    {
      const numClasses = PF_CONFIG.numClasses;
      const wq = this.weights.get("score.int4")!;
      const ws = this.weights.get("score.scales")!;
      const wz = this.weights.get("score.zp")!;
      const wb = this.weights.get("score.bias")!;
      dispatch(
        this.pipelines.matmulF32F32,
        device.createBindGroup({
          layout: this.bgls.matmul,
          entries: [
            { binding: 0, resource: { buffer: u(dims4(T, numClasses, D, 0)) } },
            { binding: 1, resource: { buffer: scratch.hiddenF32 } },
            resBinding(2, wq),
            resBinding(3, ws),
            resBinding(4, wz),
            resBinding(5, wb),
            { binding: 6, resource: { buffer: scratch.logitsOut } },
          ],
        }),
        T * numClasses,
      );
    }

    pass.end();

    const logitsBytes = T * PF_CONFIG.numClasses * 4;
    encoder.copyBufferToBuffer(scratch.logitsOut, 0, scratch.logitsReadback, 0, logitsBytes);
    device.queue.submit([encoder.finish()]);

    await scratch.logitsReadback.mapAsync(GPUMapMode.READ, 0, logitsBytes);
    const readView = scratch.logitsReadback.getMappedRange(0, logitsBytes);
    const f32 = new Float32Array(readView.slice(0));
    scratch.logitsReadback.unmap();

    // Release per-forward uniform buffers.
    for (const r of ownedResources) r.destroy();

    return {
      data: f32,
      sequenceLength: T,
      numClasses: PF_CONFIG.numClasses,
    };
  }

  dispose(): void {
    this.weights = null;
    this.pipelines = null;
    this.bgls = null;
    if (this.scratch) destroyScratch(this.scratch);
    this.scratch = null;
    this.device?.destroy();
    this.device = null;
  }

  // ---- test helper: single-matmul parity ----

  /**
   * Run `y[T, N] = x[T, K] @ dequant(W[n])^T + bias[n]` on the GPU
   * using the loaded weights keyed by `weightsKey` (e.g.
   * `"layers.0.attn.q_proj"`). `x` is f32 row-major [T, K]; returns
   * f32 [T, N] read back to the CPU. Used by the parity test.
   */
  async matmulInt4Test(
    weightsKey: string,
    x: Float32Array,
    T: number,
    N: number,
    K: number,
  ): Promise<Float32Array> {
    if (!this.device || !this.weights || !this.pipelines || !this.bgls) {
      throw new Error("WebGpuBackend.matmulInt4Test: call warmup() first");
    }
    const d = this.device;
    const w_int4 = this.weights.get(`${weightsKey}.int4`);
    const w_scales = this.weights.get(`${weightsKey}.scales`);
    const w_zp = this.weights.get(`${weightsKey}.zp`);
    const bias = this.weights.get(`${weightsKey}.bias`);
    if (!w_int4 || !w_scales || !w_zp || !bias) {
      throw new Error(`WebGpuBackend.matmulInt4Test: missing weights for ${weightsKey}`);
    }

    const xBuf = d.createBuffer({
      size: Math.max(16, x.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    d.queue.writeBuffer(xBuf, 0, x.buffer as ArrayBuffer, x.byteOffset, x.byteLength);

    const yBytes = T * N * 4;
    const yBuf = d.createBuffer({ size: yBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

    const dimsBuf = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(dimsBuf, 0, new Uint32Array([T, N, K, 0]).buffer as ArrayBuffer);

    const bindGroup = d.createBindGroup({
      layout: this.bgls.matmul,
      entries: [
        { binding: 0, resource: { buffer: dimsBuf } },
        { binding: 1, resource: { buffer: xBuf } },
        resBinding(2, w_int4),
        resBinding(3, w_scales),
        resBinding(4, w_zp),
        resBinding(5, bias),
        { binding: 6, resource: { buffer: yBuf } },
      ],
    });

    const encoder = d.createCommandEncoder({ label: "matmul_int4.encoder" });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.matmulF32F32);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.max(1, Math.ceil((T * N) / 64)));
    pass.end();

    const readBuf = d.createBuffer({ size: yBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyBufferToBuffer(yBuf, 0, readBuf, 0, yBytes);
    d.queue.submit([encoder.finish()]);

    await readBuf.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(readBuf.getMappedRange().slice(0));
    readBuf.unmap();
    readBuf.destroy();
    xBuf.destroy();
    yBuf.destroy();
    dimsBuf.destroy();
    return out;
  }

  // ---- scratch management ----

  private ensureScratch(T: number): void {
    if (this.scratch && this.scratch.allocatedT >= T) return;
    if (!this.device) throw new Error("WebGpuBackend.ensureScratch: no device");

    if (this.scratch) destroyScratch(this.scratch);

    const device = this.device;
    const D = PF_CONFIG.hiddenSize;
    const Hq = PF_CONFIG.numHeads;
    const Hkv = PF_CONFIG.numKvHeads;
    const hd = PF_CONFIG.headDim;
    const dff = PF_CONFIG.intermediateSize;
    const Kpick = PF_CONFIG.numExpertsPerTok;
    const E = this.numExperts;
    const numClasses = PF_CONFIG.numClasses;

    const mk = (label: string, size: number, usage: GPUBufferUsageFlags): GPUBuffer =>
      device.createBuffer({ label, size: Math.max(16, (size + 3) & ~3), usage });

    const STORE = GPUBufferUsage.STORAGE;
    const STORE_DST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const STORE_SRC = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;

    this.scratch = {
      allocatedT: T,
      idsBuf:        mk("ids", T * 4, STORE_DST),
      maskBuf:       mk("mask", (T + 3) & ~3, STORE_DST),
      cosBuf:        mk("cos", T * (hd / 2) * 2, STORE_DST),
      sinBuf:        mk("sin", T * (hd / 2) * 2, STORE_DST),
      h0:            mk("h0", T * D * 2, STORE),
      h1:            mk("h1", T * D * 2, STORE),
      normed1:       mk("normed1", T * D * 2, STORE),
      normed2:       mk("normed2", T * D * 2, STORE),
      hiddenF32:     mk("hiddenF32", T * D * 4, STORE),
      qBuf:          mk("q", T * Hq * hd * 2, STORE),
      kBuf:          mk("k", T * Hkv * hd * 2, STORE),
      vBuf:          mk("v", T * Hkv * hd * 2, STORE),
      attnOut:       mk("attnOut", T * Hq * hd * 2, STORE),
      oOut:          mk("oOut", T * D * 2, STORE),
      routerLogits:  mk("routerLogits", T * E * 4, STORE),
      routingIdx:    mk("routingIdx", T * Kpick * 4, STORE),
      routingScores: mk("routingScores", T * Kpick * 4, STORE),
      acc:           mk("acc", T * D * 4, STORE),
      gateUp:        mk("gateUp", T * Kpick * 2 * dff * 4, STORE),
      glu:           mk("glu", T * Kpick * dff * 4, STORE),
      moeOut:        mk("moeOut", T * D * 2, STORE),
      logitsOut:     mk("logitsOut", T * numClasses * 4, STORE_SRC),
      logitsReadback: device.createBuffer({
        label: "logitsReadback",
        size: Math.max(16, (T * numClasses * 4 + 3) & ~3),
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
    };
  }
}

// ---------- bind group layouts ----------

function createBindGroupLayouts(device: GPUDevice): BindGroupLayouts {
  const uniformEntry = (binding: number): GPUBindGroupLayoutEntry => ({
    binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" },
  });
  const rEntry = (binding: number): GPUBindGroupLayoutEntry => ({
    binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" },
  });
  const rwEntry = (binding: number): GPUBindGroupLayoutEntry => ({
    binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" },
  });

  return {
    embed: device.createBindGroupLayout({
      label: "bgl.embed",
      entries: [uniformEntry(0), rEntry(1), rEntry(2), rEntry(3), rEntry(4), rwEntry(5)],
    }),
    rmsNorm: device.createBindGroupLayout({
      label: "bgl.rmsnorm",
      entries: [uniformEntry(0), rEntry(1), rEntry(2), rwEntry(3)],
    }),
    addRmsNormToF32: device.createBindGroupLayout({
      label: "bgl.addRmsNormToF32",
      entries: [uniformEntry(0), rEntry(1), rEntry(2), rEntry(3), rwEntry(4), rwEntry(5)],
    }),
    matmul: device.createBindGroupLayout({
      label: "bgl.matmul",
      entries: [uniformEntry(0), rEntry(1), rEntry(2), rEntry(3), rEntry(4), rEntry(5), rwEntry(6)],
    }),
    cast1to1: device.createBindGroupLayout({
      label: "bgl.cast1to1",
      entries: [uniformEntry(0), rEntry(1), rwEntry(2)],
    }),
    castScaled: device.createBindGroupLayout({
      label: "bgl.castScaled",
      entries: [uniformEntry(0), uniformEntry(1), rEntry(2), rwEntry(3)],
    }),
    add: device.createBindGroupLayout({
      label: "bgl.add",
      entries: [uniformEntry(0), rEntry(1), rEntry(2), rwEntry(3)],
    }),
    zero: device.createBindGroupLayout({
      label: "bgl.zero",
      entries: [uniformEntry(0), rwEntry(1)],
    }),
    rope: device.createBindGroupLayout({
      label: "bgl.rope",
      entries: [uniformEntry(0), rwEntry(1), rEntry(2), rEntry(3)],
    }),
    banded: device.createBindGroupLayout({
      label: "bgl.banded",
      entries: [uniformEntry(0), rEntry(1), rEntry(2), rEntry(3), rEntry(4), rEntry(5), rwEntry(6)],
    }),
    routerTopk: device.createBindGroupLayout({
      label: "bgl.routerTopk",
      entries: [uniformEntry(0), rEntry(1), rwEntry(2), rwEntry(3)],
    }),
    swiglu: device.createBindGroupLayout({
      label: "bgl.swiglu",
      entries: [uniformEntry(0), rEntry(1), rwEntry(2)],
    }),
    qmoeGateUp: device.createBindGroupLayout({
      label: "bgl.qmoeGateUp",
      entries: [
        uniformEntry(0), rEntry(1), rEntry(2),
        rEntry(3), rEntry(4), rEntry(5), rEntry(6),
        rwEntry(7),
      ],
    }),
    qmoeDown: device.createBindGroupLayout({
      label: "bgl.qmoeDown",
      entries: [
        uniformEntry(0), rEntry(1), rEntry(2), rEntry(3),
        rEntry(4), rEntry(5), rEntry(6), rEntry(7),
        rwEntry(8),
      ],
    }),
  };
}

// ---------- pipeline compilation ----------

async function createPipelines(
  device: GPUDevice,
  bgls: BindGroupLayouts,
): Promise<Pipelines> {
  const mk = async (
    label: string,
    wgsl: string,
    bgl: GPUBindGroupLayout,
  ): Promise<GPUComputePipeline> => {
    const module = device.createShaderModule({ label: `${label}.wgsl`, code: wgsl });
    return device.createComputePipelineAsync({
      label: `${label}.pipeline`,
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      compute: { module, entryPoint: "main" },
    });
  };

  const [
    embed, rmsNorm, addRmsNormToF32, matmulF32F32, matmulFp16F16, castFp16ToF32,
    castF32ToFp16Scaled, addFp16, zero, rope, banded,
    routerTopk, swiglu, qmoeGateUp, qmoeDownScatter,
  ] = await Promise.all([
    mk("embed",              EMBED_LOOKUP_INT4_WGSL,       bgls.embed),
    mk("rmsnorm",            RMS_NORM_WGSL,                 bgls.rmsNorm),
    mk("add_rmsnorm_to_f32", ADD_RMSNORM_FP16_TO_F32_WGSL,  bgls.addRmsNormToF32),
    mk("matmul_int4_f32_f32",MATMUL_INT4_F32_F32_WGSL,      bgls.matmul),
    mk("matmul_int4_fp16_f16",MATMUL_INT4_FP16_F16_WGSL,    bgls.matmul),
    mk("cast_fp16_to_f32",   CAST_FP16_TO_F32_WGSL,         bgls.cast1to1),
    mk("cast_f32_to_fp16_scaled", CAST_F32_TO_FP16_SCALED_WGSL, bgls.castScaled),
    mk("add_fp16",           ADD_FP16_WGSL,                 bgls.add),
    mk("zero",               ZERO_F32_WGSL,                 bgls.zero),
    mk("rope",               ROPE_APPLY_WGSL,               bgls.rope),
    mk("banded",             BANDED_ATTENTION_WGSL,         bgls.banded),
    mk("routerTopk",         ROUTER_TOPK_WGSL,              bgls.routerTopk),
    mk("swiglu",             SWIGLU_CLAMP_WGSL,             bgls.swiglu),
    mk("qmoeGateUp",         QMOE_GATE_UP_WGSL,             bgls.qmoeGateUp),
    mk("qmoeDownScatter",    QMOE_DOWN_SCATTER_WGSL,        bgls.qmoeDown),
  ]);

  return {
    embed, rmsNorm, addRmsNormToF32, matmulF32F32, matmulFp16F16, castFp16ToF32,
    castF32ToFp16Scaled, addFp16, zero, rope, banded,
    routerTopk, swiglu, qmoeGateUp, qmoeDownScatter,
  };
}

// ---------- helpers ----------

function resBinding(binding: number, t: GpuTensor): GPUBindGroupEntry {
  // WebGPU rejects storage-buffer bindings whose size isn't a multiple of
  // 4. Round up to the next 4-byte boundary; shader reads are element-
  // indexed (f16 / u32 / f32) so the trailing padding bytes are never
  // touched. The underlying buffer was already sized to this rounding at
  // upload time.
  const alignedSize = (t.byteSize + 3) & ~3;
  return {
    binding,
    resource: { buffer: t.buffer, offset: t.byteOffset, size: alignedSize },
  };
}

function destroyScratch(s: Scratch): void {
  s.idsBuf.destroy(); s.maskBuf.destroy();
  s.cosBuf.destroy(); s.sinBuf.destroy();
  s.h0.destroy(); s.h1.destroy();
  s.normed1.destroy(); s.normed2.destroy();
  s.hiddenF32.destroy();
  s.qBuf.destroy(); s.kBuf.destroy(); s.vBuf.destroy();
  s.attnOut.destroy(); s.oOut.destroy();
  s.routerLogits.destroy();
  s.routingIdx.destroy(); s.routingScores.destroy();
  s.acc.destroy();
  s.gateUp.destroy(); s.glu.destroy(); s.moeOut.destroy();
  s.logitsOut.destroy(); s.logitsReadback.destroy();
}

function padTo4(src: Uint8Array): Uint8Array {
  const len = (src.byteLength + 3) & ~3;
  if (len === src.byteLength) return src;
  const out = new Uint8Array(len);
  out.set(src);
  return out;
}

function detectNumLayers(map: ReadonlyMap<string, GpuTensor>): number {
  let max = -1;
  for (const name of map.keys()) {
    const m = /^layers\.(\d+)\.input_layernorm$/.exec(name);
    if (m) max = Math.max(max, Number.parseInt(m[1]!, 10));
  }
  if (max < 0) throw new Error("WebGpuBackend: no transformer layers found in weight map");
  return max + 1;
}

// ---------- weight upload ----------

interface WeightsBreakdown {
  weights: Map<string, GpuTensor>;
  fetchMs: number;
  parseMs: number;
  uploadMs: number;
}

async function loadOnnxWeightsGpuTimed(
  device: GPUDevice,
  modelSource: string,
): Promise<WeightsBreakdown> {
  const tF0 = performance.now();
  const base = modelSource.endsWith("/") ? modelSource : `${modelSource}/`;
  const graphUrl = `${base}onnx/model_q4f16.onnx`;
  const dataUrl = `${base}onnx/model_q4f16.onnx_data`;
  const [graphBytes, extBytes] = await Promise.all([
    fetchBytes(graphUrl),
    fetchBytes(dataUrl),
  ]);
  const fetchMs = performance.now() - tF0;

  const tP0 = performance.now();
  const graph = parseOnnxGraph(new Uint8Array(graphBytes));
  const extData = new Uint8Array(extBytes);
  const parseMs = performance.now() - tP0;

  const tU0 = performance.now();
  const weights = uploadWeights(device, graph, extData);
  const uploadMs = performance.now() - tU0;

  return { weights, fetchMs, parseMs, uploadMs };
}

/**
 * Parse the ONNX graph + external data, convert each tensor into the
 * dtype/layout our WGSL kernels expect, and upload as a storage buffer.
 * Scales/zp/int4 layouts match the WASM path byte-for-byte.
 */
function uploadWeights(
  device: GPUDevice,
  graph: ReturnType<typeof parseOnnxGraph>,
  extData: Uint8Array,
): Map<string, GpuTensor> {

  const out = new Map<string, GpuTensor>();

  function upload(
    key: string,
    usage: GPUBufferUsageFlags,
    bytes: Uint8Array,
    shape: readonly number[],
  ): GpuTensor {
    const paddedSize = (bytes.byteLength + 3) & ~3;
    const buffer = device.createBuffer({
      label: key,
      size: Math.max(16, paddedSize),
      usage: usage | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    const mapped = new Uint8Array(buffer.getMappedRange());
    mapped.set(bytes);
    buffer.unmap();
    const info: GpuTensor = {
      name: key,
      buffer,
      byteOffset: 0,
      byteSize: bytes.byteLength,
      shape: [...shape],
    };
    out.set(key, info);
    return info;
  }
  const STORAGE = GPUBufferUsage.STORAGE;

  const bytesOf = (name: string): Uint8Array => {
    const t = graph.get(name);
    if (!t) throw new Error(`loadOnnxWeightsGpu: missing ONNX tensor "${name}"`);
    return resolveTensorBytes(t, extData);
  };
  const shapeOf = (name: string): readonly number[] => {
    const t = graph.get(name);
    if (!t) throw new Error(`loadOnnxWeightsGpu: missing ONNX tensor "${name}"`);
    return t.shape;
  };

  // Embed table.
  upload("embed.int4", STORAGE,
    bytesOf("model_embed_tokens_weight_quant"),
    shapeOf("model_embed_tokens_weight_quant"));
  upload("embed.scales", STORAGE,
    bytesOf("model_embed_tokens_weight_scales"),
    shapeOf("model_embed_tokens_weight_scales"));
  upload("embed.zp", STORAGE,
    bytesOf("model_embed_tokens_weight_zp"),
    shapeOf("model_embed_tokens_weight_zp"));

  // Final norm.
  upload("final_norm", STORAGE,
    bytesOf("model.layers.8.final_norm_layernorm.weight"),
    shapeOf("model.layers.8.final_norm_layernorm.weight"));

  // Classifier. No score bias in ONNX; synthesise zero fp16 buffer.
  {
    const quantShape = shapeOf("model_score_MatMul_weight_quant");
    upload("score.int4", STORAGE,
      bytesOf("model_score_MatMul_weight_quant"), quantShape);
    upload("score.scales", STORAGE,
      bytesOf("model_score_MatMul_weight_scales"),
      shapeOf("model_score_MatMul_weight_scales"));
    upload("score.zp", STORAGE,
      bytesOf("model_score_MatMul_weight_zp"),
      shapeOf("model_score_MatMul_weight_zp"));
    const numClasses = quantShape[0]!;
    upload("score.bias", STORAGE, new Uint8Array(numClasses * 2), [numClasses]);
  }

  // Transformer layers.
  let numLayers = 0;
  for (let L = 0; L < 64; L++) {
    if (graph.has(`model.layers.${L}.input_layernorm.weight`)) numLayers = L + 1;
  }
  for (let L = 0; L < numLayers; L++) {
    for (const n of ["input_layernorm", "post_attention_layernorm"] as const) {
      const oname = `model.layers.${L}.${n}.weight`;
      upload(`layers.${L}.${n}`, STORAGE, bytesOf(oname), shapeOf(oname));
    }
    for (const proj of ["q_proj", "k_proj", "v_proj", "o_proj"] as const) {
      const qBase = `model_layers_${L}_attn_${proj}_MatMul`;
      const k = `layers.${L}.attn.${proj}`;
      upload(`${k}.int4`, STORAGE,
        bytesOf(`${qBase}_weight_quant`),
        shapeOf(`${qBase}_weight_quant`));
      upload(`${k}.scales`, STORAGE,
        bytesOf(`${qBase}_weight_scales`),
        shapeOf(`${qBase}_weight_scales`));
      upload(`${k}.zp`, STORAGE,
        bytesOf(`${qBase}_weight_zp`),
        shapeOf(`${qBase}_weight_zp`));
      const biasName = `model.layers.${L}.attn.${proj}.Add.bias`;
      upload(`${k}.bias`, STORAGE, bytesOf(biasName), shapeOf(biasName));
    }
    {
      const sName = `model.layers.${L}.attn.sinks`;
      const flatLen = shapeOf(sName).reduce((a, b) => a * b, 1);
      upload(`layers.${L}.attn.sinks`, STORAGE, bytesOf(sName), [flatLen]);
    }
    {
      const rBase = `/model/layers_${L}/moe/router/MatMul`;
      const k = `layers.${L}.router`;
      upload(`${k}.int4`, STORAGE,
        bytesOf(`${rBase}_weight_fp32_quant`),
        shapeOf(`${rBase}_weight_fp32_quant`));
      // Router scales are f32 in ONNX — shader expects f16, down-convert.
      upload(`${k}.scales`, STORAGE,
        f32ToFp16Bytes(bytesOf(`${rBase}_weight_fp32_scales`)),
        shapeOf(`${rBase}_weight_fp32_scales`));
      upload(`${k}.zp`, STORAGE,
        bytesOf(`${rBase}_weight_fp32_zp`),
        shapeOf(`${rBase}_weight_fp32_zp`));
      const biasName = `/model/layers.${L}/moe/router/Add.bias_fp32`;
      upload(`${k}.bias`, STORAGE,
        f32ToFp16Bytes(bytesOf(biasName)), shapeOf(biasName));
    }
    // Experts: uint4 + f16 scales + synth zp=0x88.
    for (const [onnxProj, ourKey] of [
      ["gate_up_proj", "gate_up"] as const,
      ["down_proj",    "down"]    as const,
    ]) {
      const eBase = `model_layers_${L}_moe_experts_${onnxProj}`;
      const k = `layers.${L}.experts.${ourKey}`;
      const quantShape = shapeOf(`${eBase}_weight_quant`);
      const scalesShape = shapeOf(`${eBase}_weight_scales`);
      const nBlocks = scalesShape[2]!;
      const zpPerRow = (nBlocks + 1) >>> 1;
      const Ect = quantShape[0]!;
      const Nrows = quantShape[1]!;
      const zp = new Uint8Array(Ect * Nrows * zpPerRow);
      zp.fill(0x88);
      upload(`${k}.int4`, STORAGE,
        bytesOf(`${eBase}_weight_quant`), quantShape);
      upload(`${k}.scales`, STORAGE,
        bytesOf(`${eBase}_weight_scales`), scalesShape);
      upload(`${k}.zp`, STORAGE, zp, [Ect, Nrows, zpPerRow]);
      const biasName = `model.layers.${L}.moe.experts.${onnxProj}.bias`;
      upload(`${k}.bias`, STORAGE, bytesOf(biasName), shapeOf(biasName));
    }
  }

  return out;
}

const fetchBytes = fetchBytesCached;

const _cvBuf = new ArrayBuffer(4);
const _cvU32 = new Uint32Array(_cvBuf);
const _cvF32 = new Float32Array(_cvBuf);

function f32ToFp16Bytes(src: Uint8Array): Uint8Array {
  const n = src.byteLength / 4;
  const dst = new Uint8Array(n * 2);
  const sv = new DataView(src.buffer, src.byteOffset, src.byteLength);
  const dv = new DataView(dst.buffer);
  for (let i = 0; i < n; i++) {
    _cvU32[0] = sv.getUint32(i * 4, true);
    dv.setUint16(i * 2, f32ToFp16(_cvF32[0]!), true);
  }
  return dst;
}

function f32ToFp16(f: number): number {
  if (f === 0) return 0;
  _cvF32[0] = f;
  const u32 = _cvU32[0]!;
  const sign = (u32 >>> 16) & 0x8000;
  const exp32 = (u32 >>> 23) & 0xff;
  const mant23 = u32 & 0x7fffff;
  if (exp32 === 0xff) return (sign | 0x7c00 | (mant23 ? 0x200 : 0)) & 0xffff;
  let exp16 = exp32 - 127 + 15;
  if (exp16 >= 0x1f) return (sign | 0x7c00) & 0xffff;
  if (exp16 <= 0) {
    if (exp16 < -10) return sign;
    const shift = 14 - exp16;
    const mant24 = mant23 | 0x800000;
    return (sign | ((mant24 + (1 << (shift - 1))) >>> shift)) & 0xffff;
  }
  const lsb = (mant23 >>> 13) & 1;
  let m10 = (mant23 + 0xfff + lsb) >>> 13;
  if (m10 >= 0x400) {
    m10 = 0;
    exp16 += 1;
    if (exp16 >= 0x1f) return (sign | 0x7c00) & 0xffff;
  }
  return (sign | (exp16 << 10) | m10) & 0xffff;
}
