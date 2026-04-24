/**
 * Stage 2 backend: WGSL compute shaders on WebGPU.
 *
 * Uploads the same `onnx/model_q4f16.onnx` + `.onnx_data` the other two
 * backends consume and runs the full forward on the GPU. No per-kernel
 * round-trip through JS — input token IDs + attention mask go in, logits
 * come out, everything else is a storage buffer on the device.
 *
 * Requires `shader-f16`. We feature-detect during `warmup()` and throw
 * if the adapter can't enable it; callers (selectBackend) should fall
 * back to the WASM path in that case.
 *
 * This file is scaffolding + the hottest kernel (int4 matmul). Remaining
 * kernels (rms_norm, rope, banded_attention, swiglu, router, MoE, embed,
 * classifier) land in follow-up passes; each one gets a matching WGSL
 * shader + bind-group layout + pipeline. Forward composition lives in
 * JS — same shape as `modelForward` / `blockForward`, just dispatching
 * GPU compute passes instead of wasm kernel calls.
 */

import type {
  BackendConstructionOptions,
  InferenceBackend,
  Logits,
} from "./abstract.js";
import { parseOnnxGraph, resolveTensorBytes, type OnnxTensorRef } from "../model/onnx-reader.js";

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

// ---------- WGSL ----------

/**
 * int4-block32 matmul: f32 x [T, K] × packed-uint4 W [N, K] → f32 y [T, N].
 * Dequant: `(q_nibble - zp_nibble) * scale_block`.
 *
 * Weight layout matches our on-device storage (same as WASM memory):
 *   `w_int4`: [N, K/2] u8 — byte j of row n packs `(w[n, 2j])` in the
 *             low nibble and `(w[n, 2j+1])` in the high nibble.
 *   `w_scales`: [N, K/32] f16 — one scale per 32-weight block.
 *   `w_zp`: [N, ceil(K/32/2)] u8 — uint4 zero-points packed 2/byte,
 *           same even/odd scheme as the weights.
 *   `bias`: [N] f16.
 *
 * One thread per output element. Workgroup size 64 covers a single row
 * stride cleanly. D = K = 640 / 896 in this model → 20-28 blocks per
 * dot, within reach of a scalar loop.
 */
const MATMUL_INT4_WGSL = /* wgsl */ `
enable f16;

struct Dims { T: u32, N: u32, K: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w_int4: array<u32>;    // bytes packed 4/word
@group(0) @binding(3) var<storage, read> w_scales: array<f16>;
@group(0) @binding(4) var<storage, read> w_zp: array<u32>;      // bytes packed 4/word
@group(0) @binding(5) var<storage, read> bias: array<f16>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;

const BLOCK: u32 = 32u;

fn load_byte(arr: ptr<storage, array<u32>, read>, idx: u32) -> u32 {
    let word = (*arr)[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

fn load_nibble_u32(arr: ptr<storage, array<u32>, read>, nibble_idx: u32) -> u32 {
    let byte_idx = nibble_idx >> 1u;
    let byte = load_byte(arr, byte_idx);
    let hi = (nibble_idx & 1u);
    return select(byte & 0xFu, (byte >> 4u) & 0xFu, hi == 1u);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tn = gid.x;
    let total = dims.T * dims.N;
    if (tn >= total) { return; }
    let t = tn / dims.N;
    let n = tn % dims.N;
    let K = dims.K;
    let n_blocks = K / BLOCK;
    let zp_per_row = (n_blocks + 1u) >> 1u;

    let int4_nibble_base = n * K;           // K nibbles per row
    let scale_row = n * n_blocks;           // f16 elements per row
    let zp_row_nibble = n * n_blocks;       // nibble index (n_blocks nibbles per row, packed 2/byte into zp_per_row bytes)
    // Byte indices into the packed u32 storage arrays.
    let x_row = t * K;

    var acc: f32 = 0.0;
    for (var b: u32 = 0u; b < n_blocks; b = b + 1u) {
        let scale: f32 = f32(w_scales[scale_row + b]);
        // zp packing: byte (n * zp_per_row + b/2); low nibble = even b,
        // high nibble = odd b.
        let zp_byte_idx = n * zp_per_row + (b >> 1u);
        let zp_byte = load_byte(&w_zp, zp_byte_idx);
        let zp_nib = select(zp_byte & 0xFu, (zp_byte >> 4u) & 0xFu, (b & 1u) == 1u);
        let zp_f: f32 = f32(zp_nib);

        let base_nibble = int4_nibble_base + b * BLOCK;
        var block_sum: f32 = 0.0;
        // 32 weights per block. Unroll by 4.
        for (var k: u32 = 0u; k < BLOCK; k = k + 4u) {
            // Decode 4 nibbles.
            let q0 = f32(load_nibble_u32(&w_int4, base_nibble + k + 0u)) - zp_f;
            let q1 = f32(load_nibble_u32(&w_int4, base_nibble + k + 1u)) - zp_f;
            let q2 = f32(load_nibble_u32(&w_int4, base_nibble + k + 2u)) - zp_f;
            let q3 = f32(load_nibble_u32(&w_int4, base_nibble + k + 3u)) - zp_f;
            let x0 = x[x_row + b * BLOCK + k + 0u];
            let x1 = x[x_row + b * BLOCK + k + 1u];
            let x2 = x[x_row + b * BLOCK + k + 2u];
            let x3 = x[x_row + b * BLOCK + k + 3u];
            block_sum = fma(q0, x0, block_sum);
            block_sum = fma(q1, x1, block_sum);
            block_sum = fma(q2, x2, block_sum);
            block_sum = fma(q3, x3, block_sum);
        }
        acc = fma(block_sum, scale, acc);
    }
    acc = acc + f32(bias[n]);
    y[t * dims.N + n] = acc;
}
`;

// ---------- backend ----------

export interface WebGpuBackendOptions extends BackendConstructionOptions {}

export class WebGpuBackend implements InferenceBackend {
  readonly name = "webgpu" as const;
  private device: GPUDevice | null = null;
  private weights: Map<string, GpuTensor> | null = null;
  private matmulPipeline: GPUComputePipeline | null = null;
  private matmulBGL: GPUBindGroupLayout | null = null;
  private readonly opts: WebGpuBackendOptions;

  constructor(opts: WebGpuBackendOptions) {
    this.opts = opts;
  }

  async warmup(): Promise<void> {
    if (typeof navigator === "undefined" || !navigator.gpu) {
      throw new Error("WebGpuBackend: navigator.gpu not available");
    }
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) throw new Error("WebGpuBackend: no GPUAdapter");
    if (!adapter.features.has("shader-f16")) {
      throw new Error(
        "WebGpuBackend: adapter lacks shader-f16; caller should fall back to the wasm backend",
      );
    }
    const device = await adapter.requestDevice({
      requiredFeatures: ["shader-f16"],
      requiredLimits: {
        // Experts: 128 * 1280 * 320 bytes = 52 MiB per layer just for
        // gate_up quant. Max storage buffer size needs to accommodate.
        maxStorageBufferBindingSize: Math.min(
          adapter.limits.maxStorageBufferBindingSize,
          1024 * 1024 * 1024,
        ),
        maxBufferSize: Math.min(adapter.limits.maxBufferSize, 1024 * 1024 * 1024),
      },
    });
    this.device = device;

    this.weights = await loadOnnxWeightsGpu(device, this.opts.bundle.modelSource);

    this.matmulBGL = device.createBindGroupLayout({
      label: "matmul_int4.bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    const module = device.createShaderModule({
      label: "matmul_int4.wgsl",
      code: MATMUL_INT4_WGSL,
    });
    this.matmulPipeline = await device.createComputePipelineAsync({
      label: "matmul_int4.pipeline",
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.matmulBGL] }),
      compute: { module, entryPoint: "main" },
    });
  }

  async forward(_tokenIds: Int32Array, _attentionMask: Uint8Array): Promise<Logits> {
    throw new Error("WebGpuBackend.forward: not yet implemented (scaffold only)");
  }

  dispose(): void {
    // Individual GPUBuffer handles drop with the Map; nothing to release
    // explicitly. GPUDevice.destroy() is available but optional.
    this.weights = null;
    this.device?.destroy();
    this.device = null;
    this.matmulPipeline = null;
    this.matmulBGL = null;
  }

  // ---- test helper: run a single matmul via the WGSL kernel ----

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
    if (!this.device || !this.weights || !this.matmulPipeline || !this.matmulBGL) {
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
      layout: this.matmulBGL,
      entries: [
        { binding: 0, resource: { buffer: dimsBuf } },
        { binding: 1, resource: { buffer: xBuf } },
        { binding: 2, resource: { buffer: w_int4.buffer, offset: w_int4.byteOffset, size: w_int4.byteSize } },
        { binding: 3, resource: { buffer: w_scales.buffer, offset: w_scales.byteOffset, size: w_scales.byteSize } },
        { binding: 4, resource: { buffer: w_zp.buffer, offset: w_zp.byteOffset, size: w_zp.byteSize } },
        { binding: 5, resource: { buffer: bias.buffer, offset: bias.byteOffset, size: bias.byteSize } },
        { binding: 6, resource: { buffer: yBuf } },
      ],
    });

    const encoder = d.createCommandEncoder({ label: "matmul_int4.encoder" });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.matmulPipeline);
    pass.setBindGroup(0, bindGroup);
    const total = T * N;
    pass.dispatchWorkgroups(Math.ceil(total / 64));
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
}

// ---------- weight upload ----------

/**
 * Parse the ONNX graph + external data, convert each tensor into the
 * dtype/layout our WGSL kernels expect, and upload as a storage buffer.
 * Scales/zp/int4 layouts match the WASM path byte-for-byte so a single
 * name → buffer lookup covers both shader variants.
 */
async function loadOnnxWeightsGpu(
  device: GPUDevice,
  modelSource: string,
): Promise<Map<string, GpuTensor>> {
  const base = modelSource.endsWith("/") ? modelSource : `${modelSource}/`;
  const graphUrl = `${base}onnx/model_q4f16.onnx`;
  const dataUrl = `${base}onnx/model_q4f16.onnx_data`;
  const [graphBytes, extBytes] = await Promise.all([
    fetchBytes(graphUrl),
    fetchBytes(dataUrl),
  ]);
  const graph = parseOnnxGraph(new Uint8Array(graphBytes));
  const extData = new Uint8Array(extBytes);

  const out = new Map<string, GpuTensor>();

  function upload(
    key: string,
    usage: GPUBufferUsageFlags,
    bytes: Uint8Array,
    shape: readonly number[],
  ): GpuTensor {
    // WebGPU requires buffer size to be a multiple of 4.
    const paddedSize = (bytes.byteLength + 3) & ~3;
    const buffer = device.createBuffer({
      label: key,
      size: paddedSize,
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
      const E = quantShape[0]!;
      const Nrows = quantShape[1]!;
      const zp = new Uint8Array(E * Nrows * zpPerRow);
      zp.fill(0x88);
      upload(`${k}.int4`, STORAGE,
        bytesOf(`${eBase}_weight_quant`), quantShape);
      upload(`${k}.scales`, STORAGE,
        bytesOf(`${eBase}_weight_scales`), scalesShape);
      upload(`${k}.zp`, STORAGE, zp, [E, Nrows, zpPerRow]);
      const biasName = `model.layers.${L}.moe.experts.${onnxProj}.bias`;
      upload(`${k}.bias`, STORAGE, bytesOf(biasName), shapeOf(biasName));
    }
  }

  return out;
}

async function fetchBytes(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`loadOnnxWeightsGpu: fetch ${url} → ${r.status} ${r.statusText}`);
  return r.arrayBuffer();
}

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
