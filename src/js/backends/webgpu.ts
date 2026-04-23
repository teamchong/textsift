/**
 * Stage 2 backend: custom WGSL compute shaders.
 *
 * Target architecture:
 *   - Int4-quantized MoE weights uploaded to a GPU buffer.
 *   - Per-layer forward pass dispatches one kernel per op (matmul, MoE
 *     router, sparse expert dispatch, attention, layer norm).
 *   - Subgroup-free reductions so Firefox/Safari work once they ship
 *     WebGPU but before subgroups extension lands.
 *
 * Not wired yet — waiting on:
 *   1. ModelLoader → real bundle.
 *   2. Stage-1 WASM path benchmarked (establishes "to beat" number).
 *   3. `scripts/measure-experts.py` go/no-go for MoE compression
 *      (informs whether to write a shared-codebook expert matmul
 *      or a straight MoE dispatch).
 */

import type {
  BackendConstructionOptions,
  InferenceBackend,
  Logits,
} from "./abstract.js";

export class WebGPUBackend implements InferenceBackend {
  readonly name = "webgpu" as const;
  private readonly opts: BackendConstructionOptions;

  constructor(opts: BackendConstructionOptions) {
    this.opts = opts;
  }

  async warmup(): Promise<void> {
    void this.opts;
    throw new Error(
      "WebGPUBackend is pending implementation. See docs/roadmap.md — Stage 2.",
    );
  }

  async forward(_tokenIds: Int32Array, _attentionMask: Uint8Array): Promise<Logits> {
    throw new Error("WebGPUBackend.forward pending implementation.");
  }

  dispose(): void {
    // No-op until real buffers are allocated in warmup().
  }
}
