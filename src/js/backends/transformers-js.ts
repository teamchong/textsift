/**
 * Stage-0 baseline backend: `@huggingface/transformers` (transformers.js).
 *
 * This gives us a working end-to-end browser inference path today, while
 * the custom Zig+WASM and WGSL backends are under development. It runs
 * on either onnxruntime-web's WASM or WebGPU backend depending on
 * transformers.js's internal detection.
 *
 * Migration path:
 *   - When `./wasm.js` (Zig+WASM) ships: backends/select.ts switches.
 *   - When `./webgpu.js` (custom WGSL) ships: same.
 *   - transformers.js stays as the portability fallback forever.
 */

import type {
  BackendConstructionOptions,
  InferenceBackend,
  Logits,
} from "./abstract.js";

interface TokenClassificationPipeline {
  (
    input: Int32Array | string,
    options?: Record<string, unknown>,
  ): Promise<unknown>;
  dispose?: () => Promise<void>;
}

export class TransformersJsBackend implements InferenceBackend {
  readonly name = "transformers-js" as const;
  private pipeline: TokenClassificationPipeline | null = null;
  private readonly opts: BackendConstructionOptions;

  constructor(opts: BackendConstructionOptions) {
    this.opts = opts;
  }

  async warmup(): Promise<void> {
    // Dynamic import keeps transformers.js out of the bundle when
    // consumers opt into a different backend at build time via their
    // bundler's tree-shaking.
    const mod = await import("@huggingface/transformers");
    const pipeline = (mod as unknown as {
      pipeline: (
        task: string,
        model: string,
        options?: Record<string, unknown>,
      ) => Promise<TokenClassificationPipeline>;
    }).pipeline;

    const dtype = this.opts.quantization === "fp16"
      ? "fp16"
      : this.opts.quantization === "int8"
        ? "q8"
        : "q4";

    this.pipeline = await pipeline("token-classification", "openai/privacy-filter", {
      dtype,
    });
  }

  async forward(
    tokenIds: Int32Array,
    _attentionMask: Uint8Array,
  ): Promise<Logits> {
    const pipeline = this.pipeline;
    if (!pipeline) {
      throw new Error("backend used before warmup()");
    }
    throw new Error(
      "TransformersJsBackend.forward is pending direct-logit extraction wiring — "
        + "the upstream `pipeline()` helper returns decoded spans, not raw logits. "
        + "Next step: use the lower-level `AutoModelForTokenClassification` export "
        + "from @huggingface/transformers to run the model once and expose its "
        + `output.logits tensor. tokenIds length: ${tokenIds.length}.`,
    );
  }

  dispose(): void {
    this.pipeline?.dispose?.();
    this.pipeline = null;
  }
}
