/**
 * Model loader: downloads privacy-filter weights + tokenizer + calibration,
 * caches them in OPFS (browser) or the `~/.cache/pii-wasm/` dir (Node),
 * and returns a structured bundle the backends consume.
 *
 * File list fetched from the model source:
 *   - model.safetensors  (or onnx/model.onnx after export)
 *   - tokenizer.json
 *   - tokenizer_config.json
 *   - viterbi_calibration.json
 *   - config.json
 *
 * Caching:
 *   - OPFS: a directory per model URL; files live alongside a
 *     `.complete` marker written only after a successful full download.
 *   - Node: same scheme but on the filesystem.
 *
 * The `onProgress` callback fires once per file start, on each chunk as
 * bytes accumulate, and on cache-hit for each file that was already
 * present.
 */

import { PrivacyFilterError, type ProgressEvent } from "../types.js";

export interface LoadedModelBundle {
  /** Model weights (safetensors or ONNX bytes, depending on which export is present). */
  readonly weights: Uint8Array;
  /** Parsed JSON contents of `viterbi_calibration.json`. */
  readonly calibrationJson: unknown;
  /** Raw bytes of `tokenizer.json`. */
  readonly tokenizerJson: Uint8Array;
  /** Parsed JSON contents of `tokenizer_config.json`. */
  readonly tokenizerConfig: unknown;
  /** Parsed JSON contents of `config.json`. */
  readonly modelConfig: unknown;
  /** The 8 canonical span labels, in the order they appear in the model head. */
  readonly labelSet: readonly string[];
  /** Source URL the weights were fetched from. */
  readonly source: string;
}

export interface ModelLoaderOptions {
  source: string;
  signal?: AbortSignal;
  onProgress?: (event: ProgressEvent) => void;
}

export class ModelLoader {
  private readonly opts: ModelLoaderOptions;

  constructor(opts: ModelLoaderOptions) {
    if (!opts.source.endsWith("/")) {
      this.opts = { ...opts, source: opts.source + "/" };
    } else {
      this.opts = opts;
    }
  }

  async load(): Promise<LoadedModelBundle> {
    throw new PrivacyFilterError(
      "ModelLoader.load is pending wiring — this constructor builds the bundle metadata; "
        + "connecting it to OPFS + HuggingFace Hub fetches is the next step once the first "
        + "inference backend is ready to consume a bundle. See `src/js/model/loader.ts`.",
      "MODEL_DOWNLOAD_FAILED",
    );
  }
}
