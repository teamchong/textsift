/**
 * Model loader.
 *
 * Fetches the two small JSON artifacts textsift owns (parsing + decoding):
 *   - `config.json` — architecture, `id2label`, context length.
 *   - `viterbi_calibration.json` — the 6 CRF transition biases.
 *
 * Tokenizer artifact (`tokenizer.json`) and ONNX graph + external data
 * files are loaded by the WASM and WebGPU backends directly via
 * `fetchBytesCached`, which writes them to OPFS for second-visit
 * cache hits. No `@huggingface/transformers` dependency.
 */

import { PrivacyFilterError, type ProgressEvent } from "../types.js";

/** Narrow view of the `config.json` fields textsift reads. */
export interface ModelConfig {
  readonly model_type: string;
  readonly architectures: readonly string[];
  readonly id2label: Readonly<Record<string, string>>;
  readonly label2id?: Readonly<Record<string, number>>;
  readonly max_position_embeddings: number;
}

export interface LoadedModelBundle {
  /** HF Hub repo id (e.g. "openai/privacy-filter"). */
  readonly modelId: string;
  /** Resolved URL base (always ending in `/`). */
  readonly modelSource: string;
  /** Parsed `config.json`. */
  readonly modelConfig: ModelConfig;
  /** Parsed `viterbi_calibration.json` (raw JSON; caller passes to `loadCalibration`). */
  readonly calibrationJson: unknown;
  /** 8 PII span labels, in model-head order (derived from `id2label`). */
  readonly labelSet: readonly string[];
}

export interface ModelLoaderOptions {
  /** HF Hub repo id (e.g. `openai/privacy-filter`) or a full base URL. */
  source: string;
  signal?: AbortSignal;
  onProgress?: (event: ProgressEvent) => void;
}

const EXPECTED_ARCHITECTURE = "OpenAIPrivacyFilterForTokenClassification";
const EXPECTED_MODEL_TYPE = "openai_privacy_filter";
const EXPECTED_NUM_CLASSES = 33;
const EXPECTED_NUM_LABELS = 8;

export class ModelLoader {
  private readonly opts: ModelLoaderOptions;

  constructor(opts: ModelLoaderOptions) {
    this.opts = { ...opts, source: normalizeSource(opts.source) };
  }

  async load(): Promise<LoadedModelBundle> {
    const { source, signal, onProgress } = this.opts;
    const modelId = deriveModelId(source);

    const [modelConfigJson, calibrationJson] = await Promise.all([
      fetchJson(`${source}config.json`, { signal, onProgress }),
      fetchJson(`${source}viterbi_calibration.json`, { signal, onProgress }),
    ]);

    const modelConfig = parseConfig(modelConfigJson);
    const labelSet = deriveLabelSet(modelConfig);

    return Object.freeze({
      modelId,
      modelSource: source,
      modelConfig,
      calibrationJson,
      labelSet,
    });
  }
}

function normalizeSource(source: string): string {
  if (/^https?:\/\//i.test(source)) {
    return source.endsWith("/") ? source : `${source}/`;
  }
  return `https://huggingface.co/${source}/resolve/main/`;
}

function deriveModelId(source: string): string {
  const match = source.match(/huggingface\.co\/([^/]+\/[^/]+)\//i);
  if (match && match[1]) return match[1];
  return "openai/privacy-filter";
}

interface FetchOptions {
  signal?: AbortSignal;
  onProgress?: (event: ProgressEvent) => void;
}

async function fetchJson(url: string, opts: FetchOptions): Promise<unknown> {
  let res: Response;
  try {
    res = await fetch(url, { signal: opts.signal });
  } catch (e) {
    if ((e as Error).name === "AbortError") {
      throw new PrivacyFilterError("fetch aborted", "ABORTED", e as Error);
    }
    throw new PrivacyFilterError(
      `network error loading ${url}: ${(e as Error).message}`,
      "MODEL_DOWNLOAD_FAILED",
      e as Error,
    );
  }
  if (!res.ok) {
    throw new PrivacyFilterError(
      `fetch ${url} → ${res.status} ${res.statusText}`,
      "MODEL_DOWNLOAD_FAILED",
    );
  }
  const text = await res.text();
  opts.onProgress?.({
    stage: "download",
    loaded: text.length,
    total: Number(res.headers.get("content-length")) || text.length,
    url,
  });
  try {
    return JSON.parse(text);
  } catch (e) {
    throw new PrivacyFilterError(
      `invalid JSON at ${url}: ${(e as Error).message}`,
      "MODEL_DOWNLOAD_FAILED",
      e as Error,
    );
  }
}

function parseConfig(raw: unknown): ModelConfig {
  if (raw === null || typeof raw !== "object" || Array.isArray(raw)) {
    throw new PrivacyFilterError("config.json is not an object", "MODEL_DOWNLOAD_FAILED");
  }
  const cfg = raw as Record<string, unknown>;
  const modelType = cfg["model_type"];
  if (modelType !== EXPECTED_MODEL_TYPE) {
    throw new PrivacyFilterError(
      `config.json model_type is ${JSON.stringify(modelType)}; expected ${EXPECTED_MODEL_TYPE}`,
      "MODEL_DOWNLOAD_FAILED",
    );
  }
  const archs = cfg["architectures"];
  if (!Array.isArray(archs) || !archs.includes(EXPECTED_ARCHITECTURE)) {
    throw new PrivacyFilterError(
      `config.json architectures missing ${EXPECTED_ARCHITECTURE}`,
      "MODEL_DOWNLOAD_FAILED",
    );
  }
  const id2label = cfg["id2label"];
  if (id2label === null || typeof id2label !== "object" || Array.isArray(id2label)) {
    throw new PrivacyFilterError("config.json id2label missing", "MODEL_DOWNLOAD_FAILED");
  }
  const maxPos = cfg["max_position_embeddings"];
  if (typeof maxPos !== "number") {
    throw new PrivacyFilterError(
      "config.json max_position_embeddings missing",
      "MODEL_DOWNLOAD_FAILED",
    );
  }

  return Object.freeze({
    model_type: modelType,
    architectures: Object.freeze([...(archs as string[])]),
    id2label: id2label as Record<string, string>,
    label2id: cfg["label2id"] as Record<string, number> | undefined,
    max_position_embeddings: maxPos,
  });
}

/**
 * Derive the 8 span-label strings in model-head order from `id2label`.
 *
 * The head's class ordering is `[O, (B,I,E,S) × 8]` — class 0 is
 * background, then each label claims four consecutive class ids for
 * B/I/E/S. We verify the block structure while deriving the label
 * list so we fail fast if the upstream head layout ever changes.
 */
function deriveLabelSet(config: ModelConfig): readonly string[] {
  const id2label = config.id2label;
  const indices = Object.keys(id2label)
    .map((k) => Number.parseInt(k, 10))
    .sort((a, b) => a - b);

  if (indices.length !== EXPECTED_NUM_CLASSES) {
    throw new PrivacyFilterError(
      `config.json id2label has ${indices.length} entries; expected ${EXPECTED_NUM_CLASSES}`,
      "MODEL_DOWNLOAD_FAILED",
    );
  }
  if (indices[0] !== 0 || indices[indices.length - 1] !== EXPECTED_NUM_CLASSES - 1) {
    throw new PrivacyFilterError(
      "config.json id2label indices are not a contiguous [0,32] range",
      "MODEL_DOWNLOAD_FAILED",
    );
  }
  if (id2label["0"] !== "O") {
    throw new PrivacyFilterError(
      `config.json id2label[0] must be "O"; got ${JSON.stringify(id2label["0"])}`,
      "MODEL_DOWNLOAD_FAILED",
    );
  }

  const labels: string[] = [];
  const prefixes = ["B", "I", "E", "S"] as const;
  for (let lbl = 0; lbl < EXPECTED_NUM_LABELS; lbl++) {
    const base = 1 + lbl * 4;
    const tags = prefixes.map((p, k) => ({ prefix: p, tag: id2label[String(base + k)] }));
    const first = tags[0]?.tag;
    if (typeof first !== "string" || !first.startsWith("B-")) {
      throw new PrivacyFilterError(
        `config.json id2label[${base}] is not a B-tag: ${JSON.stringify(first)}`,
        "MODEL_DOWNLOAD_FAILED",
      );
    }
    const labelName = first.slice(2);
    for (const { prefix, tag } of tags) {
      if (typeof tag !== "string" || tag !== `${prefix}-${labelName}`) {
        throw new PrivacyFilterError(
          `config.json id2label block for ${labelName} broken at ${prefix}: got ${JSON.stringify(tag)}`,
          "MODEL_DOWNLOAD_FAILED",
        );
      }
    }
    labels.push(labelName);
  }
  return Object.freeze(labels);
}
