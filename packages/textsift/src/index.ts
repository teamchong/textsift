/**
 * textsift — Node native binding entry point.
 *
 * Loads a Zig-compiled `.node` shared library. Each platform ships its
 * own hand-tuned fast path (comptime-selected in src/native/napi.zig):
 *
 *   macOS  → Metal-direct (hand-written MSL via Obj-C bridge)
 *   Linux  → Vulkan-direct (hand-written GLSL → SPIR-V via glslangValidator)
 *   Windows → Dawn-direct (Tint → D3D12 via Dawn's backend selection)
 *
 *   import { PrivacyFilter } from "textsift";
 *   const filter = await PrivacyFilter.create();
 *   const { redactedText } = await filter.redact("Hi Alice, my email is alice@example.com");
 */

import {
  PrivacyFilter as BrowserPrivacyFilter,
  type BackendResolver,
} from "./browser/privacy-filter.js";
import type { CreateOptions } from "./browser/types.js";
import { NodeBackend } from "./native/backend.js";

export type {
  CreateOptions,
  DetectResult,
  DetectedSpan,
  RedactOptions,
  RedactResult,
  Rule,
  SpanLabel,
  MarkerStrategy,
} from "./browser/types.js";

export { secretRules, RULE_PRESETS, markerPresets } from "./browser/index.js";
export type { RulePresetName, MarkerPresetName } from "./browser/index.js";

/**
 * `PrivacyFilter` for Node — wraps the browser implementation but
 * injects a `NodeBackend` that talks to the platform-native NAPI
 * surface (`metal*` / `vulkan*` / `dawn*`) instead of `navigator.gpu`.
 *
 * All other stages (BPE tokenizer, Viterbi CRF decoder, regex rule
 * engine, span/redaction logic) come straight from `src/browser/` —
 * pure TS with no DOM deps.
 *
 * Graceful fallback: if the native fast path fails to initialise
 * (no Vulkan loader on Linux, no Metal on a non-Apple-Silicon Mac,
 * no Dawn-compatible adapter on Windows, or the per-platform .node
 * binary is missing because optionalDependencies didn't install for
 * this triple), `create()` falls back to the WASM/CPU backend so
 * `import { PrivacyFilter } from "textsift"` still works — just
 * slower (~5–10× slower than the native path; still faster than ORT
 * Node CPU because the WASM kernels are Zig SIMD-optimized).
 */
export class PrivacyFilter extends BrowserPrivacyFilter {
  static override async create(opts: CreateOptions = {}): Promise<PrivacyFilter> {
    // Try the native fast path first. If it fails (no GPU, missing
    // platform binary, etc.), fall back to the WASM CPU path so
    // PrivacyFilter still works — same API, slower runtime.
    const resolver: BackendResolver = {
      async resolveAuto({ bundle, quantization }) {
        try {
          const backend = new NodeBackend({
            bundle,
            quantization,
            device: "webgpu",
          });
          await backend.warmup();
          return backend;
        } catch (e) {
          const reason = (e as Error).message;
          // eslint-disable-next-line no-console
          console.warn(
            `textsift: native GPU backend unavailable (${reason}). ` +
              `Falling back to WASM CPU path. Install Vulkan drivers and rebuild ` +
              `for the fast path on Linux. See packages/textsift/src/native/HANDOFF.md.`,
          );
          // Returning null tells the base class to fall through to the
          // built-in selectBackend(), which picks WASM in Node since
          // navigator.gpu is undefined.
          return null;
        }
      },
    };
    return BrowserPrivacyFilter.createWithResolver(
      { ...opts, backend: opts.backend ?? "auto" },
      resolver,
    ) as Promise<PrivacyFilter>;
  }
}
