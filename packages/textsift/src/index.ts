/**
 * textsift — Node native binding entry point.
 *
 * Loads a Zig-compiled `.node` shared library. On macOS the fast
 * path is Metal-direct (hand-written MSL kernels via an Obj-C
 * bridge); a wgpu-native fallback ships for adapter probing and
 * non-Metal hosts. All 15 kernels are byte-equal vs the browser
 * WGSL fixtures and at T=32 the Metal-direct forward runs in
 * ~11.6 ms (vs ~22 ms browser).
 *
 *   import { PrivacyFilter } from "textsift/browser"; // works today
 *   import { PrivacyFilter } from "textsift";         // requires GPU
 *
 * The high-level `PrivacyFilter` wiring on the native entry is not
 * yet finished (see issue #79) — `create()` still throws while the
 * tokenizer + weight-loader + Viterbi pieces are ported on top of
 * the kernel layer. The kernel layer itself is validated by
 * `tests/native/forward-metal.js` (Metal-direct end-to-end forward).
 */

import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const NATIVE_PATH = resolve(HERE, "./textsift-native.node");

interface NativeBinding {
  getAdapterInfo(): {
    vendor: string;
    architecture: string;
    device: string;
    description: string;
    backendType: number;
    adapterType: number;
  };
}

let cached: NativeBinding | null = null;
function loadNative(): NativeBinding {
  if (cached) return cached;
  try {
    cached = createRequire(import.meta.url)(NATIVE_PATH) as NativeBinding;
    return cached;
  } catch (err) {
    throw new Error(
      `textsift native binding failed to load from ${NATIVE_PATH}. ` +
        `Run \`npm run build:native\` in packages/textsift, or import ` +
        `from "textsift/browser" for the WASM/WebGPU path. ` +
        `Underlying error: ${(err as Error).message}`,
    );
  }
}

/** Information about the GPU adapter the native binding selected. */
export interface AdapterInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  backendType: number;
  adapterType: number;
}

/**
 * Probe the GPU adapter available to the native binding. Throws if
 * no adapter is found — the caller should catch that and route to
 * `textsift/browser` (the WASM path), since this entry runs WebGPU
 * only and has no CPU fallback.
 */
export function getAdapterInfo(): AdapterInfo {
  return loadNative().getAdapterInfo();
}

export class PrivacyFilter {
  private constructor() {}

  static async create(): Promise<PrivacyFilter> {
    // Surface the adapter check synchronously so callers can catch
    // "no GPU" cleanly. Once the inference port lands, this becomes
    // the device-creation step instead of throwing.
    getAdapterInfo();
    throw new Error(
      "textsift native PrivacyFilter wiring is under construction " +
        '(see issue #79). Import from "textsift/browser" until it ' +
        "lands. The kernel layer is done (Metal-direct on macOS, " +
        "all 15 kernels byte-equal vs browser, ~1.9× faster " +
        "end-to-end at T=32 — see tests/native/forward-metal.js); " +
        "the missing piece is the JS-side tokenizer + weight loader " +
        "+ Viterbi wired through the new metal* / wgpu* NAPI surface.",
    );
  }
}
