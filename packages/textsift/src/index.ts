/**
 * textsift — Node native binding entry point.
 *
 * Loads a Zig-compiled `.node` shared library that talks to
 * `wgpu-native` (Mozilla's wgpu Rust crate compiled as a C library)
 * and runs the same WGSL kernels the browser entry runs against
 * `navigator.gpu`. WebGPU only — no CPU fallback. If the host lacks
 * a usable GPU adapter (Vulkan/Metal/D3D12), `create()` throws and
 * the caller is expected to import `textsift/browser` instead.
 *
 *   import { PrivacyFilter } from "textsift/browser"; // works today
 *   import { PrivacyFilter } from "textsift";         // requires GPU
 *
 * The full WebGpuBackend port (14 WGSL shaders + buffer mgmt +
 * dispatch + readback) is in progress under issue #79. Today the
 * native entry verifies adapter availability and reports it; the
 * inference path lands shader-by-shader with conformance + bench
 * tests against the browser path at every step.
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
      "textsift native inference is under construction (see issue #79). " +
        'For browser/Node-via-WASM use, import from "textsift/browser". ' +
        "The native binding probes a WebGPU adapter successfully — the " +
        "WGSL kernel port lands shader-by-shader with conformance + " +
        "bench tests against the browser path.",
    );
  }
}
