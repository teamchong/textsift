/**
 * textsift — Node native binding entry point.
 *
 * This entry is reserved for the native NAPI binding (Zig-compiled
 * shared library, exposed via Node-API). It is **not yet built** —
 * see issue #79.
 *
 * Today, importing `textsift` from Node throws. Browser bundlers
 * resolve the `./browser` subpath instead (via the `exports` field
 * in package.json) and never load this module.
 *
 *   import { PrivacyFilter } from "textsift/browser"; // works today
 *   import { PrivacyFilter } from "textsift";         // throws (until #79)
 *
 * Once the native binding lands, the API surface here will mirror
 * the browser entry exactly: same `PrivacyFilter` class, same
 * `detect()` / `redact()` shapes, same `Rule` types. The only
 * difference is the backend — native CPU/GPU instead of WASM/WebGPU.
 */

export class PrivacyFilter {
  static async create(): Promise<PrivacyFilter> {
    throw new Error(
      "textsift native binding is not built yet (see issue #79). " +
        'For browser/Node-via-WASM use, import from "textsift/browser".',
    );
  }
}
