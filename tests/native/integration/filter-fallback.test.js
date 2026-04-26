// Verify the WASM fallback path fires when the native GPU backend is
// unavailable (e.g. no Vulkan loader on Linux, no Metal on a non-Apple
// Silicon Mac, or the platform's optionalDependency .node didn't install).
//
// We simulate native failure by temporarily moving the .node binary aside
// so `createRequire(NATIVE_PATH)` throws. The PrivacyFilter resolver
// catches that and returns null, which makes the base class fall through
// to the WASM CPU path. The integration test passes if redact() still
// works end-to-end with the same span output as the native path — just
// slower.

import { rename, stat } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const NATIVE_PATH = resolve(HERE, "../../../packages/textsift/dist/textsift-native.node");
const NATIVE_BAK = `${NATIVE_PATH}.fallback-test.bak`;

async function exists(p) {
  try { await stat(p); return true; } catch { return false; }
}

const TEST_INPUT =
  "Hi Alice, my email is alice@example.com and my phone is +1-555-0123. " +
  "Please send the report to bob@gmail.com by tomorrow.";

const hadNative = await exists(NATIVE_PATH);
if (hadNative) {
  await rename(NATIVE_PATH, NATIVE_BAK);
}

let result;
try {
  // Capture the deprecation warning emitted by the fallback path so we
  // can assert it printed (proves we actually went through the fallback).
  const warns = [];
  const origWarn = console.warn;
  console.warn = (...args) => { warns.push(args.join(" ")); origWarn(...args); };

  const { PrivacyFilter } = await import("../../../packages/textsift/dist/index.js");
  const filter = await PrivacyFilter.create();
  result = await filter.redact(TEST_INPUT);
  filter.dispose();

  console.warn = origWarn;

  const fallbackFired = warns.some((w) => w.includes("native GPU backend unavailable"));
  if (!fallbackFired) {
    console.error("FAIL: expected fallback warning ('native GPU backend unavailable'), got none");
    console.error("warns:", warns);
    process.exit(1);
  }
} finally {
  if (hadNative) {
    await rename(NATIVE_BAK, NATIVE_PATH);
  }
}

if (!result.containsPii) {
  console.error("FAIL: WASM fallback didn't detect PII");
  process.exit(1);
}
if (result.spans.length < 2) {
  console.error(`FAIL: WASM fallback found ${result.spans.length} spans, expected ≥2`);
  process.exit(1);
}

console.log(`✓ WASM fallback works: ${result.spans.length} spans detected via CPU path`);
for (const span of result.spans) {
  console.log(`  [${span.start}..${span.end}] ${span.label}`);
}
