/**
 * Browser perf bench: transformers.js default (WebGPU when available)
 * vs our Zig+WASM Stage-1 backend. Collects warm-forward medians across
 * several input lengths and prints a comparison table.
 *
 * Requires the full blob at tests/fixtures/pii-weights-full.bin
 * (symlink to /tmp/pii-wasm-e2e/pii-weights-full.bin from
 * gen-full-parity-fixture.py). The transformers.js path fetches the
 * ONNX export from HuggingFace Hub on first run (~770 MB), cached by
 * the browser afterwards.
 *
 * Not a CI test — marked with `@bench` so it's skipped by default.
 */

import { test, expect } from "@playwright/test";

test("e2e perf bench vs transformers.js default", async ({ page }) => {
  test.setTimeout(10 * 60_000); // model downloads can be slow

  page.on("pageerror", (err) => {
    console.error("page error:", err.message, err.stack);
  });
  page.on("console", (msg) => {
    if (msg.type() === "error" || msg.type() === "warning") {
      console.log(`[${msg.type()}] ${msg.text()}`);
    }
  });

  await page.goto("/tests/browser/bench.html");
  await page.waitForFunction(() => (window as any).__result !== undefined, null, {
    timeout: 10 * 60_000,
  });
  const result = await page.evaluate(() => (window as any).__result);

  if (!result.ok) {
    throw new Error(`bench failed: ${result.error}\n${result.stack ?? ""}`);
  }

  console.log("\n--- browser perf bench ---");
  console.log(
    `WebGPU: ${result.hasWebGPU}  ` +
      `tjs warmup=${result.tjsWarmupMs.toFixed(0)}ms  ` +
      `wasm warmup=${result.wasmWarmupMs.toFixed(0)}ms`,
  );
  console.log(
    `  ${"text".padEnd(40)} ${"tjs (ms)".padStart(10)} ${"wasm (ms)".padStart(10)} ${"speedup".padStart(10)}`,
  );
  for (const r of result.rows) {
    console.log(
      `  ${r.text.slice(0, 40).padEnd(40)} ${r.tjsMs.toFixed(1).padStart(10)} ${r.wasmMs.toFixed(1).padStart(10)} ${r.speedup.toFixed(2).padStart(9)}x`,
    );
  }

  expect(result.rows.length).toBeGreaterThan(0);
});
