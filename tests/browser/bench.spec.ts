/**
 * Browser perf bench: steady-state per-forward latency for
 * transformers.js (forced via resolver) vs our Zig+WASM and WebGPU
 * backends. Median of 5 samples after 2 warmup iterations.
 *
 * This bench does NOT report cold-start numbers. Cold start is
 * dominated by model download + storage choice, neither of which
 * tells you anything about engine speed (and isolating engine cost
 * from storage requires Service-Worker plumbing we deliberately
 * skip — see benchmarks.mdx for the rationale).
 */

import { test, expect } from "@playwright/test";
import { skipWithoutShaderF16 } from "./helpers/skip-without-shader-f16";

test.beforeEach(skipWithoutShaderF16);

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
  console.log(`WebGPU: ${result.hasWebGPU}`);
  console.log("\n  LATENCY (median per-forward, ms — lower is better)");
  console.log(
    `  ${"text".padEnd(40)} ${"tjs".padStart(10)} ${"wasm".padStart(10)} ${"gpu".padStart(10)} ${"gpu/tjs".padStart(10)}`,
  );
  for (const r of result.rows) {
    console.log(
      `  ${r.text.slice(0, 40).padEnd(40)} ` +
        `${r.tjsMs.toFixed(1).padStart(10)} ` +
        `${r.wasmMs.toFixed(1).padStart(10)} ` +
        `${r.gpuMs.toFixed(1).padStart(10)} ` +
        `${(r.gpuMs / r.tjsMs).toFixed(2).padStart(9)}x`,
    );
  }
  console.log("\n  THROUGHPUT (sustained, tok/s — higher is better)");
  console.log(
    `  ${"text".padEnd(40)} ${"tjs".padStart(10)} ${"wasm".padStart(10)} ${"gpu".padStart(10)} ${"gpu/tjs".padStart(10)}`,
  );
  for (const r of result.rows) {
    console.log(
      `  ${r.text.slice(0, 40).padEnd(40)} ` +
        `${r.tjsTokPerSec.toFixed(0).padStart(10)} ` +
        `${r.wasmTokPerSec.toFixed(0).padStart(10)} ` +
        `${r.gpuTokPerSec.toFixed(0).padStart(10)} ` +
        `${(r.gpuTokPerSec / r.tjsTokPerSec).toFixed(2).padStart(9)}x`,
    );
  }

  expect(result.rows.length).toBeGreaterThan(0);
});
