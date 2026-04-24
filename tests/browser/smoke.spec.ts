/**
 * Browser cross-backend conformance: assert the Zig+WASM backend and
 * the transformers.js WebGPU backend produce identical spans on the
 * same input. Both load the same `onnx/model_q4f16.onnx`; any span
 * divergence is a kernel bug.
 */

import { test, expect } from "@playwright/test";

test("wasm backend span parity vs transformers.js", async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });

  await page.goto("/tests/browser/smoke.html");
  await page.waitForFunction(() => (window as any).__result !== undefined, null, {
    timeout: 120_000,
  });
  const result = await page.evaluate(() => (window as any).__result);

  expect(consoleErrors, `console errors: ${consoleErrors.join("\n")}`).toEqual([]);
  expect(result).toMatchObject({ ok: true });
  expect(result.tjsSpans).toEqual(result.wasmSpans);
  console.log(
    `[smoke] ${result.tjsSpans.length} spans agreed, ` +
      `wasm warmup=${result.tWarmupMs.toFixed(0)}ms, forward=${result.tForwardMs.toFixed(0)}ms`,
  );
});
