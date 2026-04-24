/**
 * Browser cross-backend conformance: the Stage-2 WGSL backend and the
 * transformers.js baseline must produce byte-identical spans on the
 * same input. Any divergence is a kernel port bug, not an exp-vs-poly
 * rounding issue (span decode runs on Viterbi over per-token argmax).
 */

import { test, expect } from "@playwright/test";

test("webgpu backend span parity vs transformers.js", async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });

  await page.goto("/tests/browser/webgpu-smoke.html");
  await page.waitForFunction(() => (window as any).__result !== undefined, null, {
    timeout: 180_000,
  });
  const result = await page.evaluate(() => (window as any).__result);

  expect(consoleErrors, `console errors: ${consoleErrors.join("\n")}`).toEqual([]);
  expect(result).toMatchObject({ ok: true });
  expect(result.gpuSpans).toEqual(result.tjsSpans);
  console.log(
    `[webgpu-smoke] ${result.tjsSpans.length} spans agreed, ` +
      `warmup=${result.tWarmupMs.toFixed(0)}ms detect=${result.tForwardMs.toFixed(0)}ms`,
  );
});
