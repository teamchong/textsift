/**
 * Per-forward latency across T. Single backend (webgpu) so we can
 * isolate where latency scales with T.
 */

import { test, expect } from "@playwright/test";

test("webgpu forward latency across T", async ({ page }) => {
  test.setTimeout(5 * 60_000);

  await page.goto("/tests/browser/webgpu-profile.html");
  await page.waitForFunction(() => (window as any).__result !== undefined, null, {
    timeout: 5 * 60_000,
  });
  const result = await page.evaluate(() => (window as any).__result);

  if (!result.ok) throw new Error(`profile failed: ${result.error}\n${result.stack ?? ""}`);

  console.log("\n--- webgpu per-forward latency ---");
  for (const r of result.rows) {
    console.log(`  ${r.label.padEnd(8)} forward=${r.forwardMs.toFixed(1)}ms`);
  }
  expect(result.rows.length).toBeGreaterThan(0);
});
