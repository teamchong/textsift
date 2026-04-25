/**
 * Empirically probe transformers.js device/dtype combinations for this
 * model: which actually work on CPU? Settles the question vs relying on
 * whatever ORT-Web or transformers.js version-specific behaviour we
 * documented in CLAUDE.md.
 */

import { test, expect } from "@playwright/test";

test("transformers.js CPU path probe", async ({ page }) => {
  test.setTimeout(15 * 60_000);
  page.on("console", (msg) => {
    const text = msg.text();
    if (msg.type() === "error" || msg.type() === "warning") {
      console.log(`[${msg.type()}] ${text}`);
    }
  });

  await page.goto("/tests/browser/tjs-cpu-probe.html");
  await page.waitForFunction(() => (window as any).__result !== undefined, null, {
    timeout: 15 * 60_000,
  });
  const result = await page.evaluate(() => (window as any).__result);

  if (!result.ok) throw new Error(`probe failed: ${result.error}\n${result.stack ?? ""}`);

  console.log("\n--- transformers.js device/dtype probe ---");
  for (const r of result.results) {
    if (r.ok) {
      console.log(`  ${r.label.padEnd(30)} create=${r.createMs.toFixed(0).padStart(6)}ms  forward=${r.medianMs.toFixed(1).padStart(7)}ms`);
    } else {
      console.log(`  ${r.label.padEnd(30)} FAIL at ${r.stage}: ${r.error.slice(0, 120)}`);
    }
  }
  expect(result.results.length).toBeGreaterThan(0);
});
