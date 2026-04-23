/**
 * Browser test for the public API path: `PrivacyFilter.create({ backend: "wasm" })`.
 * Verifies the full wiring — tokenizer + chunking + viterbi + Stage-1
 * backend — runs end-to-end in a real Chromium tab and returns the
 * documented result shape.
 *
 * The committed weight blob is a 1-layer truncation, so span quality is
 * poor; this test asserts the pipeline RUNS, not what it decides.
 */

import { test, expect } from "@playwright/test";

test("PrivacyFilter.create(backend:wasm) end-to-end", async ({ page }) => {
  page.on("pageerror", (err) => {
    throw new Error(`page error: ${err.message}\n${err.stack ?? ""}`);
  });
  await page.goto("/tests/browser/public-api.html");
  await page.waitForFunction(() => (window as any).__result !== undefined, null, {
    timeout: 120_000,
  });
  const result = await page.evaluate(() => (window as any).__result);
  expect(result.ok, JSON.stringify(result, null, 2)).toBe(true);
  expect(result.detect).toMatchObject({
    inputMatches: true,
    spansIsArray: true,
    containsPiiType: "boolean",
    summaryType: "object",
  });
  expect(result.redact).toMatchObject({
    inputMatches: true,
    hasRedactedText: true,
    spansIsArray: true,
  });
  console.log(
    `[public-api] create=${result.tCreateMs.toFixed(1)}ms ` +
      `detect=${result.tDetectMs.toFixed(1)}ms ` +
      `redact=${result.tRedactMs.toFixed(1)}ms`,
  );
});
