/**
 * Custom rule engine: regex + match-fn rules merged with model spans.
 * Verifies batch and streaming paths agree, and that rule spans
 * land with the expected severity / marker / source discriminator.
 */

import { test, expect } from "@playwright/test";

test("custom rules merge with model spans (batch + stream)", async ({ page }) => {
  test.setTimeout(5 * 60_000);

  const consoleErrors: string[] = [];
  page.on("pageerror", (err) => {
    consoleErrors.push(`page error: ${err.message}\n${err.stack ?? ""}`);
  });
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });

  await page.goto("/tests/browser/rules.html");
  await page.waitForFunction(() => (window as any).__result !== undefined, null, {
    timeout: 4 * 60_000,
  });
  const result = await page.evaluate(() => (window as any).__result);

  if (!result.ok) {
    if (result.error) {
      throw new Error(`rules test errored: ${result.error}\n${result.stack ?? ""}`);
    }
    throw new Error(
      `rules divergence:\n` +
        `  cardSpans=${result.cardSpanCount} (severity=${JSON.stringify(result.cardSeverity)})\n` +
        `  ticketSpans=${result.ticketSpanCount} (severity=${JSON.stringify(result.ticketSeverity)})\n` +
        `  cardRedacted=${result.cardRedacted} ticketRedacted=${result.ticketRedacted} cardLeaked=${result.cardLeaked}\n` +
        `  textsAgree=${result.textsAgree} ruleCountsAgree=${result.ruleCountsAgree}\n` +
        `  batch=${JSON.stringify(result.batchExcerpt)}\n` +
        `  stream=${JSON.stringify(result.streamExcerpt)}`,
    );
  }
  expect(result.ok).toBe(true);
  expect(result.cardSpanCount).toBe(2);
  expect(result.ticketSpanCount).toBe(1);
  expect(result.cardLeaked).toBe(false);
  expect(result.cardRedacted).toBe(true);
  expect(result.ticketRedacted).toBe(true);
  expect(result.textsAgree).toBe(true);
  expect(result.ruleCountsAgree).toBe(true);
  expect(consoleErrors, `console errors: ${consoleErrors.join("\n")}`).toEqual([]);
  console.log(
    `[rules] card=${result.cardSpanCount} ticket=${result.ticketSpanCount} ` +
      `batch=${result.batchTextLen}c stream=${result.streamTextLen}c`,
  );
});
