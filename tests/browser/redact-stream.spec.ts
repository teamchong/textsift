/**
 * redact(AsyncIterable<string>) overload: the concatenation of
 * `textStream` chunks must equal the `redactedText` a batch
 * `redact(fullText)` call produces. Span counts match too.
 */

import { test, expect } from "@playwright/test";

test("redact(stream) matches batch redact() output", async ({ page }) => {
  test.setTimeout(5 * 60_000);

  const consoleErrors: string[] = [];
  page.on("pageerror", (err) => {
    consoleErrors.push(`page error: ${err.message}\n${err.stack ?? ""}`);
  });
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });

  await page.goto("/tests/browser/redact-stream.html");
  await page.waitForFunction(() => (window as any).__result !== undefined, null, {
    timeout: 4 * 60_000,
  });
  const result = await page.evaluate(() => (window as any).__result);

  if (!result.ok) {
    if (result.error) {
      throw new Error(`redact-stream test errored: ${result.error}\n${result.stack ?? ""}`);
    }
    throw new Error(
      `redact-stream divergence:\n` +
        `  textSame=${result.textSame} spanCountSame=${result.spanCountSame}\n` +
        `  batch text (${result.batchTextLen} chars):  ${JSON.stringify(result.batchExcerpt)}\n` +
        `  stream text (${result.streamTextLen} chars): ${JSON.stringify(result.streamExcerpt)}\n` +
        `  batch spans=${result.batchSpanCount}, stream spans=${result.streamSpanCount}, final=${result.finalResultSpanCount}`,
    );
  }
  expect(result.ok).toBe(true);
  expect(result.textSame).toBe(true);
  expect(result.streamTextLen).toBe(result.batchTextLen);
  expect(consoleErrors, `console errors: ${consoleErrors.join("\n")}`).toEqual([]);
  console.log(
    `[redact-stream] text matches (${result.streamTextLen} chars), ` +
      `${result.streamSpanCount} spans  ` +
      `(batch=${result.tBatchMs.toFixed(0)}ms, stream=${result.tStreamMs.toFixed(0)}ms)`,
  );
});
