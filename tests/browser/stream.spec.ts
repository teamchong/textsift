/**
 * DetectStream session: streaming detection over a chunked input must
 * produce the same span set as a single batch `detect()` call on the
 * full text.
 */

import { test, expect } from "@playwright/test";

test("DetectStream yields the same spans as batch detect()", async ({ page }) => {
  test.setTimeout(5 * 60_000);

  const consoleErrors: string[] = [];
  page.on("pageerror", (err) => {
    consoleErrors.push(`page error: ${err.message}\n${err.stack ?? ""}`);
  });
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });

  await page.goto("/tests/browser/stream.html");
  await page.waitForFunction(() => (window as any).__result !== undefined, null, {
    timeout: 4 * 60_000,
  });
  const result = await page.evaluate(() => (window as any).__result);

  if (!result.ok) {
    if (result.error) {
      throw new Error(`stream test errored: ${result.error}\n${result.stack ?? ""}`);
    }
    throw new Error(
      `stream span set differs from batch:\n` +
        `  only-batch:  ${JSON.stringify(result.onlyBatch)}\n` +
        `  only-stream: ${JSON.stringify(result.onlyStream)}`,
    );
  }
  expect(result.ok).toBe(true);
  expect(result.batchCount).toBe(result.streamCount);
  expect(consoleErrors, `console errors: ${consoleErrors.join("\n")}`).toEqual([]);
  console.log(
    `[stream] ${result.batchCount} spans match  ` +
      `(batch=${result.tBatchMs.toFixed(0)}ms, stream=${result.tStreamMs.toFixed(0)}ms)`,
  );
});
