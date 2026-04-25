/**
 * Native tokenizer conformance vs transformers.js AutoTokenizer.
 *
 * The native textsift-core tokenizer must produce byte-for-byte
 * identical token-id sequences to AutoTokenizer across a varied
 * corpus (English, Unicode, code, edge whitespace, special tokens).
 * Any divergence is a tokenizer bug — fail the build.
 */

import { test, expect } from "@playwright/test";

test("textsift-core tokenizer matches AutoTokenizer byte-for-byte", async ({ page }) => {
  test.setTimeout(5 * 60_000);

  const consoleErrors: string[] = [];
  page.on("pageerror", (err) => {
    consoleErrors.push(`page error: ${err.message}\n${err.stack ?? ""}`);
  });
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });

  await page.goto("/tests/browser/tokenizer-conformance.html");
  await page.waitForFunction(() => (window as any).__result !== undefined, null, {
    timeout: 5 * 60_000,
  });
  const result = await page.evaluate(() => (window as any).__result);

  if (!result.ok) {
    const detail = result.error
      ? `${result.error}\n${result.stack ?? ""}`
      : `${result.failureCount}/${result.totalCases} cases failed:\n` +
        result.failures
          .map(
            (f: any) =>
              `  case ${f.i} (${JSON.stringify((f.text ?? "").slice(0, 60))}): ` +
              `${f.reason}\n    tjs:    ${JSON.stringify(f.tjs)}\n    native: ${JSON.stringify(f.native)}`,
          )
          .join("\n");
    throw new Error(`tokenizer conformance failed:\n${detail}`);
  }

  expect(result.failureCount).toBe(0);
  console.log(
    `[tokenizer-conformance] ${result.totalCases}/${result.totalCases} cases match  ` +
      `(native ready ${result.nativeReadyMs.toFixed(0)}ms, tjs ready ${result.tjsReadyMs.toFixed(0)}ms)`,
  );
  expect(consoleErrors, `console errors: ${consoleErrors.join("\n")}`).toEqual([]);
});
