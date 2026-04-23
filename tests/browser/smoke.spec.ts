/**
 * Browser smoke test for the Stage-1 WasmBackend.
 *
 * Loads the bundled library + truncated weight blob + committed PyTorch
 * reference fixtures, runs one forward pass in a real Chromium tab, and
 * asserts that every token's argmax class matches the reference. That
 * gives us a real-world confirmation — beyond the Node tests — that the
 * wasm module, the typed-array views, and the bump allocator all behave
 * correctly under a browser's JS engine.
 */

import { test, expect } from "@playwright/test";

test("wasm backend argmax parity vs PyTorch reference", async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });

  await page.goto("/tests/browser/smoke.html");
  await page.waitForFunction(() => (window as any).__result !== undefined, null, {
    timeout: 60_000,
  });
  const result = await page.evaluate(() => (window as any).__result);

  expect(consoleErrors, `console errors: ${consoleErrors.join("\n")}`).toEqual([]);
  expect(result).toMatchObject({
    ok: true,
    T: 16,
    C: 33,
    numClasses: 33,
  });
  expect(result.classMatches).toBe(result.T);
  console.log(
    `[smoke] argmax ${result.classMatches}/${result.T}, ` +
      `maxAbs=${result.maxAbs.toExponential(2)}, rms=${result.rms.toExponential(2)}, ` +
      `warmup=${result.tWarmupMs.toFixed(1)}ms, forward=${result.tForwardMs.toFixed(1)}ms`,
  );
});
