/**
 * Full-forward parity: runs the same token sequence through the WebGPU
 * backend and the WASM backend, compares logits + per-token argmax.
 * Any divergence beyond fp16 tolerance (or any argmax disagreement) is
 * a kernel port bug.
 */

import { test, expect } from "@playwright/test";

test("webgpu forward parity vs wasm", async ({ page }) => {
  // Cold-cache run downloads ~770 MB across two backends; allow 5 min.
  test.setTimeout(5 * 60_000);

  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });

  await page.goto("/tests/browser/webgpu-forward.html");
  await page.waitForFunction(() => (window as any).__result !== undefined, null, {
    timeout: 4 * 60_000,
  });
  const result = await page.evaluate(() => (window as any).__result);

  console.log(
    `[webgpu-forward] T=${result.T}  argmax=${result.argmaxAgree}/${result.T}  ` +
      `maxAbs=${result.maxAbs.toExponential(2)}  rms=${result.rms.toExponential(2)}  ` +
      `gpu=${result.tGpuMs.toFixed(0)}ms wasm=${result.tWasmMs.toFixed(0)}ms`,
  );
  if (result.row0Gpu && result.row0Wasm) {
    const g = (result.row0Gpu as number[]).map((x: number) => x.toFixed(3));
    const w = (result.row0Wasm as number[]).map((x: number) => x.toFixed(3));
    console.log(`  row0 gpu: [${g.slice(0, 8).join(", ")}…]`);
    console.log(`  row0 wasm: [${w.slice(0, 8).join(", ")}…]`);
    console.log(`  outlier: row t=${result.outlierRowT} c=${result.outlierRowC} gpu=${result.outlierGpu} wasm=${result.outlierWasm}`);
  }
  expect(consoleErrors, `console errors: ${consoleErrors.join("\n")}`).toEqual([]);
  expect(result).toMatchObject({ ok: true });
  expect(result.argmaxAgree).toBe(result.T);
  expect(result.maxAbs).toBeLessThan(3.0);
  expect(result.rms).toBeLessThan(0.5);
});
