/**
 * WebGPU matmul parity: asserts the new `matmul_int4` WGSL kernel
 * produces the same output as the WASM `matmul_f32_x_int4block_out_f32`
 * on the same input + real q_proj weights. fp16/f32 tolerance applies.
 */

import { test, expect } from "@playwright/test";
import { skipWithoutShaderF16 } from "./helpers/skip-without-shader-f16";

test.beforeEach(skipWithoutShaderF16);

test("webgpu matmul_int4 parity vs wasm", async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });

  await page.goto("/tests/browser/webgpu-matmul.html");
  await page.waitForFunction(() => (window as any).__result !== undefined, null, {
    // Cold-cache runs download ~770 MB across two backends.
    timeout: 4 * 60_000,
  });
  const result = await page.evaluate(() => (window as any).__result);

  expect(consoleErrors, `console errors: ${consoleErrors.join("\n")}`).toEqual([]);
  expect(result).toMatchObject({ ok: true });
  expect(result.maxAbs).toBeLessThan(0.05);
  expect(result.rms).toBeLessThan(1e-3);
  console.log(
    `[webgpu-matmul] T=${result.T} N=${result.N} K=${result.K}  ` +
      `maxAbs=${result.maxAbs.toExponential(2)}  maxRel=${result.maxRel.toExponential(2)}  rms=${result.rms.toExponential(2)}  ` +
      `gpu=${result.tGpuMs.toFixed(1)}ms wasm=${result.tWasmMs.toFixed(1)}ms`,
  );
});
