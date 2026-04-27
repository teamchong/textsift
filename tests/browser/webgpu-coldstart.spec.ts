/**
 * Cold-start breakdown: splits WebGpuBackend.warmup into adapter/device
 * acquisition, ONNX fetch + parse, weight upload, and pipeline compile.
 * Run twice — first call is cold (download), second is cache-hit.
 */

import { test, expect } from "@playwright/test";
import { skipWithoutShaderF16 } from "./helpers/skip-without-shader-f16";

test.beforeEach(skipWithoutShaderF16);

test("webgpu cold-start timing breakdown", async ({ page }) => {
  test.setTimeout(5 * 60_000);
  page.on("console", (msg) => {
    const text = msg.text();
    if (text.includes("[textsift]")) console.log(text);
  });

  await page.goto("/tests/browser/webgpu-coldstart.html");
  await page.waitForFunction(() => (window as any).__result !== undefined, null, {
    timeout: 5 * 60_000,
  });
  const result = await page.evaluate(() => (window as any).__result);

  if (!result.ok) {
    throw new Error(`coldstart failed: ${result.error}\n${result.stack ?? ""}`);
  }
  const dumpRun = (label: string, run: { warmupMs: number; timings: any }) => {
    const t = run.timings;
    console.log(`  ${label}: ${run.warmupMs.toFixed(0)}ms`);
    console.log(`    adapter:        ${t.adapterMs.toFixed(0)}ms`);
    console.log(`    device:         ${t.deviceMs.toFixed(0)}ms`);
    console.log(`    onnx fetch:     ${t.onnxFetchMs.toFixed(0)}ms`);
    console.log(`    onnx parse:     ${t.onnxParseMs.toFixed(0)}ms`);
    console.log(`    weight upload:  ${t.weightUploadMs.toFixed(0)}ms`);
    console.log(`    pipeline compile: ${t.pipelineCompileMs.toFixed(0)}ms`);
  };
  console.log("\n--- webgpu cold-start breakdown ---");
  console.log(`  bundle load:    ${result.bundleMs.toFixed(0)}ms`);
  dumpRun("first warmup (cold)", result.cold);
  dumpRun("second warmup (cache API hit)", result.warm);
  console.log(`  grand total:    ${result.totalMs.toFixed(0)}ms`);
  expect(result.ok).toBe(true);
});
