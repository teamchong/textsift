/**
 * Playwright helper: skip a test when the running browser's WebGPU
 * adapter doesn't expose `shader-f16`. The textsift WebGPU backend
 * requires it; CI runners often use a software WebGPU adapter
 * (SwiftShader, lavapipe) that doesn't.
 *
 * Usage at the top of a webgpu spec:
 *   test.beforeEach(skipWithoutShaderF16);
 */
import type { Page, TestInfo } from "@playwright/test";
import { test } from "@playwright/test";

export async function skipWithoutShaderF16(
  { page }: { page: Page },
  testInfo: TestInfo,
): Promise<void> {
  await page.goto("about:blank");
  const probe = await page.evaluate(async (): Promise<
    { ok: true; hasF16: boolean } | { ok: false; reason: string }
  > => {
    const gpu = (navigator as unknown as { gpu?: { requestAdapter(): Promise<{ features: { has(n: string): boolean } } | null> } }).gpu;
    if (!gpu) return { ok: false, reason: "no navigator.gpu" };
    try {
      const adapter = await gpu.requestAdapter();
      if (!adapter) return { ok: false, reason: "requestAdapter returned null" };
      return { ok: true, hasF16: adapter.features.has("shader-f16") };
    } catch (e) {
      return { ok: false, reason: (e as Error).message };
    }
  });

  if (!probe.ok) {
    test.skip(true, `WebGPU unavailable: ${probe.reason}`);
    return;
  }
  if (!probe.hasF16) {
    test.skip(true, "WebGPU adapter lacks shader-f16 (likely software adapter on CI)");
  }
}
