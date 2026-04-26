/**
 * Run every registered shader through browser WebGPU on a
 * deterministic seeded input and dump `(uniform, extraUniforms,
 * inputs, output)` to `tests/native/conformance/fixtures/<shader>/`.
 * The native conformance tests load these binaries and assert
 * byte-equal (or within 1 fp16 ULP) against the wgpu-native dispatch.
 *
 * One Playwright run dumps fixtures for every registered shader.
 * Browser source of truth is `tests/conformance/shaders.html`.
 */

import { test, expect } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES_ROOT = resolve(HERE, "../native/conformance/fixtures");

test("dump shader conformance fixtures from browser WebGPU", async ({ page }) => {
  test.setTimeout(2 * 60_000);

  page.on("pageerror", (err) => {
    console.error("page error:", err.message, err.stack);
  });
  page.on("console", (msg) => {
    if (msg.type() === "error") console.log(`[console.error] ${msg.text()}`);
  });

  await page.goto("/tests/conformance/shaders.html");
  await page.waitForFunction(() => (window as any).__fixtures !== undefined, null, {
    timeout: 60_000,
  });
  const fixtures = await page.evaluate(() => (window as any).__fixtures);

  if (fixtures.error) {
    throw new Error(`fixture dump errored: ${fixtures.error}\n${fixtures.stack ?? ""}`);
  }

  for (const [name, f] of Object.entries(fixtures as Record<string, any>)) {
    const dir = resolve(FIXTURES_ROOT, name);
    mkdirSync(dir, { recursive: true });
    writeFileSync(resolve(dir, "uniform.bin"), Buffer.from(f.uniform.bytes));
    for (const eu of f.extraUniforms as Array<{ name: string; bytes: number[] }>) {
      writeFileSync(resolve(dir, `_uniform_${eu.name}.bin`), Buffer.from(eu.bytes));
    }
    for (const input of f.inputs as Array<{ name: string; bytes: number[] }>) {
      writeFileSync(resolve(dir, `${input.name}.bin`), Buffer.from(input.bytes));
    }
    writeFileSync(resolve(dir, "expected.bin"), Buffer.from(f.output.bytes));
    if (f.output.initial) {
      writeFileSync(resolve(dir, "_output_initial.bin"), Buffer.from(f.output.initial));
    }
    writeFileSync(
      resolve(dir, "meta.json"),
      JSON.stringify(
        {
          shader: name,
          uniform: { binding: f.uniform.binding, byteLength: f.uniform.bytes.length },
          extraUniforms: f.extraUniforms.map((eu: any) => ({
            binding: eu.binding,
            name: eu.name,
            byteLength: eu.bytes.length,
          })),
          inputs: f.inputs.map((i: any) => ({
            binding: i.binding,
            name: i.name,
            byteLength: i.bytes.length,
          })),
          output: {
            binding: f.output.binding,
            dtype: f.output.dtype,
            byteLength: f.output.bytes.length,
            hasInitial: !!f.output.initial,
          },
          dispatch: f.dispatch,
        },
        null,
        2,
      ),
    );
    console.log(
      `[${name}] uniform=${f.uniform.bytes.length}B, ` +
        `extra=[${f.extraUniforms.map((e: any) => `${e.name}:${e.bytes.length}B`).join(",")}], ` +
        `inputs=[${f.inputs.map((i: any) => `${i.name}:${i.bytes.length}B`).join(",")}], ` +
        `output=${f.output.bytes.length}B (${f.output.dtype})`,
    );
  }

  expect(Object.keys(fixtures).length).toBeGreaterThan(0);
});
