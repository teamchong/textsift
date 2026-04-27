/**
 * Browser-side per-shader microbench. Runs the same pipeline / bind
 * group / buffer setup the native backend uses, times N dispatches,
 * dumps medians for the matching native bench script to compare
 * against.
 */
import { test, expect } from "@playwright/test";
import { writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { skipWithoutShaderF16 } from "../browser/helpers/skip-without-shader-f16";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = resolve(HERE, "../native/conformance/fixtures/_browser-bench.json");

const CHAIN_LEN = process.env.CHAIN ?? "1";

test.beforeEach(skipWithoutShaderF16);

test(`browser shader microbench (chain=${CHAIN_LEN})`, async ({ page }) => {
  test.setTimeout(2 * 60_000);

  page.on("pageerror", (err) => console.error("page error:", err.message));

  await page.goto(`/tests/conformance/bench-shaders.html?chain=${CHAIN_LEN}`);
  await page.waitForFunction(() => (window as any).__bench !== undefined, null, {
    timeout: 90_000,
  });
  const results = await page.evaluate(() => (window as any).__bench);
  if (results.error) throw new Error(`bench errored: ${results.error}\n${results.stack ?? ""}`);

  mkdirSync(dirname(OUT), { recursive: true });
  writeFileSync(OUT, JSON.stringify(results, null, 2));
  console.log("\n--- browser shader microbench (median ms / dispatch) ---");
  for (const [name, r] of Object.entries(results as Record<string, any>)) {
    console.log(`  ${name.padEnd(28)} median=${r.median.toFixed(3)}ms  min=${r.min.toFixed(3)}ms`);
  }

  expect(Object.keys(results).length).toBeGreaterThan(0);
});
