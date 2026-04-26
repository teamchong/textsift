import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "tests/browser",
  fullyParallel: false,
  // Cold-cache runs download ~770 MB. Set a generous default timeout
  // so tests don't fail the first time they touch a new model source.
  timeout: 5 * 60_000,
  expect: { timeout: 10_000 },
  reporter: "list",
  use: {
    baseURL: "http://localhost:8123",
    // Required so WebAssembly.Memory + SharedArrayBuffer work for any
    // future multi-threaded test — harmless for the current single-
    // threaded setup.
    contextOptions: { bypassCSP: true },
  },
  webServer: {
    // Custom server adds Cross-Origin-Opener-Policy: same-origin and
    // Cross-Origin-Embedder-Policy: require-corp so SharedArrayBuffer
    // is available — required for the multi-threaded WASM backend.
    command: "python3 packages/textsift/scripts/serve-coi.py 8123",
    url: "http://localhost:8123/packages/textsift/dist/browser/textsift.wasm",
    reuseExistingServer: !process.env.CI,
    cwd: ".",
    timeout: 15_000,
  },
  projects: [
    {
      name: "chromium",
      use: {
        browserName: "chromium",
        launchOptions: {
          args: [
            // Required for WebGPU in Chromium — both in headless and headed
            // modes. On macOS with ANGLE-over-Metal this surfaces the real
            // GPU; on Linux CI you'd want swiftshader or vulkan instead.
            "--enable-unsafe-webgpu",
            "--enable-features=Vulkan",
            "--use-angle=metal",
            "--enable-webgpu-developer-features",
          ],
        },
      },
    },
  ],
});
