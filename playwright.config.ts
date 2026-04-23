import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "tests/browser",
  fullyParallel: false,
  timeout: 120_000,
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
    command: "python3 -m http.server 8123",
    url: "http://localhost:8123/dist/pii.wasm",
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
