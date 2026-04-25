import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

export default defineConfig({
  site: "https://teamchong.github.io",
  base: "/pii-wasm",
  integrations: [
    starlight({
      title: "pii-wasm",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/teamchong/pii-wasm",
        },
      ],
      sidebar: [
        { label: "Overview", slug: "index" },
        { label: "Quickstart", slug: "quickstart" },
        { label: "API Reference", slug: "api" },
        { label: "Backends", slug: "backends" },
        { label: "Benchmarks", slug: "benchmarks" },
        { label: "Architecture", slug: "architecture" },
        { label: "Caveats", slug: "caveats" },
      ],
    }),
  ],
});
