import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import { fileURLToPath } from "node:url";
import { resolve as pathResolve, dirname } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  site: "https://teamchong.github.io",
  base: "/textsift",
  vite: {
    // Vite's client-side bundler doesn't traverse subpath exports
    // through the workspace symlink reliably (only finds the bare
    // entry, then errors on the subpath). Alias to the dist file
    // directly — same target the exports field resolves to.
    resolve: {
      alias: {
        "textsift/browser": pathResolve(
          HERE,
          "../packages/textsift/dist/browser/index.js",
        ),
      },
    },
  },
  integrations: [
    starlight({
      title: "textsift",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/teamchong/textsift",
        },
      ],
      sidebar: [
        { label: "Overview", slug: "index" },
        { label: "Quickstart", slug: "quickstart" },
        { label: "Playground", slug: "playground" },
        { label: "API Reference", slug: "api" },
        { label: "Backends", slug: "backends" },
        { label: "Benchmarks", slug: "benchmarks" },
        { label: "Architecture", slug: "architecture" },
        { label: "Caveats", slug: "caveats" },
      ],
    }),
  ],
});
