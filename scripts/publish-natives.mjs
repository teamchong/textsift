#!/usr/bin/env node
// Publish per-triple subpackages from the build artifacts collected by
// the release workflow. Each subpackage contains a single .node binary
// matching the platform/arch in its name; the umbrella `textsift`
// package lists them all under `optionalDependencies` so npm picks the
// right one at install time.
//
// Triples expected in ./native-bins/textsift-${triple}/textsift-native.node:
//   linux-x64, linux-arm64, darwin-x64, darwin-arm64, windows-x64
//
// The umbrella package's runtime loader reads
// `process.platform + "-" + process.arch` and requires the matching
// `textsift-${triple}` from node_modules.

import { readFileSync, writeFileSync, mkdirSync, copyFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { execSync } from "node:child_process";

const RELEASE_TAG = process.env.RELEASE_TAG || "v0.0.0";
const VERSION = RELEASE_TAG.replace(/^v/, "");

const TRIPLES = [
  { triple: "linux-x64",     os: "linux",  cpu: "x64",   libc: "glibc" },
  { triple: "linux-arm64",   os: "linux",  cpu: "arm64", libc: "glibc" },
  { triple: "darwin-x64",    os: "darwin", cpu: "x64" },
  { triple: "darwin-arm64",  os: "darwin", cpu: "arm64" },
  { triple: "windows-x64",   os: "win32",  cpu: "x64" },
];

const ROOT = resolve(".");
const BINS_DIR = resolve(ROOT, "native-bins");

for (const t of TRIPLES) {
  const src = resolve(BINS_DIR, `textsift-${t.triple}`, "textsift-native.node");
  if (!existsSync(src)) {
    console.warn(`SKIP: no artifact for ${t.triple} (missing ${src})`);
    continue;
  }
  const pkgDir = resolve(ROOT, `dist-pkg/native-${t.triple}`);
  mkdirSync(pkgDir, { recursive: true });
  copyFileSync(src, resolve(pkgDir, "textsift-native.node"));

  const pkgJson = {
    name: `textsift-${t.triple}`,
    version: VERSION,
    description: `textsift native binding for ${t.triple}`,
    main: "textsift-native.node",
    files: ["textsift-native.node"],
    os: [t.os],
    cpu: [t.cpu],
    ...(t.libc ? { libc: [t.libc] } : {}),
    license: "Apache-2.0",
    repository: { type: "git", url: "git+https://github.com/teamchong/textsift.git" },
  };
  writeFileSync(resolve(pkgDir, "package.json"), JSON.stringify(pkgJson, null, 2));

  console.log(`publishing ${pkgJson.name}@${VERSION} from ${pkgDir}`);
  execSync(`npm publish --access public --provenance`, {
    cwd: pkgDir,
    stdio: "inherit",
    env: process.env,
  });
}

// Update the umbrella textsift package.json to list the just-published
// natives as optionalDependencies before its own publish step runs.
const umbrella = resolve(ROOT, "packages/textsift/package.json");
const pkg = JSON.parse(readFileSync(umbrella, "utf8"));
pkg.version = VERSION;
pkg.optionalDependencies = pkg.optionalDependencies || {};
for (const t of TRIPLES) {
  pkg.optionalDependencies[`textsift-${t.triple}`] = VERSION;
}
writeFileSync(umbrella, JSON.stringify(pkg, null, 2));
console.log(`updated ${umbrella} to v${VERSION} with ${TRIPLES.length} optionalDependencies`);
