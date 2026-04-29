#!/usr/bin/env bash
# One-time publish of v0.1.0 from the local machine, using the
# `teamch` account (regular `npm login` token). Mirrors the pattern
# that worked for turboquant-wasm and vectorjson: real first version
# published manually, OIDC takes over from v0.1.1+.
#
# Prerequisite: the 5 .node artifacts are already downloaded to
# /tmp/v010-natives/textsift-<triple>/textsift-native.node, which the
# release CI run uploaded as artifacts.
#
# After this script succeeds:
#   1. Bump packages/textsift/package.json version to 0.1.1
#   2. Push to main → CI publishes 0.1.1 via OIDC and the OIDC binding
#      finally takes effect.
#
# Usage:
#   npm login                            # logged in as `teamch`
#   bash scripts/publish-v010-locally.sh

set -euo pipefail

VERSION="0.1.0"
ARTIFACTS_DIR="/tmp/v010-natives"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

WHOAMI="$(npm whoami)"
if [[ "$WHOAMI" != "teamch" ]]; then
  echo "expected npm user 'teamch', got '$WHOAMI'" >&2
  exit 1
fi

if [[ ! -d "$ARTIFACTS_DIR" ]]; then
  echo "artifacts not found at $ARTIFACTS_DIR" >&2
  echo "run: gh run download <run-id> --pattern 'textsift-*' --dir $ARTIFACTS_DIR" >&2
  exit 1
fi

TRIPLES=(
  "linux-x64    linux  x64   glibc"
  "linux-arm64  linux  arm64 glibc"
  "darwin-x64   darwin x64   -"
  "darwin-arm64 darwin arm64 -"
  "windows-x64  win32  x64   -"
)

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

# Step 1: publish each native subpackage at v0.1.0 with its real
# .node binary, no --provenance (provenance requires OIDC; we're
# token-authed locally).
for entry in "${TRIPLES[@]}"; do
  read -r triple os cpu libc <<< "$entry"
  echo
  if npm view "textsift-$triple@$VERSION" version >/dev/null 2>&1; then
    echo "==> textsift-$triple@$VERSION already on npm, skipping"
    continue
  fi
  echo "==> publish textsift-$triple@$VERSION"

  src="$ARTIFACTS_DIR/textsift-$triple/textsift-native.node"
  if [[ ! -f "$src" ]]; then
    echo "missing artifact: $src" >&2
    exit 1
  fi

  pkgdir="$WORKDIR/textsift-$triple"
  mkdir -p "$pkgdir"
  cp "$src" "$pkgdir/textsift-native.node"

  libc_field=""
  if [[ "$libc" != "-" ]]; then
    libc_field=$',\n  "libc": ["'"$libc"'"]'
  fi

  cat > "$pkgdir/package.json" <<EOF
{
  "name": "textsift-$triple",
  "version": "$VERSION",
  "description": "textsift native binding for $triple",
  "main": "textsift-native.node",
  "files": ["textsift-native.node"],
  "os": ["$os"],
  "cpu": ["$cpu"]$libc_field,
  "license": "Apache-2.0",
  "repository": { "type": "git", "url": "git+https://github.com/teamchong/textsift.git" }
}
EOF

  ( cd "$pkgdir" && npm publish --access public )
done

# Step 2: build the umbrella IN-PLACE in the workspace so it sees the
# hoisted node_modules (`@types/node`, `@webgpu/types`, etc. live at
# the workspace root, not in the umbrella subpackage). Then stage a
# temp copy with patched package.json for publish.
echo
echo "==> build umbrella in-place"
( cd "$ROOT" && npm run build:browser -w textsift && npm run build:native -w textsift )

echo
if npm view "textsift@$VERSION" version >/dev/null 2>&1; then
  echo "==> textsift@$VERSION already on npm, done."
  exit 0
fi
echo "==> publish textsift@$VERSION"
umbrella_src="$ROOT/packages/textsift"
umbrella_stage="$WORKDIR/textsift-umbrella"
mkdir -p "$umbrella_stage"

# Copy only what npm would actually publish (per the package.json
# `files` allowlist) plus package.json itself. Avoids dragging
# node_modules / src / scripts into the publish staging.
cp "$umbrella_src/package.json" "$umbrella_stage/package.json"
[[ -f "$umbrella_src/LICENSE" ]] && cp "$umbrella_src/LICENSE" "$umbrella_stage/LICENSE"
cp "$umbrella_src/README.md" "$umbrella_stage/README.md"
cp -R "$umbrella_src/dist" "$umbrella_stage/dist"

# Inject optionalDependencies + correct version into the staged copy.
node <<NODE
const fs = require("node:fs");
const p = "$umbrella_stage/package.json";
const pkg = JSON.parse(fs.readFileSync(p, "utf8"));
pkg.version = "$VERSION";
pkg.optionalDependencies = {};
for (const t of ["linux-x64", "linux-arm64", "darwin-x64", "darwin-arm64", "windows-x64"]) {
  pkg.optionalDependencies["textsift-" + t] = "$VERSION";
}
fs.writeFileSync(p, JSON.stringify(pkg, null, 2) + "\n");
NODE

( cd "$umbrella_stage" && npm publish --access public )

echo
echo "v$VERSION published. Next:"
echo "  1. Bump packages/textsift/package.json version to 0.1.1"
echo "  2. git commit + push to main"
echo "  3. CI publishes 0.1.1 via OIDC; future versions are CI-driven"
