#!/usr/bin/env bash
# One-time: publish a v0.0.0 of each of the 6 npm names so they exist
# in the registry. npm's per-package Trusted Publisher UI only appears
# on packages that already exist; once these v0.0.0 versions are up,
# the OIDC config can be added per package on npmjs.com, and the real
# v0.1.0 release goes out via CI without needing NPM_TOKEN.
#
# Run as the npm user that should own all 6 names (i.e. `teamch`).
#
# Usage:
#   npm login                   # if not already logged in as `teamch`
#   bash scripts/claim-npm-names.sh

set -euo pipefail

NAMES=(
  textsift
  textsift-linux-x64
  textsift-linux-arm64
  textsift-darwin-x64
  textsift-darwin-arm64
  textsift-win32-x64
)

WHOAMI="$(npm whoami)"
if [[ "$WHOAMI" != "teamch" ]]; then
  echo "expected npm user 'teamch', got '$WHOAMI'" >&2
  exit 1
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

for name in "${NAMES[@]}"; do
  echo
  echo "==> publish $name@0.0.0"
  pkgdir="$WORKDIR/$name"
  mkdir -p "$pkgdir"

  cat > "$pkgdir/package.json" <<EOF
{
  "name": "$name",
  "version": "0.0.0",
  "description": "See https://github.com/teamchong/textsift",
  "license": "Apache-2.0",
  "repository": { "type": "git", "url": "git+https://github.com/teamchong/textsift.git" },
  "homepage": "https://github.com/teamchong/textsift"
}
EOF

  ( cd "$pkgdir" && npm publish --access public )
done

echo
echo "All 6 names published at v0.0.0. Next steps:"
echo
echo "  1. Visit each package on npmjs.com and configure Trusted Publisher"
echo "     (type GitHub Actions, repo teamchong/textsift, workflow release.yml):"
for name in "${NAMES[@]}"; do
  echo "     https://www.npmjs.com/package/$name/access"
done
echo
echo "  2. Once all 6 have OIDC configured, tag the real release:"
echo "     git tag v0.1.0 && git push origin v0.1.0"
