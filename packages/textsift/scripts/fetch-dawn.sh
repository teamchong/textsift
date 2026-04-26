#!/usr/bin/env bash
# Fetch Google Dawn (Chromium's WebGPU implementation) prebuilt
# binaries + headers for the current host. Dawn ships Tint, the WGSL
# compiler used by Chrome — measurably faster MSL/SPIR-V codegen
# than Naga (which wgpu-native uses).
#
# Output: vendor/dawn/<target>/ containing:
#   include/dawn/webgpu.h, include/dawn/webgpu_cpp.h, include/webgpu/webgpu.h, ...
#   lib/libdawn_native.{dylib,so,dll}  (release)

set -euo pipefail

VERSION="${DAWN_VERSION:-v20260423.175430}"
RELEASE_URL="https://github.com/google/dawn/releases/download/${VERSION}"

HOST_ARCH="$(uname -m)"
HOST_OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
case "${TARGET:-}" in
  "")
    case "$HOST_OS-$HOST_ARCH" in
      darwin-arm64)  TARGET="macos-latest-Release" ;;
      darwin-x86_64) TARGET="macos-15-intel-Release" ;;
      linux-x86_64)  TARGET="ubuntu-latest-Release" ;;
      *) echo "unsupported host: $HOST_OS-$HOST_ARCH" >&2; exit 1 ;;
    esac
    ;;
esac

# The asset name embeds the commit hash. Resolve it from the release
# manifest so we don't have to hard-code it.
COMMIT="$(curl -fsSL "https://api.github.com/repos/google/dawn/releases/tags/${VERSION}" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); names=[a['name'] for a in d['assets']]; \
import re; m=[re.match(r'Dawn-([0-9a-f]+)-', n) for n in names]; \
print(next(x.group(1) for x in m if x))")"
ASSET="Dawn-${COMMIT}-${TARGET}.tar.gz"

OUT_DIR="$(cd "$(dirname "$0")/.." && pwd)/vendor/dawn/${TARGET}"
if [[ -f "${OUT_DIR}/include/dawn/webgpu.h" || -f "${OUT_DIR}/include/webgpu/webgpu.h" ]]; then
  echo "Dawn ${VERSION} for ${TARGET} already vendored at ${OUT_DIR}"
  exit 0
fi

mkdir -p "${OUT_DIR}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "fetching ${RELEASE_URL}/${ASSET}"
curl -fsSL "${RELEASE_URL}/${ASSET}" -o "${TMP}/dawn.tar.gz"
tar -xzf "${TMP}/dawn.tar.gz" -C "${TMP}"

# Move whatever the archive structure is into our standard layout.
EXTRACTED_ROOT="$(find "${TMP}" -maxdepth 2 -type d -name 'Dawn-*' | head -1)"
if [[ -z "$EXTRACTED_ROOT" ]]; then
  EXTRACTED_ROOT="$(find "${TMP}" -mindepth 1 -maxdepth 1 -type d ! -path "${TMP}" | head -1)"
fi
echo "extracted to: $EXTRACTED_ROOT"

mkdir -p "${OUT_DIR}/include" "${OUT_DIR}/lib"
if [[ -d "${EXTRACTED_ROOT}/include" ]]; then
  cp -R "${EXTRACTED_ROOT}/include/." "${OUT_DIR}/include/"
fi
# Find shared/static libs in the extracted tree.
find "${EXTRACTED_ROOT}" -type f \( -name '*.dylib' -o -name '*.so' -o -name '*.so.*' -o -name '*.dll' -o -name '*.lib' -o -name '*.a' \) \
  -exec cp {} "${OUT_DIR}/lib/" \;

echo "vendored Dawn ${VERSION} for ${TARGET}:"
echo "  include:"
ls "${OUT_DIR}/include" 2>/dev/null | head -10 || true
echo "  lib:"
ls "${OUT_DIR}/lib" 2>/dev/null | head -10 || true
