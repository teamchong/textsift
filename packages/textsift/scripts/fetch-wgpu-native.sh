#!/usr/bin/env bash
# Fetch the wgpu-native prebuilt library + headers for the current
# host (or for an explicit target via the TARGET env var).
#
# Output: vendor/wgpu-native/<target>/ containing:
#   include/webgpu/webgpu.h
#   include/webgpu/wgpu.h
#   lib/libwgpu_native.{dylib,so,dll}  (release variant)
#
# Why prebuilt: wgpu is a Rust project; pulling cargo + the entire
# wgpu/naga toolchain into our build is overkill when upstream
# already publishes platform binaries on every release.

set -euo pipefail

VERSION="${WGPU_VERSION:-v29.0.0.0}"
RELEASE_URL="https://github.com/gfx-rs/wgpu-native/releases/download/${VERSION}"

# Detect target if not explicitly set.
HOST_ARCH="$(uname -m)"
HOST_OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
case "${TARGET:-}" in
  "")
    case "$HOST_OS-$HOST_ARCH" in
      darwin-arm64)  TARGET="macos-aarch64" ;;
      darwin-x86_64) TARGET="macos-x86_64" ;;
      linux-x86_64)  TARGET="linux-x86_64" ;;
      linux-aarch64) TARGET="linux-aarch64" ;;
      *) echo "unsupported host: $HOST_OS-$HOST_ARCH" >&2; exit 1 ;;
    esac
    ;;
esac

ASSET="wgpu-${TARGET}-release.zip"
OUT_DIR="$(cd "$(dirname "$0")/.." && pwd)/vendor/wgpu-native/${TARGET}"

if [[ -f "${OUT_DIR}/include/webgpu/webgpu.h" ]]; then
  echo "wgpu-native ${VERSION} for ${TARGET} already vendored at ${OUT_DIR}"
  exit 0
fi

mkdir -p "${OUT_DIR}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "fetching ${RELEASE_URL}/${ASSET}"
curl -fsSL "${RELEASE_URL}/${ASSET}" -o "${TMP}/wgpu.zip"
unzip -q "${TMP}/wgpu.zip" -d "${TMP}/extract"

mkdir -p "${OUT_DIR}/include/webgpu" "${OUT_DIR}/lib"
cp "${TMP}/extract/include/webgpu/"*.h "${OUT_DIR}/include/webgpu/"
# Copy whatever libs the archive shipped — Linux .so, macOS .dylib, Windows .dll + .lib.
find "${TMP}/extract" -type f \( -name '*.dylib' -o -name '*.so' -o -name '*.dll' -o -name '*.lib' -o -name '*.a' \) \
  -exec cp {} "${OUT_DIR}/lib/" \;

echo "vendored wgpu-native ${VERSION} for ${TARGET}:"
ls "${OUT_DIR}/include/webgpu/"
ls "${OUT_DIR}/lib/"
