#!/usr/bin/env bash
# Compile the Zig native binding to a Node-loadable .node shared
# library, linking against the vendored wgpu-native release for the
# current host. Run `fetch-wgpu-native.sh` first.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PKG_ROOT="$(cd "$HERE/.." && pwd)"

HOST_ARCH="$(uname -m)"
HOST_OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
case "$HOST_OS-$HOST_ARCH" in
  darwin-arm64)  TARGET="macos-aarch64" ;;
  darwin-x86_64) TARGET="macos-x86_64" ;;
  linux-x86_64)  TARGET="linux-x86_64" ;;
  linux-aarch64) TARGET="linux-aarch64" ;;
  *) echo "unsupported host: $HOST_OS-$HOST_ARCH" >&2; exit 1 ;;
esac

NODE_INC="$(node -p "require('path').join(process.execPath, '../../include/node')")"
WGPU_DIR="${PKG_ROOT}/vendor/wgpu-native/${TARGET}"
OUT="${PKG_ROOT}/dist/textsift-native.node"

if [[ ! -d "$NODE_INC" ]]; then
  echo "node headers not found at $NODE_INC" >&2
  exit 1
fi
if [[ ! -d "$WGPU_DIR/include" ]]; then
  echo "wgpu-native not vendored at $WGPU_DIR — run fetch-wgpu-native.sh first" >&2
  exit 1
fi

mkdir -p "${PKG_ROOT}/dist"

# Platform-specific link flags. wgpu-native links against system GPU
# frameworks; on macOS that's Metal/Foundation/QuartzCore, on Linux
# vulkan-loader, on Windows d3d12/dxgi.
case "$HOST_OS" in
  darwin)
    EXTRA_LINK="-framework Metal -framework Foundation -framework QuartzCore -framework IOKit -framework CoreGraphics -framework MetalKit -framework AppKit"
    ;;
  linux)
    EXTRA_LINK=""  # wgpu-native libs already declare needs
    ;;
  *) EXTRA_LINK="" ;;
esac

mise exec -- zig build-lib \
  "${PKG_ROOT}/src/native/napi.zig" \
  -dynamic -lc \
  -I "$NODE_INC" \
  -I "$WGPU_DIR/include" \
  -L "$WGPU_DIR/lib" \
  -lwgpu_native \
  -fPIC \
  -O ReleaseSafe \
  -fallow-shlib-undefined \
  -z undefs \
  $EXTRA_LINK \
  -femit-bin="$OUT"

echo "built: $OUT ($(stat -f%z "$OUT" 2>/dev/null || stat -c%s "$OUT") bytes)"
