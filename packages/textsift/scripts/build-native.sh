#!/usr/bin/env bash
# Compile the Zig native binding to a Node-loadable .node shared library.
#
# Per-platform fast path — comptime-selected at compile time:
#   macOS  → Metal-direct (hand-written MSL via Obj-C bridge)
#   Linux  → Vulkan-direct (hand-written GLSL → SPIR-V via glslangValidator)

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PKG_ROOT="$(cd "$HERE/.." && pwd)"

HOST_ARCH="$(uname -m)"
HOST_OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
case "$HOST_OS-$HOST_ARCH" in
  darwin-arm64|darwin-x86_64|linux-x86_64|linux-aarch64) ;;
  *) echo "unsupported host: $HOST_OS-$HOST_ARCH" >&2; exit 1 ;;
esac

NODE_INC="$(node -p "require('path').join(process.execPath, '../../include/node')")"
OUT="${PKG_ROOT}/dist/textsift-native.node"

if [[ ! -d "$NODE_INC" ]]; then
  echo "node headers not found at $NODE_INC" >&2
  exit 1
fi

mkdir -p "${PKG_ROOT}/dist"

# Platform-specific link flags. macOS pulls in the Metal frameworks;
# Linux links libvulkan (provided by libvulkan-dev / libvulkan.so.1).
case "$HOST_OS" in
  darwin)
    EXTRA_LINK="-framework Metal -framework Foundation -framework QuartzCore -framework IOKit -framework CoreGraphics -framework MetalKit -framework AppKit"
    ;;
  linux)
    EXTRA_LINK="-lvulkan"
    ;;
  *) EXTRA_LINK="" ;;
esac

# On macOS we compile the Metal Obj-C bridge alongside napi.zig.
# On Linux we compile the Vulkan-direct C bridge AND pre-compile every
# GLSL kernel to SPIR-V so Zig @embedFile picks them up.
EXTRA_SRC=()
case "$HOST_OS" in
  darwin)
    EXTRA_SRC+=( -cflags -fobjc-arc -- "${PKG_ROOT}/src/native/metal/bridge.m" )
    ;;
  linux)
    GLSL_DIR="${PKG_ROOT}/src/native/vulkan/shaders"
    if ! command -v glslangValidator >/dev/null 2>&1; then
      echo "glslangValidator not found — install glslang-tools (apt) or shaderc" >&2
      exit 1
    fi
    shopt -s nullglob
    for glsl in "${GLSL_DIR}"/*.comp.glsl; do
      spv="${glsl%.glsl}.spv"
      if [[ -z "${TS_VK_QUIET:-}" ]]; then echo "glslangValidator -V ${glsl##*/}"; fi
      glslangValidator -V --target-env vulkan1.2 "$glsl" -o "$spv" >/dev/null
    done
    shopt -u nullglob
    EXTRA_SRC+=( "${PKG_ROOT}/src/native/vulkan/bridge.c" )
    ;;
esac

mise exec -- zig build-lib \
  "${PKG_ROOT}/src/native/napi.zig" \
  "${EXTRA_SRC[@]}" \
  -dynamic -lc \
  -I "$NODE_INC" \
  -I "${PKG_ROOT}/src/native" \
  -fPIC \
  -O ReleaseSafe \
  -fallow-shlib-undefined \
  -z undefs \
  $EXTRA_LINK \
  -femit-bin="$OUT"

echo "built: $OUT ($(stat -f%z "$OUT" 2>/dev/null || stat -c%s "$OUT") bytes)"
