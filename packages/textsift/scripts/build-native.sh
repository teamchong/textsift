#!/usr/bin/env bash
# Compile the Zig native binding to a Node-loadable .node shared library.
#
# Platform routing — comptime-selected via napi.zig's `is_macos / is_linux /
# is_windows` gates AND mirrored here in the build script's case statements:
#
#   macOS    → Metal-direct (hand-written MSL via Obj-C bridge)
#   Linux    → Vulkan-direct (hand-written GLSL → SPIR-V via glslangValidator)
#   Windows  → Dawn-direct (Tint → D3D12 via Dawn's backend selection;
#                no Vulkan-direct because glslang isn't standard on Win and
#                no Metal/Obj-C)
#
# Dawn was previously also compiled into the Linux .node, but Dawn on Linux
# uses Vulkan internally — if the loader is missing, Dawn fails the same
# way Vulkan-direct does. The fallback was redundant; the real Linux
# fallback when no GPU is present is the WASM CPU path.
#
# Each case below contributes its own EXTRA_LINK and EXTRA_SRC; if a case
# is empty (e.g. an unsupported host) we fail loud rather than silently
# build something broken.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PKG_ROOT="$(cd "$HERE/.." && pwd)"

HOST_ARCH="$(uname -m)"
HOST_OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
# Treat MINGW/MSYS/Cygwin uname as "windows" for routing purposes.
case "$HOST_OS" in
  mingw*|msys*|cygwin*) HOST_OS="windows" ;;
esac
case "$HOST_OS-$HOST_ARCH" in
  darwin-arm64|darwin-x86_64|linux-x86_64|linux-aarch64|windows-x86_64) ;;
  *) echo "unsupported host: $HOST_OS-$HOST_ARCH" >&2; exit 1 ;;
esac

NODE_INC="$(node -p "require('path').join(process.execPath, '../../include/node')")"
OUT="${PKG_ROOT}/dist/textsift-native.node"

if [[ ! -d "$NODE_INC" ]]; then
  echo "node headers not found at $NODE_INC" >&2
  exit 1
fi

mkdir -p "${PKG_ROOT}/dist"

# Platform-specific link flags.
case "$HOST_OS" in
  darwin)
    # Metal frameworks for the Obj-C bridge.
    EXTRA_LINK="-framework Metal -framework Foundation -framework QuartzCore -framework IOKit -framework CoreGraphics -framework MetalKit -framework AppKit"
    ;;
  linux)
    # libvulkan for Vulkan-direct. Dawn is no longer linked on Linux —
    # it would have used Vulkan internally anyway, so the fallback was
    # redundant.
    EXTRA_LINK="-lvulkan -lpthread -ldl -lm"
    ;;
  windows)
    # Dawn brings its own D3D12 backend; we just link the static lib +
    # Windows system libs Dawn references (d3d12, dxgi, d3dcompiler).
    # See vendor/dawn/lib/libwebgpu_dawn.a (built per-platform on the
    # respective CI runner — same hidden-visibility CMake flags as Linux).
    EXTRA_LINK="-L${PKG_ROOT}/vendor/dawn/lib -lwebgpu_dawn -ld3d12 -ldxgi -ld3dcompiler -lstdc++"
    ;;
  *) echo "no link config for $HOST_OS" >&2; exit 1 ;;
esac

# Per-platform extra source files compiled with napi.zig.
EXTRA_SRC=()
case "$HOST_OS" in
  darwin)
    EXTRA_SRC+=( -cflags -fobjc-arc -- "${PKG_ROOT}/src/native/metal/bridge.m" )
    ;;
  linux)
    # Pre-compile every GLSL kernel to SPIR-V so Zig @embedFile picks them
    # up at compile time.
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
  windows)
    # Windows ships only the Dawn-direct bridge — no Vulkan-direct (no
    # glslangValidator step), no Metal-direct (no Obj-C). Dawn handles
    # WGSL → HLSL → D3D12 internally via Tint at runtime.
    EXTRA_SRC+=( "${PKG_ROOT}/src/native/dawn/bridge.c" )
    ;;
esac

mise exec -- zig build-lib \
  "${PKG_ROOT}/src/native/napi.zig" \
  "${EXTRA_SRC[@]}" \
  -dynamic -lc \
  -I "$NODE_INC" \
  -I "${PKG_ROOT}/src/native" \
  -I "${PKG_ROOT}/vendor/dawn/include" \
  -fPIC \
  -O ReleaseSafe \
  -fallow-shlib-undefined \
  -z undefs \
  $EXTRA_LINK \
  -femit-bin="$OUT"

echo "built: $OUT ($(stat -f%z "$OUT" 2>/dev/null || stat -c%s "$OUT") bytes)"
