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

OUT="${PKG_ROOT}/dist/textsift-native.node"

# Locate Node headers. macOS/Linux ship them next to node.exe; Windows
# distributions don't, so we download the official headers tarball
# under the package's vendor/ dir and reuse on subsequent runs.
NODE_INC="$(node -p "require('path').join(process.execPath, '../../include/node')")"
NODE_LIB_DIR=""   # Windows-only: holds node.lib for napi import resolution.
if [[ ! -d "$NODE_INC" ]]; then
  if [[ "$HOST_OS" == "windows" ]]; then
    NODE_VERSION="$(node -p 'process.versions.node')"
    VENDOR_HEADERS="${PKG_ROOT}/vendor/node-headers/${NODE_VERSION}"
    NODE_INC="${VENDOR_HEADERS}/include/node"
    NODE_LIB_DIR="${VENDOR_HEADERS}/lib"
    if [[ ! -d "$NODE_INC" ]]; then
      echo "fetching Node ${NODE_VERSION} headers (Windows distros omit them)..." >&2
      mkdir -p "$VENDOR_HEADERS" "$NODE_LIB_DIR"
      TARBALL="https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-headers.tar.gz"
      curl -fsSL "$TARBALL" | tar -xz -C "$VENDOR_HEADERS" --strip-components=1
      # node.lib is the Windows import library that exposes node.exe's
      # napi_* exports. Without it, lld-link leaves every napi_* symbol
      # undefined and the resulting .node segfaults at LoadLibrary.
      echo "fetching Node ${NODE_VERSION} import lib (node.lib)..." >&2
      curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/win-x64/node.lib" \
        -o "${NODE_LIB_DIR}/node.lib"
    fi
  fi
fi
if [[ ! -d "$NODE_INC" ]]; then
  echo "node headers not found at $NODE_INC" >&2
  exit 1
fi

mkdir -p "${PKG_ROOT}/dist"

# Platform-specific link flags. Use an array so paths with spaces
# (Windows SDK paths under "Program Files (x86)") survive expansion.
EXTRA_LINK_ARGS=()
case "$HOST_OS" in
  darwin)
    # Metal frameworks for the Obj-C bridge.
    EXTRA_LINK_ARGS=(
      -framework Metal -framework Foundation -framework QuartzCore
      -framework IOKit -framework CoreGraphics -framework MetalKit
      -framework AppKit
    )
    ;;
  linux)
    # libvulkan for Vulkan-direct. Dawn is no longer linked on Linux —
    # it would have used Vulkan internally anyway, so the fallback was
    # redundant.
    EXTRA_LINK_ARGS=( -lvulkan -lpthread -ldl -lm )
    ;;
  windows)
    # Dawn brings its own D3D12 backend; we just link the static lib +
    # Windows system libs Dawn references (d3d12, dxgi, d3dcompiler).
    # See vendor/dawn/lib/webgpu_dawn.lib (built on the Windows CI
    # runner — same hidden-visibility CMake flags as Linux).
    # No -lstdc++: that's a MinGW-ism. With -target windows-msvc Zig
    # links MSVC's C++ runtime (msvcprt) automatically.
    # Explicit -lucrt -lvcruntime: Zig's `-lc` on windows-msvc pulls
    # msvcrt.lib (the static-startup wrapper) but doesn't always pull
    # ucrt.lib (the UCRT imports). Without it, msvcrt's vcstartup
    # utility.obj references `__acrt_thread_attach` etc. as undefined
    # — lld-link warns at link time, but the resulting .node segfaults
    # at LoadLibrary because the unresolved imports fire from the DLL
    # entry on every thread attach.
    EXTRA_LINK_ARGS=(
      "-L${PKG_ROOT}/vendor/dawn/lib"
      "-L${NODE_LIB_DIR}"
      -lwebgpu_dawn -ld3d12 -ldxgi -ld3dcompiler
      -lucrt -lvcruntime
      -lnode
    )
    # Zig's link step on Windows (msvc ABI) doesn't pick up the LIB env
    # var that vcvars64.bat / ilammy/msvc-dev-cmd populates. Translate
    # each entry into an explicit `-L` so Zig finds the Windows SDK
    # system libraries (d3dcompiler.lib, d3d12.lib, dxgi.lib live under
    # `Windows Kits\10\Lib\<sdk-ver>\um\x64`).
    if [[ -n "${LIB:-}" ]]; then
      IFS=';' read -ra LIB_DIRS <<< "$LIB"
      for d in "${LIB_DIRS[@]}"; do
        [[ -n "$d" ]] && EXTRA_LINK_ARGS+=( "-L$d" )
      done
    fi
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

# On Windows, target MSVC ABI explicitly. Zig's default
# `native-native` resolves to `windows-gnu` (MinGW) — but Dawn's
# webgpu_dawn.lib is built with MSVC ABI (clang on Windows defaults
# to MSVC). Mixing them produces duplicate-symbol link errors on
# Control Flow Guard intrinsics (libmingw32 vs msvcrt). Forcing
# `windows-msvc` makes Zig link MSVC's C runtime exclusively.
ZIG_TARGET_ARGS=()
if [[ "$HOST_OS" == "windows" ]]; then
  ZIG_TARGET_ARGS=( -target "${HOST_ARCH}-windows-msvc" )
fi

mise exec -- zig build-lib \
  "${PKG_ROOT}/src/native/napi.zig" \
  "${EXTRA_SRC[@]}" \
  ${ZIG_TARGET_ARGS[@]+"${ZIG_TARGET_ARGS[@]}"} \
  -dynamic -lc \
  -I "$NODE_INC" \
  -I "${PKG_ROOT}/src/native" \
  -I "${PKG_ROOT}/vendor/dawn/include" \
  -fPIC \
  -O ReleaseSafe \
  -fallow-shlib-undefined \
  -z undefs \
  "${EXTRA_LINK_ARGS[@]}" \
  -femit-bin="$OUT"

echo "built: $OUT ($(stat -f%z "$OUT" 2>/dev/null || stat -c%s "$OUT") bytes)"
