// Wrap wasm's `f32x4.relaxed_madd` so Zig can call it. Zig's
// `@mulAdd(@Vector(4, f32), ...)` falls back to per-lane software
// FMA on wasm, defeating the whole point. Clang's builtin emits the
// relaxed-SIMD instruction directly when the target has
// +relaxed_simd. Built with -flto so the Zig/C link-time optimizer
// inlines this one-instruction body into every matmul MAC.

#include <wasm_simd128.h>

__attribute__((always_inline)) v128_t
relaxed_madd_f32x4(v128_t a, v128_t b, v128_t c) {
    return __builtin_wasm_relaxed_madd_f32x4(a, b, c);
}
