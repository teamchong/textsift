# Changelog

## v0.1.0 — 2026-04-27

First public release.

### What ships

- **Browser** — `import { PrivacyFilter } from "textsift/browser"`. WebGPU when the adapter has `shader-f16`, WASM (Zig + SIMD128 + relaxed_simd, multi-threaded via SharedArrayBuffer) otherwise. WASM bytes inlined into the JS bundle — no second fetch.
- **Node native** — `import { PrivacyFilter } from "textsift"`. Per-platform `.node` binary picked at install time via npm `optionalDependencies`:
  - macOS arm64/x64 → Metal-direct (hand-written MSL via Obj-C bridge).
  - Linux x86_64/arm64 (glibc) → Vulkan-direct (hand-written GLSL → SPIR-V).
  - Windows x64 → Dawn-direct (Tint → D3D12 via Google Dawn).
  - Anything else (Alpine/musl, no GPU adapter, etc.) → WASM CPU fallback through the same API.
- **CLI** — `npx textsift redact <file>` for one-off use; `--write` for in-place.
- **Pre-commit hook** — pre-commit framework integration; gates commits with detected secrets.
- **GitHub Action** — drop-in workflow step with `::error file=…` annotations.
- **SARIF export** — `import { detectResultToSarif, toSarif } from "textsift/sarif"` for GitHub Code Scanning / GitLab SAST.

### What's in the API

- `PrivacyFilter.create()`, `redact()`, `detect()`, `redactBatch()`.
- `classifyColumns()` / `redactTable()` for tabular data audits and per-column drop / synth / redact.
- Custom `rules` (regex or match-fn) merged with model spans; built-in `secrets` preset for JWT / GitHub PATs / AWS / Stripe / OpenAI / etc.
- `markerPresets.faker()` for cross-row consistent fake values.
- `Symbol.dispose` / `using` for automatic cleanup.

### Measured performance

Hardware named on every number; CI-runner numbers are not republished.

| Path | T=32 latency | Hardware |
|---|---:|---|
| Browser WebGPU | 11.8 ms | M3 Pro, Chromium |
| Mac Metal-direct | 10.8 ms | M2 Pro |
| Linux Vulkan-direct | ~25 ms (32× over ORT-CPU) | Intel Iris Xe, Mesa |

End-to-end `redact()` on a 122-char input with 4 PII spans: ~50–110 ms on real hardware.

### Pre-1.0 caveats

- Surface is not stable. Breaking changes can land before 1.0.
- Native-binary CI matrix covers `linux-x64`, `linux-arm64`, `darwin-x64`, `darwin-arm64`, `windows-x64`. Other targets (Alpine/musl, Windows ARM, FreeBSD) fall through to the WASM CPU path.
- llvmpipe / software-fallback Vulkan adapters are tolerated (`rope_apply` allows 32 fp16 ULPs of cross-driver drift), but not benchmarked as if they were real GPUs.
