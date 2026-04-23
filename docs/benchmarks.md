# Benchmarks

Stage 0 baseline numbers. Replace when Stage 1 (Zig+WASM) lands and whenever
transformers.js or ort-web bump versions.

## Stage 0 — transformers.js + WebGPU + model_q4f16

Default configuration for `PrivacyFilter.create()`:

- Model: `openai/privacy-filter` via `AutoModelForTokenClassification`.
- ONNX export: `onnx/model_q4f16.onnx` (~772 MB, block-int4 weights + fp16
  activations; smallest exported variant that fits in a browser).
- Execution provider: WebGPU (ORT Web). `backend: "wasm"` forces the WASM CPU
  path instead but is not measured here.
- Viterbi decoder + BIOES → spans + whitespace trim run on JS (no WASM).

### Chrome 147 / Apple Silicon M-series

Input ≈ 3 340 chars, ≈ 700–800 BPE tokens.

| Run | Latency |
|-----|---------|
| 1 (short input, 4 spans — warmup includes pipeline JIT) | 220 ms |
| 2 (first ~700-token run, still warm-ish) | 1 134 ms |
| 3 | 363 ms |
| 4 | 322 ms |
| 5 | 312 ms |

Median of runs 3–5 (steady-state warm): **322 ms**.

Peak JS heap at end of session: **≈ 872 MB** (`performance.memory.usedJSHeapSize`,
Chrome-only).

### Firefox

Not measured yet. The q4f16 path requires WebGPU + `MatMulNBits` and
`GatherBlockQuantized` contrib ops in the asyncify WASM bundle. Firefox 127+
ships a WebGPU implementation good enough to try; whether ORT Web's contrib
ops dispatch on that backend is the open question. Benchmark when the demo
gets tested there.

## Notes

- "Cold" (first browser visit, full model download) is dominated by the
  772 MB ONNX download, not compute. Cache-API hit reduces the second page
  load to the compile + warmup cost (~1–2 s on M-series).
- `int4` and `int8` both map to `model_q4f16.onnx` today. See
  `src/js/backends/transformers-js.ts::dtypeFor` for the rationale.
- `fp16` maps to `model_fp16.onnx` (~2 GB). Only use when memory permits and
  you want maximum accuracy.
