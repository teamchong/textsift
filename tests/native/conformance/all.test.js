// Aggregate native conformance: every shader in the registry, run
// against the browser-dumped fixture. Adding a shader = one entry
// in the SHADERS list (and a matching entry in
// tests/conformance/shaders.html).
import { runConformance } from "./harness.js";

const SHADERS = [
  "rms_norm",
  "zero_f32",
  "cast_fp16_to_f32",
  "cast_f32_to_fp16_scaled",
  "add_fp16",
  "swiglu_clamp",
  "rope_apply",
  "matmul_int4_fp16_f16",
  "matmul_int4_f32_f32",
  "embed_lookup_int4",
  "add_rmsnorm_fp16_to_f32",
  "router_topk",
  "banded_attention",
  "qmoe_gate_up",
  "qmoe_down_scatter",
];

let failed = 0;
for (const name of SHADERS) {
  try {
    runConformance(name);
  } catch (e) {
    failed++;
    console.error(`FAIL [${name}]: ${e.message}`);
  }
}
if (failed > 0) {
  console.error(`\n${failed}/${SHADERS.length} shaders failed conformance`);
  process.exit(1);
}
console.log(`\nALL ${SHADERS.length} shaders pass conformance`);
