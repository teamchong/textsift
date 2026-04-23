#!/usr/bin/env python3
"""
Regenerate kernel parity fixtures + matching weight blob under tests/fixtures/.

Fixtures are committed. This script exists so anyone can reproduce the
bytes from first principles against a known PyTorch version + seed. If
a PyTorch upgrade shifts bf16 rounding or RNG state, the regenerated
fixtures won't match the committed ones; update the committed files and
the `baseline_matches` in `tests/fixtures/manifest.json` together.

Run from repo root:

    python3 scripts/gen-kernel-fixtures.py

Expected PyTorch: 2.4.x or newer (needs torch.nn.functional.rms_norm).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
SEED = 42

T = 8
D = 640
N_CLASSES = 33  # matches score.weight / score.bias
V_TEST = 1024   # truncated embed table — full vocab is 200 064, too big for tests
INT4_BLOCK = 32
EXPERTS_KEEP = 4  # MoE weights truncated to first N experts for test fixtures


# Share the quantizer with convert_weights.py so blob + fixture byte
# patterns stay identical under any tweak.
def _load_convert_weights():
    path = REPO_ROOT / "scripts" / "convert_weights.py"
    spec = importlib.util.spec_from_file_location("convert_weights", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bf16_bytes(tensor: torch.Tensor) -> bytes:
    assert tensor.dtype == torch.bfloat16
    return tensor.detach().contiguous().cpu().view(torch.uint16).numpy().tobytes()


def write(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    print(f"  wrote {path.relative_to(REPO_ROOT)}  ({len(data)} bytes)", file=sys.stderr)


def dequantize_int4block(packed: bytes, n: int, d: int) -> torch.Tensor:
    """
    Decode the bytes produced by `convert_weights.quantize_int4_block32_sym`
    back to an fp32 [N, D] tensor. Reference path for computing expected
    outputs — exactly what the WASM kernel computes on-the-fly.
    """
    import numpy as np
    block = INT4_BLOCK
    n_blocks = d // block
    int4_bytes = n * d // 2
    # Split packed bytes.
    int4_flat = np.frombuffer(packed[:int4_bytes], dtype=np.uint8).reshape(n, d // 2)
    scale_u16 = np.frombuffer(packed[int4_bytes:], dtype=np.uint16).reshape(n, n_blocks)
    scales = torch.from_numpy(scale_u16.view(np.uint16).copy()).view(torch.float16).to(torch.float32)

    # Unpack nibbles: low (even d) and high (odd d). Sign-extend 4 bits.
    lo = int4_flat & 0x0F
    hi = (int4_flat >> 4) & 0x0F
    # Sign-extend: treat as int8 via two's complement of a nibble.
    lo = np.where(lo & 0x08, lo.astype(np.int16) - 16, lo.astype(np.int16)).astype(np.int8)
    hi = np.where(hi & 0x08, hi.astype(np.int16) - 16, hi.astype(np.int16)).astype(np.int8)
    q = np.empty((n, d), dtype=np.int8)
    q[:, 0::2] = lo
    q[:, 1::2] = hi
    q_fp = torch.from_numpy(q).to(torch.float32)     # [N, D]
    q_blocks = q_fp.view(n, n_blocks, block)
    dequant = q_blocks * scales.unsqueeze(-1)
    return dequant.view(n, d)


def main() -> int:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    print("[gen] loading openai/privacy-filter …", file=sys.stderr)
    from transformers import AutoModelForTokenClassification
    model = AutoModelForTokenClassification.from_pretrained("openai/privacy-filter")
    sd = model.state_dict()

    cw = _load_convert_weights()

    # Load the RoPE layer once — reused by several fixtures below.
    from transformers.models.openai_privacy_filter.modeling_openai_privacy_filter import (
        OpenAIPrivacyFilterRotaryEmbedding,
        _apply_rotary_emb,
    )
    rope_layer = OpenAIPrivacyFilterRotaryEmbedding(model.config)

    # ----- 1. Weight blob -----
    #
    # Starts from the Phase-B subset (classifier head + 2 RMSNorms) +
    # embed table truncated to V_TEST rows. Extended with layer 0's
    # attention weights so the attention-forward composition test can
    # run against real, not random, parameters.
    print("[gen] building weight blob via scripts/convert_weights.py …", file=sys.stderr)
    convert = REPO_ROOT / "scripts" / "convert_weights.py"
    output_blob = FIXTURES_DIR / "pii-weights.bin"
    attn_tensors = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.k_proj.bias",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.v_proj.bias",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.o_proj.bias",
        "model.layers.0.self_attn.sinks",
        "model.layers.0.post_attention_layernorm.weight",
    ]
    moe_tensors = [
        "model.layers.0.mlp.router.weight",
        "model.layers.0.mlp.router.bias",
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.gate_up_proj_bias",
        "model.layers.0.mlp.experts.down_proj",
        "model.layers.0.mlp.experts.down_proj_bias",
    ]
    cmd = [
        sys.executable, str(convert),
        "--output", str(output_blob),
        "--embed-rows", str(V_TEST),
        "--expert-keep", str(EXPERTS_KEEP),
        # Expert weights: transpose last two dims before quantizing so the
        # stored shape matches our x @ W.T matmul convention.
        "--quant-transpose", "mlp.experts.gate_up_proj",
        "--quant-transpose", "mlp.experts.down_proj",
    ]
    for name in attn_tensors + moe_tensors:
        cmd += ["--include", name]
    subprocess.check_call(cmd)

    # ----- 2. RMSNorm fixture -----
    torch.manual_seed(SEED)
    x_rms = torch.randn(T, D, dtype=torch.bfloat16)
    gamma = sd["model.norm.weight"].to(torch.bfloat16)
    y_rms = F.rms_norm(x_rms, [D], weight=gamma, eps=1e-5)
    write(FIXTURES_DIR / "rms_x.bf16", bf16_bytes(x_rms))
    write(FIXTURES_DIR / "rms_y.bf16", bf16_bytes(y_rms.to(torch.bfloat16)))

    # ----- 3. Matmul fixture (classifier head, bf16 path) -----
    torch.manual_seed(SEED + 1)
    x_mm = torch.randn(T, D, dtype=torch.bfloat16)
    w_bf = sd["score.weight"].to(torch.bfloat16)    # [N, D]
    b_bf = sd["score.bias"].to(torch.bfloat16)      # [N]
    y_mm = F.linear(x_mm, w_bf, b_bf)
    write(FIXTURES_DIR / "matmul_x.bf16", bf16_bytes(x_mm))
    write(FIXTURES_DIR / "matmul_y.bf16", bf16_bytes(y_mm.to(torch.bfloat16)))

    # ----- 4. Embed fixture -----
    g = torch.Generator().manual_seed(SEED + 2)
    ids = torch.randint(0, V_TEST, (T,), generator=g, dtype=torch.int32)
    embed = sd["model.embed_tokens.weight"][:V_TEST].to(torch.bfloat16)
    y_embed = embed[ids.long()]
    write(FIXTURES_DIR / "embed_ids.i32", ids.numpy().tobytes())
    write(FIXTURES_DIR / "embed_out.bf16", bf16_bytes(y_embed))

    # ----- matmul_bf16_out_f32 fixture (router projection path) -----
    # Same x / W / bias as the bf16 test, different expected dtype.
    # Upstream router runs in fp32: this kernel skips the bf16 round.
    x_router = x_mm.clone()
    w_router = w_bf.clone()
    b_router = b_bf.clone()
    y_router = F.linear(x_router.to(torch.float32), w_router.to(torch.float32), b_router.to(torch.float32))
    write(FIXTURES_DIR / "matmul_out_f32_x.bf16", bf16_bytes(x_router))
    write(FIXTURES_DIR / "matmul_out_f32_y.f32", y_router.contiguous().numpy().tobytes())

    # ----- topk_partial_f32 fixture (router top-4) -----
    # 128 scores per row, top-4 extraction. Spread the values so ties
    # are unlikely and we actually exercise the selection logic.
    torch.manual_seed(SEED + 8)
    topk_rows = 32
    topk_cols = 128
    topk_x = torch.randn(topk_rows, topk_cols, dtype=torch.float32) * 2.0
    topk_vals, topk_idx = torch.topk(topk_x, k=4, dim=-1)
    # Confirm descending — our kernel emits "largest first" too.
    assert (topk_vals[..., 0] >= topk_vals[..., -1]).all()
    write(FIXTURES_DIR / "topk_x.f32", topk_x.contiguous().numpy().tobytes())
    write(FIXTURES_DIR / "topk_idx.i32", topk_idx.to(torch.int32).contiguous().numpy().tobytes())
    write(FIXTURES_DIR / "topk_val.f32", topk_vals.contiguous().numpy().tobytes())

    # ----- SwiGLU-with-clamp fixture -----
    # Matches OpenAIPrivacyFilterExperts._apply_gate. Inputs spread in a
    # range that exercises both clamp branches (some values > 7 and < -7).
    torch.manual_seed(SEED + 7)
    swiglu_T = 16
    swiglu_D = 640                                # = intermediate_size
    gate_up = torch.randn(swiglu_T, 2 * swiglu_D, dtype=torch.float32) * 6.0
    gate_ref, up_ref = gate_up.chunk(2, dim=-1)
    gate_ref = gate_ref.clamp(min=None, max=7.0)
    up_ref = up_ref.clamp(min=-7.0, max=7.0)
    glu = gate_ref * torch.sigmoid(gate_ref * 1.702)
    swiglu_ref = (up_ref + 1.0) * glu
    write(FIXTURES_DIR / "swiglu_x.f32", gate_up.numpy().tobytes())
    write(FIXTURES_DIR / "swiglu_y.f32", swiglu_ref.contiguous().numpy().tobytes())

    # ----- Banded attention fixture -----
    # Reference via upstream `eager_attention_forward` on sliding-window mask.
    # Shapes picked so boundary queries (near t=0 and t=T-1) have truncated
    # windows and mid-sequence queries have full windows — exercises all paths.
    from transformers.masking_utils import create_bidirectional_sliding_window_mask

    torch.manual_seed(SEED + 9)
    att_T = 32
    att_Hq = model.config.num_attention_heads  # 14
    att_Hkv = model.config.num_key_value_heads # 2
    att_hd = model.config.head_dim             # 64
    att_win = 4
    att_sliding = att_win + 1  # upstream quirk: sliding_window = config + 1

    # Q/K already scaled by head_dim ** -0.25 per upstream convention.
    scale = att_hd ** -0.25
    Q_in = torch.randn(att_T, att_Hq, att_hd, dtype=torch.bfloat16)
    K_in = torch.randn(att_T, att_Hkv, att_hd, dtype=torch.bfloat16)
    V_in = torch.randn(att_T, att_Hkv, att_hd, dtype=torch.bfloat16)
    sinks = torch.randn(att_Hq, dtype=torch.float32) * 0.5
    Q_scaled = (Q_in.float() * scale).to(torch.bfloat16)
    K_scaled = (K_in.float() * scale).to(torch.bfloat16)

    # Reference: manually replicate eager_attention_forward semantics.
    # Inputs reshaped to (1, H, T, head_dim).
    Q_b = Q_scaled.unsqueeze(0).transpose(1, 2).contiguous().float()   # (1, Hq, T, hd)
    K_b = K_scaled.unsqueeze(0).transpose(1, 2).contiguous().float()   # (1, Hkv, T, hd)
    V_b = V_in.unsqueeze(0).transpose(1, 2).contiguous().float()       # (1, Hkv, T, hd)
    n_kv_groups = att_Hq // att_Hkv
    K_rep = K_b.repeat_interleave(n_kv_groups, dim=1)                  # (1, Hq, T, hd)
    V_rep = V_b.repeat_interleave(n_kv_groups, dim=1)
    attn_weights = torch.matmul(Q_b, K_rep.transpose(-2, -1))           # (1, Hq, T, T), scaling already applied
    # Sliding window mask (additive, 0 inside window, -inf outside).
    dtype_big = torch.finfo(torch.float32).min
    mask = torch.zeros(att_T, att_T)
    for i in range(att_T):
        for j in range(att_T):
            if abs(i - j) > att_win:
                mask[i, j] = dtype_big
    attn_weights = attn_weights + mask
    sinks_col = sinks.float().view(1, -1, 1, 1).expand(1, att_Hq, att_T, 1)
    combined = torch.cat([attn_weights, sinks_col], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = torch.nn.functional.softmax(combined, dim=-1, dtype=torch.float32)
    scores_ref = probs[..., :-1]                                        # drop sink
    out_ref = torch.matmul(scores_ref, V_rep)                           # (1, Hq, T, hd)
    out_ref = out_ref.squeeze(0).transpose(0, 1).contiguous().to(torch.bfloat16)  # (T, Hq, hd)

    # Reshape Q/K/V to [T, H*hd] layout expected by the kernel.
    write(FIXTURES_DIR / "attn_q.bf16", bf16_bytes(Q_scaled.view(att_T, att_Hq * att_hd)))
    write(FIXTURES_DIR / "attn_k.bf16", bf16_bytes(K_scaled.view(att_T, att_Hkv * att_hd)))
    write(FIXTURES_DIR / "attn_v.bf16", bf16_bytes(V_in.view(att_T, att_Hkv * att_hd)))
    write(FIXTURES_DIR / "attn_sinks.f32", sinks.numpy().tobytes())
    write(FIXTURES_DIR / "attn_out.bf16", bf16_bytes(out_ref.view(att_T, att_Hq * att_hd)))

    # Masked variant: second half of tokens marked padding. We expect the
    # first half's outputs to match a `T=att_T/2` unmasked attention on
    # the same Q/K/V prefix — padding keys should contribute nothing.
    pad_T = att_T // 2
    mask_vec = torch.cat([torch.ones(pad_T, dtype=torch.uint8), torch.zeros(att_T - pad_T, dtype=torch.uint8)])
    # Reference: recompute attention with a mask that -∞s all columns j ≥ pad_T.
    mask_pad = torch.zeros(att_T, att_T)
    for i in range(att_T):
        for j in range(att_T):
            if abs(i - j) > att_win or j >= pad_T:
                mask_pad[i, j] = dtype_big
    attn_weights2 = torch.matmul(Q_b, K_rep.transpose(-2, -1)) + mask_pad
    combined2 = torch.cat([attn_weights2, sinks_col], dim=-1)
    combined2 = combined2 - combined2.max(dim=-1, keepdim=True).values
    probs2 = torch.nn.functional.softmax(combined2, dim=-1, dtype=torch.float32)
    scores2 = probs2[..., :-1]
    out_ref2 = torch.matmul(scores2, V_rep).squeeze(0).transpose(0, 1).contiguous().to(torch.bfloat16)
    write(FIXTURES_DIR / "attn_mask.u8", mask_vec.numpy().tobytes())
    write(FIXTURES_DIR / "attn_out_masked.bf16", bf16_bytes(out_ref2.view(att_T, att_Hq * att_hd)))

    # ----- Softmax fixture (attention-shape) -----
    #
    # Shape picked to exercise both the "long" row (attention scores +
    # sink = window*2 + 1 cells) and a small even one (router top-k).
    torch.manual_seed(SEED + 5)
    x_sm_att = torch.randn(8, 257, dtype=torch.float32) * 3.0  # spread logits
    y_sm_att = F.softmax(x_sm_att, dim=-1)
    write(FIXTURES_DIR / "softmax_att_x.f32", x_sm_att.numpy().tobytes())
    write(FIXTURES_DIR / "softmax_att_y.f32", y_sm_att.numpy().tobytes())

    torch.manual_seed(SEED + 6)
    x_sm_top = torch.randn(16, 4, dtype=torch.float32)
    y_sm_top = F.softmax(x_sm_top, dim=-1)
    write(FIXTURES_DIR / "softmax_topk_x.f32", x_sm_top.numpy().tobytes())
    write(FIXTURES_DIR / "softmax_topk_y.f32", y_sm_top.numpy().tobytes())

    # ----- Expert dispatch fixture (MoE, layer 0, first 4 experts) -----
    #
    # Synthetic routing: every token routes to experts [0,1,2,3] with
    # uniform scores 1/K=0.25. Exercises every in-blob expert on every
    # token and keeps the score math predictable for debugging. The
    # reference computes expert forward from our BLOB's dequantized
    # weights (not the model's original fp32), so the kernel test is
    # bounded by kernel rounding, not quant error.
    torch.manual_seed(SEED + 11)
    exp_T = 16
    exp_D = model.config.hidden_size        # 640
    exp_dff = model.config.intermediate_size  # 640
    K = model.config.num_experts_per_tok     # 4

    hidden_exp = torch.randn(exp_T, exp_D, dtype=torch.bfloat16)

    routing_idx = torch.zeros(exp_T, K, dtype=torch.int32)
    for t in range(exp_T):
        for k in range(K):
            routing_idx[t, k] = k % EXPERTS_KEEP  # cycle 0..3

    routing_scores = torch.full((exp_T, K), 1.0 / K, dtype=torch.float32)

    # Quantize + dequantize the first-4 experts ourselves so the reference
    # uses exactly the weights that end up in the blob.
    gu_full = sd["model.layers.0.mlp.experts.gate_up_proj"][:EXPERTS_KEEP]  # [4, D, 2*dff]
    gu_bias_full = sd["model.layers.0.mlp.experts.gate_up_proj_bias"][:EXPERTS_KEEP]  # [4, 2*dff]
    d_full = sd["model.layers.0.mlp.experts.down_proj"][:EXPERTS_KEEP]      # [4, dff, D]
    d_bias_full = sd["model.layers.0.mlp.experts.down_proj_bias"][:EXPERTS_KEEP]  # [4, D]

    # Quantize with transpose_last so stored shape is [E, 2*dff, D] / [E, D, dff].
    gu_packed, gu_shape = cw.quantize_int4_block32_sym(gu_full, transpose_last=True)
    d_packed, d_shape = cw.quantize_int4_block32_sym(d_full, transpose_last=True)
    assert gu_shape == (EXPERTS_KEEP, 2 * exp_dff, exp_D), gu_shape
    assert d_shape == (EXPERTS_KEEP, exp_D, exp_dff), d_shape

    # Dequantize each expert slice back to fp32.
    def dequant_expert_3d(packed: bytes, E: int, N: int, D: int) -> torch.Tensor:
        out = torch.zeros(E, N, D, dtype=torch.float32)
        int4_per_slice = N * D // 2
        scale_per_slice_bytes = N * D // 32 * 2
        total_int4 = E * int4_per_slice
        for e in range(E):
            s_int4 = e * int4_per_slice
            s_scale = total_int4 + e * scale_per_slice_bytes
            slice_bytes = packed[s_int4 : s_int4 + int4_per_slice] + \
                          packed[s_scale : s_scale + scale_per_slice_bytes]
            out[e] = dequantize_int4block(slice_bytes, N, D)
        return out

    gu_dequant = dequant_expert_3d(gu_packed, EXPERTS_KEEP, 2 * exp_dff, exp_D)   # [E, 2*dff, D]
    d_dequant = dequant_expert_3d(d_packed, EXPERTS_KEEP, exp_D, exp_dff)          # [E, D, dff]

    # Reference expert forward. All f32 intermediates, matching upstream.
    hidden_fp32 = hidden_exp.to(torch.float32)
    gu_bias_fp32 = gu_bias_full.to(torch.float32)
    d_bias_fp32 = d_bias_full.to(torch.float32)
    acc_ref = torch.zeros(exp_T, exp_D, dtype=torch.float32)
    for e in range(EXPERTS_KEEP):
        tok_idx = []
        scores_e = []
        for t in range(exp_T):
            for k in range(K):
                if routing_idx[t, k].item() == e:
                    tok_idx.append(t)
                    scores_e.append(routing_scores[t, k].item())
        if not tok_idx:
            continue
        x_e = hidden_fp32[tok_idx]                              # [m, D]
        # stored gu_dequant[e]: [2*dff, D], compute x @ W.T + b.
        gate_up = x_e @ gu_dequant[e].T + gu_bias_fp32[e]        # [m, 2*dff]
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=7.0)
        up = up.clamp(min=-7.0, max=7.0)
        glu = gate * torch.sigmoid(gate * 1.702)
        gated = (up + 1.0) * glu                                 # [m, dff]
        # stored d_dequant[e]: [D, dff], compute gated @ W.T + b.
        out = gated @ d_dequant[e].T + d_bias_fp32[e]            # [m, D]
        for i, t in enumerate(tok_idx):
            acc_ref[t] += scores_e[i] * out[i]
    acc_ref *= K
    exp_out = acc_ref.to(torch.bfloat16)

    write(FIXTURES_DIR / "moe_hidden.bf16", bf16_bytes(hidden_exp))
    write(FIXTURES_DIR / "moe_routing_idx.i32", routing_idx.numpy().tobytes())
    write(FIXTURES_DIR / "moe_routing_scores.f32", routing_scores.numpy().tobytes())
    write(FIXTURES_DIR / "moe_out.bf16", bf16_bytes(exp_out))

    # ----- Model forward fixture (1-layer-truncated) -----
    #
    # Shrinks the real model to a single block (layer 0) + first N
    # experts so the fixture doesn't need 8 layers' × 128 experts'
    # weights. Our WASM modelForward runs the same single-block
    # pipeline against the committed blob. IDs are bounded by V_TEST.
    torch.manual_seed(SEED + 13)
    mdl_T = 16
    mdl_ids = torch.randint(0, V_TEST, (mdl_T,), dtype=torch.int32)

    # Patch layer-0 experts to the dequantized versions of our blob so
    # the reference's fp32 compute sees the same weights we carry. Then
    # truncate the experts + router to the first EXPERTS_KEEP experts —
    # otherwise the real router would emit indices in [0, 128) that our
    # 4-expert blob can't serve.
    layer0 = model.model.layers[0]
    with torch.no_grad():
        layer0.mlp.experts.gate_up_proj.data[:EXPERTS_KEEP] = \
            gu_dequant.transpose(-1, -2).to(torch.bfloat16)
        layer0.mlp.experts.down_proj.data[:EXPERTS_KEEP] = \
            d_dequant.transpose(-1, -2).to(torch.bfloat16)
    layer0.mlp.experts.num_experts = EXPERTS_KEEP
    layer0.mlp.experts.gate_up_proj = torch.nn.Parameter(
        layer0.mlp.experts.gate_up_proj.data[:EXPERTS_KEEP].clone()
    )
    layer0.mlp.experts.gate_up_proj_bias = torch.nn.Parameter(
        layer0.mlp.experts.gate_up_proj_bias.data[:EXPERTS_KEEP].clone()
    )
    layer0.mlp.experts.down_proj = torch.nn.Parameter(
        layer0.mlp.experts.down_proj.data[:EXPERTS_KEEP].clone()
    )
    layer0.mlp.experts.down_proj_bias = torch.nn.Parameter(
        layer0.mlp.experts.down_proj_bias.data[:EXPERTS_KEEP].clone()
    )
    layer0.mlp.router.num_experts = EXPERTS_KEEP
    layer0.mlp.router.weight = torch.nn.Parameter(
        layer0.mlp.router.weight.data[:EXPERTS_KEEP].clone()
    )
    layer0.mlp.router.bias = torch.nn.Parameter(
        layer0.mlp.router.bias.data[:EXPERTS_KEEP].clone()
    )
    # Truncate to 1 layer.
    model.model.layers = torch.nn.ModuleList([model.model.layers[0]])
    model.config.num_hidden_layers = 1
    model.config.num_local_experts = EXPERTS_KEEP

    with torch.no_grad():
        out = model(input_ids=mdl_ids.unsqueeze(0).long())
    mdl_logits = out.logits.squeeze(0).contiguous().to(torch.bfloat16)

    write(FIXTURES_DIR / "mdl_input_ids.i32", mdl_ids.numpy().tobytes())
    write(FIXTURES_DIR / "mdl_logits.bf16", bf16_bytes(mdl_logits))

    # ----- Block forward fixture (layer 0, synthetic routing) -----
    #
    # Exercises `blockForwardWithRouting`: norm + attention + residual +
    # norm + MLP (expert dispatch with fixed routing) + residual.
    # Uses the actual layer-0 attention module plus the layer-0 experts
    # module, but the experts' weights are pre-patched with the
    # dequantized versions of our blob — so the reference is bound by
    # kernel rounding only, not quantization error.
    torch.manual_seed(SEED + 12)
    blk_T = 16
    d_model = model.config.hidden_size
    sliding = model.config.sliding_window
    K = model.config.num_experts_per_tok

    # Model-forward fixture already patched + truncated layer0 above.
    # The block-forward fixture reuses that same truncated module, so
    # we don't repeat the patch here.

    hidden_blk = torch.randn(1, blk_T, d_model, dtype=torch.bfloat16)
    routing_idx_blk = torch.zeros(blk_T, K, dtype=torch.int32)
    for t in range(blk_T):
        for k in range(K):
            routing_idx_blk[t, k] = k % EXPERTS_KEEP
    routing_scores_blk = torch.full((blk_T, K), 1.0 / K, dtype=torch.float32)

    pos_ids_blk = torch.arange(blk_T, dtype=torch.long).unsqueeze(0)
    cos_blk, sin_blk = rope_layer(hidden_blk, pos_ids_blk)
    mask_blk = torch.zeros(blk_T, blk_T, dtype=torch.float32)
    for i in range(blk_T):
        for j in range(blk_T):
            if abs(i - j) > sliding:
                mask_blk[i, j] = torch.finfo(torch.float32).min
    mask_blk_4d = mask_blk.view(1, 1, blk_T, blk_T).to(torch.bfloat16)

    with torch.no_grad():
        residual1 = hidden_blk
        normed1 = layer0.input_layernorm(hidden_blk)
        attn_out, _ = layer0.self_attn(normed1, (cos_blk, sin_blk), attention_mask=mask_blk_4d)
        h1 = residual1 + attn_out
        residual2 = h1
        normed2 = layer0.post_attention_layernorm(h1)
        # Run experts module directly with our synthetic routing.
        experts_out_flat = layer0.mlp.experts(
            normed2.reshape(-1, d_model), routing_idx_blk.long(), routing_scores_blk,
        )
        moe_out = experts_out_flat.reshape(1, blk_T, d_model) * K
        h_blk_out = residual2 + moe_out

    write(FIXTURES_DIR / "blk_hidden.bf16", bf16_bytes(hidden_blk.squeeze(0).contiguous()))
    write(FIXTURES_DIR / "blk_routing_idx.i32", routing_idx_blk.numpy().tobytes())
    write(FIXTURES_DIR / "blk_routing_scores.f32", routing_scores_blk.numpy().tobytes())
    write(FIXTURES_DIR / "blk_out.bf16", bf16_bytes(h_blk_out.squeeze(0).contiguous().to(torch.bfloat16)))

    # ----- Attention forward fixture (layer 0) -----
    # Uses the real attention submodule with layer-0 weights so the
    # test exercises a production-shaped parameterisation.
    torch.manual_seed(SEED + 10)
    att_fwd_T = 32
    d_model = model.config.hidden_size
    sliding = model.config.sliding_window  # 128 upstream

    hidden_att = torch.randn(1, att_fwd_T, d_model, dtype=torch.bfloat16)
    pos_ids_att = torch.arange(att_fwd_T, dtype=torch.long).unsqueeze(0)
    cos_att, sin_att = rope_layer(hidden_att, pos_ids_att)   # each: [1, T, head_dim/2], bf16

    attn_mod = model.model.layers[0].self_attn.eval()
    # Build sliding window mask manually — additive, bf16. Mirrors what
    # upstream's masking helpers produce for an encoder layer.
    neg_inf = torch.finfo(torch.float32).min
    mask = torch.zeros(att_fwd_T, att_fwd_T, dtype=torch.float32)
    for i in range(att_fwd_T):
        for j in range(att_fwd_T):
            if abs(i - j) > sliding:
                mask[i, j] = neg_inf
    mask_4d = mask.view(1, 1, att_fwd_T, att_fwd_T).to(torch.bfloat16)

    with torch.no_grad():
        attn_out, _ = attn_mod(hidden_att, (cos_att, sin_att), attention_mask=mask_4d)
    # attn_out: [1, T, d_model] bf16
    attn_out = attn_out.squeeze(0).contiguous()

    write(FIXTURES_DIR / "attn_fwd_hidden.bf16", bf16_bytes(hidden_att.squeeze(0).contiguous()))
    write(FIXTURES_DIR / "attn_fwd_out.bf16", bf16_bytes(attn_out))

    # ----- RoPE yarn apply fixture -----
    position_ids = torch.arange(T, dtype=torch.long).unsqueeze(0)
    dummy = torch.empty(1, T, model.config.hidden_size, dtype=torch.bfloat16)
    cos_full, sin_full = rope_layer(dummy, position_ids)         # each: [1, T, head_dim/2], bf16
    # Layer already downcasts to the input dtype (bf16). Store bytes directly
    # so there's no f32 ↔ bf16 round-trip ambiguity.
    cos_bf = cos_full.squeeze(0).contiguous()                    # bf16 [T, head_dim/2]
    sin_bf = sin_full.squeeze(0).contiguous()

    torch.manual_seed(SEED + 4)
    H = model.config.num_attention_heads
    head_dim = model.config.head_dim
    q_in = torch.randn(T, H, head_dim, dtype=torch.bfloat16)

    # Reference expects (batch, num_heads, seq, head_dim); cos/sin (batch, 1, seq, head_dim/2).
    q_b = q_in.unsqueeze(0).transpose(1, 2).contiguous()
    cos_b = cos_bf.unsqueeze(0).unsqueeze(1)
    sin_b = sin_bf.unsqueeze(0).unsqueeze(1)
    q_ref = _apply_rotary_emb(q_b, cos_b, sin_b)
    q_out = q_ref.squeeze(0).transpose(0, 1).contiguous()         # [T, H, head_dim]

    write(FIXTURES_DIR / "rope_qk_in.bf16", bf16_bytes(q_in))
    write(FIXTURES_DIR / "rope_cos.bf16", bf16_bytes(cos_bf))
    write(FIXTURES_DIR / "rope_sin.bf16", bf16_bytes(sin_bf))
    write(FIXTURES_DIR / "rope_qk_out.bf16", bf16_bytes(q_out.to(torch.bfloat16)))

    # ----- 5. Int4-blockwise matmul fixture (quantized score.weight) -----
    torch.manual_seed(SEED + 3)
    x_i4 = torch.randn(T, D, dtype=torch.bfloat16)
    # Quantize W using the same code convert_weights.py will use in prod.
    # Source fp32 to get a clean quant (not twice-rounded through bf16).
    w_fp = sd["score.weight"].to(torch.float32)
    packed, w_shape = cw.quantize_int4_block32_sym(w_fp)
    assert w_shape == (N_CLASSES, D)
    # Reference expected output: dequantize W exactly, then F.linear in fp32.
    w_dequant = dequantize_int4block(packed, N_CLASSES, D)           # [N, D] fp32
    x_fp = x_i4.to(torch.float32)
    b_fp = sd["score.bias"].to(torch.float32)
    y_i4 = F.linear(x_fp, w_dequant, b_fp).to(torch.bfloat16)
    write(FIXTURES_DIR / "int4_matmul_x.bf16", bf16_bytes(x_i4))
    write(FIXTURES_DIR / "int4_matmul_w.i4block", packed)
    write(FIXTURES_DIR / "int4_matmul_bias.bf16", bf16_bytes(sd["score.bias"].to(torch.bfloat16)))
    write(FIXTURES_DIR / "int4_matmul_y.bf16", bf16_bytes(y_i4))

    print("[gen] done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
