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

    # ----- 1. Weight blob (5 tensors, with embed table truncated to V_TEST rows) -----
    print("[gen] building weight blob via scripts/convert_weights.py …", file=sys.stderr)
    convert = REPO_ROOT / "scripts" / "convert_weights.py"
    output_blob = FIXTURES_DIR / "pii-weights.bin"
    cmd = [
        sys.executable, str(convert),
        "--output", str(output_blob),
        "--include", "model.embed_tokens.weight",
        "--embed-rows", str(V_TEST),
    ]
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
    sinks = torch.randn(att_Hq, dtype=torch.bfloat16) * 0.5
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
    write(FIXTURES_DIR / "attn_sinks.bf16", bf16_bytes(sinks))
    write(FIXTURES_DIR / "attn_out.bf16", bf16_bytes(out_ref.view(att_T, att_Hq * att_hd)))

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

    # ----- RoPE yarn apply fixture -----
    #
    # Runs the upstream `OpenAIPrivacyFilterRotaryEmbedding` layer once
    # to get cos/sin tables with yarn scaling + attention_factor baked
    # in, then applies `_apply_rotary_emb` for the expected output.
    from transformers.models.openai_privacy_filter.modeling_openai_privacy_filter import (
        OpenAIPrivacyFilterRotaryEmbedding,
        _apply_rotary_emb,
    )
    rope_layer = OpenAIPrivacyFilterRotaryEmbedding(model.config)
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
