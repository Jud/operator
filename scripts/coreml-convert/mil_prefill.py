"""Build Qwen3.5 prefill model directly in MIL.

No tracing — constructs the computation graph programmatically.
Model-agnostic: works with any Qwen3.5 size via model_id.

Phase 1: All 24 layers, single token, no while_loop (verify correctness)
Phase 2: Add while_loop for multi-token prefill

Usage:
    /tmp/coreml-venv/bin/python scripts/coreml-convert/mil_prefill.py \
        Qwen/Qwen3.5-0.8B --output models/prefill_mil/
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb, types
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


# ── MIL op helpers ──────────────────────────────────────────────

def rms_norm(x, weight, name, eps=1e-6):
    """Qwen3.5 RMSNorm: output = x/rms(x) * (1 + weight)."""
    sq = mb.mul(x=x, y=x, name=f"{name}_sq")
    ms = mb.reduce_mean(x=sq, axes=[-1], keep_dims=True, name=f"{name}_ms")
    rs = mb.rsqrt(x=mb.add(x=ms, y=np.float32(eps), name=f"{name}_eps"), name=f"{name}_rs")
    normed = mb.mul(x=x, y=rs, name=f"{name}_rms")
    return mb.mul(x=normed, y=mb.add(x=weight, y=np.float32(1.0), name=f"{name}_wp1"), name=f"{name}_out")


def rms_norm_plain(x, weight, name, eps=1e-6):
    """Standard RMSNorm: output = x/rms(x) * weight (for Qwen3_5RMSNormGated)."""
    sq = mb.mul(x=x, y=x, name=f"{name}_sq")
    ms = mb.reduce_mean(x=sq, axes=[-1], keep_dims=True, name=f"{name}_ms")
    rs = mb.rsqrt(x=mb.add(x=ms, y=np.float32(eps), name=f"{name}_eps"), name=f"{name}_rs")
    return mb.mul(x=mb.mul(x=x, y=rs, name=f"{name}_rms"), y=weight, name=f"{name}_out")


def silu(x, name):
    return mb.mul(x=x, y=mb.sigmoid(x=x, name=f"{name}_sig"), name=f"{name}_silu")


def l2_normalize(x, name, eps=1e-12):
    """L2 normalize along last axis."""
    sq = mb.reduce_sum(x=mb.mul(x=x, y=x, name=f"{name}_sq"), axes=[-1], keep_dims=True, name=f"{name}_ss")
    return mb.mul(x=x, y=mb.rsqrt(x=mb.add(x=sq, y=np.float32(eps), name=f"{name}_e"), name=f"{name}_rs"), name=f"{name}_n")


def mlp_block(h, w_ln, w_gate, w_up, w_down, name):
    """RMSNorm + SwiGLU MLP."""
    residual = h
    h = rms_norm(h, w_ln, f"{name}_ln")
    g = silu(mb.linear(x=h, weight=w_gate, name=f"{name}_gate"), f"{name}_gsilu")
    u = mb.linear(x=h, weight=w_up, name=f"{name}_up")
    m = mb.linear(x=mb.mul(x=g, y=u, name=f"{name}_gu"), weight=w_down, name=f"{name}_down")
    return mb.add(x=residual, y=m, name=f"{name}_res")


# ── DeltaNet layer ──────────────────────────────────────────────

def deltanet_recurrent_step(query, key, value, beta, g_exp, rec_state, name):
    """One step of DeltaNet recurrence. Returns (output, new_state)."""
    n = name
    g_4d = mb.expand_dims(x=mb.expand_dims(x=g_exp, axes=[-1], name=f"{n}_g3"), axes=[-1], name=f"{n}_g4")
    decayed = mb.mul(x=rec_state, y=g_4d, name=f"{n}_decay")
    k_4d = mb.expand_dims(x=key, axes=[-1], name=f"{n}_k4d")
    kv_mem = mb.reduce_sum(x=mb.mul(x=decayed, y=k_4d, name=f"{n}_sk"), axes=[2], keep_dims=False, name=f"{n}_kvmem")
    delta = mb.mul(x=mb.sub(x=value, y=kv_mem, name=f"{n}_diff"), y=beta, name=f"{n}_delta")
    update = mb.mul(x=mb.expand_dims(x=key, axes=[-1], name=f"{n}_ko"),
                    y=mb.expand_dims(x=delta, axes=[-2], name=f"{n}_do"), name=f"{n}_upd")
    new_state = mb.add(x=decayed, y=update, name=f"{n}_newrec")
    q_4d = mb.expand_dims(x=query, axes=[-1], name=f"{n}_q4d")
    out = mb.reduce_sum(x=mb.mul(x=new_state, y=q_4d, name=f"{n}_sq"), axes=[2], keep_dims=False, name=f"{n}_out")
    return out, new_state


def build_deltanet_layer(h, conv_state, rec_state, w, cfg, name):
    """Full DeltaNet decoder layer. Returns (output, new_conv, new_rec)."""
    n = name
    n_heads, k_dim, v_dim = cfg['dn_heads'], cfg['dn_k_dim'], cfg['dn_v_dim']
    key_total = k_dim * n_heads
    qkv_total = key_total * 2 + v_dim * n_heads
    conv_k = cfg['conv_k']

    residual = h
    h = rms_norm(h, w['ln'], f"{n}_ln")

    # QKV + conv1d update
    mixed = mb.linear(x=h, weight=w['qkv'], name=f"{n}_qkv")
    mixed_3d = mb.expand_dims(x=mixed, axes=[2], name=f"{n}_m3d")
    cat = mb.concat(values=[conv_state, mixed_3d], axis=2, name=f"{n}_ccat")
    new_conv = mb.slice_by_index(x=cat, begin=[0, 0, 1], end=[1, qkv_total, conv_k + 1], name=f"{n}_nconv")
    w_c3d = mb.expand_dims(x=mb.squeeze(x=w['conv_w'], axes=[1], name=f"{n}_csq"), axes=[0], name=f"{n}_c3d")
    conv_win = mb.slice_by_index(x=cat, begin=[0, 0, 1], end=[1, qkv_total, conv_k + 1], name=f"{n}_cwin")
    conv_out = silu(mb.reduce_sum(x=mb.mul(x=conv_win, y=w_c3d, name=f"{n}_cm"), axes=[-1], keep_dims=False, name=f"{n}_cs"), f"{n}_csilu")

    # Split Q, K, V → heads
    query = mb.reshape(x=mb.slice_by_index(x=conv_out, begin=[0, 0], end=[1, key_total], name=f"{n}_q"),
                       shape=np.array([1, n_heads, k_dim], dtype=np.int32), name=f"{n}_qh")
    key = mb.reshape(x=mb.slice_by_index(x=conv_out, begin=[0, key_total], end=[1, key_total * 2], name=f"{n}_k"),
                     shape=np.array([1, n_heads, k_dim], dtype=np.int32), name=f"{n}_kh")
    value = mb.reshape(x=mb.slice_by_index(x=conv_out, begin=[0, key_total * 2], end=[1, qkv_total], name=f"{n}_v"),
                       shape=np.array([1, n_heads, v_dim], dtype=np.int32), name=f"{n}_vh")

    # Z gate, beta, gate g
    z = mb.reshape(x=mb.linear(x=h, weight=w['z'], name=f"{n}_z"), shape=np.array([1, n_heads, v_dim], dtype=np.int32), name=f"{n}_zh")
    beta = mb.expand_dims(x=mb.sigmoid(x=mb.linear(x=h, weight=w['b'], name=f"{n}_bp"), name=f"{n}_beta"), axes=[-1], name=f"{n}_b3d")
    a_out = mb.linear(x=h, weight=w['a'], name=f"{n}_ap")
    sp = mb.log(x=mb.add(x=mb.exp(x=mb.add(x=a_out, y=w['dt_bias'], name=f"{n}_ab"), name=f"{n}_spe"), y=np.float32(1.0), name=f"{n}_sp1"), name=f"{n}_sp")
    g_exp = mb.exp(x=mb.mul(x=mb.mul(x=mb.exp(x=w['A_log'], name=f"{n}_Ae"), y=np.float32(-1.0), name=f"{n}_nA"), y=sp, name=f"{n}_g"), name=f"{n}_ge")

    # L2 norm + scale
    scale = np.float32(1.0 / (k_dim ** 0.5))
    query = mb.mul(x=l2_normalize(query, f"{n}_ql2"), y=scale, name=f"{n}_qs")
    key = l2_normalize(key, f"{n}_kl2")

    # Recurrence
    attn_out, new_rec = deltanet_recurrent_step(query, key, value, beta, g_exp, rec_state, f"{n}_rec")

    # Gated RMSNorm (plain weight, not 1+weight)
    out_flat = mb.reshape(x=attn_out, shape=np.array([n_heads, v_dim], dtype=np.int32), name=f"{n}_of")
    z_flat = mb.reshape(x=z, shape=np.array([n_heads, v_dim], dtype=np.int32), name=f"{n}_zf")
    out_normed = rms_norm_plain(out_flat, w['norm'], f"{n}_onorm")
    gated = mb.mul(x=out_normed, y=silu(z_flat, f"{n}_zsilu"), name=f"{n}_gated")

    # Output projection + residual
    gated_2d = mb.reshape(x=gated, shape=np.array([1, v_dim * n_heads], dtype=np.int32), name=f"{n}_g2d")
    h_mid = mb.add(x=residual, y=mb.linear(x=gated_2d, weight=w['out'], name=f"{n}_oproj"), name=f"{n}_res1")

    # MLP
    output = mlp_block(h_mid, w['pln'], w['gate'], w['up'], w['down'], f"{n}_mlp")
    return output, new_conv, new_rec


# ── Attention layer ─────────────────────────────────────────────

def rotate_half(x, n_heads, rope_dim, name):
    """HF rotate_half: split in half, swap with negation."""
    half = rope_dim // 2
    x1 = mb.slice_by_index(x=x, begin=[0, 0, 0], end=[1, n_heads, half], name=f"{name}_x1")
    x2 = mb.slice_by_index(x=x, begin=[0, 0, half], end=[1, n_heads, rope_dim], name=f"{name}_x2")
    return mb.concat(values=[mb.mul(x=x2, y=np.float32(-1.0), name=f"{name}_neg"), x1], axis=-1, name=f"{name}_rot")


def build_attention_layer(h, cos, sin, key_cache, value_cache, attn_mask, cache_pos, w, cfg, name):
    """Full attention decoder layer with KV cache.

    Args:
        key_cache/value_cache: [1, n_kv, max_kv_len, head_dim] padded cache
        attn_mask: [1, 1, 1, max_kv_len] with 0=attend, -1e4=mask
        cache_pos: scalar int32 — current position for cache write

    Returns: (output, new_key_cache, new_value_cache)
    """
    n = name
    n_q, n_kv, hd = cfg['attn_n_q'], cfg['attn_n_kv'], cfg['attn_hd']
    ng = n_q // n_kv
    rope_dim = cfg['rope_dim']
    max_kv = cfg['max_kv_len']

    residual = h
    h = rms_norm(h, w['ln'], f"{n}_ln")

    # Q(+gate), K, V projections
    # HF interleaves Q and gate per head: reshape to [n_q, hd*2] then chunk
    qg = mb.linear(x=h, weight=w['q'], name=f"{n}_qg")
    qg_heads = mb.reshape(x=qg, shape=np.array([1, n_q, hd * 2], dtype=np.int32), name=f"{n}_qgh")
    query = mb.slice_by_index(x=qg_heads, begin=[0, 0, 0], end=[1, n_q, hd], name=f"{n}_qf")
    gate_heads = mb.slice_by_index(x=qg_heads, begin=[0, 0, hd], end=[1, n_q, hd * 2], name=f"{n}_gfh")
    g_flat = mb.reshape(x=gate_heads, shape=np.array([1, n_q * hd], dtype=np.int32), name=f"{n}_gf")

    k_flat = mb.linear(x=h, weight=w['k'], name=f"{n}_kf")
    v_flat = mb.linear(x=h, weight=w['v'], name=f"{n}_vf")

    key_new = mb.reshape(x=k_flat, shape=np.array([1, n_kv, hd], dtype=np.int32), name=f"{n}_kh")
    value_new = mb.reshape(x=v_flat, shape=np.array([1, n_kv, hd], dtype=np.int32), name=f"{n}_vh")

    # Q/K norms (Qwen3.5RMSNorm with 1+weight)
    query = rms_norm(query, w['qn'], f"{n}_qnorm")
    key_new = rms_norm(key_new, w['kn'], f"{n}_knorm")

    # RoPE
    qr = mb.slice_by_index(x=query, begin=[0, 0, 0], end=[1, n_q, rope_dim], name=f"{n}_qr")
    qp = mb.slice_by_index(x=query, begin=[0, 0, rope_dim], end=[1, n_q, hd], name=f"{n}_qp")
    kr = mb.slice_by_index(x=key_new, begin=[0, 0, 0], end=[1, n_kv, rope_dim], name=f"{n}_kr")
    kp = mb.slice_by_index(x=key_new, begin=[0, 0, rope_dim], end=[1, n_kv, hd], name=f"{n}_kp")
    qe = mb.add(x=mb.mul(x=qr, y=cos, name=f"{n}_qc"), y=mb.mul(x=rotate_half(qr, n_q, rope_dim, f"{n}_qrh"), y=sin, name=f"{n}_qs"), name=f"{n}_qe")
    ke = mb.add(x=mb.mul(x=kr, y=cos, name=f"{n}_kc"), y=mb.mul(x=rotate_half(kr, n_kv, rope_dim, f"{n}_krh"), y=sin, name=f"{n}_ks"), name=f"{n}_ke")
    query = mb.concat(values=[qe, qp], axis=-1, name=f"{n}_qfull")     # [1, n_q, hd]
    key_new = mb.concat(values=[ke, kp], axis=-1, name=f"{n}_kfull")   # [1, n_kv, hd]

    # KV cache update: write new K,V at cache_pos using comparison mask
    positions = mb.const(val=np.arange(max_kv, dtype=np.int32), name=f"{n}_positions")
    cp_scalar = mb.gather(x=cache_pos, indices=np.int32(0), axis=0, name=f"{n}_cp0")
    pos_mask = mb.cast(x=mb.equal(x=positions, y=cp_scalar, name=f"{n}_eq"), dtype="fp32", name=f"{n}_pmask")
    pos_mask_4d = mb.reshape(x=pos_mask, shape=np.array([1, 1, max_kv, 1], dtype=np.int32), name=f"{n}_pm4d")

    k_new_4d = mb.expand_dims(x=key_new, axes=[2], name=f"{n}_kn4d")   # [1, n_kv, 1, hd]
    v_new_4d = mb.expand_dims(x=value_new, axes=[2], name=f"{n}_vn4d")

    # where(pos_mask, new_kv_broadcast, old_cache)
    new_key_cache = mb.add(
        x=mb.mul(x=key_cache, y=mb.sub(x=np.float32(1.0), y=pos_mask_4d, name=f"{n}_inv"), name=f"{n}_kold"),
        y=mb.mul(x=k_new_4d, y=pos_mask_4d, name=f"{n}_knew2"), name=f"{n}_nkc")
    new_value_cache = mb.add(
        x=mb.mul(x=value_cache, y=mb.sub(x=np.float32(1.0), y=pos_mask_4d, name=f"{n}_inv2"), name=f"{n}_vold"),
        y=mb.mul(x=v_new_4d, y=pos_mask_4d, name=f"{n}_vnew2"), name=f"{n}_nvc")

    # GQA repeat full KV cache
    # key_cache: [1, n_kv, max_kv, hd] → [1, n_q, max_kv, hd]
    k_rep = mb.reshape(x=mb.tile(x=mb.expand_dims(x=new_key_cache, axes=[2], name=f"{n}_kce"),
                                  reps=[1, 1, ng, 1, 1], name=f"{n}_kct"),
                       shape=np.array([1, n_q, max_kv, hd], dtype=np.int32), name=f"{n}_kcr")
    v_rep = mb.reshape(x=mb.tile(x=mb.expand_dims(x=new_value_cache, axes=[2], name=f"{n}_vce"),
                                  reps=[1, 1, ng, 1, 1], name=f"{n}_vct"),
                       shape=np.array([1, n_q, max_kv, hd], dtype=np.int32), name=f"{n}_vcr")

    # Attention: Q [1, n_q, hd] @ K^T [1, n_q, hd, max_kv] → [1, n_q, max_kv]
    sc = np.float32(1.0 / (hd ** 0.5))
    q_4d = mb.expand_dims(x=query, axes=[2], name=f"{n}_q4d")  # [1, n_q, 1, hd]
    # Q @ K^T: [1, n_q, 1, hd] @ [1, n_q, max_kv, hd]^T
    # Use matmul: [1, n_q, 1, hd] @ [1, n_q, hd, max_kv] → [1, n_q, 1, max_kv]
    k_t = mb.transpose(x=k_rep, perm=[0, 1, 3, 2], name=f"{n}_kt")  # [1, n_q, hd, max_kv]
    attn_scores = mb.mul(x=mb.matmul(x=q_4d, y=k_t, name=f"{n}_qkt"), y=sc, name=f"{n}_scaled")  # [1, n_q, 1, max_kv]

    # Apply attention mask
    attn_scores = mb.add(x=attn_scores, y=attn_mask, name=f"{n}_masked")  # [1, n_q, 1, max_kv]
    attn_weights = mb.softmax(x=attn_scores, axis=-1, name=f"{n}_sm")

    # attn_weights @ V: [1, n_q, 1, max_kv] @ [1, n_q, max_kv, hd] → [1, n_q, 1, hd]
    attn_out = mb.matmul(x=attn_weights, y=v_rep, name=f"{n}_av")  # [1, n_q, 1, hd]
    attn_out = mb.squeeze(x=attn_out, axes=[2], name=f"{n}_ao")  # [1, n_q, hd]

    # Gate + output projection
    gate = mb.reshape(x=g_flat, shape=np.array([1, n_q, hd], dtype=np.int32), name=f"{n}_gh")
    gated = mb.mul(x=attn_out, y=mb.sigmoid(x=gate, name=f"{n}_gsig"), name=f"{n}_gated")
    oproj = mb.linear(x=mb.reshape(x=gated, shape=np.array([1, n_q * hd], dtype=np.int32), name=f"{n}_gfl"), weight=w['o'], name=f"{n}_oproj")
    h_mid = mb.add(x=residual, y=oproj, name=f"{n}_res1")

    # MLP
    output = mlp_block(h_mid, w['pln'], w['gate'], w['up'], w['down'], f"{n}_mlp")
    return output, new_key_cache, new_value_cache


# ── Main ────────────────────────────────────────────────────────

def extract_layer_weights(model, layer_idx, layer_type):
    """Extract weights as numpy for a single layer."""
    sd = {k: v.detach().numpy().astype(np.float32) for k, v in model.model.layers[layer_idx].named_parameters()}
    if layer_type == "linear_attention":
        return {
            'ln': sd['input_layernorm.weight'], 'qkv': sd['linear_attn.in_proj_qkv.weight'],
            'conv_w': sd['linear_attn.conv1d.weight'], 'z': sd['linear_attn.in_proj_z.weight'],
            'b': sd['linear_attn.in_proj_b.weight'], 'a': sd['linear_attn.in_proj_a.weight'],
            'out': sd['linear_attn.out_proj.weight'], 'norm': sd['linear_attn.norm.weight'],
            'dt_bias': sd['linear_attn.dt_bias'], 'A_log': sd['linear_attn.A_log'],
            'pln': sd['post_attention_layernorm.weight'], 'gate': sd['mlp.gate_proj.weight'],
            'up': sd['mlp.up_proj.weight'], 'down': sd['mlp.down_proj.weight'],
        }
    else:  # full_attention
        return {
            'ln': sd['input_layernorm.weight'], 'q': sd['self_attn.q_proj.weight'],
            'k': sd['self_attn.k_proj.weight'], 'v': sd['self_attn.v_proj.weight'],
            'o': sd['self_attn.o_proj.weight'], 'qn': sd['self_attn.q_norm.weight'],
            'kn': sd['self_attn.k_norm.weight'],
            'pln': sd['post_attention_layernorm.weight'], 'gate': sd['mlp.gate_proj.weight'],
            'up': sd['mlp.up_proj.weight'], 'down': sd['mlp.down_proj.weight'],
        }


def build_while_loop(args, model, tokenizer, all_weights, embed_w, final_ln_w, lm_head_w,
                     all_cos, all_sin, cfg, delta_indices, attn_indices,
                     num_layers, layer_types, H, qkv_total, conv_k,
                     dn_heads, dn_k, dn_v, attn_nq, attn_nkv, attn_hd, rope_dim):
    """Build multi-token prefill model with mb.while_loop."""
    max_seq = args.max_seq_len
    max_kv = max_seq  # KV cache sized to fit full sequence
    cfg['max_kv_len'] = max_kv

    print(f"Building MIL program (while_loop, max_seq={max_seq}, max_kv={max_kv})...")

    # Pre-compute RoPE as numpy constants
    cos_np = all_cos[0, :max_seq, :].numpy().astype(np.float32)  # [max_seq, rope_dim]
    sin_np = all_sin[0, :max_seq, :].numpy().astype(np.float32)

    input_specs = [
        mb.TensorSpec(shape=(max_seq,), dtype=types.int32),   # token_ids (padded)
        mb.TensorSpec(shape=(1,), dtype=types.int32),          # seq_len
    ]

    @mb.program(input_specs=input_specs, opset_version=ct.target.iOS17)
    def prefill_prog(token_ids, seq_len):
        # ── Constants ──
        embed_c = mb.const(val=embed_w, name="embed_w")
        cos_c = mb.const(val=cos_np, name="all_cos")
        sin_c = mb.const(val=sin_np, name="all_sin")
        final_ln_c = mb.const(val=final_ln_w, name="final_ln_w")
        lm_head_c = mb.const(val=lm_head_w, name="lm_head_w")

        layer_w = {}
        for i in range(num_layers):
            lw = {}
            for k, v in all_weights[i].items():
                lw[k] = mb.const(val=v, name=f"L{i}_w_{k}")
            layer_w[i] = lw

        # ── Pre-compute all embeddings ──
        all_embeddings = mb.gather(x=embed_c, indices=token_ids, axis=0, name="all_emb")  # [max_seq, H]

        # ── Initial loop vars ──
        seq_len_s = mb.gather(x=seq_len, indices=np.int32(0), axis=0, name="seq_len_s")
        init_counter = mb.const(val=np.int32(0), name="init_ctr")
        init_hidden = mb.const(val=np.zeros((1, H), dtype=np.float32), name="init_h")

        init_loop = [init_counter, init_hidden]
        for j in range(len(delta_indices)):
            init_loop.append(mb.const(val=np.zeros((1, qkv_total, conv_k), dtype=np.float32), name=f"init_conv_{j}"))
            init_loop.append(mb.const(val=np.zeros((1, dn_heads, dn_k, dn_v), dtype=np.float32), name=f"init_rec_{j}"))
        for j in range(len(attn_indices)):
            init_loop.append(mb.const(val=np.zeros((1, attn_nkv, max_kv, attn_hd), dtype=np.float32), name=f"init_kc_{j}"))
            init_loop.append(mb.const(val=np.zeros((1, attn_nkv, max_kv, attn_hd), dtype=np.float32), name=f"init_vc_{j}"))

        print(f"  {len(init_loop)} loop vars (counter + hidden + {len(delta_indices)*2} DeltaNet + {len(attn_indices)*2} attn)")

        # ── Loop condition ──
        def cond(counter, hidden, *states):
            return mb.less(x=counter, y=seq_len_s, name="loop_cond")

        # ── Loop body ──
        def body(counter, hidden, *states):
            # Gather embedding for current token
            h = mb.expand_dims(
                x=mb.gather(x=all_embeddings, indices=counter, axis=0, name="emb_g"),
                axes=[0], name="h_emb")  # [1, H]

            # Gather cos/sin for current position
            cos = mb.reshape(
                x=mb.gather(x=cos_c, indices=counter, axis=0, name="cos_g"),
                shape=np.array([1, 1, rope_dim], dtype=np.int32), name="cos_r")
            sin = mb.reshape(
                x=mb.gather(x=sin_c, indices=counter, axis=0, name="sin_g"),
                shape=np.array([1, 1, rope_dim], dtype=np.int32), name="sin_r")

            # Attention mask: 0 for pos <= counter, -1e4 for pos > counter
            pos_range = mb.const(val=np.arange(max_kv, dtype=np.int32), name="pos_rng")
            mask_bool = mb.less_equal(x=pos_range, y=counter, name="mask_le")
            mask_f = mb.cast(x=mask_bool, dtype="fp32", name="mask_f")
            attn_mask = mb.reshape(
                x=mb.mul(x=mb.sub(x=mask_f, y=np.float32(1.0), name="mask_sub"),
                          y=np.float32(1e4), name="mask_mul"),
                shape=np.array([1, 1, 1, max_kv], dtype=np.int32), name="attn_mask")

            cache_pos = mb.expand_dims(x=counter, axes=[0], name="cache_pos")

            # Unpack states
            si = 0
            conv_s, rec_s = {}, {}
            for j, idx in enumerate(delta_indices):
                conv_s[idx] = states[si]; rec_s[idx] = states[si + 1]; si += 2
            kv_s = {}
            for j, idx in enumerate(attn_indices):
                kv_s[idx] = (states[si], states[si + 1]); si += 2

            # Run all layers
            new_conv, new_rec, new_kv = {}, {}, {}
            for i in range(num_layers):
                if layer_types[i] == "linear_attention":
                    h, nc, nr = build_deltanet_layer(h, conv_s[i], rec_s[i], layer_w[i], cfg, f"L{i}")
                    new_conv[i] = nc; new_rec[i] = nr
                else:
                    kc, vc = kv_s[i]
                    h, nkc, nvc = build_attention_layer(h, cos, sin, kc, vc, attn_mask, cache_pos, layer_w[i], cfg, f"L{i}")
                    new_kv[i] = (nkc, nvc)

            # Pack new states
            new_ctr = mb.add(x=counter, y=np.int32(1), name="inc_ctr")
            new_states = []
            for idx in delta_indices:
                new_states.append(new_conv[idx]); new_states.append(new_rec[idx])
            for idx in attn_indices:
                new_states.append(new_kv[idx][0]); new_states.append(new_kv[idx][1])

            return (new_ctr, h, *new_states)

        # ── Execute loop ──
        results = mb.while_loop(_cond=cond, _body=body, loop_vars=tuple(init_loop))

        # ── After loop: logits from final hidden ──
        final_hidden = results[1]  # [1, H]
        h = rms_norm(final_hidden, final_ln_c, "final_ln")
        logits = mb.linear(x=h, weight=lm_head_c, name="lm_head")

        # Output: logits + all final states
        outputs = [logits]
        ri = 2  # skip counter and hidden
        for idx in delta_indices:
            outputs.append(results[ri]); outputs.append(results[ri + 1]); ri += 2
        for idx in attn_indices:
            outputs.append(results[ri]); outputs.append(results[ri + 1]); ri += 2
        return tuple(outputs)

    precision = ct.precision.FLOAT16 if args.fp16 else ct.precision.FLOAT32
    t0 = time.time()
    print(f"Converting to CoreML ({'FP16' if args.fp16 else 'FP32'})...")
    mlmodel = ct.convert(prefill_prog, convert_to="mlprogram",
                         minimum_deployment_target=ct.target.iOS18,
                         compute_precision=precision)
    print(f"Converted in {time.time() - t0:.1f}s")

    suffix = "_fp16" if args.fp16 else ""
    out_path = os.path.join(args.output, f"prefill_loop{suffix}.mlpackage")
    mlmodel.save(out_path)
    print(f"Saved {out_path}")

    # ── Verify ──
    print("\nVerifying while_loop model...")
    sys.path.insert(0, os.path.dirname(__file__))
    import patches  # noqa
    from functional_deltanet import patch_transformers
    patch_transformers()

    prompt = "Clean up: um so I was like going to the uh store"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        pt_logits = model(input_ids).logits[0, -1].numpy().copy()
    pt_token = int(np.argmax(pt_logits))
    print(f"PyTorch token: {pt_token} ({repr(tokenizer.decode([pt_token]))})")

    # Pad token IDs to max_seq_len
    token_np = np.zeros(max_seq, dtype=np.int32)
    token_np[:prompt_len] = input_ids[0].numpy()

    mil_inputs = {
        "token_ids": token_np,
        "seq_len": np.array([prompt_len], dtype=np.int32),
    }
    mil_out = mlmodel.predict(mil_inputs)
    logits_key = [k for k in mil_out.keys() if 'lm_head' in k or mil_out[k].shape[-1] > 10000]
    mil_logits = mil_out[logits_key[0]].flatten() if logits_key else max(mil_out.values(), key=lambda v: v.shape[-1]).flatten()
    mil_token = int(np.argmax(mil_logits))
    print(f"MIL token:     {mil_token} ({repr(tokenizer.decode([mil_token]))})")
    print(f"Match: {'✓' if mil_token == pt_token else '✗'}")

    diff = np.abs(mil_logits[:len(pt_logits)] - pt_logits).max()
    print(f"Max logit diff: {diff:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--output", default="models/prefill_mil/")
    parser.add_argument("--while-loop", action="store_true", help="Build multi-token prefill with while_loop")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length for while_loop model")
    parser.add_argument("--fp16", action="store_true", help="Convert while_loop model to FP16")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"Loading {args.model_id}...")
    config = AutoConfig.from_pretrained(args.model_id)
    config.text_config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32, config=config)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tc = config.text_config

    H = tc.hidden_size
    num_layers = tc.num_hidden_layers
    layer_types = tc.layer_types

    # DeltaNet config
    dn_heads = tc.linear_num_value_heads
    dn_k = tc.linear_key_head_dim
    dn_v = tc.linear_value_head_dim
    conv_k = tc.linear_conv_kernel_dim
    qkv_total = (dn_k * dn_heads * 2) + (dn_v * dn_heads)

    # Attention config
    attn_nq = tc.num_attention_heads
    attn_nkv = tc.num_key_value_heads
    attn_hd = tc.head_dim

    max_rope_pos = max(32, args.max_seq_len) if args.while_loop else 32
    with torch.no_grad():
        all_cos, all_sin = model.model.rotary_emb(torch.zeros(1, max_rope_pos, H), torch.arange(max_rope_pos).unsqueeze(0))
    rope_dim = all_cos.shape[-1]

    cfg = {
        'hidden_size': H, 'dn_heads': dn_heads, 'dn_k_dim': dn_k, 'dn_v_dim': dn_v,
        'conv_k': conv_k, 'mlp_dim': tc.intermediate_size,
        'attn_n_q': attn_nq, 'attn_n_kv': attn_nkv, 'attn_hd': attn_hd, 'rope_dim': rope_dim,
    }

    delta_indices = [i for i in range(num_layers) if layer_types[i] == "linear_attention"]
    attn_indices = [i for i in range(num_layers) if layer_types[i] == "full_attention"]
    print(f"  {len(delta_indices)} DeltaNet + {len(attn_indices)} attention = {num_layers} layers")

    # Extract all weights
    print("Extracting weights...")
    all_weights = {}
    for i in range(num_layers):
        all_weights[i] = extract_layer_weights(model, i, layer_types[i])

    embed_w = model.model.embed_tokens.weight.detach().numpy().astype(np.float32)
    final_ln_w = model.model.norm.weight.detach().numpy().astype(np.float32)
    lm_head_w = model.lm_head.weight.detach().numpy().astype(np.float32)

    if args.while_loop:
        return build_while_loop(args, model, tokenizer, all_weights, embed_w, final_ln_w, lm_head_w,
                                all_cos, all_sin, cfg, delta_indices, attn_indices,
                                num_layers, layer_types, H, qkv_total, conv_k,
                                dn_heads, dn_k, dn_v, attn_nq, attn_nkv, attn_hd, rope_dim)

    # ── Build all 24 layers (single token, no while_loop) ──
    print("Building MIL program (all layers, single token)...")

    max_kv = 512  # KV cache size
    cfg['max_kv_len'] = max_kv

    # Build input specs — MIL needs exact parameter count in function signature
    param_names = ["hidden", "cos", "sin", "cache_pos", "attn_mask"]
    input_specs = [
        mb.TensorSpec(shape=(1, H), dtype=types.fp32),
        mb.TensorSpec(shape=(1, 1, rope_dim), dtype=types.fp32),
        mb.TensorSpec(shape=(1, 1, rope_dim), dtype=types.fp32),
        mb.TensorSpec(shape=(1,), dtype=types.int32),  # cache position
        mb.TensorSpec(shape=(1, 1, 1, max_kv), dtype=types.fp32),
    ]
    for j, i in enumerate(delta_indices):
        param_names.extend([f"conv_{j}", f"rec_{j}"])
        input_specs.extend([
            mb.TensorSpec(shape=(1, qkv_total, conv_k), dtype=types.fp32),
            mb.TensorSpec(shape=(1, dn_heads, dn_k, dn_v), dtype=types.fp32),
        ])
    for j, i in enumerate(attn_indices):
        param_names.extend([f"kcache_{j}", f"vcache_{j}"])
        input_specs.extend([
            mb.TensorSpec(shape=(1, attn_nkv, max_kv, attn_hd), dtype=types.fp32),
            mb.TensorSpec(shape=(1, attn_nkv, max_kv, attn_hd), dtype=types.fp32),
        ])
    print(f"  {len(input_specs)} inputs")

    # Dynamically create function with correct signature for @mb.program
    def _build(args_dict):
        h = args_dict["hidden"]
        cos = args_dict["cos"]
        sin = args_dict["sin"]
        cache_pos = args_dict["cache_pos"]
        attn_mask = args_dict["attn_mask"]

        layer_w = {}
        for i in range(num_layers):
            lw = {}
            for k, v in all_weights[i].items():
                lw[k] = mb.const(val=v, name=f"L{i}_w_{k}")  # "w_" prefix avoids collision with op names
            layer_w[i] = lw
        final_ln = mb.const(val=final_ln_w, name="final_ln")
        lm_head_c = mb.const(val=lm_head_w, name="lm_head_w")

        conv_states, rec_states = {}, {}
        for j, idx in enumerate(delta_indices):
            conv_states[idx] = args_dict[f"conv_{j}"]
            rec_states[idx] = args_dict[f"rec_{j}"]

        kv_caches = {}
        for j, idx in enumerate(attn_indices):
            kv_caches[idx] = (args_dict[f"kcache_{j}"], args_dict[f"vcache_{j}"])

        new_conv, new_rec, new_kv = {}, {}, {}
        for i in range(num_layers):
            if layer_types[i] == "linear_attention":
                h, nc, nr = build_deltanet_layer(h, conv_states[i], rec_states[i], layer_w[i], cfg, f"L{i}")
                new_conv[i] = nc
                new_rec[i] = nr
            else:
                kc, vc = kv_caches[i]
                h, nkc, nvc = build_attention_layer(h, cos, sin, kc, vc, attn_mask, cache_pos, layer_w[i], cfg, f"L{i}")
                new_kv[i] = (nkc, nvc)

        h = rms_norm(h, final_ln, "final_ln")
        logits = mb.linear(x=h, weight=lm_head_c, name="lm_head")

        outputs = [logits]
        for idx in delta_indices:
            outputs.append(new_conv[idx])
            outputs.append(new_rec[idx])
        for idx in attn_indices:
            outputs.append(new_kv[idx][0])
            outputs.append(new_kv[idx][1])
        return tuple(outputs)

    # Generate function with exact parameter names matching input_specs
    sig = ", ".join(param_names)
    pack = "{" + ", ".join(f"'{p}': {p}" for p in param_names) + "}"
    fn_globals = {"_build": _build}
    exec(f"def _full_model({sig}):\n return _build({pack})", fn_globals)
    full_model_fn = fn_globals["_full_model"]

    full_model = mb.program(input_specs=input_specs, opset_version=ct.target.iOS17)(full_model_fn)

    t0 = time.time()
    print("Converting to CoreML...")
    mlmodel = ct.convert(full_model, convert_to="mlprogram",
                         minimum_deployment_target=ct.target.iOS18,
                         compute_precision=ct.precision.FLOAT32)
    print(f"Converted in {time.time() - t0:.1f}s")

    out_path = os.path.join(args.output, "full_model.mlpackage")
    mlmodel.save(out_path)
    print(f"Saved {out_path}")

    # ── Verify against PyTorch ──
    print("\nVerifying against PyTorch...")
    sys.path.insert(0, os.path.dirname(__file__))
    import patches  # noqa
    from functional_deltanet import patch_transformers
    patch_transformers()

    prompt = "Clean up: um so I was like going to the uh store"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        # Prefill tokens 0..N-2 to get prior cache
        prior_out = model(input_ids[:, :-1], use_cache=True)
        prior_cache = prior_out.past_key_values
        # Clone ALL cache state BEFORE single-token forward (which modifies cache in-place)
        saved_conv = [s.clone() if s is not None else None for s in prior_cache.conv_states]
        saved_rec = [s.clone() if s is not None else None for s in prior_cache.recurrent_states]
        saved_key = [s.clone() if s is not None else None for s in prior_cache.key_cache]
        saved_val = [s.clone() if s is not None else None for s in prior_cache.value_cache]
        # Process last token with cache → reference
        last_out = model(input_ids[:, -1:], past_key_values=prior_cache, use_cache=True)
        pt_token = torch.argmax(last_out.logits[0, -1]).item()
    print(f"PyTorch token: {pt_token} ({repr(tokenizer.decode([pt_token]))})")

    # Build MIL inputs from SAVED (pre-modification) cache
    last_hidden = model.model.embed_tokens(input_ids[:, -1:]).squeeze(1).detach().numpy()
    pos = prompt_len - 1
    mil_inputs = {
        "hidden": last_hidden,
        "cos": all_cos[:, pos:pos+1, :].numpy(),
        "sin": all_sin[:, pos:pos+1, :].numpy(),
        "cache_pos": np.array([pos], dtype=np.int32),
        "attn_mask": np.where(np.arange(max_kv) <= pos, 0.0, -1e4).reshape(1, 1, 1, max_kv).astype(np.float32),
    }
    for j, i in enumerate(delta_indices):
        mil_inputs[f"conv_{j}"] = np.ascontiguousarray(saved_conv[i].numpy())
        mil_inputs[f"rec_{j}"] = np.ascontiguousarray(saved_rec[i].numpy())
    for j, i in enumerate(attn_indices):
        k = np.ascontiguousarray(saved_key[i].numpy().astype(np.float32))
        v = np.ascontiguousarray(saved_val[i].numpy().astype(np.float32))
        k_pad = np.zeros((1, attn_nkv, max_kv, attn_hd), dtype=np.float32)
        v_pad = np.zeros((1, attn_nkv, max_kv, attn_hd), dtype=np.float32)
        k_pad[:, :, :k.shape[2], :] = k
        v_pad[:, :, :v.shape[2], :] = v
        mil_inputs[f"kcache_{j}"] = k_pad
        mil_inputs[f"vcache_{j}"] = v_pad

    print("Running MIL model...")
    mil_out = mlmodel.predict(mil_inputs)
    # Find the logits output by name (outputs may be reordered by CoreML)
    spec = mlmodel.get_spec()
    output_names = [o.name for o in spec.description.output]
    print(f"  Output names: {output_names[:5]}...")
    logits_key = [k for k in mil_out.keys() if 'lm_head' in k or mil_out[k].shape[-1] > 10000]
    if logits_key:
        mil_logits = mil_out[logits_key[0]]
    else:
        # Fallback: find the largest output (vocab-sized)
        mil_logits = max(mil_out.values(), key=lambda v: v.shape[-1])
    mil_token = np.argmax(mil_logits.flatten())
    print(f"MIL token:     {mil_token} ({repr(tokenizer.decode([int(mil_token)]))})")
    print(f"Match: {'✓' if mil_token == pt_token else '✗'}")

    pt_logits = last_out.logits[0, -1].detach().numpy()
    diff = np.abs(mil_logits.flatten()[:len(pt_logits)] - pt_logits).max()
    print(f"Max logit diff: {diff:.6f}")


if __name__ == "__main__":
    main()
