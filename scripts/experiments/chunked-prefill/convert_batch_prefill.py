"""Batch prefill model: process N tokens in a single forward pass.

Wraps the HF Qwen3.5 model with:
- Conv1d prepending (initial conv_state from cached system prompt)
- Chunked DeltaNet recurrence with initial rec_state
- Attention KV cache scatter for multi-position writes to padded cache

Usage:
    /tmp/coreml-venv/bin/python scripts/experiments/chunked-prefill/convert_batch_prefill.py \
        Qwen/Qwen3.5-0.8B --output models/batch_prefill_fp16/
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import patches  # noqa: coremltools fixes
from functional_deltanet import patch_transformers
from functional_deltanet_chunked import torch_chunk_gated_delta_rule_functional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DynamicCache,
    apply_rotary_pos_emb,
)


class BatchPrefillWrapper(nn.Module):
    """Process N tokens with initial states from cached system prompt."""

    def __init__(self, hf_model, config, fixed_n=64, max_kv=256, use_neumann=True):
        super().__init__()
        self.embed = hf_model.model.embed_tokens
        self.layers = hf_model.model.layers
        self.norm = hf_model.model.norm
        self.lm_head = hf_model.lm_head

        tc = config.text_config if hasattr(config, "text_config") else config
        self.num_layers = tc.num_hidden_layers
        self.layer_types = tc.layer_types
        self.delta_indices = [i for i in range(self.num_layers) if tc.layer_types[i] == "linear_attention"]
        self.attn_indices = [i for i in range(self.num_layers) if tc.layer_types[i] == "full_attention"]
        self.fixed_n = fixed_n
        self.max_kv = max_kv
        self.use_neumann = use_neumann

        # DeltaNet config
        self.n_v_heads = tc.linear_num_value_heads
        self.k_head_dim = tc.linear_key_head_dim
        self.v_head_dim = tc.linear_value_head_dim
        self.conv_kernel = tc.linear_conv_kernel_dim

    def forward(self, input_ids, seq_len, start_pos, rope_cos, rope_sin, attn_mask, *states):
        """
        Args:
            input_ids: [1, fixed_n] padded token IDs
            seq_len: [1] actual number of tokens
            start_pos: [1] starting position (= system prompt length)
            rope_cos: [1, fixed_n, rope_dim]
            rope_sin: [1, fixed_n, rope_dim]
            attn_mask: [1, 1, fixed_n, max_kv] causal mask
            *states: conv_0, rec_0, conv_1, rec_1, ..., key_0, val_0, ...
        """
        hidden = self.embed(input_ids)  # [1, N, H]

        # Force cos/sin into the graph (prevents trace from treating as constant)
        cos = rope_cos * (rope_cos.sum() * 0.0 + 1.0)
        sin = rope_sin * (rope_sin.sum() * 0.0 + 1.0)

        # Unpack states
        si = 0
        conv_states = {}
        rec_states = {}
        for i in self.delta_indices:
            conv_states[i] = states[si]
            rec_states[i] = states[si + 1]
            si += 2
        n_delta_states = len(self.delta_indices) * 2
        kv_caches = {}
        for j, i in enumerate(self.attn_indices):
            kv_caches[i] = (states[n_delta_states + j * 2], states[n_delta_states + j * 2 + 1])

        # Process layers — both DeltaNet and attention handled manually
        new_conv = {}
        new_rec = {}
        new_kv = {}
        for i, layer in enumerate(self.layers):
            if self.layer_types[i] == "linear_attention":
                hidden, nc, nr = self._deltanet_forward(
                    layer, hidden, (cos, sin), conv_states[i], rec_states[i]
                )
                new_conv[i] = nc
                new_rec[i] = nr
            else:
                kc, vc = kv_caches[i]
                hidden, nkc, nvc = self._attention_forward(
                    layer, hidden, cos, sin, kc, vc, attn_mask, start_pos
                )
                new_kv[i] = (nkc, nvc)

        # Logits from last real position only
        # For tracing, always compute logits for all positions, then slice later in Python
        logits = self.lm_head(self.norm(hidden))  # [1, N, vocab]

        # Pack output states (same format as decode model)
        updated = []
        for i in self.delta_indices:
            updated.append(new_conv[i])
            updated.append(new_rec[i])
        for i in self.attn_indices:
            updated.append(new_kv[i][0])
            updated.append(new_kv[i][1])

        return (logits,) + tuple(updated)

    def _deltanet_forward(self, layer, hidden_states, position_embeddings, init_conv, init_rec):
        """DeltaNet layer forward with initial state support for batch prefill."""
        dn = layer.linear_attn  # Qwen3_5GatedDeltaNet module
        batch_size, seq_len, _ = hidden_states.shape

        # Input layernorm + attention
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        # QKV projection
        mixed_qkv = dn.in_proj_qkv(hidden_states).transpose(1, 2)  # [B, D, N]

        # Conv1d with initial state prepended
        padded = torch.cat([init_conv, mixed_qkv], dim=-1)  # [B, D, conv_k + N]
        conv_weight = dn.conv1d.weight  # [D, 1, conv_k]
        conv_out = F.conv1d(padded, conv_weight, bias=dn.conv1d.bias, padding=0, groups=mixed_qkv.shape[1])
        mixed_qkv = F.silu(conv_out[:, :, -seq_len:])  # last N outputs
        new_conv = padded[:, :, -self.conv_kernel:]  # new conv state

        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, N, D]

        # Split Q, K, V
        query, key, value = torch.split(
            mixed_qkv,
            [dn.key_dim, dn.key_dim, dn.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, self.k_head_dim)
        key = key.reshape(batch_size, seq_len, -1, self.k_head_dim)
        value = value.reshape(batch_size, seq_len, -1, self.v_head_dim)

        # Z gate, beta, g — z uses the layernormed hidden (same as QKV)
        z = dn.in_proj_z(hidden_states)  # hidden_states is post-layernorm here
        z = z.reshape(batch_size, seq_len, -1, self.v_head_dim)
        beta = dn.in_proj_b(hidden_states).sigmoid()
        a = dn.in_proj_a(hidden_states)
        g = -dn.A_log.float().exp() * F.softplus(a.float() + dn.dt_bias)

        # GQA repeat if needed
        if self.n_v_heads // (dn.key_dim // self.k_head_dim) > 1:
            n_k_heads = dn.key_dim // self.k_head_dim
            repeat_factor = self.n_v_heads // n_k_heads
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)

        # Chunked DeltaNet recurrence with initial state
        core_attn_out, new_rec = torch_chunk_gated_delta_rule_functional(
            query, key, value, g=g, beta=beta,
            chunk_size=64,
            initial_state=init_rec,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_neumann=self.use_neumann,
        )

        # Gated RMSNorm
        core_attn_out = core_attn_out.reshape(-1, self.v_head_dim)
        z_flat = z.reshape(-1, self.v_head_dim)
        core_attn_out = dn.norm(core_attn_out, z_flat)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        # Output projection + residual
        hidden_states = residual + dn.out_proj(core_attn_out)

        # MLP
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + layer.mlp(hidden_states)

        return hidden_states, new_conv, new_rec

    def _attention_forward(self, layer, hidden_states, cos, sin, key_cache, value_cache, attn_mask, start_pos):
        """Attention layer forward with manual KV cache scatter for batch prefill."""
        attn = layer.self_attn
        batch_size, seq_len, _ = hidden_states.shape
        n_q = attn.config.num_attention_heads
        n_kv = attn.config.num_key_value_heads
        hd = attn.head_dim
        ng = n_q // n_kv

        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        # Q(+gate), K, V projections — Q/gate interleaved per head
        qg = attn.q_proj(hidden_states)
        qg_heads = qg.reshape(1, seq_len, n_q, hd * 2)  # [B, N, n_q, hd*2]
        query, gate = qg_heads.chunk(2, dim=-1)  # each [B, N, n_q, hd]
        gate = gate.reshape(1, seq_len, n_q * hd)  # [B, N, n_q*hd]

        query = attn.q_norm(query).transpose(1, 2)  # [B, n_q, N, hd]
        key_new = attn.k_norm(
            attn.k_proj(hidden_states).reshape(1, seq_len, n_kv, hd)
        ).transpose(1, 2)  # [B, n_kv, N, hd]
        value_new = attn.v_proj(hidden_states).reshape(1, seq_len, n_kv, hd).transpose(1, 2)

        # RoPE
        query, key_new = apply_rotary_pos_emb(query, key_new, cos, sin)

        # Write new K, V into padded cache using scatter
        # Use torch.cat for index broadcasting (avoids expand op issues in coremltools)
        sp = start_pos[0]
        offset = torch.arange(self.fixed_n, device=hidden_states.device)
        idx = (offset + sp).long().reshape(1, 1, self.fixed_n, 1)
        # Broadcast with cat (coremltools-friendly, avoids expand)
        idx = torch.cat([idx] * n_kv, dim=1)     # [1, n_kv, N, 1]
        idx = torch.cat([idx] * hd, dim=3)        # [1, n_kv, N, hd]
        new_key_cache = key_cache.scatter(2, idx, key_new)
        new_value_cache = value_cache.scatter(2, idx, value_new)

        # GQA repeat
        k_rep = new_key_cache.repeat_interleave(ng, dim=1)  # [B, n_q, max_kv, hd]
        v_rep = new_value_cache.repeat_interleave(ng, dim=1)

        # Attention: Q @ K^T * scale + mask → softmax → @ V
        scale = hd ** -0.5
        attn_weights = torch.matmul(query, k_rep.transpose(2, 3)) * scale  # [B, n_q, N, max_kv]
        attn_weights = attn_weights + attn_mask  # [B, 1, N, max_kv] broadcasts
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, v_rep)  # [B, n_q, N, hd]

        # Reshape + gate + output projection
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)  # [B, N, n_q*hd]
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = attn.o_proj(attn_output)

        hidden_states = residual + attn_output

        # MLP
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + layer.mlp(hidden_states)

        return hidden_states, new_key_cache, new_value_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--output", default="models/batch_prefill_fp16/")
    parser.add_argument("--fixed-n", type=int, default=64)
    parser.add_argument("--max-kv", type=int, default=256)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading {args.model_id}...")
    patch_transformers()  # Need this for attention cache update and RMSNormGated

    config = AutoConfig.from_pretrained(args.model_id)
    config.text_config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32, config=config)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tc = config.text_config

    # Dimensions
    hidden_size = tc.hidden_size
    conv_dim = tc.linear_key_head_dim * tc.linear_num_key_heads * 2 + tc.linear_value_head_dim * tc.linear_num_value_heads
    conv_kernel = tc.linear_conv_kernel_dim
    n_v_heads = tc.linear_num_value_heads
    k_head_dim = tc.linear_key_head_dim
    v_head_dim = tc.linear_value_head_dim
    n_kv_heads = tc.num_key_value_heads
    head_dim = tc.head_dim

    with torch.no_grad():
        dummy = torch.zeros(1, args.max_kv, hidden_size)
        all_cos, all_sin = model.model.rotary_emb(dummy, torch.arange(args.max_kv).unsqueeze(0))
    rope_dim = all_cos.shape[-1]

    # Use Neumann for FP32 (fast), row-by-row for FP16 (numerically stable)
    use_neumann = args.fp32
    wrapper = BatchPrefillWrapper(model, config, fixed_n=args.fixed_n, max_kv=args.max_kv, use_neumann=use_neumann)
    print(f"Resolution: {'Neumann series' if use_neumann else 'row-by-row (FP16 safe)'}")
    wrapper.eval()

    delta_indices = wrapper.delta_indices
    attn_indices = wrapper.attn_indices
    print(f"Layers: {len(delta_indices)} DeltaNet + {len(attn_indices)} attention")

    # ── PyTorch verification ──
    print("\nPyTorch verification...")
    prompt = "Clean up: um so I was like going to the uh store"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]

    # Get reference from full forward
    with torch.no_grad():
        pt_ref = model(input_ids).logits[0, -1].numpy().copy()
    pt_token = int(np.argmax(pt_ref))
    print(f"  PT reference token: {pt_token} ({repr(tokenizer.decode([pt_token]))})")

    # Get system prompt cache (first 5 tokens as "system prompt")
    split = 5
    with torch.no_grad():
        cache_out = model(input_ids[:, :split], use_cache=True)
        sys_cache = cache_out.past_key_values

    # Batch prefill the remaining tokens
    remaining = input_ids[:, split:]
    n_remaining = remaining.shape[1]
    padded_ids = F.pad(remaining, (0, args.fixed_n - n_remaining))

    cos = all_cos[:, split:split + args.fixed_n, :]
    sin = all_sin[:, split:split + args.fixed_n, :]

    # Build attention mask: [1, 1, N, max_kv]
    # Position i (of the new tokens) can attend to:
    #   - cached positions 0..split-1
    #   - new positions 0..i
    # Padding positions (i >= n_remaining) attend to position 0 only (avoids NaN softmax)
    mask = torch.full((1, 1, args.fixed_n, args.max_kv), float("-inf"))
    for i in range(args.fixed_n):
        if i < n_remaining:
            mask[0, 0, i, :split + i + 1] = 0.0
        else:
            mask[0, 0, i, 0] = 0.0  # attend to position 0 (avoids all-inf → NaN)

    # Pack initial states
    init_states = []
    for i in delta_indices:
        init_states.append(sys_cache.conv_states[i].contiguous())
        init_states.append(sys_cache.recurrent_states[i].contiguous())
    for i in attn_indices:
        k = sys_cache.key_cache[i].contiguous()
        v = sys_cache.value_cache[i].contiguous()
        k_pad = torch.zeros(1, n_kv_heads, args.max_kv, head_dim)
        v_pad = torch.zeros(1, n_kv_heads, args.max_kv, head_dim)
        k_pad[:, :, :k.shape[2], :] = k
        v_pad[:, :, :v.shape[2], :] = v
        init_states.append(k_pad)
        init_states.append(v_pad)

    with torch.no_grad():
        result = wrapper(
            padded_ids,
            torch.tensor([n_remaining]),
            torch.tensor([split]),
            cos, sin, mask,
            *init_states,
        )

    batch_logits = result[0][0, n_remaining - 1].numpy()
    batch_token = int(np.argmax(batch_logits))
    diff = np.abs(batch_logits - pt_ref).max()
    print(f"  Batch prefill token: {batch_token} ({repr(tokenizer.decode([batch_token]))})")
    print(f"  Match: {'PASS' if batch_token == pt_token else 'FAIL'}")
    print(f"  Max logit diff: {diff:.6f}")

    if batch_token != pt_token:
        print("ABORTING — PyTorch verification failed")
        return

    # ── Trace ──
    print("\nTracing...")
    sample_inputs = (
        padded_ids,
        torch.tensor([n_remaining]),
        torch.tensor([split]),
        cos, sin, mask,
        *init_states,
    )
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, sample_inputs, strict=False)
    trace_time = time.time() - t0
    print(f"  Traced in {trace_time:.1f}s")

    # Verify trace matches eager
    with torch.no_grad():
        traced_result = traced(*sample_inputs)
    traced_logits = traced_result[0][0, n_remaining - 1].numpy()
    traced_token = int(np.argmax(traced_logits))
    trace_diff = np.abs(traced_logits - pt_ref).max()
    print(f"  Traced token: {traced_token} ({repr(tokenizer.decode([traced_token]))})")
    print(f"  Trace vs PT diff: {trace_diff:.6f}")

    if traced_token != pt_token:
        print("WARNING — traced model disagrees with PT reference")

    # ── Convert to CoreML ──
    precision = ct.precision.FLOAT32 if args.fp32 else ct.precision.FLOAT16
    print(f"\nConverting to CoreML ({'FP32' if args.fp32 else 'FP16'})...")

    # Build input names for CoreML
    input_names = ["input_ids", "seq_len", "start_pos", "rope_cos", "rope_sin", "attn_mask"]
    for j in range(len(delta_indices)):
        input_names.extend([f"conv_state_{j}", f"rec_state_{j}"])
    for j in range(len(attn_indices)):
        input_names.extend([f"key_cache_{j}", f"value_cache_{j}"])

    # Build ct input types
    ct_inputs = [
        ct.TensorType(name="input_ids", shape=padded_ids.shape, dtype=np.int32),
        ct.TensorType(name="seq_len", shape=(1,), dtype=np.int32),
        ct.TensorType(name="start_pos", shape=(1,), dtype=np.int32),
        ct.TensorType(name="rope_cos", shape=cos.shape),
        ct.TensorType(name="rope_sin", shape=sin.shape),
        ct.TensorType(name="attn_mask", shape=mask.shape),
    ]
    for j in range(len(delta_indices)):
        ct_inputs.append(ct.TensorType(name=f"conv_state_{j}", shape=init_states[j * 2].shape))
        ct_inputs.append(ct.TensorType(name=f"rec_state_{j}", shape=init_states[j * 2 + 1].shape))
    n_delta_s = len(delta_indices) * 2
    for j in range(len(attn_indices)):
        ct_inputs.append(ct.TensorType(name=f"key_cache_{j}", shape=init_states[n_delta_s + j * 2].shape))
        ct_inputs.append(ct.TensorType(name=f"value_cache_{j}", shape=init_states[n_delta_s + j * 2 + 1].shape))

    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=precision,
    )
    convert_time = time.time() - t0
    print(f"  Converted in {convert_time:.1f}s")

    out_path = os.path.join(args.output, "model.mlpackage")
    mlmodel.save(out_path)
    print(f"  Saved {out_path}")

    # ── CoreML verification ──
    print("\nCoreML verification...")
    mil_inputs = {}
    mil_inputs["input_ids"] = padded_ids.numpy().astype(np.int32)
    mil_inputs["seq_len"] = np.array([n_remaining], dtype=np.int32)
    mil_inputs["start_pos"] = np.array([split], dtype=np.int32)
    mil_inputs["rope_cos"] = cos.numpy().astype(np.float32)
    mil_inputs["rope_sin"] = sin.numpy().astype(np.float32)
    mil_inputs["attn_mask"] = mask.numpy().astype(np.float32)
    for j in range(len(delta_indices)):
        mil_inputs[f"conv_state_{j}"] = init_states[j * 2].numpy().astype(np.float32)
        mil_inputs[f"rec_state_{j}"] = init_states[j * 2 + 1].numpy().astype(np.float32)
    for j in range(len(attn_indices)):
        mil_inputs[f"key_cache_{j}"] = init_states[n_delta_s + j * 2].numpy().astype(np.float32)
        mil_inputs[f"value_cache_{j}"] = init_states[n_delta_s + j * 2 + 1].numpy().astype(np.float32)

    mil_out = mlmodel.predict(mil_inputs)
    # Find logits output
    logits_key = [k for k in mil_out.keys() if mil_out[k].shape[-1] > 10000]
    if logits_key:
        mil_logits = mil_out[logits_key[0]]
    else:
        mil_logits = max(mil_out.values(), key=lambda v: v.shape[-1] if hasattr(v, 'shape') else 0)

    # Extract logits at position n_remaining-1
    if len(mil_logits.shape) == 3:
        mil_last = mil_logits[0, n_remaining - 1]
    elif len(mil_logits.shape) == 2:
        mil_last = mil_logits[0]
    else:
        mil_last = mil_logits.flatten()

    mil_token = int(np.argmax(mil_last))
    mil_diff = np.abs(mil_last[:len(pt_ref)] - pt_ref).max()
    print(f"  CoreML token: {mil_token} ({repr(tokenizer.decode([mil_token]))})")
    print(f"  CoreML vs PT diff: {mil_diff:.6f}")
    print(f"  Match: {'PASS' if mil_token == pt_token else 'FAIL'}")

    # ── Summary ──
    print(f"\n{'=' * 50}")
    print(f"Batch prefill model: {args.fixed_n} tokens, {'FP16' if not args.fp32 else 'FP32'}")
    print(f"Trace: {trace_time:.1f}s, Convert: {convert_time:.1f}s")
    print(f"Token match: PT={pt_token}, Traced={traced_token}, CoreML={mil_token}")
    print(f"Logit diff: Traced={trace_diff:.6f}, CoreML={mil_diff:.6f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
