#!/usr/bin/env python3
"""Verify padded KV cache + external mask produces correct decode output.

Compares three approaches:
  1. HF native decode (model.forward with growing cache) — ground truth
  2. Manual layer-by-layer decode with growing cache — isolates layer calling
  3. Manual layer-by-layer decode with PADDED KV cache + external mask — the CoreML approach

If #2 matches #1, our layer-calling is correct.
If #3 matches #2, our padded KV + mask approach is correct.

Key insight: get_seq_length() is NOT called inside individual layers.
The attention mask is passed externally, completely bypassing get_seq_length().

Usage:
    /tmp/coreml-venv/bin/python scripts/coreml-convert/verify_kv_decode.py [model_id]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import patches  # noqa: E402, F401

import torch
from functional_deltanet import patch_transformers

patch_transformers()

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache  # noqa: E402


def build_cache(num_layers, layer_types, conv_states_dict, recurrent_states_dict,
                key_cache_dict=None, value_cache_dict=None):
    """Build a Qwen3_5DynamicCache with explicit state tensors."""
    cache = Qwen3_5DynamicCache.__new__(Qwen3_5DynamicCache)
    cache.conv_states = [None] * num_layers
    cache.recurrent_states = [None] * num_layers
    cache.key_cache = [None] * num_layers
    cache.value_cache = [None] * num_layers
    cache.layer_types = layer_types
    cache.transformer_layers = [j for j in range(num_layers) if layer_types[j] == "full_attention"]
    cache.last_linear_layer = num_layers - 1 - layer_types[::-1].index("linear_attention")

    for i in range(num_layers):
        if layer_types[i] == "linear_attention":
            cache.conv_states[i] = conv_states_dict[i]
            cache.recurrent_states[i] = recurrent_states_dict[i]
        elif key_cache_dict and i in key_cache_dict:
            cache.key_cache[i] = key_cache_dict[i]
            cache.value_cache[i] = value_cache_dict[i]

    return cache


def manual_decode_step(model, layer_types, hidden, cos, sin, cache, pos, mask=None):
    """Run one decode step through all layers manually."""
    for i, layer in enumerate(model.model.layers):
        if layer_types[i] == "full_attention":
            result = layer(
                hidden,
                position_embeddings=(cos, sin),
                past_key_values=cache,
                cache_position=torch.tensor([pos]),
                attention_mask=mask,
            )
        else:
            result = layer(
                hidden,
                position_embeddings=(cos, sin),
                past_key_values=cache,
                cache_position=torch.tensor([pos]),
            )
        hidden = result[0] if isinstance(result, tuple) else result

    logits = model.lm_head(model.model.norm(hidden))
    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", nargs="?", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--num-tokens", type=int, default=10)
    args = parser.parse_args()

    print(f"Loading {args.model_id}...")
    config = AutoConfig.from_pretrained(args.model_id)
    config.text_config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32, config=config)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    tc = config.text_config
    layer_types = tc.layer_types
    num_layers = tc.num_hidden_layers

    prompt = "Clean up: um so I was like going to the uh store"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]
    print(f"Prompt: {repr(prompt)} ({prompt_len} tokens)")

    # Pre-compute RoPE for all positions
    with torch.no_grad():
        dummy = torch.zeros(1, args.max_seq_len, tc.hidden_size)
        all_pos = torch.arange(args.max_seq_len).unsqueeze(0)
        all_cos, all_sin = model.model.rotary_emb(dummy, all_pos)

    # ═══ Approach 1: HF native decode ═══
    print("\n=== Approach 1: HF native (model.forward, growing cache) ===")
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        hf_cache = out.past_key_values
        first_token = torch.argmax(out.logits[0, -1]).item()

        # Generate num_tokens DECODE outputs (not including the prefill token)
        hf_tokens = []
        current = torch.tensor([[first_token]])
        for _ in range(args.num_tokens):
            out = model(current, past_key_values=hf_cache, use_cache=True)
            hf_cache = out.past_key_values
            tok = torch.argmax(out.logits[0, -1]).item()
            hf_tokens.append(tok)
            current = torch.tensor([[tok]])

    print(f"  Tokens: {hf_tokens}")
    print(f"  Text:   {tokenizer.decode(hf_tokens)}")

    # ═══ Prefill (shared by approaches 2 and 3) ═══
    with torch.no_grad():
        prefill_out = model(input_ids, use_cache=True)
        prefill_cache = prefill_out.past_key_values

    assert torch.argmax(prefill_out.logits[0, -1]).item() == first_token, "Prefill first token mismatch"

    # ═══ Approach 2: Manual layer-by-layer with GROWING cache ═══
    print("\n=== Approach 2: Manual layers, growing cache (no mask) ===")

    conv_s = {i: prefill_cache.conv_states[i].clone() for i in range(num_layers) if layer_types[i] == "linear_attention"}
    rec_s = {i: prefill_cache.recurrent_states[i].clone() for i in range(num_layers) if layer_types[i] == "linear_attention"}
    key_c = {i: prefill_cache.key_cache[i].clone() for i in range(num_layers) if layer_types[i] == "full_attention"}
    val_c = {i: prefill_cache.value_cache[i].clone() for i in range(num_layers) if layer_types[i] == "full_attention"}

    growing_tokens = []
    current_token = first_token
    pos = prompt_len

    with torch.no_grad():
        for step in range(args.num_tokens):
            cache = build_cache(num_layers, layer_types, conv_s, rec_s, key_c, val_c)
            hidden = model.model.embed_tokens(torch.tensor([[current_token]]))
            cos = all_cos[:, pos:pos+1, :]
            sin = all_sin[:, pos:pos+1, :]

            logits = manual_decode_step(model, layer_types, hidden, cos, sin, cache, pos)
            tok = torch.argmax(logits[0, -1]).item()
            growing_tokens.append(tok)

            # Extract updated states
            for i in range(num_layers):
                if layer_types[i] == "linear_attention":
                    conv_s[i] = cache.conv_states[i]
                    rec_s[i] = cache.recurrent_states[i]
                else:
                    key_c[i] = cache.key_cache[i]
                    val_c[i] = cache.value_cache[i]

            current_token = tok
            pos += 1

    print(f"  Tokens: {growing_tokens}")
    print(f"  Text:   {tokenizer.decode(growing_tokens)}")
    grow_match = sum(h == g for h, g in zip(hf_tokens, growing_tokens))
    print(f"  vs HF:  {grow_match}/{args.num_tokens} match")

    # ═══ Approach 3: Manual layer-by-layer with PADDED KV cache + external mask ═══
    print(f"\n=== Approach 3: Manual layers, PADDED KV cache ({args.max_seq_len}), external mask ===")

    conv_s = {i: prefill_cache.conv_states[i].clone() for i in range(num_layers) if layer_types[i] == "linear_attention"}
    rec_s = {i: prefill_cache.recurrent_states[i].clone() for i in range(num_layers) if layer_types[i] == "linear_attention"}

    # Pad KV cache to max_seq_len
    padded_key = {}
    padded_val = {}
    for i in range(num_layers):
        if layer_types[i] == "full_attention":
            k = prefill_cache.key_cache[i]
            v = prefill_cache.value_cache[i]
            n_kv_heads, head_dim = k.shape[1], k.shape[3]
            k_pad = torch.zeros(1, n_kv_heads, args.max_seq_len, head_dim)
            v_pad = torch.zeros(1, n_kv_heads, args.max_seq_len, head_dim)
            k_pad[:, :, :prompt_len, :] = k
            v_pad[:, :, :prompt_len, :] = v
            padded_key[i] = k_pad
            padded_val[i] = v_pad

    padded_tokens = []
    current_token = first_token
    pos = prompt_len

    with torch.no_grad():
        for step in range(args.num_tokens):
            # External attention mask: 0 for valid, -inf for padding
            mask = torch.full((1, 1, 1, args.max_seq_len), float("-inf"))
            mask[0, 0, 0, :pos + 1] = 0.0

            cache = build_cache(num_layers, layer_types, conv_s, rec_s, padded_key, padded_val)
            hidden = model.model.embed_tokens(torch.tensor([[current_token]]))
            cos = all_cos[:, pos:pos+1, :]
            sin = all_sin[:, pos:pos+1, :]

            logits = manual_decode_step(model, layer_types, hidden, cos, sin, cache, pos, mask=mask)
            tok = torch.argmax(logits[0, -1]).item()
            padded_tokens.append(tok)

            # Extract updated states (KV cache was updated via scatter in cache.update)
            for i in range(num_layers):
                if layer_types[i] == "linear_attention":
                    conv_s[i] = cache.conv_states[i]
                    rec_s[i] = cache.recurrent_states[i]
                else:
                    padded_key[i] = cache.key_cache[i]
                    padded_val[i] = cache.value_cache[i]

            current_token = tok
            pos += 1

    print(f"  Tokens: {padded_tokens}")
    print(f"  Text:   {tokenizer.decode(padded_tokens)}")
    pad_match = sum(h == p for h, p in zip(hf_tokens, padded_tokens))
    print(f"  vs HF:  {pad_match}/{args.num_tokens} match")

    # ═══ Summary ═══
    print("\n" + "=" * 60)
    if hf_tokens == growing_tokens == padded_tokens:
        print(f"PERFECT MATCH: all 3 approaches produce identical {args.num_tokens} tokens")
        print("Padded KV cache with external mask is CORRECT.")
        print("Ready for CoreML conversion.")
    else:
        if hf_tokens != growing_tokens:
            print("ISSUE: Manual growing cache != HF native (layer-calling bug)")
            for i, (h, g) in enumerate(zip(hf_tokens, growing_tokens)):
                if h != g:
                    print(f"  Step {i}: HF={h} ({repr(tokenizer.decode([h]))}), grow={g} ({repr(tokenizer.decode([g]))})")
        if growing_tokens != padded_tokens:
            print("ISSUE: Padded cache != growing cache (KV padding/mask bug)")
            for i, (g, p) in enumerate(zip(growing_tokens, padded_tokens)):
                if g != p:
                    print(f"  Step {i}: grow={g} ({repr(tokenizer.decode([g]))}), padded={p} ({repr(tokenizer.decode([p]))})")


if __name__ == "__main__":
    main()
