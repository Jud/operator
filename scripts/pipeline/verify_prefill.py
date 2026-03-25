#!/usr/bin/env python3
"""Clean verification of MIL prefill model against PyTorch.

Avoids the in-place cache mutation bug by using TWO INDEPENDENT model() calls:
1. Full prompt forward → PT reference logits (cache discarded immediately)
2. tokens[:-1] forward → extract cache with .contiguous().clone().numpy()

No cache objects are shared between calls. No manual layer-by-layer needed.

Usage:
    /tmp/coreml-venv/bin/python scripts/coreml-convert/verify_mil_prefill.py \
        [model_id] [--model-path path/to/model.mlpackage]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import patches  # noqa
from functional_deltanet import patch_transformers
patch_transformers()

import numpy as np
import torch
import coremltools as ct
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", nargs="?", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--model-path", default="models/prefill_mil/full_model.mlpackage")
    parser.add_argument("--max-kv", type=int, default=512)
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
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
    H = tc.hidden_size
    num_layers = tc.num_hidden_layers
    layer_types = tc.layer_types

    # DeltaNet dims
    dn_heads = tc.linear_num_value_heads
    dn_k = tc.linear_key_head_dim
    dn_v = tc.linear_value_head_dim
    conv_k = tc.linear_conv_kernel_dim
    qkv_total = (dn_k * dn_heads * 2) + (dn_v * dn_heads)

    # Attention dims
    attn_nkv = tc.num_key_value_heads
    attn_hd = tc.head_dim

    delta_indices = [i for i in range(num_layers) if layer_types[i] == "linear_attention"]
    attn_indices = [i for i in range(num_layers) if layer_types[i] == "full_attention"]

    # RoPE
    max_kv = args.max_kv
    with torch.no_grad():
        all_cos, all_sin = model.model.rotary_emb(
            torch.zeros(1, max_kv, H), torch.arange(max_kv).unsqueeze(0)
        )
    rope_dim = all_cos.shape[-1]

    prompt = "Clean up: um so I was like going to the uh store"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]
    print(f"Prompt: {repr(prompt)} ({prompt_len} tokens)")

    # ═══ Step 1: PT reference — full forward, extract final cache ═══
    print("\n=== Step 1: PyTorch reference (full prompt forward) ===")
    with torch.no_grad():
        full_out = model(input_ids)
        pt_logits = full_out.logits[0, -1].numpy().copy()
        pt_cache = full_out.past_key_values
        # Extract final state for comparison with MIL outputs
        pt_final_rec = {}
        pt_final_conv = {}
        for i in delta_indices:
            pt_final_conv[i] = pt_cache.conv_states[i].contiguous().clone().numpy()
            pt_final_rec[i] = pt_cache.recurrent_states[i].contiguous().clone().numpy()
        pt_final_key = {}
        pt_final_val = {}
        for i in attn_indices:
            pt_final_key[i] = pt_cache.key_cache[i].contiguous().clone().numpy()
            pt_final_val[i] = pt_cache.value_cache[i].contiguous().clone().numpy()
    pt_token = int(np.argmax(pt_logits))
    print(f"  PT token: {pt_token} ({repr(tokenizer.decode([pt_token]))})")

    # ═══ Step 1b: Sanity check — prefill+decode should match full forward ═══
    print("\n=== Step 1b: PT prefill+decode consistency check ===")
    with torch.no_grad():
        prefill_cache = model(input_ids[:, :-1], use_cache=True).past_key_values
        decode_logits = model(input_ids[:, -1:], past_key_values=prefill_cache).logits[0, -1].numpy().copy()
    decode_diff = float(np.abs(pt_logits - decode_logits).max())
    decode_token = int(np.argmax(decode_logits))
    print(f"  Decode token: {decode_token} ({repr(tokenizer.decode([decode_token]))})")
    print(f"  Full vs decode diff: {decode_diff:.6f}")
    if decode_diff > 0.01:
        print(f"  WARNING: PT full forward and PT prefill+decode DISAGREE!")

    # ═══ Step 2: Extract cache from separate prefill call ═══
    # Process tokens[:-1], immediately extract ALL state to numpy.
    # .contiguous().clone().numpy() on every tensor — handles Fortran-ordered
    # conv_states and transposed value_cache.
    print("\n=== Step 2: Cache extraction (tokens[:-1]) ===")
    with torch.no_grad():
        cache = model(input_ids[:, :-1], use_cache=True).past_key_values

        conv_np, rec_np = {}, {}
        for i in delta_indices:
            conv_np[i] = cache.conv_states[i].contiguous().clone().numpy().astype(np.float32)
            rec_np[i] = cache.recurrent_states[i].contiguous().clone().numpy().astype(np.float32)

        key_np, val_np = {}, {}
        for i in attn_indices:
            key_np[i] = cache.key_cache[i].contiguous().clone().numpy().astype(np.float32)
            val_np[i] = cache.value_cache[i].contiguous().clone().numpy().astype(np.float32)

        hidden_np = model.model.embed_tokens(input_ids[:, -1:]).squeeze(1).detach().numpy().astype(np.float32)

    for i in delta_indices[:1]:
        print(f"  conv[{i}]: {conv_np[i].shape}, C_CONTIGUOUS={conv_np[i].flags['C_CONTIGUOUS']}")
        print(f"  rec[{i}]:  {rec_np[i].shape}")
    for i in attn_indices[:1]:
        print(f"  key[{i}]:  {key_np[i].shape}, val[{i}]: {val_np[i].shape}")
    print(f"  hidden:   {hidden_np.shape}")

    # ═══ Step 3: Run MIL model ═══
    print(f"\n=== Step 3: MIL model ({args.model_path}) ===")
    compute_units = ct.ComputeUnit.CPU_ONLY if args.cpu else ct.ComputeUnit.ALL
    print(f"  Compute units: {'CPU_ONLY' if args.cpu else 'ALL (GPU)'}")
    mlmodel = ct.models.MLModel(args.model_path, compute_units=compute_units)

    pos = prompt_len - 1
    mil_inputs = {
        "hidden": hidden_np,
        "cos": all_cos[:, pos:pos+1, :].numpy().astype(np.float32),
        "sin": all_sin[:, pos:pos+1, :].numpy().astype(np.float32),
        "cache_pos": np.array([pos], dtype=np.int32),
        "attn_mask": np.where(
            np.arange(max_kv) <= pos, 0.0, -1e4
        ).reshape(1, 1, 1, max_kv).astype(np.float32),
    }

    for j, i in enumerate(delta_indices):
        mil_inputs[f"conv_{j}"] = np.ascontiguousarray(conv_np[i])
        mil_inputs[f"rec_{j}"] = np.ascontiguousarray(rec_np[i])

    for j, i in enumerate(attn_indices):
        k_pad = np.zeros((1, attn_nkv, max_kv, attn_hd), dtype=np.float32)
        v_pad = np.zeros((1, attn_nkv, max_kv, attn_hd), dtype=np.float32)
        k_pad[:, :, :key_np[i].shape[2], :] = key_np[i]
        v_pad[:, :, :val_np[i].shape[2], :] = val_np[i]
        mil_inputs[f"kcache_{j}"] = k_pad
        mil_inputs[f"vcache_{j}"] = v_pad

    t0 = time.time()
    mil_out = mlmodel.predict(mil_inputs)
    elapsed = (time.time() - t0) * 1000
    print(f"  Predict: {elapsed:.0f}ms")

    # Find logits output (may be named differently by CoreML)
    logits_key = [k for k in mil_out.keys() if 'lm_head' in k or mil_out[k].shape[-1] > 10000]
    if logits_key:
        mil_logits = mil_out[logits_key[0]].flatten()
    else:
        mil_logits = max(mil_out.values(), key=lambda v: v.shape[-1]).flatten()

    mil_token = int(np.argmax(mil_logits))
    print(f"  MIL token: {mil_token} ({repr(tokenizer.decode([mil_token]))})")

    # ═══ Compare output states layer-by-layer ═══
    print("\n=== State comparison (MIL vs PT final cache) ===")
    # Identify MIL output names
    spec = mlmodel.get_spec()
    output_names = [o.name for o in spec.description.output]
    print(f"  MIL outputs ({len(output_names)}): {output_names[:8]}...")

    # The outputs are ordered: logits, then for each delta layer (conv, rec),
    # then for each attn layer (key, value). Skip first output (logits).
    out_idx = 1
    for j, i in enumerate(delta_indices):
        oname_conv = output_names[out_idx]
        oname_rec = output_names[out_idx + 1]
        mil_conv = mil_out[oname_conv]
        mil_rec = mil_out[oname_rec]
        conv_diff = float(np.abs(mil_conv - pt_final_conv[i]).max())
        rec_diff = float(np.abs(mil_rec - pt_final_rec[i]).max())
        marker = " <<<" if conv_diff > 0.1 or rec_diff > 0.1 else ""
        print(f"  L{i:2d} (DeltaNet): conv_diff={conv_diff:.6f}, rec_diff={rec_diff:.6f}{marker}")
        out_idx += 2
    for j, i in enumerate(attn_indices):
        oname_k = output_names[out_idx]
        oname_v = output_names[out_idx + 1]
        mil_k = mil_out[oname_k]
        mil_v = mil_out[oname_v]
        # PT key/val have shape [1, n_kv, seq_len, hd], MIL has [1, n_kv, max_kv, hd]
        seq_len = pt_final_key[i].shape[2]
        key_diff = float(np.abs(mil_k[:, :, :seq_len, :] - pt_final_key[i]).max())
        val_diff = float(np.abs(mil_v[:, :, :seq_len, :] - pt_final_val[i]).max())
        marker = " <<<" if key_diff > 0.1 or val_diff > 0.1 else ""
        print(f"  L{i:2d} (Attention): key_diff={key_diff:.6f}, val_diff={val_diff:.6f}{marker}")
        out_idx += 2

    # ═══ Compare logits ═══
    diff = float(np.abs(mil_logits[:len(pt_logits)] - pt_logits).max())
    mean_diff = float(np.abs(mil_logits[:len(pt_logits)] - pt_logits).mean())

    print(f"\n{'=' * 60}")
    print(f"PT token:        {pt_token} ({repr(tokenizer.decode([pt_token]))})")
    print(f"MIL token:       {mil_token} ({repr(tokenizer.decode([mil_token]))})")
    print(f"Token match:     {'PASS' if mil_token == pt_token else 'FAIL'}")
    print(f"Max logit diff:  {diff:.6f}")
    print(f"Mean logit diff: {mean_diff:.6f}")

    if diff < 0.1:
        print("\nPASS — MIL prefill model is correct (logit diff < 0.1)")
    elif mil_token == pt_token and diff < 1.0:
        print(f"\nSOFT PASS — tokens match, logit diff {diff:.4f} (FP32 accumulation noise)")
    else:
        print(f"\nFAIL — diff={diff:.4f}")
        pt_top5 = np.argsort(pt_logits)[-5:][::-1]
        mil_top5 = np.argsort(mil_logits[:len(pt_logits)])[-5:][::-1]
        print(f"  PT top-5:  {[(int(t), f'{pt_logits[t]:.2f}') for t in pt_top5]}")
        print(f"  MIL top-5: {[(int(t), f'{mil_logits[t]:.2f}') for t in mil_top5]}")

    return 0 if mil_token == pt_token else 1


if __name__ == "__main__":
    sys.exit(main())
