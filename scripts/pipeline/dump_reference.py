#!/usr/bin/env python3
"""Dump per-layer reference activations from PyTorch for MPSGraph verification.

Runs a single decode step through the Qwen 3.5 2B model and saves every
intermediate activation as .npy files. The MPSGraph implementation can then
compare its outputs against these references layer by layer.

Usage:
    /tmp/coreml-venv/bin/python scripts/pipeline/dump_reference.py \
        --model-id Qwen/Qwen3.5-2B \
        --output models/reference_activations/
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import patches  # noqa
from functional_deltanet import patch_transformers  # noqa

patch_transformers()

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def dump_reference(model_id, output_dir, prompt_cache_dir=None):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {model_id}...")
    config = AutoConfig.from_pretrained(model_id)
    config.text_config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, config=config)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tc = config.text_config

    # System prompt + test input
    SYSTEM_PROMPT = (
        "<|im_start|>system\n"
        "You clean up raw speech-to-text transcriptions. Remove filler words "
        "(um, uh, like), fix grammar, and handle self-corrections. Keep the "
        "original meaning. Output ONLY the cleaned text.\n"
        "<|im_end|>\n<|im_start|>user\n<transcription>"
    )
    TEST_INPUT = "um so I was like going to the uh store"
    SUFFIX = "</transcription>\n<|im_end|>\n<|im_start|>assistant\n"

    full_prompt = SYSTEM_PROMPT + TEST_INPUT + SUFFIX
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]
    print(f"Prompt: {prompt_len} tokens")

    # --- Phase 1: Run full prefill to get KV cache state ---
    print("Running prefill...")
    with torch.no_grad():
        prefill_out = model(input_ids, use_cache=True)
        cache = prefill_out.past_key_values
        first_token = prefill_out.logits[0, -1].argmax().item()

    print(f"First generated token: {first_token} = '{tokenizer.decode([first_token])}'")

    # --- Phase 2: Single decode step with hooks to capture activations ---
    print("Running decode step with activation hooks...")

    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                # Save each tensor in the tuple
                for i, t in enumerate(output):
                    if isinstance(t, torch.Tensor):
                        activations[f"{name}_out_{i}"] = t.detach().float().numpy()
            elif isinstance(output, torch.Tensor):
                activations[f"{name}_out"] = output.detach().float().numpy()
            # Also save input
            if isinstance(input, tuple):
                for i, t in enumerate(input):
                    if isinstance(t, torch.Tensor):
                        activations[f"{name}_in_{i}"] = t.detach().float().numpy()
            elif isinstance(input, torch.Tensor):
                activations[f"{name}_in"] = input.detach().float().numpy()
        return hook

    hooks = []

    # Hook embedding
    hooks.append(model.model.embed_tokens.register_forward_hook(make_hook("embed")))

    # Hook each layer
    layer_types = tc.layer_types
    delta_idx = 0
    attn_idx = 0
    for i, layer in enumerate(model.model.layers):
        layer_type = layer_types[i]
        prefix = f"layer_{i}"

        # RMSNorm before attention
        hooks.append(layer.input_layernorm.register_forward_hook(make_hook(f"{prefix}_input_norm")))

        if layer_type == "linear_attention":
            dn = layer.linear_attn
            # DeltaNet submodule hooks
            hooks.append(dn.in_proj_qkv.register_forward_hook(make_hook(f"{prefix}_deltanet_qkv")))
            hooks.append(dn.in_proj_z.register_forward_hook(make_hook(f"{prefix}_deltanet_z")))
            hooks.append(dn.in_proj_b.register_forward_hook(make_hook(f"{prefix}_deltanet_b")))
            hooks.append(dn.in_proj_a.register_forward_hook(make_hook(f"{prefix}_deltanet_a")))
            hooks.append(dn.norm.register_forward_hook(make_hook(f"{prefix}_deltanet_norm")))
            hooks.append(dn.out_proj.register_forward_hook(make_hook(f"{prefix}_deltanet_out")))

            # Monkey-patch forward to capture internal intermediates
            original_forward = dn.forward
            layer_prefix = prefix  # capture for closure

            def make_patched_forward(orig, pfx):
                def patched_forward(hidden_states, cache_params=None, cache_position=None, attention_mask=None):
                    import torch
                    import torch.nn.functional as F
                    # Run original forward but capture intermediates
                    # We intercept after conv1d_update and after the recurrence
                    batch_size, seq_len, _ = hidden_states.shape
                    use_precomputed = (cache_params is not None and cache_params.has_previous_state
                                       and seq_len == 1 and cache_position is not None)

                    mixed_qkv = dn.in_proj_qkv(hidden_states).transpose(1, 2)
                    z = dn.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, dn.head_v_dim)
                    b = dn.in_proj_b(hidden_states)
                    a = dn.in_proj_a(hidden_states)

                    if use_precomputed:
                        conv_state = cache_params.conv_states[dn.layer_idx]
                        recurrent_state = cache_params.recurrent_states[dn.layer_idx]
                        conv_result = dn.causal_conv1d_update(
                            mixed_qkv, conv_state, dn.conv1d.weight.squeeze(1),
                            dn.conv1d.bias, dn.activation)
                        # causal_conv1d_update may return (output, new_state) or just output
                        if isinstance(conv_result, tuple):
                            mixed_qkv = conv_result[0]
                        else:
                            mixed_qkv = conv_result
                        activations[f"{pfx}_after_conv1d"] = mixed_qkv.detach().float().numpy()
                    else:
                        if cache_params is not None:
                            conv_state = F.pad(mixed_qkv, (dn.conv_kernel_size - mixed_qkv.shape[-1], 0))
                            cache_params.conv_states[dn.layer_idx] = conv_state
                        mixed_qkv = F.silu(dn.conv1d(mixed_qkv)[:, :, :seq_len])

                    mixed_qkv = mixed_qkv.transpose(1, 2)
                    query, key, value = torch.split(mixed_qkv, [dn.key_dim, dn.key_dim, dn.value_dim], dim=-1)
                    query = query.reshape(batch_size, seq_len, -1, dn.head_k_dim)
                    key = key.reshape(batch_size, seq_len, -1, dn.head_k_dim)
                    value = value.reshape(batch_size, seq_len, -1, dn.head_v_dim)

                    activations[f"{pfx}_query"] = query.detach().float().numpy()
                    activations[f"{pfx}_key"] = key.detach().float().numpy()
                    activations[f"{pfx}_value"] = value.detach().float().numpy()

                    beta = b.sigmoid()
                    g = -dn.A_log.float().exp() * F.softplus(a.float() + dn.dt_bias)

                    activations[f"{pfx}_beta"] = beta.detach().float().numpy()
                    activations[f"{pfx}_g"] = g.detach().float().numpy()

                    if dn.num_v_heads // dn.num_k_heads > 1:
                        query = query.repeat_interleave(dn.num_v_heads // dn.num_k_heads, dim=2)
                        key = key.repeat_interleave(dn.num_v_heads // dn.num_k_heads, dim=2)

                    if use_precomputed:
                        core_attn_out, last_recurrent_state = dn.recurrent_gated_delta_rule(
                            query, key, value, g=g, beta=beta,
                            initial_state=recurrent_state, output_final_state=True,
                            use_qk_l2norm_in_kernel=True)
                    else:
                        core_attn_out, last_recurrent_state = dn.chunk_gated_delta_rule(
                            query, key, value, g=g, beta=beta,
                            initial_state=None, output_final_state=cache_params is not None,
                            use_qk_l2norm_in_kernel=True)

                    if isinstance(core_attn_out, torch.Tensor):
                        activations[f"{pfx}_recurrence_out"] = core_attn_out.detach().float().numpy()
                    if last_recurrent_state is not None and isinstance(last_recurrent_state, torch.Tensor):
                        activations[f"{pfx}_new_rec_state"] = last_recurrent_state.detach().float().numpy()

                    if cache_params is not None:
                        cache_params.recurrent_states[dn.layer_idx] = last_recurrent_state

                    core_attn_out = core_attn_out.reshape(-1, dn.head_v_dim)
                    z_flat = z.reshape(-1, dn.head_v_dim)
                    core_attn_out = dn.norm(core_attn_out, z_flat)
                    activations[f"{pfx}_after_norm_gate"] = core_attn_out.detach().float().numpy()

                    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
                    output = dn.out_proj(core_attn_out)
                    return output
                return patched_forward

            dn.forward = make_patched_forward(original_forward, layer_prefix)
            delta_idx += 1
        else:
            # Standard attention
            hooks.append(layer.self_attn.q_proj.register_forward_hook(make_hook(f"{prefix}_attn_q")))
            hooks.append(layer.self_attn.k_proj.register_forward_hook(make_hook(f"{prefix}_attn_k")))
            hooks.append(layer.self_attn.v_proj.register_forward_hook(make_hook(f"{prefix}_attn_v")))
            hooks.append(layer.self_attn.o_proj.register_forward_hook(make_hook(f"{prefix}_attn_o")))
            attn_idx += 1

        # Post-attention norm
        hooks.append(layer.post_attention_layernorm.register_forward_hook(make_hook(f"{prefix}_post_norm")))

        # FFN
        hooks.append(layer.mlp.gate_proj.register_forward_hook(make_hook(f"{prefix}_ffn_gate")))
        hooks.append(layer.mlp.up_proj.register_forward_hook(make_hook(f"{prefix}_ffn_up")))
        hooks.append(layer.mlp.down_proj.register_forward_hook(make_hook(f"{prefix}_ffn_down")))

    # Hook final norm and lm_head
    hooks.append(model.model.norm.register_forward_hook(make_hook("final_norm")))
    hooks.append(model.lm_head.register_forward_hook(make_hook("lm_head")))

    # Run single decode step
    decode_input = torch.tensor([[first_token]])
    with torch.no_grad():
        decode_out = model(decode_input, past_key_values=cache, use_cache=True)
        second_token = decode_out.logits[0, -1].argmax().item()

    print(f"Second generated token: {second_token} = '{tokenizer.decode([second_token])}'")
    print(f"Captured {len(activations)} activation tensors")

    # Remove hooks
    for h in hooks:
        h.remove()

    # --- Phase 3: Save everything ---
    print(f"\nSaving to {output_dir}/...")

    # Save activations
    act_dir = os.path.join(output_dir, "activations")
    os.makedirs(act_dir, exist_ok=True)
    for name, arr in sorted(activations.items()):
        path = os.path.join(act_dir, f"{name}.npy")
        np.save(path, arr.astype(np.float16))

    # Save ALL model weights (FP16)
    weight_dir = os.path.join(output_dir, "weights")
    os.makedirs(weight_dir, exist_ok=True)
    for name, param in model.named_parameters():
        safe_name = name.replace(".", "_")
        np.save(os.path.join(weight_dir, f"{safe_name}.npy"),
                param.detach().float().numpy().astype(np.float16))

    # Save decode step inputs
    inputs_dir = os.path.join(output_dir, "decode_inputs")
    os.makedirs(inputs_dir, exist_ok=True)
    np.save(os.path.join(inputs_dir, "token_id.npy"), np.array([first_token], dtype=np.int32))
    np.save(os.path.join(inputs_dir, "position.npy"), np.array([prompt_len], dtype=np.int32))

    # Save KV cache state (from prefill)
    cache_dir = os.path.join(output_dir, "cache_state")
    os.makedirs(cache_dir, exist_ok=True)
    delta_idx = 0
    attn_idx = 0
    for i in range(tc.num_hidden_layers):
        if layer_types[i] == "linear_attention":
            np.save(os.path.join(cache_dir, f"conv_state_{delta_idx}.npy"),
                    cache.conv_states[i].detach().float().numpy().astype(np.float16))
            np.save(os.path.join(cache_dir, f"rec_state_{delta_idx}.npy"),
                    cache.recurrent_states[i].detach().float().numpy().astype(np.float16))
            delta_idx += 1
        else:
            k = cache.key_cache[i].detach().float().numpy().astype(np.float16)
            v = cache.value_cache[i].detach().float().numpy().astype(np.float16)
            np.save(os.path.join(cache_dir, f"key_cache_{attn_idx}.npy"), k)
            np.save(os.path.join(cache_dir, f"value_cache_{attn_idx}.npy"), v)
            attn_idx += 1

    # Save RoPE
    with torch.no_grad():
        dummy = torch.zeros(1, 256, tc.hidden_size)
        all_cos, all_sin = model.model.rotary_emb(dummy, torch.arange(256).unsqueeze(0))
    np.save(os.path.join(inputs_dir, "rope_cos.npy"), all_cos[0].numpy().astype(np.float32))
    np.save(os.path.join(inputs_dir, "rope_sin.npy"), all_sin[0].numpy().astype(np.float32))

    # Save expected output
    np.save(os.path.join(output_dir, "expected_token.npy"), np.array([second_token], dtype=np.int32))
    np.save(os.path.join(output_dir, "expected_logits.npy"),
            decode_out.logits[0, -1].detach().float().numpy().astype(np.float16))

    # Save metadata
    meta = {
        "model_id": model_id,
        "prompt": full_prompt,
        "prompt_len": prompt_len,
        "first_token": first_token,
        "second_token": second_token,
        "first_token_str": tokenizer.decode([first_token]),
        "second_token_str": tokenizer.decode([second_token]),
        "n_layers": tc.num_hidden_layers,
        "n_delta": delta_idx,
        "n_attn": attn_idx,
        "hidden_size": tc.hidden_size,
        "n_activations": len(activations),
        "layer_types": layer_types,
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Summary
    total_files = sum(len(files) for _, _, files in os.walk(output_dir))
    total_size = sum(os.path.getsize(os.path.join(dp, f))
                     for dp, _, fns in os.walk(output_dir) for f in fns)
    print(f"\nSaved {total_files} files ({total_size / 1e6:.1f} MB)")
    print(f"  Activations: {len(activations)} tensors")
    print(f"  Expected: token={second_token} ('{tokenizer.decode([second_token])}')")
    print(f"\nVerification usage:")
    print(f"  1. Build MPSGraph layer by layer")
    print(f"  2. Feed decode_inputs/ as input")
    print(f"  3. Compare each layer's output against activations/")
    print(f"  4. Final logits should match expected_logits.npy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--output", default="models/reference_activations/")
    args = parser.parse_args()

    dump_reference(args.model_id, args.output)


if __name__ == "__main__":
    main()
