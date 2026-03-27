"""Convert Qwen3.5 into 2 CoreML chunks with KV cache support.

Chunk A: embed + layers 0-11
Chunk B: layers 12-23 + norm + lm_head

Each chunk takes explicit KV cache tensors (padded to max_seq_len) for its
attention layers, plus DeltaNet states for linear attention layers.
An external attention mask handles the variable-length decode position.

This solves the KV cache blocker: we bypass get_seq_length() entirely by
passing the mask as an input and using scatter-based cache updates.

Usage:
    /tmp/coreml-venv/bin/python scripts/coreml-convert/convert_split2_kv.py \\
        Qwen/Qwen3.5-0.8B --output models/split2_kv/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import patches  # noqa: E402, F401

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
from functional_deltanet import patch_transformers

patch_transformers()

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache  # noqa: E402


class ChunkModelKV(nn.Module):
    """A chunk of Qwen3.5 with explicit KV cache for attention layers."""

    def __init__(self, layers, layer_indices, layer_types, config,
                 include_embed=None, include_head=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.layer_indices = layer_indices
        self.layer_types = layer_types
        self.num_model_layers = config.num_hidden_layers
        self.embed = include_embed
        self.norm = include_head[0] if include_head else None
        self.lm_head = include_head[1] if include_head else None

        # Which layers in this chunk are DeltaNet vs attention?
        self.delta_local = [(idx, li) for idx, li in enumerate(layer_indices)
                           if layer_types[li] == "linear_attention"]
        self.attn_local = [(idx, li) for idx, li in enumerate(layer_indices)
                          if layer_types[li] == "full_attention"]

    def forward(self, hidden_or_id, rope_cos, rope_sin, cache_position, attn_mask, *states):
        """Decode one token with explicit state management.

        States order: DeltaNet states first, then KV cache.
          DeltaNet: (conv_0, rec_0, conv_1, rec_1, ...)
          KV cache: (key_0, val_0, key_1, val_1, ...)
        """
        if self.embed is not None:
            hidden = self.embed(hidden_or_id)
        else:
            hidden = hidden_or_id

        # Anti-constant-fold: make RoPE depend on input data
        cos = rope_cos * (rope_cos.sum() * 0.0 + 1.0)
        sin = rope_sin * (rope_sin.sum() * 0.0 + 1.0)

        # Build cache object
        cache = Qwen3_5DynamicCache.__new__(Qwen3_5DynamicCache)
        cache.conv_states = [None] * (self.num_model_layers + 1)
        cache.recurrent_states = [None] * (self.num_model_layers + 1)
        cache.key_cache = [None] * (self.num_model_layers + 1)
        cache.value_cache = [None] * (self.num_model_layers + 1)

        delta_indices = [li for li in self.layer_indices
                        if self.layer_types[li] == "linear_attention"]
        cache.last_linear_layer = delta_indices[-1] if delta_indices else 0
        cache.layer_types = self.layer_types
        cache.transformer_layers = [j for j in range(self.num_model_layers)
                                   if self.layer_types[j] == "full_attention"]

        # Populate DeltaNet states
        state_idx = 0
        for _, li in self.delta_local:
            cache.conv_states[li] = states[state_idx]
            cache.recurrent_states[li] = states[state_idx + 1]
            state_idx += 2

        # Populate KV cache (padded)
        n_delta_states = len(self.delta_local) * 2
        kv_idx = n_delta_states
        for _, li in self.attn_local:
            cache.key_cache[li] = states[kv_idx]
            cache.value_cache[li] = states[kv_idx + 1]
            kv_idx += 2

        # Run all layers — same call for both types
        # DeltaNet: uses cache.conv_states/recurrent_states, ignores attn_mask (shape[1]==1)
        # Attention: uses cache.key_cache/value_cache + attn_mask, scatter-based update
        for idx, layer in enumerate(self.layers):
            li = self.layer_indices[idx]
            if self.layer_types[li] == "linear_attention":
                result = layer(
                    hidden,
                    position_embeddings=(cos, sin),
                    past_key_values=cache,
                    cache_position=cache_position,
                )
            else:
                result = layer(
                    hidden,
                    position_embeddings=(cos, sin),
                    past_key_values=cache,
                    cache_position=cache_position,
                    attention_mask=attn_mask,
                )
            hidden = result[0] if isinstance(result, tuple) else result

        # Head if included
        if self.norm is not None:
            hidden = self.lm_head(self.norm(hidden))

        # Collect updated states: DeltaNet first, then KV
        updated = []
        for _, li in self.delta_local:
            updated.append(cache.conv_states[li])
            updated.append(cache.recurrent_states[li])
        for _, li in self.attn_local:
            updated.append(cache.key_cache[li])
            updated.append(cache.value_cache[li])

        return (hidden,) + tuple(updated)


def export_test_inputs(model, tokenizer, config, output_dir, max_seq_len):
    """Export test inputs for Swift validation."""
    tc = config.text_config
    test_dir = os.path.join(output_dir, "test_inputs")
    os.makedirs(test_dir, exist_ok=True)

    prompt = "Clean up: um so I was like going to the uh store"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]

    # Prefill
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        cache = out.past_key_values
        first_token = torch.argmax(out.logits[0, -1]).item()

    print(f"  Prompt: {repr(prompt)} ({prompt_len} tokens)")
    print(f"  First decode token: {first_token} ({repr(tokenizer.decode([first_token]))})")

    # RoPE for first decode position
    with torch.no_grad():
        dummy = torch.zeros(1, max_seq_len, tc.hidden_size)
        all_pos = torch.arange(max_seq_len).unsqueeze(0)
        all_cos, all_sin = model.model.rotary_emb(dummy, all_pos)

    cos = all_cos[:, prompt_len:prompt_len+1, :].numpy().astype(np.float32)
    sin = all_sin[:, prompt_len:prompt_len+1, :].numpy().astype(np.float32)
    np.save(os.path.join(test_dir, "rope_cos.npy"), cos)
    np.save(os.path.join(test_dir, "rope_sin.npy"), sin)

    # Attention mask for first decode position
    mask = np.full((1, 1, 1, max_seq_len), -1e4, dtype=np.float32)
    mask[0, 0, 0, :prompt_len + 1] = 0.0
    np.save(os.path.join(test_dir, "attn_mask.npy"), mask)

    # Cache position
    np.save(os.path.join(test_dir, "cache_position.npy"),
            np.array([prompt_len], dtype=np.int32))

    # Input token
    np.save(os.path.join(test_dir, "input_id.npy"),
            np.array([[first_token]], dtype=np.int32))

    # Split at midpoint
    mid = tc.num_hidden_layers // 2
    chunk_a_indices = list(range(mid))
    chunk_b_indices = list(range(mid, tc.num_hidden_layers))

    # DeltaNet states
    for chunk_name, indices in [("a", chunk_a_indices), ("b", chunk_b_indices)]:
        delta_j = 0
        attn_j = 0
        for li in indices:
            if tc.layer_types[li] == "linear_attention":
                np.save(os.path.join(test_dir, f"{chunk_name}_conv_{delta_j}.npy"),
                        np.ascontiguousarray(cache.conv_states[li].numpy().astype(np.float32)))
                np.save(os.path.join(test_dir, f"{chunk_name}_rec_{delta_j}.npy"),
                        np.ascontiguousarray(cache.recurrent_states[li].numpy().astype(np.float32)))
                delta_j += 1
            else:
                # KV cache: pad to max_seq_len
                k = cache.key_cache[li].numpy().astype(np.float32)
                v = cache.value_cache[li].numpy().astype(np.float32)
                n_kv_h, cur_len, h_dim = k.shape[1], k.shape[2], k.shape[3]
                k_pad = np.zeros((1, n_kv_h, max_seq_len, h_dim), dtype=np.float32)
                v_pad = np.zeros((1, n_kv_h, max_seq_len, h_dim), dtype=np.float32)
                k_pad[:, :, :cur_len, :] = k
                v_pad[:, :, :cur_len, :] = v
                np.save(os.path.join(test_dir, f"{chunk_name}_key_{attn_j}.npy"), k_pad)
                np.save(os.path.join(test_dir, f"{chunk_name}_val_{attn_j}.npy"), v_pad)
                attn_j += 1

    # Reference: run one manual decode step for verification
    from verify_kv_decode import build_cache, manual_decode_step

    conv_s = {i: cache.conv_states[i].clone() for i in range(tc.num_hidden_layers) if tc.layer_types[i] == "linear_attention"}
    rec_s = {i: cache.recurrent_states[i].clone() for i in range(tc.num_hidden_layers) if tc.layer_types[i] == "linear_attention"}
    padded_key = {}
    padded_val = {}
    for i in range(tc.num_hidden_layers):
        if tc.layer_types[i] == "full_attention":
            k = cache.key_cache[i]
            v = cache.value_cache[i]
            n_kv, h_dim = k.shape[1], k.shape[3]
            kp = torch.zeros(1, n_kv, max_seq_len, h_dim)
            vp = torch.zeros(1, n_kv, max_seq_len, h_dim)
            kp[:, :, :prompt_len, :] = k
            vp[:, :, :prompt_len, :] = v
            padded_key[i] = kp
            padded_val[i] = vp

    with torch.no_grad():
        mask_t = torch.full((1, 1, 1, max_seq_len), -1e4)
        mask_t[0, 0, 0, :prompt_len + 1] = 0.0
        ref_cache = build_cache(tc.num_hidden_layers, tc.layer_types, conv_s, rec_s, padded_key, padded_val)
        hidden = model.model.embed_tokens(torch.tensor([[first_token]]))
        ref_cos = all_cos[:, prompt_len:prompt_len+1, :]
        ref_sin = all_sin[:, prompt_len:prompt_len+1, :]
        ref_logits = manual_decode_step(model, tc.layer_types, hidden, ref_cos, ref_sin,
                                        ref_cache, prompt_len, mask_t)
        ref_token = torch.argmax(ref_logits[0, -1]).item()

    np.save(os.path.join(test_dir, "ref_token.npy"), np.array([ref_token], dtype=np.int32))
    print(f"  Reference next token: {ref_token} ({repr(tokenizer.decode([ref_token]))})")
    print(f"  Saved test inputs to {test_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--output", default="models/split2_kv/")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--skip-export", action="store_true", help="Skip test input export")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    precision = ct.precision.FLOAT32 if args.fp32 else ct.precision.FLOAT16

    print(f"Loading {args.model_id}...")
    config = AutoConfig.from_pretrained(args.model_id)
    config.text_config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32, config=config)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tc = config.text_config

    # Dimensions
    conv_dim = tc.linear_key_head_dim * tc.linear_num_key_heads * 2 + tc.linear_value_head_dim * tc.linear_num_value_heads
    conv_kernel = tc.linear_conv_kernel_dim
    n_v_heads = tc.linear_num_value_heads
    k_head_dim = tc.linear_key_head_dim
    v_head_dim = tc.linear_value_head_dim
    n_kv_heads = tc.num_key_value_heads
    head_dim = tc.head_dim
    hidden_size = tc.hidden_size
    max_seq_len = args.max_seq_len

    # RoPE
    with torch.no_grad():
        dummy = torch.zeros(1, max_seq_len, hidden_size)
        all_cos, all_sin = model.model.rotary_emb(dummy, torch.arange(max_seq_len).unsqueeze(0))
    rope_dim = all_cos.shape[-1]

    # Split at midpoint
    mid = tc.num_hidden_layers // 2
    chunk_a_indices = list(range(mid))
    chunk_b_indices = list(range(mid, tc.num_hidden_layers))

    chunk_a_delta = [i for i in chunk_a_indices if tc.layer_types[i] == "linear_attention"]
    chunk_a_attn = [i for i in chunk_a_indices if tc.layer_types[i] == "full_attention"]
    chunk_b_delta = [i for i in chunk_b_indices if tc.layer_types[i] == "linear_attention"]
    chunk_b_attn = [i for i in chunk_b_indices if tc.layer_types[i] == "full_attention"]

    print(f"Chunk A: layers {chunk_a_indices[0]}-{chunk_a_indices[-1]}")
    print(f"  {len(chunk_a_delta)} DeltaNet + {len(chunk_a_attn)} attention + embed")
    print(f"Chunk B: layers {chunk_b_indices[0]}-{chunk_b_indices[-1]}")
    print(f"  {len(chunk_b_delta)} DeltaNet + {len(chunk_b_attn)} attention + head")
    print(f"Max seq len: {max_seq_len}")

    # --- Export test inputs ---
    if not args.skip_export:
        print("\nExporting test inputs...")
        export_test_inputs(model, tokenizer, config, args.output, max_seq_len)

    # --- Chunk A ---
    print("\n--- Chunk A ---")
    chunk_a = ChunkModelKV(
        [model.model.layers[i] for i in chunk_a_indices],
        chunk_a_indices, tc.layer_types, tc,
        include_embed=model.model.embed_tokens,
    )
    chunk_a.eval()

    # Sample inputs for tracing
    sample_id = torch.zeros((1, 1), dtype=torch.long)
    sample_cos = all_cos[:, 0:1, :]
    sample_sin = all_sin[:, 0:1, :]
    sample_cache_pos = torch.tensor([1], dtype=torch.long)
    sample_mask = torch.zeros((1, 1, 1, max_seq_len))

    # DeltaNet states for chunk A
    sample_states_a = []
    ct_in_a = [
        ct.TensorType(name="input_id", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="rope_cos", shape=(1, 1, rope_dim)),
        ct.TensorType(name="rope_sin", shape=(1, 1, rope_dim)),
        ct.TensorType(name="cache_position", shape=(1,), dtype=np.int32),
        ct.TensorType(name="attn_mask", shape=(1, 1, 1, max_seq_len)),
    ]
    ct_out_a = [ct.TensorType(name="hidden")]

    for j, li in enumerate([(idx, li) for idx, li in enumerate(chunk_a_indices) if tc.layer_types[li] == "linear_attention"]):
        sample_states_a.append(torch.zeros(1, conv_dim, conv_kernel))
        sample_states_a.append(torch.zeros(1, n_v_heads, k_head_dim, v_head_dim))
        ct_in_a.append(ct.TensorType(name=f"conv_state_{j}", shape=(1, conv_dim, conv_kernel)))
        ct_in_a.append(ct.TensorType(name=f"rec_state_{j}", shape=(1, n_v_heads, k_head_dim, v_head_dim)))
        ct_out_a.append(ct.TensorType(name=f"conv_state_{j}_out"))
        ct_out_a.append(ct.TensorType(name=f"rec_state_{j}_out"))

    # KV cache for chunk A attention layers
    for j in range(len(chunk_a_attn)):
        sample_states_a.append(torch.zeros(1, n_kv_heads, max_seq_len, head_dim))
        sample_states_a.append(torch.zeros(1, n_kv_heads, max_seq_len, head_dim))
        ct_in_a.append(ct.TensorType(name=f"key_cache_{j}", shape=(1, n_kv_heads, max_seq_len, head_dim)))
        ct_in_a.append(ct.TensorType(name=f"value_cache_{j}", shape=(1, n_kv_heads, max_seq_len, head_dim)))
        ct_out_a.append(ct.TensorType(name=f"key_cache_{j}_out"))
        ct_out_a.append(ct.TensorType(name=f"value_cache_{j}_out"))

    print(f"  Inputs: {len(ct_in_a)}, Outputs: {len(ct_out_a)}")
    print(f"  DeltaNet states: {len(chunk_a_delta) * 2}, KV caches: {len(chunk_a_attn) * 2}")

    print("  Tracing...")
    with torch.no_grad():
        traced_a = torch.jit.trace(
            chunk_a,
            (sample_id, sample_cos, sample_sin, sample_cache_pos, sample_mask, *sample_states_a),
            strict=False,
        )

    # Verify trace
    with torch.no_grad():
        ref_a = chunk_a(sample_id, sample_cos, sample_sin, sample_cache_pos, sample_mask, *sample_states_a)
        tr_a = traced_a(sample_id, sample_cos, sample_sin, sample_cache_pos, sample_mask, *sample_states_a)
        diff_a = (ref_a[0] - tr_a[0]).abs().max().item()
        print(f"  Trace verification: max diff = {diff_a:.6f}")

    print("  Converting to CoreML...")
    path_a = os.path.join(args.output, "chunk_a.mlpackage")
    ct.convert(
        traced_a,
        inputs=ct_in_a,
        outputs=ct_out_a,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    ).save(path_a)
    print(f"  Saved {path_a}")

    # --- Chunk B ---
    print("\n--- Chunk B ---")
    chunk_b = ChunkModelKV(
        [model.model.layers[i] for i in chunk_b_indices],
        chunk_b_indices, tc.layer_types, tc,
        include_head=(model.model.norm, model.lm_head),
    )
    chunk_b.eval()

    sample_hidden = torch.randn(1, 1, hidden_size)
    sample_states_b = []
    ct_in_b = [
        ct.TensorType(name="hidden", shape=(1, 1, hidden_size)),
        ct.TensorType(name="rope_cos", shape=(1, 1, rope_dim)),
        ct.TensorType(name="rope_sin", shape=(1, 1, rope_dim)),
        ct.TensorType(name="cache_position", shape=(1,), dtype=np.int32),
        ct.TensorType(name="attn_mask", shape=(1, 1, 1, max_seq_len)),
    ]
    ct_out_b = [ct.TensorType(name="logits")]

    for j in range(len(chunk_b_delta)):
        sample_states_b.append(torch.zeros(1, conv_dim, conv_kernel))
        sample_states_b.append(torch.zeros(1, n_v_heads, k_head_dim, v_head_dim))
        ct_in_b.append(ct.TensorType(name=f"conv_state_{j}", shape=(1, conv_dim, conv_kernel)))
        ct_in_b.append(ct.TensorType(name=f"rec_state_{j}", shape=(1, n_v_heads, k_head_dim, v_head_dim)))
        ct_out_b.append(ct.TensorType(name=f"conv_state_{j}_out"))
        ct_out_b.append(ct.TensorType(name=f"rec_state_{j}_out"))

    for j in range(len(chunk_b_attn)):
        sample_states_b.append(torch.zeros(1, n_kv_heads, max_seq_len, head_dim))
        sample_states_b.append(torch.zeros(1, n_kv_heads, max_seq_len, head_dim))
        ct_in_b.append(ct.TensorType(name=f"key_cache_{j}", shape=(1, n_kv_heads, max_seq_len, head_dim)))
        ct_in_b.append(ct.TensorType(name=f"value_cache_{j}", shape=(1, n_kv_heads, max_seq_len, head_dim)))
        ct_out_b.append(ct.TensorType(name=f"key_cache_{j}_out"))
        ct_out_b.append(ct.TensorType(name=f"value_cache_{j}_out"))

    print(f"  Inputs: {len(ct_in_b)}, Outputs: {len(ct_out_b)}")
    print(f"  DeltaNet states: {len(chunk_b_delta) * 2}, KV caches: {len(chunk_b_attn) * 2}")

    print("  Tracing...")
    with torch.no_grad():
        traced_b = torch.jit.trace(
            chunk_b,
            (sample_hidden, sample_cos, sample_sin, sample_cache_pos, sample_mask, *sample_states_b),
            strict=False,
        )

    with torch.no_grad():
        ref_b = chunk_b(sample_hidden, sample_cos, sample_sin, sample_cache_pos, sample_mask, *sample_states_b)
        tr_b = traced_b(sample_hidden, sample_cos, sample_sin, sample_cache_pos, sample_mask, *sample_states_b)
        diff_b = (ref_b[0] - tr_b[0]).abs().max().item()
        print(f"  Trace verification: max diff = {diff_b:.6f}")

    print("  Converting to CoreML...")
    path_b = os.path.join(args.output, "chunk_b.mlpackage")
    ct.convert(
        traced_b,
        inputs=ct_in_b,
        outputs=ct_out_b,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    ).save(path_b)
    print(f"  Saved {path_b}")

    print(f"\n=== Done: 2 models + test inputs in {args.output} ===")
    print(f"KV cache: {len(chunk_a_attn)} attn layers in A, {len(chunk_b_attn)} in B")
    print(f"Each KV tensor: [1, {n_kv_heads}, {max_seq_len}, {head_dim}]")


if __name__ == "__main__":
    main()
