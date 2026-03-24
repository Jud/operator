"""Convert Qwen3.5 into 2 CoreML chunks split at the midpoint.

Chunk A: embed + layers 0-11 (blocks 0-2)
Chunk B: layers 12-23 + norm + lm_head (blocks 3-5)

Only 2 predict() calls per token. DeltaNet states pass in/out
of each chunk. No KV cache (attention layers are stateless).

Usage:
    /tmp/coreml-venv/bin/python scripts/coreml-convert/convert_split2.py \
        Qwen/Qwen3.5-0.8B --output models/split2/
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import patches  # noqa

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
from functional_deltanet import patch_transformers

patch_transformers()

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache


class ChunkModel(nn.Module):
    """A chunk of the Qwen3.5 model (contiguous layers)."""

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

        # Which layers in this chunk are DeltaNet?
        self.delta_local = [(idx, li) for idx, li in enumerate(layer_indices)
                           if layer_types[li] == "linear_attention"]

    def forward(self, hidden_or_id, rope_cos, rope_sin, *deltanet_states):
        if self.embed is not None:
            hidden = self.embed(hidden_or_id)
        else:
            hidden = hidden_or_id

        # Force rope through a data-dependent op so torch.jit.trace
        # doesn't constant-fold it. Multiply by 1.0 derived from the
        # input's own norm — tracer can't fold this away.
        cos = rope_cos * (rope_cos.sum() * 0.0 + 1.0)
        sin = rope_sin * (rope_sin.sum() * 0.0 + 1.0)

        # Build cache for DeltaNet layers in this chunk
        cache = Qwen3_5DynamicCache.__new__(Qwen3_5DynamicCache)
        cache.conv_states = [None] * (self.num_model_layers + 1)
        cache.recurrent_states = [None] * (self.num_model_layers + 1)
        cache.key_cache = [None] * (self.num_model_layers + 1)
        cache.value_cache = [None] * (self.num_model_layers + 1)

        # The last DeltaNet layer in this chunk
        delta_indices = [li for li in self.layer_indices
                        if self.layer_types[li] == "linear_attention"]
        cache.last_linear_layer = delta_indices[-1] if delta_indices else 0
        cache.layer_types = self.layer_types
        cache.transformer_layers = [j for j in range(self.num_model_layers)
                                   if self.layer_types[j] == "full_attention"]

        # Populate DeltaNet states
        state_idx = 0
        for _, li in self.delta_local:
            cache.conv_states[li] = deltanet_states[state_idx]
            cache.recurrent_states[li] = deltanet_states[state_idx + 1]
            state_idx += 2

        # Run layers
        for idx, layer in enumerate(self.layers):
            li = self.layer_indices[idx]
            if self.layer_types[li] == "linear_attention":
                result = layer(hidden, position_embeddings=(cos, sin),
                              past_key_values=cache,
                              cache_position=torch.tensor([0]))
            else:
                result = layer(hidden, position_embeddings=(cos, sin),
                              cache_position=torch.tensor([0]))
            hidden = result[0] if isinstance(result, tuple) else result

        # Head if included
        if self.norm is not None:
            hidden = self.lm_head(self.norm(hidden))

        # Collect updated DeltaNet states
        updated = []
        for _, li in self.delta_local:
            updated.append(cache.conv_states[li])
            updated.append(cache.recurrent_states[li])

        return (hidden,) + tuple(updated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--output", default="models/split2/")
    parser.add_argument("--fp32", action="store_true")
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

    conv_dim = tc.linear_key_head_dim * tc.linear_num_key_heads * 2 + tc.linear_value_head_dim * tc.linear_num_value_heads
    conv_kernel = tc.linear_conv_kernel_dim
    n_v_heads = tc.linear_num_value_heads
    k_head_dim = tc.linear_key_head_dim
    v_head_dim = tc.linear_value_head_dim
    hidden_size = tc.hidden_size

    # RoPE
    with torch.no_grad():
        dummy = torch.zeros(1, 300, hidden_size)
        all_cos, all_sin = model.model.rotary_emb(dummy, torch.arange(300).unsqueeze(0))
    rope_dim = all_cos.shape[-1]

    # Split at midpoint
    mid = tc.num_hidden_layers // 2
    chunk_a_indices = list(range(mid))
    chunk_b_indices = list(range(mid, tc.num_hidden_layers))

    chunk_a_delta = [i for i in chunk_a_indices if tc.layer_types[i] == "linear_attention"]
    chunk_b_delta = [i for i in chunk_b_indices if tc.layer_types[i] == "linear_attention"]

    print(f"Chunk A: layers {chunk_a_indices[0]}-{chunk_a_indices[-1]} ({len(chunk_a_delta)} DeltaNet + {len(chunk_a_indices)-len(chunk_a_delta)} attention) + embed")
    print(f"Chunk B: layers {chunk_b_indices[0]}-{chunk_b_indices[-1]} ({len(chunk_b_delta)} DeltaNet + {len(chunk_b_indices)-len(chunk_b_delta)} attention) + head")

    # --- Chunk A ---
    print("\n--- Chunk A ---")
    chunk_a = ChunkModel(
        [model.model.layers[i] for i in chunk_a_indices],
        chunk_a_indices, tc.layer_types, tc,
        include_embed=model.model.embed_tokens,
    )
    chunk_a.eval()

    sample_id = torch.zeros((1, 1), dtype=torch.long)
    sample_cos = all_cos[:, 0:1, :]
    sample_sin = all_sin[:, 0:1, :]
    sample_states_a = []
    ct_in_a = [
        ct.TensorType(name="input_id", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="rope_cos", shape=(1, 1, rope_dim)),
        ct.TensorType(name="rope_sin", shape=(1, 1, rope_dim)),
    ]
    ct_out_a = [ct.TensorType(name="hidden")]

    for j, li in enumerate(chunk_a_delta):
        sample_states_a.append(torch.zeros(1, conv_dim, conv_kernel))
        sample_states_a.append(torch.zeros(1, n_v_heads, k_head_dim, v_head_dim))
        ct_in_a.append(ct.TensorType(name=f"conv_state_{j}", shape=(1, conv_dim, conv_kernel)))
        ct_in_a.append(ct.TensorType(name=f"rec_state_{j}", shape=(1, n_v_heads, k_head_dim, v_head_dim)))
        ct_out_a.append(ct.TensorType(name=f"conv_state_{j}_out"))
        ct_out_a.append(ct.TensorType(name=f"rec_state_{j}_out"))

    with torch.no_grad():
        traced_a = torch.jit.trace(chunk_a, (sample_id, sample_cos, sample_sin, *sample_states_a), strict=False)

    path_a = os.path.join(args.output, "chunk_a.mlpackage")
    ct.convert(traced_a, inputs=ct_in_a, outputs=ct_out_a,
               compute_precision=precision, compute_units=ct.ComputeUnit.ALL,
               minimum_deployment_target=ct.target.iOS17,
               convert_to="mlprogram").save(path_a)
    print(f"  Saved {path_a}")

    # --- Chunk B ---
    print("\n--- Chunk B ---")
    chunk_b = ChunkModel(
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
    ]
    ct_out_b = [ct.TensorType(name="logits")]

    for j, li in enumerate(chunk_b_delta):
        sample_states_b.append(torch.zeros(1, conv_dim, conv_kernel))
        sample_states_b.append(torch.zeros(1, n_v_heads, k_head_dim, v_head_dim))
        ct_in_b.append(ct.TensorType(name=f"conv_state_{j}", shape=(1, conv_dim, conv_kernel)))
        ct_in_b.append(ct.TensorType(name=f"rec_state_{j}", shape=(1, n_v_heads, k_head_dim, v_head_dim)))
        ct_out_b.append(ct.TensorType(name=f"conv_state_{j}_out"))
        ct_out_b.append(ct.TensorType(name=f"rec_state_{j}_out"))

    with torch.no_grad():
        traced_b = torch.jit.trace(chunk_b, (sample_hidden, sample_cos, sample_sin, *sample_states_b), strict=False)

    path_b = os.path.join(args.output, "chunk_b.mlpackage")
    ct.convert(traced_b, inputs=ct_in_b, outputs=ct_out_b,
               compute_precision=precision, compute_units=ct.ComputeUnit.ALL,
               minimum_deployment_target=ct.target.iOS17,
               convert_to="mlprogram").save(path_b)
    print(f"  Saved {path_b}")

    print(f"\n=== Done: 2 models in {args.output} ===")


if __name__ == "__main__":
    main()
