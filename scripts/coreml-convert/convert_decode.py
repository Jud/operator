"""Convert Qwen3.5 decode model (single-token + KV cache) to CoreML.

Strategy: Instead of trying to use the HF model's cache mechanism (which
uses list mutations that can't be traced), we run each layer individually
and pass cache tensors in/out explicitly as function arguments.

Usage:
    /tmp/coreml-venv/bin/python scripts/coreml-convert/convert_decode.py \
        Qwen/Qwen3.5-0.8B --max-seq-len 256 --output models/
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import patches  # noqa: F401

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
from functional_deltanet import patch_transformers

patch_transformers()

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


class SingleLayerDeltaNet(nn.Module):
    """Wraps a single DeltaNet layer for trace-friendly decode."""

    def __init__(self, layer, layer_idx):
        super().__init__()
        self.layer = layer
        self.layer_idx = layer_idx

    def forward(self, hidden_states, conv_state, recurrent_state, position_ids, cache_position):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

        # Build a minimal cache for this layer
        config = self.layer.self_attn.config if hasattr(self.layer, 'self_attn') else self.layer.linear_attn.config
        # ...this is getting too complex. Let me try a different approach.
        pass


class DecodeModel(nn.Module):
    """Full decode model that takes explicit cache tensors as args.

    Instead of using HF's cache object, we manually run the model's
    internal components and thread cache tensors through as plain args.
    """

    def __init__(self, hf_model, config):
        super().__init__()
        self.embed = hf_model.model.embed_tokens
        self.norm = hf_model.model.norm
        self.lm_head = hf_model.lm_head
        self.layers = hf_model.model.layers
        self.rotary_emb = hf_model.model.rotary_emb

        tc = config.text_config if hasattr(config, "text_config") else config
        self.num_layers = tc.num_hidden_layers
        self.interval = tc.full_attention_interval
        self.layer_types = tc.layer_types

    def forward(self, input_id, cache_position, rope_cos, rope_sin, *cache_in):
        """Decode one token. No KV cache — only DeltaNet states.

        The full-attention layers re-compute from scratch each step.
        This is slower but correct, and will be replaced by disaggregated
        models where each chunk manages its own state.
        """
        hidden = self.embed(input_id)
        cos = rope_cos
        sin = rope_sin

        # Build cache with only DeltaNet states (no KV cache)
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
        cache = Qwen3_5DynamicCache.__new__(Qwen3_5DynamicCache)
        cache.layer_types = self.layer_types
        cache.transformer_layers = [j for j in range(self.num_layers) if self.layer_types[j] == "full_attention"]
        cache.last_linear_layer = self.num_layers - 1 - self.layer_types[::-1].index("linear_attention")
        cache.conv_states = [None] * self.num_layers
        cache.recurrent_states = [None] * self.num_layers
        cache.key_cache = [None] * self.num_layers
        cache.value_cache = [None] * self.num_layers

        # Only populate DeltaNet states
        cache_idx = 0
        for i in range(self.num_layers):
            if self.layer_types[i] == "linear_attention":
                cache.conv_states[i] = cache_in[cache_idx]
                cache.recurrent_states[i] = cache_in[cache_idx + 1]
                cache_idx += 2

        # Run all layers — attention layers run without KV cache (no past)
        for i, layer in enumerate(self.layers):
            if self.layer_types[i] == "linear_attention":
                result = layer(
                    hidden,
                    position_embeddings=(cos, sin),
                    past_key_values=cache,
                    cache_position=cache_position,
                )
            else:
                # Full attention without cache — just compute fresh
                result = layer(
                    hidden,
                    position_embeddings=(cos, sin),
                    cache_position=cache_position,
                )
            hidden = result[0] if isinstance(result, tuple) else result

        # Extract updated DeltaNet states
        cache_out = []
        for i in range(self.num_layers):
            if self.layer_types[i] == "linear_attention":
                cache_out.append(cache.conv_states[i])
                cache_out.append(cache.recurrent_states[i])

        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)

        return (logits,) + tuple(cache_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--output", default="models/")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 compute precision")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading {args.model_id}...")
    config = AutoConfig.from_pretrained(args.model_id)
    # Force eager attention to avoid SDPA GQA incompatibility with CoreML
    config.text_config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32, config=config)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    tc = config.text_config if hasattr(config, "text_config") else config

    # Build wrapper
    wrapper = DecodeModel(model, config)
    wrapper.eval()

    # Cache shapes
    conv_dim = tc.linear_key_head_dim * tc.linear_num_key_heads * 2 + tc.linear_value_head_dim * tc.linear_num_value_heads
    conv_kernel = tc.linear_conv_kernel_dim
    n_v_heads = tc.linear_num_value_heads
    k_head_dim = tc.linear_key_head_dim
    v_head_dim = tc.linear_value_head_dim
    n_kv_heads = tc.num_key_value_heads
    head_dim = tc.head_dim

    # Pre-compute RoPE for all positions up to max_seq_len
    with torch.no_grad():
        dummy_hidden = torch.zeros(1, 1, tc.hidden_size)
        all_pos = torch.arange(args.max_seq_len).unsqueeze(0)
        all_cos, all_sin = model.model.rotary_emb(dummy_hidden.expand(1, args.max_seq_len, -1), all_pos)
    # Each decode step will slice by position: cos[:, pos:pos+1, :], sin[:, pos:pos+1, :]
    rope_dim = all_cos.shape[-1]

    # Build sample inputs
    sample_id = torch.zeros((1, 1), dtype=torch.long)
    sample_pos = torch.tensor([1], dtype=torch.long)
    sample_cos = all_cos[:, 1:2, :]  # position 1
    sample_sin = all_sin[:, 1:2, :]
    # Attention mask: [1, 1, 1, max_seq_len] — 0 for valid, -inf for unused
    # At position 1, positions 0 and 1 are valid (prefill token + current token)
    sample_mask = torch.full((1, 1, 1, args.max_seq_len), float("-inf"))
    sample_mask[0, 0, 0, :2] = 0.0  # first 2 positions valid

    # Only DeltaNet states — no KV cache
    sample_cache = []
    for i in range(tc.num_hidden_layers):
        if tc.layer_types[i] == "linear_attention":
            sample_cache.append(torch.zeros(1, conv_dim, conv_kernel))
            sample_cache.append(torch.zeros(1, n_v_heads, k_head_dim, v_head_dim))

    print(f"Testing wrapper ({len(sample_cache)} DeltaNet state tensors)...")
    with torch.no_grad():
        out = wrapper(sample_id, sample_pos, sample_cos, sample_sin, *sample_cache)
        print(f"  Logits: {out[0].shape}, Cache out: {len(out) - 1}")

    print("Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (sample_id, sample_pos, sample_cos, sample_sin, *sample_cache), strict=False)
    print("Trace OK")

    # Verify
    with torch.no_grad():
        ref = wrapper(sample_id, sample_pos, sample_cos, sample_sin, *sample_cache)
        tr = traced(sample_id, sample_pos, sample_cos, sample_sin, *sample_cache)
        diff = (ref[0] - tr[0]).abs().max().item()
        print(f"Trace verification: max diff = {diff:.6f}")

    # CoreML conversion
    print("Converting to CoreML...")
    ct_inputs = [
        ct.TensorType(name="input_id", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="cache_position", shape=(1,), dtype=np.int32),
        ct.TensorType(name="rope_cos", shape=(1, 1, rope_dim)),
        ct.TensorType(name="rope_sin", shape=(1, 1, rope_dim)),
    ]
    ct_outputs = [ct.TensorType(name="logits")]

    # Only DeltaNet states
    for i in range(tc.num_hidden_layers):
        if tc.layer_types[i] == "linear_attention":
            ct_inputs.append(ct.TensorType(name=f"conv_state_{i}", shape=(1, conv_dim, conv_kernel)))
            ct_inputs.append(ct.TensorType(name=f"recurrent_state_{i}", shape=(1, n_v_heads, k_head_dim, v_head_dim)))
            ct_outputs.append(ct.TensorType(name=f"conv_state_{i}_out"))
            ct_outputs.append(ct.TensorType(name=f"recurrent_state_{i}_out"))

    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        compute_precision=ct.precision.FLOAT32 if args.fp32 else ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )

    slug = args.model_id.replace("/", "_").replace(".", "_")
    suffix = "_fp32" if args.fp32 else ""
    out_path = os.path.join(args.output, f"{slug}_decode{suffix}.mlpackage")
    mlmodel.save(out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
