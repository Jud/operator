"""Convert Qwen3.5 into disaggregated CoreML models.

Splits the model into DeltaNet chunks (3 layers each, target ANE) and
Attention chunks (1 layer each, target GPU) that can run concurrently.

Architecture: 6 blocks of [3 DeltaNet + 1 Attention]
  - 6 DeltaNet models: deltanet_block_{0-5}.mlpackage
  - 6 Attention models: attention_block_{0-5}.mlpackage
  - 1 Embed model: embed.mlpackage
  - 1 Head model: head.mlpackage (norm + lm_head)

Inference loop:
  hidden = embed(token)
  for i in range(6):
      hidden, states = deltanet_block_i(hidden, states)  # ANE
      hidden = attention_block_i(hidden, rope, kv)        # GPU (concurrent with next ANE)
  logits = head(hidden)

Usage:
    /tmp/coreml-venv/bin/python scripts/coreml-convert/convert_disaggregated.py \
        Qwen/Qwen3.5-0.8B --output models/disaggregated/
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


class EmbedModel(nn.Module):
    def __init__(self, embed_tokens):
        super().__init__()
        self.embed = embed_tokens

    def forward(self, input_id):
        return self.embed(input_id)


class DeltaNetBlock(nn.Module):
    """3 consecutive DeltaNet layers."""

    def __init__(self, layers, layer_indices):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.layer_indices = layer_indices

    def forward(self, hidden, *deltanet_states):
        # deltanet_states: [conv0, rec0, conv1, rec1, conv2, rec2]
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

        # Build a single cache with all this block's DeltaNet states.
        # Must set conv_states for ALL layer indices including last_linear_layer
        # so has_previous_state returns True (decode path).
        num_layers = self.layer_indices[-1] + 1  # enough slots
        cache = Qwen3_5DynamicCache.__new__(Qwen3_5DynamicCache)
        cache.conv_states = [None] * (num_layers + 1)
        cache.recurrent_states = [None] * (num_layers + 1)
        cache.key_cache = [None] * (num_layers + 1)
        cache.value_cache = [None] * (num_layers + 1)
        cache.last_linear_layer = self.layer_indices[-1]  # last layer in this block
        cache.layer_types = ["linear_attention"] * (num_layers + 1)
        cache.transformer_layers = []

        for i, li in enumerate(self.layer_indices):
            cache.conv_states[li] = deltanet_states[i * 2]
            cache.recurrent_states[li] = deltanet_states[i * 2 + 1]

        dummy_rope = (torch.zeros(1), torch.zeros(1))

        for i, layer in enumerate(self.layers):
            result = layer(hidden, position_embeddings=dummy_rope,
                          past_key_values=cache, cache_position=torch.tensor([0]))
            hidden = result[0] if isinstance(result, tuple) else result

        updated_states = []
        for i, li in enumerate(self.layer_indices):
            updated_states.append(cache.conv_states[li])
            updated_states.append(cache.recurrent_states[li])

        return (hidden,) + tuple(updated_states)


class AttentionBlock(nn.Module):
    """1 full-attention layer."""

    def __init__(self, layer, layer_idx):
        super().__init__()
        self.layer = layer
        self.layer_idx = layer_idx

    def forward(self, hidden, rope_cos, rope_sin):
        # No KV cache for now — fresh attention each step
        result = self.layer(hidden, position_embeddings=(rope_cos, rope_sin),
                           cache_position=torch.tensor([0]))
        return result[0] if isinstance(result, tuple) else result


class HeadModel(nn.Module):
    def __init__(self, norm, lm_head):
        super().__init__()
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, hidden):
        return self.lm_head(self.norm(hidden))


def convert_and_save(module, sample_inputs, ct_inputs, ct_outputs, path, precision):
    module.eval()
    with torch.no_grad():
        traced = torch.jit.trace(module, sample_inputs, strict=False)

    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )
    mlmodel.save(path)
    return mlmodel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--output", default="models/disaggregated/")
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

    # Pre-compute RoPE
    with torch.no_grad():
        dummy = torch.zeros(1, 300, hidden_size)
        all_cos, all_sin = model.model.rotary_emb(dummy, torch.arange(300).unsqueeze(0))

    # Identify blocks
    blocks = []
    delta_layers = []
    for i in range(tc.num_hidden_layers):
        if tc.layer_types[i] == "linear_attention":
            delta_layers.append(i)
        else:
            blocks.append({"delta": delta_layers, "attn": i})
            delta_layers = []

    print(f"Blocks: {len(blocks)}")
    for b in blocks:
        print(f"  DeltaNet {b['delta']} -> Attention {b['attn']}")

    # 1. Embed model
    print("\n--- Embed ---")
    embed = EmbedModel(model.model.embed_tokens)
    sample_id = torch.zeros((1, 1), dtype=torch.long)
    path = os.path.join(args.output, "embed.mlpackage")
    convert_and_save(
        embed, (sample_id,),
        [ct.TensorType(name="input_id", shape=(1, 1), dtype=np.int32)],
        [ct.TensorType(name="hidden")],
        path, precision,
    )
    print(f"  Saved {path}")

    # 2. DeltaNet blocks
    for bi, block in enumerate(blocks):
        print(f"\n--- DeltaNet block {bi} (layers {block['delta']}) ---")
        layers = [model.model.layers[i] for i in block["delta"]]
        dn = DeltaNetBlock(layers, block["delta"])

        sample_hidden = torch.randn(1, 1, hidden_size)
        sample_states = []
        ct_in = [ct.TensorType(name="hidden", shape=(1, 1, hidden_size))]
        ct_out = [ct.TensorType(name="hidden_out")]

        for j, li in enumerate(block["delta"]):
            sample_states.append(torch.zeros(1, conv_dim, conv_kernel))
            sample_states.append(torch.zeros(1, n_v_heads, k_head_dim, v_head_dim))
            ct_in.append(ct.TensorType(name=f"conv_state_{j}", shape=(1, conv_dim, conv_kernel)))
            ct_in.append(ct.TensorType(name=f"recurrent_state_{j}", shape=(1, n_v_heads, k_head_dim, v_head_dim)))
            ct_out.append(ct.TensorType(name=f"conv_state_{j}_out"))
            ct_out.append(ct.TensorType(name=f"recurrent_state_{j}_out"))

        path = os.path.join(args.output, f"deltanet_block_{bi}.mlpackage")
        convert_and_save(
            dn, (sample_hidden, *sample_states),
            ct_in, ct_out, path, precision,
        )
        print(f"  Saved {path}")

    # 3. Attention blocks
    rope_dim = all_cos.shape[-1]
    for bi, block in enumerate(blocks):
        li = block["attn"]
        print(f"\n--- Attention block {bi} (layer {li}) ---")
        attn = AttentionBlock(model.model.layers[li], li)

        sample_hidden = torch.randn(1, 1, hidden_size)
        sample_cos = all_cos[:, 0:1, :]
        sample_sin = all_sin[:, 0:1, :]

        path = os.path.join(args.output, f"attention_block_{bi}.mlpackage")
        convert_and_save(
            attn, (sample_hidden, sample_cos, sample_sin),
            [
                ct.TensorType(name="hidden", shape=(1, 1, hidden_size)),
                ct.TensorType(name="rope_cos", shape=(1, 1, rope_dim)),
                ct.TensorType(name="rope_sin", shape=(1, 1, rope_dim)),
            ],
            [ct.TensorType(name="hidden_out")],
            path, precision,
        )
        print(f"  Saved {path}")

    # 4. Head model
    print(f"\n--- Head (norm + lm_head) ---")
    head = HeadModel(model.model.norm, model.lm_head)
    sample_hidden = torch.randn(1, 1, hidden_size)
    path = os.path.join(args.output, "head.mlpackage")
    convert_and_save(
        head, (sample_hidden,),
        [ct.TensorType(name="hidden", shape=(1, 1, hidden_size))],
        [ct.TensorType(name="logits")],
        path, precision,
    )
    print(f"  Saved {path}")

    print(f"\n=== Done: {2 + len(blocks) * 2} models in {args.output} ===")


if __name__ == "__main__":
    main()
