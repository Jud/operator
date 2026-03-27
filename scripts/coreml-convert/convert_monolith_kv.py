"""Convert Qwen3.5 monolith decode model with KV cache to CoreML.

Single model: embed + all 24 layers + norm + lm_head.
Same KV cache approach as convert_split2_kv.py but in one predict() call.

Usage:
    /tmp/coreml-venv/bin/python scripts/coreml-convert/convert_monolith_kv.py \
        Qwen/Qwen3.5-0.8B --output models/monolith_kv/
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


class MonolithDecodeKV(nn.Module):
    """Full decode model with explicit KV cache for attention layers."""

    def __init__(self, hf_model, config):
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

    def forward(self, input_id, rope_cos, rope_sin, cache_position, attn_mask, *states):
        hidden = self.embed(input_id)
        cos = rope_cos * (rope_cos.sum() * 0.0 + 1.0)
        sin = rope_sin * (rope_sin.sum() * 0.0 + 1.0)

        cache = Qwen3_5DynamicCache.__new__(Qwen3_5DynamicCache)
        cache.conv_states = [None] * (self.num_layers + 1)
        cache.recurrent_states = [None] * (self.num_layers + 1)
        cache.key_cache = [None] * (self.num_layers + 1)
        cache.value_cache = [None] * (self.num_layers + 1)
        cache.last_linear_layer = self.delta_indices[-1]
        cache.layer_types = self.layer_types
        cache.transformer_layers = self.attn_indices

        # Populate DeltaNet states, then KV cache
        si = 0
        for i in self.delta_indices:
            cache.conv_states[i] = states[si]
            cache.recurrent_states[i] = states[si + 1]
            si += 2
        n_delta = len(self.delta_indices) * 2
        for j, i in enumerate(self.attn_indices):
            cache.key_cache[i] = states[n_delta + j * 2]
            cache.value_cache[i] = states[n_delta + j * 2 + 1]

        for i, layer in enumerate(self.layers):
            if self.layer_types[i] == "linear_attention":
                result = layer(hidden, position_embeddings=(cos, sin),
                              past_key_values=cache, cache_position=cache_position)
            else:
                result = layer(hidden, position_embeddings=(cos, sin),
                              past_key_values=cache, cache_position=cache_position,
                              attention_mask=attn_mask)
            hidden = result[0] if isinstance(result, tuple) else result

        logits = self.lm_head(self.norm(hidden))

        updated = []
        for i in self.delta_indices:
            updated.append(cache.conv_states[i])
            updated.append(cache.recurrent_states[i])
        for i in self.attn_indices:
            updated.append(cache.key_cache[i])
            updated.append(cache.value_cache[i])

        return (logits,) + tuple(updated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--output", default="models/monolith_kv/")
    parser.add_argument("--max-seq-len", type=int, default=256)
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
    n_kv_heads = tc.num_key_value_heads
    head_dim = tc.head_dim
    hidden_size = tc.hidden_size
    max_seq_len = args.max_seq_len

    with torch.no_grad():
        dummy = torch.zeros(1, max_seq_len, hidden_size)
        all_cos, all_sin = model.model.rotary_emb(dummy, torch.arange(max_seq_len).unsqueeze(0))
    rope_dim = all_cos.shape[-1]

    wrapper = MonolithDecodeKV(model, config)
    wrapper.eval()

    delta_indices = [i for i in range(tc.num_hidden_layers) if tc.layer_types[i] == "linear_attention"]
    attn_indices = [i for i in range(tc.num_hidden_layers) if tc.layer_types[i] == "full_attention"]
    print(f"Layers: {len(delta_indices)} DeltaNet + {len(attn_indices)} attention = {tc.num_hidden_layers}")

    # Sample inputs
    sample = [
        torch.zeros((1, 1), dtype=torch.long),
        all_cos[:, 0:1, :], all_sin[:, 0:1, :],
        torch.tensor([1], dtype=torch.long),
        torch.zeros((1, 1, 1, max_seq_len)),
    ]
    for _ in delta_indices:
        sample.append(torch.zeros(1, conv_dim, conv_kernel))
        sample.append(torch.zeros(1, n_v_heads, k_head_dim, v_head_dim))
    for _ in attn_indices:
        sample.append(torch.zeros(1, n_kv_heads, max_seq_len, head_dim))
        sample.append(torch.zeros(1, n_kv_heads, max_seq_len, head_dim))

    print("Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, tuple(sample), strict=False)
        ref = wrapper(*sample)
        tr = traced(*sample)
        print(f"  Trace verification: max diff = {(ref[0] - tr[0]).abs().max().item():.6f}")

    # CoreML inputs/outputs
    ct_in = [
        ct.TensorType(name="input_id", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="rope_cos", shape=(1, 1, rope_dim)),
        ct.TensorType(name="rope_sin", shape=(1, 1, rope_dim)),
        ct.TensorType(name="cache_position", shape=(1,), dtype=np.int32),
        ct.TensorType(name="attn_mask", shape=(1, 1, 1, max_seq_len)),
    ]
    ct_out = [ct.TensorType(name="logits")]

    for j in range(len(delta_indices)):
        ct_in.append(ct.TensorType(name=f"conv_state_{j}", shape=(1, conv_dim, conv_kernel)))
        ct_in.append(ct.TensorType(name=f"rec_state_{j}", shape=(1, n_v_heads, k_head_dim, v_head_dim)))
        ct_out.append(ct.TensorType(name=f"conv_state_{j}_out"))
        ct_out.append(ct.TensorType(name=f"rec_state_{j}_out"))
    for j in range(len(attn_indices)):
        ct_in.append(ct.TensorType(name=f"key_cache_{j}", shape=(1, n_kv_heads, max_seq_len, head_dim)))
        ct_in.append(ct.TensorType(name=f"value_cache_{j}", shape=(1, n_kv_heads, max_seq_len, head_dim)))
        ct_out.append(ct.TensorType(name=f"key_cache_{j}_out"))
        ct_out.append(ct.TensorType(name=f"value_cache_{j}_out"))

    print(f"Converting ({len(ct_in)} inputs, {len(ct_out)} outputs)...")
    mlmodel = ct.convert(
        traced, inputs=ct_in, outputs=ct_out,
        compute_precision=precision, compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS17, convert_to="mlprogram",
    )

    out_path = os.path.join(args.output, "model.mlpackage")
    mlmodel.save(out_path)
    print(f"Saved {out_path}")

    # Export test inputs
    print("Exporting test inputs...")
    test_dir = os.path.join(args.output, "test_inputs")
    os.makedirs(test_dir, exist_ok=True)

    prompt = "Clean up: um so I was like going to the uh store"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        prefill_cache = out.past_key_values
        first_token = torch.argmax(out.logits[0, -1]).item()

    np.save(f"{test_dir}/input_id.npy", np.array([[first_token]], dtype=np.int32))
    np.save(f"{test_dir}/cache_position.npy", np.array([prompt_len], dtype=np.int32))

    cos = all_cos[:, prompt_len:prompt_len+1, :].numpy().astype(np.float32)
    sin = all_sin[:, prompt_len:prompt_len+1, :].numpy().astype(np.float32)
    np.save(f"{test_dir}/rope_cos.npy", cos)
    np.save(f"{test_dir}/rope_sin.npy", sin)
    np.save(f"{test_dir}/all_rope_cos.npy", all_cos.numpy().astype(np.float32))
    np.save(f"{test_dir}/all_rope_sin.npy", all_sin.numpy().astype(np.float32))

    mask = np.full((1, 1, 1, max_seq_len), -1e4, dtype=np.float32)
    mask[0, 0, 0, :prompt_len + 1] = 0.0
    np.save(f"{test_dir}/attn_mask.npy", mask)

    for j, li in enumerate(delta_indices):
        np.save(f"{test_dir}/conv_state_{j}.npy",
                np.ascontiguousarray(prefill_cache.conv_states[li].numpy().astype(np.float32)))
        np.save(f"{test_dir}/rec_state_{j}.npy",
                np.ascontiguousarray(prefill_cache.recurrent_states[li].numpy().astype(np.float32)))
    for j, li in enumerate(attn_indices):
        k = prefill_cache.key_cache[li].numpy().astype(np.float32)
        v = prefill_cache.value_cache[li].numpy().astype(np.float32)
        k_pad = np.zeros((1, n_kv_heads, max_seq_len, head_dim), dtype=np.float32)
        v_pad = np.zeros((1, n_kv_heads, max_seq_len, head_dim), dtype=np.float32)
        k_pad[:, :, :prompt_len, :] = k
        v_pad[:, :, :prompt_len, :] = v
        np.save(f"{test_dir}/key_cache_{j}.npy", k_pad)
        np.save(f"{test_dir}/value_cache_{j}.npy", v_pad)

    # Reference token
    from verify_kv_decode import build_cache, manual_decode_step
    conv_s = {i: prefill_cache.conv_states[i].clone() for i in delta_indices}
    rec_s = {i: prefill_cache.recurrent_states[i].clone() for i in delta_indices}
    padded_key = {}
    padded_val = {}
    for i in attn_indices:
        k = prefill_cache.key_cache[i]
        v = prefill_cache.value_cache[i]
        kp = torch.zeros(1, n_kv_heads, max_seq_len, head_dim)
        vp = torch.zeros(1, n_kv_heads, max_seq_len, head_dim)
        kp[:, :, :prompt_len, :] = k
        vp[:, :, :prompt_len, :] = v
        padded_key[i] = kp
        padded_val[i] = vp

    with torch.no_grad():
        mask_t = torch.full((1, 1, 1, max_seq_len), -1e4)
        mask_t[0, 0, 0, :prompt_len + 1] = 0.0
        ref_cache = build_cache(tc.num_hidden_layers, tc.layer_types, conv_s, rec_s, padded_key, padded_val)
        hidden = model.model.embed_tokens(torch.tensor([[first_token]]))
        ref_logits = manual_decode_step(model, tc.layer_types, hidden,
                                        all_cos[:, prompt_len:prompt_len+1, :],
                                        all_sin[:, prompt_len:prompt_len+1, :],
                                        ref_cache, prompt_len, mask_t)
        ref_token = torch.argmax(ref_logits[0, -1]).item()

    np.save(f"{test_dir}/ref_token.npy", np.array([ref_token], dtype=np.int32))
    print(f"  Prompt: {repr(prompt)} ({prompt_len} tokens)")
    print(f"  First token: {first_token}, ref next: {ref_token}")
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
