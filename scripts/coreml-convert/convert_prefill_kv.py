"""Convert Qwen3.5 prefill model with cache state outputs to CoreML.

Takes input_ids (padded to seq_len), outputs:
  - logits (last token only)
  - All DeltaNet states (conv + recurrent)
  - All KV cache (padded to max_kv_len)

These outputs feed directly into the monolith decode model.

Model-agnostic: works with any Qwen3.5 size (0.8B, 4B, etc.)

Usage:
    /tmp/coreml-venv/bin/python scripts/coreml-convert/convert_prefill_kv.py \\
        Qwen/Qwen3.5-0.8B --seq-len 512 --output models/prefill_kv/
"""

import argparse
import json
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


class PrefillWithCache(nn.Module):
    """Prefill wrapper that outputs cache states for decode."""

    def __init__(self, model, config, seq_len, max_kv_len):
        super().__init__()
        self.model = model
        tc = config.text_config if hasattr(config, "text_config") else config
        self.num_layers = tc.num_hidden_layers
        self.layer_types = tc.layer_types
        self.delta_indices = [i for i in range(self.num_layers) if tc.layer_types[i] == "linear_attention"]
        self.attn_indices = [i for i in range(self.num_layers) if tc.layer_types[i] == "full_attention"]
        self.seq_len = seq_len
        self.max_kv_len = max_kv_len
        self.n_kv_heads = tc.num_key_value_heads
        self.head_dim = tc.head_dim

        # Pre-compute buffers
        self.register_buffer("position_ids", torch.arange(seq_len).unsqueeze(0))
        self.register_buffer("cache_position", torch.arange(seq_len))

        causal = torch.tril(torch.ones(1, 1, seq_len, seq_len))
        attn_mask = torch.where(causal.bool(), 0.0, float("-inf"))
        self.register_buffer("attention_mask", attn_mask)

        with torch.no_grad():
            dummy_pos = torch.arange(seq_len).unsqueeze(0)
            cos, sin = model.model.rotary_emb(
                torch.zeros(1, seq_len, tc.hidden_size), dummy_pos)
            self.register_buffer("rope_cos", cos)
            self.register_buffer("rope_sin", sin)

        # Patch RoPE to use pre-computed buffers
        wrapper = self
        def patched_rope(x, position_ids):
            return wrapper.rope_cos, wrapper.rope_sin
        self.model.model.rotary_emb.forward = patched_rope

    def forward(self, input_ids):
        out = self.model(
            input_ids=input_ids,
            position_ids=self.position_ids,
            cache_position=self.cache_position,
            attention_mask=self.attention_mask,
            use_cache=True,
        )
        logits = out.logits[:, -1:, :]
        cache = out.past_key_values

        # Collect states in same order as decode model
        states = []
        for i in self.delta_indices:
            states.append(cache.conv_states[i])
            states.append(cache.recurrent_states[i])
        for i in self.attn_indices:
            # Pad KV cache to max_kv_len using F.pad (avoids torch.zeros conversion issue)
            k = cache.key_cache[i]   # [1, n_kv_heads, seq_len, head_dim]
            v = cache.value_cache[i]
            pad_amount = self.max_kv_len - k.shape[2]
            if pad_amount > 0:
                k = torch.nn.functional.pad(k, (0, 0, 0, pad_amount))
                v = torch.nn.functional.pad(v, (0, 0, 0, pad_amount))
            states.append(k)
            states.append(v)

        return (logits,) + tuple(states)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--seq-len", type=int, default=512, help="Fixed input sequence length")
    parser.add_argument("--max-kv-len", type=int, default=256, help="Max KV cache length (must match decode model)")
    parser.add_argument("--output", default="models/prefill_kv/")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu",
                       help="Device for tracing (mps for GPU-accelerated trace)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    precision = ct.precision.FLOAT32 if args.fp32 else ct.precision.FLOAT16
    device = torch.device(args.device)

    print(f"Loading {args.model_id} (device={device})...")
    config = AutoConfig.from_pretrained(args.model_id)
    config.text_config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32, config=config)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
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
    delta_indices = [i for i in range(tc.num_hidden_layers) if tc.layer_types[i] == "linear_attention"]
    attn_indices = [i for i in range(tc.num_hidden_layers) if tc.layer_types[i] == "full_attention"]

    print(f"  {len(delta_indices)} DeltaNet + {len(attn_indices)} attention = {tc.num_hidden_layers} layers")
    print(f"  seq_len={args.seq_len}, max_kv_len={args.max_kv_len}")

    # Init wrapper on CPU (buffer creation), then move to device for trace
    model_cpu = model.cpu()
    wrapper = PrefillWithCache(model_cpu, config, args.seq_len, args.max_kv_len)
    wrapper.eval()
    wrapper = wrapper.to(device)

    # Trace
    sample = torch.zeros((1, args.seq_len), dtype=torch.long, device=device)
    print(f"Tracing on {device}...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (sample,), strict=False)
        ref = wrapper(sample)
        tr = traced(sample)
        print(f"  Trace logits diff: {(ref[0] - tr[0]).abs().max().item():.6f}")
        print(f"  Outputs: 1 logits + {len(ref) - 1} state tensors")

    # CoreML inputs/outputs
    ct_in = [ct.TensorType(name="input_ids", shape=(1, args.seq_len), dtype=np.int32)]
    ct_out = [ct.TensorType(name="logits")]

    for j in range(len(delta_indices)):
        ct_out.append(ct.TensorType(name=f"conv_state_{j}"))
        ct_out.append(ct.TensorType(name=f"rec_state_{j}"))
    for j in range(len(attn_indices)):
        ct_out.append(ct.TensorType(name=f"key_cache_{j}"))
        ct_out.append(ct.TensorType(name=f"value_cache_{j}"))

    print(f"Converting ({len(ct_in)} inputs, {len(ct_out)} outputs)...")
    mlmodel = ct.convert(
        traced, inputs=ct_in, outputs=ct_out,
        compute_precision=precision, compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS17, convert_to="mlprogram",
    )

    out_path = os.path.join(args.output, "prefill.mlpackage")
    mlmodel.save(out_path)
    print(f"Saved {out_path}")

    # Export model config for Swift
    model_config = {
        "model_id": args.model_id,
        "hidden_size": hidden_size,
        "num_layers": tc.num_hidden_layers,
        "n_delta": len(delta_indices),
        "n_attn": len(attn_indices),
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "conv_dim": conv_dim,
        "conv_kernel": conv_kernel,
        "n_v_heads": n_v_heads,
        "k_head_dim": k_head_dim,
        "v_head_dim": v_head_dim,
        "vocab_size": tc.vocab_size,
        "seq_len": args.seq_len,
        "max_kv_len": args.max_kv_len,
    }
    config_path = os.path.join(args.output, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"Saved {config_path}")

    # Verify with a real prompt
    print("\nVerifying with test prompt...")
    prompt = "Clean up: um so I was like going to the uh store"
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    padded = torch.zeros(1, args.seq_len, dtype=torch.long, device=device)
    padded[0, :ids.shape[1]] = ids[0]

    with torch.no_grad():
        pt_out = wrapper(padded)
        pt_token = torch.argmax(pt_out[0][0, 0]).item()

    ml_input = {"input_ids": padded.cpu().numpy().astype(np.int32)}
    ml_out = mlmodel.predict(ml_input)
    ml_logits = ml_out["logits"]
    ml_token = np.argmax(ml_logits.flatten())

    print(f"  PyTorch token: {pt_token} ({repr(tokenizer.decode([pt_token]))})")
    print(f"  CoreML token:  {ml_token} ({repr(tokenizer.decode([ml_token]))})")
    print(f"  Match: {'✓' if pt_token == ml_token else '✗'}")

    # Check state shapes
    for j in range(min(2, len(delta_indices))):
        cs = ml_out[f"conv_state_{j}"]
        rs = ml_out[f"rec_state_{j}"]
        print(f"  conv_state_{j}: {cs.shape}, rec_state_{j}: {rs.shape}")
    for j in range(min(2, len(attn_indices))):
        ks = ml_out[f"key_cache_{j}"]
        print(f"  key_cache_{j}: {ks.shape}")

    print(f"\nDone. Models in {args.output}")


if __name__ == "__main__":
    main()
