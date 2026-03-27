"""Convert batch prefill model with untied embedding/lm_head, then 6-bit palettize lm_head only.

The default HF model ties embed_tokens and lm_head weights. When converted to CoreML,
this becomes a single shared weight. Palettizing it affects both embedding lookup AND
logit projection, compounding error through all 24 layers.

This script:
1. Unties the weights (lm_head gets its own copy)
2. Converts to CoreML (two separate 248320x1024 weights in the graph)
3. Palettizes ONLY the lm_head weight to 6-bit

Usage:
    /tmp/coreml-venv/bin/python scripts/experiments/quip-hadamard-lmhead/convert_untied_6bit.py \
        Qwen/Qwen3.5-0.8B --output models/batch_prefill_untied_6bit/
"""

import argparse
import os
import sys
import time

# Reuse the chunked-prefill pipeline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../experiments/chunked-prefill"))
sys.path.insert(0, os.path.dirname(__file__))
import patches  # noqa
from functional_deltanet import patch_transformers
from functional_deltanet_chunked import torch_chunk_gated_delta_rule_functional  # noqa

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
import coremltools.optimize as cto
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DynamicCache,
    apply_rotary_pos_emb,
)

# Import the wrapper class from the existing conversion script
from convert_batch_prefill import BatchPrefillWrapper


def find_lm_head_weight(mlmodel):
    """Find the lm_head weight — should be distinct from embed after untying."""
    metadata = cto.coreml.get_weights_metadata(mlmodel, weight_threshold=2048)

    vocab_weights = []
    for name, meta in metadata.items():
        shape = tuple(meta.val.shape) if hasattr(meta, 'val') and meta.val is not None else None
        if shape and len(shape) == 2 and 248320 in shape:
            vocab_weights.append((name, shape))
            print(f"  {name}: {shape}")

    if len(vocab_weights) == 2:
        # Two vocab-sized weights = untied successfully
        # The lm_head is the one that's NOT the embed
        embed = [n for n, _ in vocab_weights if 'embed' in n.lower()]
        head = [n for n, _ in vocab_weights if 'embed' not in n.lower()]
        if head:
            return head[0]
        # If both or neither have 'embed', pick the second one (lm_head comes after embed in graph)
        return vocab_weights[1][0]
    elif len(vocab_weights) == 1:
        print("  WARNING: Only one vocab-sized weight found — weights may still be tied!")
        return vocab_weights[0][0]
    else:
        raise RuntimeError(f"Expected 1-2 vocab-sized weights, found {len(vocab_weights)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--output", default="models/batch_prefill_untied_6bit/")
    parser.add_argument("--fixed-n", type=int, default=64)
    parser.add_argument("--max-kv", type=int, default=256)
    parser.add_argument("--nbits", type=int, default=6)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading {args.model_id}...")
    patch_transformers()
    config = AutoConfig.from_pretrained(args.model_id)
    config.text_config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32, config=config)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tc = config.text_config

    # === UNTIE WEIGHTS ===
    print("\nUntying embedding/lm_head weights...")
    print(f"  Before: embed.data_ptr={model.model.embed_tokens.weight.data_ptr()}, "
          f"lm_head.data_ptr={model.lm_head.weight.data_ptr()}, "
          f"same={model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()}")

    with torch.no_grad():
        # Clone + tiny perturbation to prevent CoreML compiler from deduplicating
        # the two constants back into one. The perturbation is ~1e-7 which is
        # below FP16 precision (5e-4) so it has zero effect on output.
        untied_weight = model.lm_head.weight.data.clone()
        untied_weight += torch.randn_like(untied_weight) * 1e-7
        model.lm_head.weight = nn.Parameter(untied_weight)

    print(f"  After:  embed.data_ptr={model.model.embed_tokens.weight.data_ptr()}, "
          f"lm_head.data_ptr={model.lm_head.weight.data_ptr()}, "
          f"same={model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()}")

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

    wrapper = BatchPrefillWrapper(model, config, fixed_n=args.fixed_n, max_kv=args.max_kv, use_neumann=True)
    wrapper.eval()

    delta_indices = wrapper.delta_indices
    attn_indices = wrapper.attn_indices
    print(f"Layers: {len(delta_indices)} DeltaNet + {len(attn_indices)} attention")

    # PyTorch verification
    print("\nPyTorch verification...")
    prompt = "Clean up: um so I was like going to the uh store"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        pt_ref = model(input_ids).logits[0, -1].numpy().copy()
    pt_token = int(np.argmax(pt_ref))
    print(f"  PT reference token: {pt_token} ({repr(tokenizer.decode([pt_token]))})")

    split = 5
    with torch.no_grad():
        cache_out = model(input_ids[:, :split], use_cache=True)
        sys_cache = cache_out.past_key_values

    remaining = input_ids[:, split:]
    n_remaining = remaining.shape[1]
    padded_ids = F.pad(remaining, (0, args.fixed_n - n_remaining))

    cos = all_cos[:, split:split + args.fixed_n, :]
    sin = all_sin[:, split:split + args.fixed_n, :]

    mask = torch.full((1, 1, args.fixed_n, args.max_kv), float("-inf"))
    for i in range(args.fixed_n):
        if i < n_remaining:
            mask[0, 0, i, :split + i + 1] = 0.0
        else:
            mask[0, 0, i, 0] = 0.0

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
        result = wrapper(padded_ids, torch.tensor([n_remaining]), torch.tensor([split]),
                        cos, sin, mask, *init_states)

    batch_logits = result[0][0, n_remaining - 1].numpy()
    batch_token = int(np.argmax(batch_logits))
    diff = np.abs(batch_logits - pt_ref).max()
    print(f"  Batch prefill token: {batch_token} ({repr(tokenizer.decode([batch_token]))})")
    print(f"  Match: {'PASS' if batch_token == pt_token else 'FAIL'}, max diff: {diff:.6f}")

    if batch_token != pt_token:
        print("ABORTING — PyTorch verification failed")
        return

    # Trace
    print("\nTracing...")
    sample_inputs = (padded_ids, torch.tensor([n_remaining]), torch.tensor([split]),
                    cos, sin, mask, *init_states)
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, sample_inputs, strict=False)
    print(f"  Traced in {time.time() - t0:.1f}s")

    # Convert
    precision = ct.precision.FLOAT32 if args.fp32 else ct.precision.FLOAT16
    print(f"\nConverting to CoreML ({'FP32' if args.fp32 else 'FP16'})...")

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
        traced, inputs=ct_inputs, convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18, compute_precision=precision,
    )
    print(f"  Converted in {time.time() - t0:.1f}s")

    # Find and palettize lm_head only
    print(f"\nFinding vocab-sized weights (should be 2 after untying)...")
    lm_head_name = find_lm_head_weight(mlmodel)
    print(f"  Palettizing: {lm_head_name}")

    t0 = time.time()
    op_config = cto.coreml.OpPalettizerConfig(mode="kmeans", nbits=args.nbits)
    opt_config = cto.coreml.OptimizationConfig()
    opt_config.set_op_name(lm_head_name, op_config)
    mlmodel = cto.coreml.palettize_weights(mlmodel, opt_config)
    print(f"  Palettized in {time.time() - t0:.1f}s")

    out_path = os.path.join(args.output, "model.mlpackage")
    mlmodel.save(out_path)

    def dir_size(path):
        return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fns in os.walk(path) for f in fns)

    size_mb = dir_size(out_path) / 1e6
    print(f"\nSaved {out_path} ({size_mb:.1f} MB)")

    # CoreML verification
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
    logits_key = max(mil_out.keys(), key=lambda k: mil_out[k].size if hasattr(mil_out[k], 'size') else 0)
    mil_logits = mil_out[logits_key]

    if len(mil_logits.shape) == 3:
        mil_last = mil_logits[0, n_remaining - 1]
    else:
        mil_last = mil_logits[n_remaining - 1]

    mil_token = int(np.argmax(mil_last))
    mil_diff = np.abs(mil_last[:len(pt_ref)].astype(np.float32) - pt_ref).max()
    print(f"  CoreML token: {mil_token} ({repr(tokenizer.decode([mil_token]))})")
    print(f"  CoreML vs PT max diff: {mil_diff:.6f}")
    print(f"  Match: {'PASS' if mil_token == pt_token else 'FAIL'}")

    # Compare against FP16 tied model
    print(f"\n{'=' * 50}")
    print(f"Untied + 6-bit lm_head model: {size_mb:.1f} MB")
    print(f"Token: PT={pt_token}, CoreML={mil_token}, match={'YES' if mil_token == pt_token else 'NO'}")
    print(f"Max logit diff vs PT: {mil_diff:.6f}")


if __name__ == "__main__":
    main()
