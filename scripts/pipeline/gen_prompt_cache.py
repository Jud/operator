#!/usr/bin/env python3
"""Generate pre-computed system prompt cache.

Runs the system prompt through PyTorch and saves all DeltaNet + attention
states as FP16 .npy files. These are loaded at runtime to skip the
131-token system prompt prefill.

Usage:
    /tmp/coreml-venv/bin/python scripts/pipeline/gen_prompt_cache.py \
        --output models/monolith_kv_fp16/prompt_cache/
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import patches  # noqa
from functional_deltanet import patch_transformers

patch_transformers()

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


SYSTEM_PROMPT = """<|im_start|>system
You clean up raw speech-to-text transcriptions. Remove filler words (um, uh, like), fix grammar, and handle self-corrections. Keep the original meaning. Output ONLY the cleaned text.

Example:
<transcription>um so I was like going to the uh store and I needed to get some like groceries</transcription>
I was going to the store and I needed to get some groceries.

Example:
<transcription>the the meeting is at uh three no wait four o'clock</transcription>
The meeting is at four o'clock.
<|im_end|>
<|im_start|>user
<transcription>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--output", default="models/monolith_kv_fp16/prompt_cache/")
    parser.add_argument("--max-kv", type=int, default=256)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"Loading {args.model_id}...")
    config = AutoConfig.from_pretrained(args.model_id)
    config.text_config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32, config=config)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tc = config.text_config

    input_ids = tokenizer(SYSTEM_PROMPT, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]
    print(f"System prompt: {prompt_len} tokens")

    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        cache = out.past_key_values

    num_layers = tc.num_hidden_layers
    layer_types = tc.layer_types
    delta_indices = [i for i in range(num_layers) if layer_types[i] == "linear_attention"]
    attn_indices = [i for i in range(num_layers) if layer_types[i] == "full_attention"]
    attn_nkv, attn_hd = tc.num_key_value_heads, tc.head_dim

    for j, i in enumerate(delta_indices):
        np.save(f"{args.output}/conv_state_{j}.npy", cache.conv_states[i].contiguous().numpy().astype(np.float16))
        np.save(f"{args.output}/rec_state_{j}.npy", cache.recurrent_states[i].contiguous().numpy().astype(np.float16))

    for j, i in enumerate(attn_indices):
        k = cache.key_cache[i].contiguous().numpy().astype(np.float16)
        v = cache.value_cache[i].contiguous().numpy().astype(np.float16)
        k_pad = np.zeros((1, attn_nkv, args.max_kv, attn_hd), dtype=np.float16)
        v_pad = np.zeros((1, attn_nkv, args.max_kv, attn_hd), dtype=np.float16)
        k_pad[:, :, : k.shape[2], :] = k
        v_pad[:, :, : v.shape[2], :] = v
        np.save(f"{args.output}/key_cache_{j}.npy", k_pad)
        np.save(f"{args.output}/value_cache_{j}.npy", v_pad)

    with torch.no_grad():
        all_cos, all_sin = model.model.rotary_emb(
            torch.zeros(1, args.max_kv, tc.hidden_size), torch.arange(args.max_kv).unsqueeze(0)
        )
    rope_dim = int(all_cos.shape[-1])
    np.save(f"{args.output}/rope_cos.npy", all_cos[0].numpy().astype(np.float32))
    np.save(f"{args.output}/rope_sin.npy", all_sin[0].numpy().astype(np.float32))

    conv_dim = tc.linear_key_head_dim * tc.linear_num_key_heads * 2 + tc.linear_value_head_dim * tc.linear_num_value_heads

    meta = {
        "prompt_len": prompt_len,
        "max_kv": args.max_kv,
        "rope_dim": rope_dim,
        "n_delta": len(delta_indices),
        "n_attn": len(attn_indices),
        "conv_dim": conv_dim,
        "hidden_size": tc.hidden_size,
        "model_id": args.model_id,
    }
    with open(f"{args.output}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    total = sum(os.path.getsize(f"{args.output}/{fn}") for fn in os.listdir(args.output))
    print(f"Saved {len(os.listdir(args.output))} files ({total / 1024 / 1024:.1f} MB) to {args.output}")


if __name__ == "__main__":
    main()
