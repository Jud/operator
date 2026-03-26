#!/usr/bin/env python3
"""Diagnose the full cleanup pipeline: compare every state handoff against PyTorch.

Tests:
  1. Prompt cache states vs PyTorch prefill
  2. Batch prefill output states vs PyTorch (token-by-token through all layers)
  3. Decode model first step vs PyTorch
  4. Multi-token generation: PyTorch vs CoreML pipeline

Usage:
    /tmp/coreml-venv/bin/python scripts/pipeline/diagnose_pipeline.py
"""

import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import patches  # noqa
from functional_deltanet import patch_transformers

patch_transformers()

import torch
import torch.nn.functional as F
import coremltools as ct
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ── Setup ──

MODEL_ID = "Qwen/Qwen3.5-0.8B"
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

TRANSCRIPTION = "um so I was like going to the uh store"
SUFFIX = "</transcription>\n<|im_end|>\n<|im_start|>assistant\n"
MAX_KV = 256
FIXED_N = 64

print("Loading model...")
config = AutoConfig.from_pretrained(MODEL_ID)
config.text_config._attn_implementation = "eager"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, config=config)
model.eval()
for p in model.parameters():
    p.requires_grad = False
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tc = config.text_config

num_layers = tc.num_hidden_layers
layer_types = tc.layer_types
delta_indices = [i for i in range(num_layers) if layer_types[i] == "linear_attention"]
attn_indices = [i for i in range(num_layers) if layer_types[i] == "full_attention"]
n_kv = tc.num_key_value_heads
hd = tc.head_dim

# Tokenize
sys_ids = tokenizer(SYSTEM_PROMPT, return_tensors="pt").input_ids
text_ids = tokenizer.encode(TRANSCRIPTION)
suffix_ids = tokenizer.encode(SUFFIX)
full_ids = tokenizer(SYSTEM_PROMPT + TRANSCRIPTION + SUFFIX, return_tensors="pt").input_ids
sys_len = sys_ids.shape[1]
n_text = len(text_ids)
n_suffix = len(suffix_ids)
full_len = full_ids.shape[1]

print(f"System: {sys_len} tokens, Text: {n_text}, Suffix: {n_suffix}, Full: {full_len}")

with torch.no_grad():
    all_cos, all_sin = model.model.rotary_emb(
        torch.zeros(1, MAX_KV, tc.hidden_size), torch.arange(MAX_KV).unsqueeze(0)
    )
    rope_dim = int(all_cos.shape[-1])

# ═══════════════════════════════════════════════════════
# TEST 1: PyTorch ground truth — full prompt generation
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 1: PyTorch ground truth")
print("=" * 60)

with torch.no_grad():
    pt_out = model(full_ids, use_cache=True)
    pt_cache = pt_out.past_key_values
    pt_first_token = torch.argmax(pt_out.logits[0, -1]).item()

    # Generate 10 tokens
    pt_tokens = [pt_first_token]
    gen_cache = model(full_ids, use_cache=True).past_key_values
    current = torch.tensor([[pt_first_token]])
    for _ in range(9):
        out = model(current, past_key_values=gen_cache, use_cache=True)
        gen_cache = out.past_key_values
        tok = torch.argmax(out.logits[0, -1]).item()
        pt_tokens.append(tok)
        current = torch.tensor([[tok]])

pt_text = tokenizer.decode(pt_tokens)
print(f"  First token: {pt_first_token} ({repr(tokenizer.decode([pt_first_token]))})")
print(f"  Generated: {repr(pt_text[:80])}")

# ═══════════════════════════════════════════════════════
# TEST 2: Verify prompt cache against PyTorch
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 2: Prompt cache vs PyTorch")
print("=" * 60)

with torch.no_grad():
    sys_out = model(sys_ids, use_cache=True)
    sys_cache = sys_out.past_key_values

cache_dir = "models/monolith_kv_fp16/prompt_cache"
if not os.path.exists(cache_dir):
    print("  SKIP — prompt cache not found")
else:
    max_diff_conv = 0
    max_diff_rec = 0
    max_diff_key = 0
    max_diff_val = 0
    for j, i in enumerate(delta_indices):
        npy_conv = np.load(f"{cache_dir}/conv_state_{j}.npy").astype(np.float32)
        npy_rec = np.load(f"{cache_dir}/rec_state_{j}.npy").astype(np.float32)
        pt_conv = sys_cache.conv_states[i].contiguous().numpy()
        pt_rec = sys_cache.recurrent_states[i].contiguous().numpy()
        max_diff_conv = max(max_diff_conv, np.abs(npy_conv - pt_conv).max())
        max_diff_rec = max(max_diff_rec, np.abs(npy_rec - pt_rec).max())
    for j, i in enumerate(attn_indices):
        npy_key = np.load(f"{cache_dir}/key_cache_{j}.npy").astype(np.float32)
        npy_val = np.load(f"{cache_dir}/value_cache_{j}.npy").astype(np.float32)
        pt_key = sys_cache.key_cache[i].contiguous().numpy()
        pt_val = sys_cache.value_cache[i].contiguous().numpy()
        # Cache is padded, compare only filled positions
        kl = pt_key.shape[2]
        max_diff_key = max(max_diff_key, np.abs(npy_key[:, :, :kl, :] - pt_key).max())
        max_diff_val = max(max_diff_val, np.abs(npy_val[:, :, :kl, :] - pt_val).max())

    print(f"  Conv state diff:  {max_diff_conv:.6f} {'OK' if max_diff_conv < 0.01 else 'BAD'}")
    print(f"  Rec state diff:   {max_diff_rec:.6f} {'OK' if max_diff_rec < 0.01 else 'BAD'}")
    print(f"  Key cache diff:   {max_diff_key:.6f} {'OK' if max_diff_key < 0.01 else 'BAD'}")
    print(f"  Value cache diff: {max_diff_val:.6f} {'OK' if max_diff_val < 0.01 else 'BAD'}")

# ═══════════════════════════════════════════════════════
# TEST 3: Batch prefill output vs PyTorch
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 3: Batch prefill CoreML vs PyTorch")
print("=" * 60)

prefill_path = "models/batch_prefill_fp16/model.mlpackage"
if not os.path.exists(prefill_path):
    print("  SKIP — batch prefill model not found")
else:
    # PyTorch reference: process variable tokens after system prompt
    with torch.no_grad():
        # Full forward up to the end of the prompt
        pt_full = model(full_ids, use_cache=True)
        pt_full_cache = pt_full.past_key_values

    # CoreML batch prefill
    prefill_model = ct.models.MLModel(prefill_path)
    var_ids = text_ids + suffix_ids
    n_var = len(var_ids)
    padded = np.zeros(FIXED_N, dtype=np.int32)
    padded[:n_var] = var_ids

    cos_np = all_cos[0, sys_len:sys_len + FIXED_N, :].numpy().reshape(1, FIXED_N, rope_dim)
    sin_np = all_sin[0, sys_len:sys_len + FIXED_N, :].numpy().reshape(1, FIXED_N, rope_dim)

    mask = np.full((1, 1, FIXED_N, MAX_KV), -1e4, dtype=np.float32)
    for i in range(FIXED_N):
        if i < n_var:
            mask[0, 0, i, : sys_len + i + 1] = 0.0
        else:
            mask[0, 0, i, 0] = 0.0

    mil_in = {
        "input_ids": padded.reshape(1, FIXED_N),
        "seq_len": np.array([n_var], dtype=np.int32),
        "start_pos": np.array([sys_len], dtype=np.int32),
        "rope_cos": cos_np.astype(np.float32),
        "rope_sin": sin_np.astype(np.float32),
        "attn_mask": mask,
    }
    # Load cached states
    for j in range(len(delta_indices)):
        mil_in[f"conv_state_{j}"] = np.load(f"{cache_dir}/conv_state_{j}.npy").astype(np.float32)
        mil_in[f"rec_state_{j}"] = np.load(f"{cache_dir}/rec_state_{j}.npy").astype(np.float32)
    for j in range(len(attn_indices)):
        mil_in[f"key_cache_{j}"] = np.load(f"{cache_dir}/key_cache_{j}.npy").astype(np.float32)
        mil_in[f"value_cache_{j}"] = np.load(f"{cache_dir}/value_cache_{j}.npy").astype(np.float32)

    mil_out = prefill_model.predict(mil_in)

    # Compare logits
    mil_logits = mil_out["logits"]
    pt_logits = pt_full.logits[0, -1].numpy()
    mil_last = mil_logits[0, n_var - 1] if len(mil_logits.shape) == 3 else mil_logits.flatten()
    mil_token = int(np.argmax(mil_last))
    logit_diff = np.abs(mil_last[:len(pt_logits)] - pt_logits).max()
    print(f"  Logit diff: {logit_diff:.6f}")
    print(f"  PT token:   {pt_first_token} ({repr(tokenizer.decode([pt_first_token]))})")
    print(f"  ML token:   {mil_token} ({repr(tokenizer.decode([mil_token]))})")
    print(f"  Match: {'OK' if mil_token == pt_first_token else 'MISMATCH'}")

    # Compare states
    print("\n  State comparison (prefill output vs PT full-forward cache):")
    for j, i in enumerate(delta_indices):
        conv_out = mil_out[f"conv_state_{j}_out"]
        rec_out = mil_out[f"rec_state_{j}_out"]
        pt_conv = pt_full_cache.conv_states[i].contiguous().numpy()
        pt_rec = pt_full_cache.recurrent_states[i].contiguous().numpy()

        # Conv: extract at real token boundary (position n_var, not tail)
        w = conv_out.shape[2]
        conv_sliced = conv_out[:, :, n_var:n_var + 4]
        conv_diff = np.abs(conv_sliced - pt_conv).max()
        rec_diff = np.abs(rec_out - pt_rec).max()

        # Also check what the WRONG extraction gives
        conv_tail = conv_out[:, :, -4:]
        conv_tail_diff = np.abs(conv_tail - pt_conv).max()

        if j < 3 or conv_diff > 0.1:
            print(f"    L{i} conv: correct_slice={conv_diff:.4f}, wrong_tail={conv_tail_diff:.4f}, rec={rec_diff:.4f}")

    for j, i in enumerate(attn_indices):
        key_out = mil_out[f"key_cache_{j}_out"]
        val_out = mil_out[f"value_cache_{j}_out"]
        pt_key = pt_full_cache.key_cache[i].contiguous().numpy()
        pt_val = pt_full_cache.value_cache[i].contiguous().numpy()
        kl = pt_key.shape[2]
        key_diff = np.abs(key_out[:, :, :kl, :] - pt_key).max()
        val_diff = np.abs(val_out[:, :, :kl, :] - pt_val).max()
        if j < 2 or key_diff > 0.1:
            print(f"    L{i} key={key_diff:.4f}, val={val_diff:.4f}")

# ═══════════════════════════════════════════════════════
# TEST 4: Decode model first step with prefill states
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 4: Decode first step — prefill states → decode model")
print("=" * 60)

decode_path = "models/monolith_kv_fp16/model.mlpackage"
if not os.path.exists(decode_path) or not os.path.exists(prefill_path):
    print("  SKIP — models not found")
else:
    decode_model = ct.models.MLModel(decode_path)
    pos = sys_len + n_var

    # Build decode input from prefill output
    dec_in = {
        "input_id": np.array([[mil_token]], dtype=np.int32),
        "rope_cos": all_cos[0, pos:pos + 1, :].numpy().reshape(1, 1, rope_dim).astype(np.float32),
        "rope_sin": all_sin[0, pos:pos + 1, :].numpy().reshape(1, 1, rope_dim).astype(np.float32),
        "cache_position": np.array([pos], dtype=np.int32),
        "attn_mask": np.where(np.arange(MAX_KV) <= pos, 0.0, -1e4).reshape(1, 1, 1, MAX_KV).astype(np.float32),
    }

    # Transfer states from prefill to decode
    for j in range(len(delta_indices)):
        conv_full = mil_out[f"conv_state_{j}_out"]
        # Correct extraction: at real boundary
        dec_in[f"conv_state_{j}"] = conv_full[:, :, n_var:n_var + 4].astype(np.float32)
        dec_in[f"rec_state_{j}"] = mil_out[f"rec_state_{j}_out"].astype(np.float32)
    for j in range(len(attn_indices)):
        dec_in[f"key_cache_{j}"] = mil_out[f"key_cache_{j}_out"].astype(np.float32)
        dec_in[f"value_cache_{j}"] = mil_out[f"value_cache_{j}_out"].astype(np.float32)

    dec_out = decode_model.predict(dec_in)
    dec_logits = dec_out["logits"]
    dec_token = int(np.argmax(dec_logits.flatten()))

    # PyTorch reference: generate one more token after full prompt
    with torch.no_grad():
        pt_next = model(torch.tensor([[pt_first_token]]), past_key_values=pt_full_cache, use_cache=True)
        pt_next_token = torch.argmax(pt_next.logits[0, -1]).item()

    print(f"  Decode token: {dec_token} ({repr(tokenizer.decode([dec_token]))})")
    print(f"  PT ref token: {pt_next_token} ({repr(tokenizer.decode([pt_next_token]))})")
    print(f"  Match: {'OK' if dec_token == pt_next_token else 'MISMATCH'}")

    # Generate 10 tokens with the full CoreML pipeline
    print("\n  Full pipeline generation (10 tokens):")
    pipeline_tokens = [mil_token]
    current_tok = mil_token
    current_pos = pos

    # States for decode loop
    conv_s = {j: dec_in[f"conv_state_{j}"] for j in range(len(delta_indices))}
    rec_s = {j: dec_in[f"rec_state_{j}"] for j in range(len(delta_indices))}
    key_s = {j: dec_in[f"key_cache_{j}"] for j in range(len(attn_indices))}
    val_s = {j: dec_in[f"value_cache_{j}"] for j in range(len(attn_indices))}

    for step in range(9):
        d = {
            "input_id": np.array([[current_tok]], dtype=np.int32),
            "rope_cos": all_cos[0, current_pos:current_pos + 1, :].numpy().reshape(1, 1, rope_dim).astype(np.float32),
            "rope_sin": all_sin[0, current_pos:current_pos + 1, :].numpy().reshape(1, 1, rope_dim).astype(np.float32),
            "cache_position": np.array([current_pos], dtype=np.int32),
            "attn_mask": np.where(np.arange(MAX_KV) <= current_pos, 0.0, -1e4).reshape(1, 1, 1, MAX_KV).astype(np.float32),
        }
        for j in range(len(delta_indices)):
            d[f"conv_state_{j}"] = conv_s[j]
            d[f"rec_state_{j}"] = rec_s[j]
        for j in range(len(attn_indices)):
            d[f"key_cache_{j}"] = key_s[j]
            d[f"value_cache_{j}"] = val_s[j]

        out = decode_model.predict(d)
        tok = int(np.argmax(out["logits"].flatten()))
        if tok == 248044 or tok == 248046:
            break
        pipeline_tokens.append(tok)
        current_tok = tok
        current_pos += 1

        for j in range(len(delta_indices)):
            conv_s[j] = out[f"conv_state_{j}_out"]
            rec_s[j] = out[f"rec_state_{j}_out"]
        for j in range(len(attn_indices)):
            key_s[j] = out[f"key_cache_{j}_out"]
            val_s[j] = out[f"value_cache_{j}_out"]

    pipeline_text = tokenizer.decode(pipeline_tokens)
    print(f"    PT:       {repr(tokenizer.decode(pt_tokens[:10]))}")
    print(f"    Pipeline: {repr(pipeline_text[:80])}")
    print(f"    Token-by-token match: {sum(a == b for a, b in zip(pt_tokens, pipeline_tokens))}/{min(len(pt_tokens), len(pipeline_tokens))}")
