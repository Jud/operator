#!/usr/bin/env python3
"""MTP probe for Qwen3.5-0.8B: measure hit rate and latency of speculative decoding.

Loads the main model, extracts MTP weights from the checkpoint, and runs
autoregressive generation on a cleanup task.  At each step the MTP head
predicts the *next-next* token; we compare that with what the main model
actually produces on the following step.

The MTP module requires full-sequence context (its attention layer is causal
over the full sequence of fused hidden+embedding representations).  During
prefill we process all prompt positions; during generation we accumulate
the growing sequence and re-run MTP over the full context (cheap since total
sequence length is ~100 tokens for this task).

Run:  /tmp/coreml-venv/bin/python mtp_probe.py
"""

import sys, os, time

# Find project root: walk up from this file to the first dir containing scripts/
_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
for _ in range(10):
    if os.path.isdir(os.path.join(_root, "scripts", "coreml-convert")):
        break
    _root = os.path.dirname(_root)
else:
    # Fallback: the main repo checkout (worktree may not have scripts/)
    _root = os.path.expanduser("~/Projects/operator")
sys.path.insert(0, os.path.join(_root, "scripts", "coreml-convert"))

import patches                          # noqa: F401  (registers coremltools ops)
from functional_deltanet import patch_transformers
patch_transformers()

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# 1. Qwen3.5 RMSNorm: (1 + weight) * rms_norm(x), weight init = 0
# ---------------------------------------------------------------------------

class Qwen35RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * (1.0 + self.weight.float())).type_as(x)


# ---------------------------------------------------------------------------
# 2. MTP Attention (full GQA with output gate)
# ---------------------------------------------------------------------------

class MTPAttention(nn.Module):
    """
    Full GQA attention matching Qwen3.5 Attention with output gate.

    q_proj: [4096, 1024] = 8 heads * 256 head_dim * 2 (query + gate)
    k_proj: [512, 1024]  = 2 KV heads * 256 head_dim
    v_proj: [512, 1024]  = 2 KV heads * 256 head_dim
    o_proj: [1024, 2048] = hidden_size, 8 heads * 256 head_dim
    """

    def __init__(self, hidden_size=1024, num_heads=8, num_kv_heads=2, head_dim=256, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads  # 4
        self.scaling = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.q_norm = Qwen35RMSNorm(head_dim, eps=eps)
        self.k_norm = Qwen35RMSNorm(head_dim, eps=eps)

    def forward(self, hidden_states, position_embeddings):
        """
        hidden_states: (batch, seq, hidden_size)
        position_embeddings: (cos, sin) each (batch, seq, rotary_dim)
        """
        bsz, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        # Q projection -> split into query and gate
        qg = self.q_proj(hidden_states)  # (B, S, num_heads * head_dim * 2)
        qg = qg.view(bsz, seq_len, self.num_heads, self.head_dim * 2)
        query_states, gate = qg.chunk(2, dim=-1)  # each (B, S, num_heads, head_dim)
        gate = gate.reshape(bsz, seq_len, -1)  # (B, S, num_heads * head_dim)

        # QK norms (applied per-head before RoPE)
        query_states = self.q_norm(query_states)  # (B, S, num_heads, head_dim)
        query_states = query_states.transpose(1, 2)  # (B, num_heads, S, head_dim)

        # K projection + norm
        key_states = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        key_states = self.k_norm(key_states)
        key_states = key_states.transpose(1, 2)  # (B, num_kv_heads, S, head_dim)

        # V projection
        value_states = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        value_states = value_states.transpose(1, 2)  # (B, num_kv_heads, S, head_dim)

        # Apply partial RoPE (only first rotary_dim dims get rotated)
        rotary_dim = cos.shape[-1]
        q_rot, q_pass = query_states[..., :rotary_dim], query_states[..., rotary_dim:]
        k_rot, k_pass = key_states[..., :rotary_dim], key_states[..., rotary_dim:]

        cos_unsq = cos.unsqueeze(1)  # (B, 1, S, rotary_dim)
        sin_unsq = sin.unsqueeze(1)

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = q_rot * cos_unsq + rotate_half(q_rot) * sin_unsq
        k_rot = k_rot * cos_unsq + rotate_half(k_rot) * sin_unsq

        query_states = torch.cat([q_rot, q_pass], dim=-1)
        key_states = torch.cat([k_rot, k_pass], dim=-1)

        # Expand KV heads for GQA
        key_states = key_states.repeat_interleave(self.num_kv_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention (with causal mask)
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            is_causal=(seq_len > 1),
            scale=self.scaling,
        )

        # Reshape and apply gate
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)

        # Output projection
        attn_output = self.o_proj(attn_output)
        return attn_output


# ---------------------------------------------------------------------------
# 3. MTP Transformer Block (pre-norm with residual)
# ---------------------------------------------------------------------------

class MTPTransformerBlock(nn.Module):
    def __init__(self, hidden_size=1024, intermediate_size=3584, num_heads=8,
                 num_kv_heads=2, head_dim=256, eps=1e-6):
        super().__init__()
        self.input_layernorm = Qwen35RMSNorm(hidden_size, eps=eps)
        self.self_attn = MTPAttention(hidden_size, num_heads, num_kv_heads, head_dim, eps)
        self.post_attention_layernorm = Qwen35RMSNorm(hidden_size, eps=eps)

        # SwiGLU MLP
        self.mlp_gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.mlp_up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.mlp_down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states, position_embeddings):
        # Pre-norm + attention + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        # Pre-norm + MLP + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        gate = F.silu(self.mlp_gate_proj(hidden_states))
        up = self.mlp_up_proj(hidden_states)
        hidden_states = self.mlp_down_proj(gate * up)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# 4. Full MTP Module
# ---------------------------------------------------------------------------

class MTPModule(nn.Module):
    def __init__(self, hidden_size=1024, intermediate_size=3584, num_heads=8,
                 num_kv_heads=2, head_dim=256, eps=1e-6):
        super().__init__()
        self.pre_fc_norm_hidden = Qwen35RMSNorm(hidden_size, eps=eps)
        self.pre_fc_norm_embedding = Qwen35RMSNorm(hidden_size, eps=eps)
        self.fc = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.layer = MTPTransformerBlock(
            hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, eps
        )
        self.norm = Qwen35RMSNorm(hidden_size, eps=eps)

    def fuse(self, hidden_states, embeddings):
        """Fuse hidden states and embeddings through norm + fc.

        hidden_states: (B, S, hidden_size)
        embeddings:    (B, S, hidden_size)
        Returns: (B, S, hidden_size) fused representations.

        NOTE: The fc input order is cat(e_norm, h_norm), matching the
        Deepseek-V3 eh_proj convention (embed first, hidden second).
        """
        h = self.pre_fc_norm_hidden(hidden_states)
        e = self.pre_fc_norm_embedding(embeddings)
        return self.fc(torch.cat([e, h], dim=-1))

    def forward(self, fused_seq, position_embeddings, embed_weight):
        """Run transformer + head on the full fused sequence.

        fused_seq: (B, S, hidden_size) — accumulated fused representations
        position_embeddings: (cos, sin) for all S positions
        embed_weight: (vocab_size, hidden_size) — tied lm_head weight
        Returns: logits (B, S, vocab_size)
        """
        refined = self.layer(fused_seq, position_embeddings)
        normed = self.norm(refined)
        return F.linear(normed, embed_weight)  # x @ embed.T


# ---------------------------------------------------------------------------
# 5. Weight loading
# ---------------------------------------------------------------------------

def load_mtp_weights(mtp_module: MTPModule, safetensors_path: str):
    """Load MTP weights from safetensors checkpoint."""
    with safe_open(safetensors_path, framework="pt") as f:
        weight_map = {
            "mtp.pre_fc_norm_hidden.weight": mtp_module.pre_fc_norm_hidden.weight,
            "mtp.pre_fc_norm_embedding.weight": mtp_module.pre_fc_norm_embedding.weight,
            "mtp.fc.weight": mtp_module.fc.weight,
            "mtp.layers.0.input_layernorm.weight": mtp_module.layer.input_layernorm.weight,
            "mtp.layers.0.self_attn.q_proj.weight": mtp_module.layer.self_attn.q_proj.weight,
            "mtp.layers.0.self_attn.k_proj.weight": mtp_module.layer.self_attn.k_proj.weight,
            "mtp.layers.0.self_attn.v_proj.weight": mtp_module.layer.self_attn.v_proj.weight,
            "mtp.layers.0.self_attn.o_proj.weight": mtp_module.layer.self_attn.o_proj.weight,
            "mtp.layers.0.self_attn.q_norm.weight": mtp_module.layer.self_attn.q_norm.weight,
            "mtp.layers.0.self_attn.k_norm.weight": mtp_module.layer.self_attn.k_norm.weight,
            "mtp.layers.0.post_attention_layernorm.weight": mtp_module.layer.post_attention_layernorm.weight,
            "mtp.layers.0.mlp.gate_proj.weight": mtp_module.layer.mlp_gate_proj.weight,
            "mtp.layers.0.mlp.up_proj.weight": mtp_module.layer.mlp_up_proj.weight,
            "mtp.layers.0.mlp.down_proj.weight": mtp_module.layer.mlp_down_proj.weight,
            "mtp.norm.weight": mtp_module.norm.weight,
        }
        loaded = 0
        for key, param in weight_map.items():
            tensor = f.get_tensor(key)
            param.data.copy_(tensor)
            loaded += 1

    print(f"Loaded {loaded} MTP weight tensors from checkpoint")


# ---------------------------------------------------------------------------
# 6. Main probe
# ---------------------------------------------------------------------------

def main():
    device = "cpu"  # CPU for deterministic results

    # ------------------------------------------------------------------
    # Load tokenizer and model
    # ------------------------------------------------------------------
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", torch_dtype=torch.float32
    )
    model.eval()
    model.to(device)
    text_model = model.model  # Qwen3_5TextModel

    # ------------------------------------------------------------------
    # Build MTP module and load weights
    # ------------------------------------------------------------------
    print("Loading MTP weights...")
    mtp = MTPModule(
        hidden_size=1024, intermediate_size=3584,
        num_heads=8, num_kv_heads=2, head_dim=256, eps=1e-6,
    )
    snapshot_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots"
    )
    snapshot_hash = os.listdir(snapshot_dir)[0]
    safetensors_path = os.path.join(
        snapshot_dir, snapshot_hash,
        "model.safetensors-00001-of-00001.safetensors"
    )
    load_mtp_weights(mtp, safetensors_path)
    mtp.eval()
    mtp.to(device)

    embed_weight = text_model.embed_tokens.weight   # (vocab, 1024)
    rotary_emb = text_model.rotary_emb

    # ------------------------------------------------------------------
    # Build prompt
    # ------------------------------------------------------------------
    prompt = (
        '<|im_start|>user\n'
        'Clean up this transcription: '
        '"Um, I think we should, like, you know, delete the local copy, um, '
        "because it's, it's going to be, uh, symlinked anyway.\"\n"
        '<|im_end|>\n'
        '<|im_start|>assistant\n'
    )
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(input_ids)
    print(f"Prompt: {prompt_len} tokens")

    # ------------------------------------------------------------------
    # Hook to capture the FULL pre-norm hidden state (all positions)
    # ------------------------------------------------------------------
    pre_norm_hidden = {}

    def hook_fn(module, inp, out):
        pre_norm_hidden["state"] = inp[0].detach().clone()

    handle = text_model.norm.register_forward_hook(hook_fn)

    # ------------------------------------------------------------------
    # Phase 1: Prefill — get hidden states for all prompt positions
    # ------------------------------------------------------------------
    print("Running prefill...")

    ids = torch.tensor([input_ids], device=device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)

    # pre_norm_hidden["state"] is (1, prompt_len, 1024) — all positions, pre-norm
    prompt_hidden = pre_norm_hidden["state"]  # (1, prompt_len, 1024)

    # Build MTP fused sequence for prompt positions.
    # MTP at position i takes:
    #   hidden = main model's hidden state at position i
    #   embed  = embedding of the token that the main model predicts at position i
    #           (which is the input token at position i+1 during teacher forcing,
    #            or equivalently, embed_tokens(input_ids[i+1]) for prompt positions)
    #
    # For prompt positions 0..prompt_len-2, the "next token" is input_ids[i+1].
    # Position prompt_len-1 uses the model's actual prediction (first generated token).
    first_token = out.logits[0, -1].argmax().item()

    # Shifted embeddings for MTP: embed(input_ids[1]), ..., embed(input_ids[-1]), embed(first_token)
    shifted_ids = input_ids[1:] + [first_token]
    shifted_embeds = text_model.embed_tokens(
        torch.tensor([shifted_ids], device=device)
    )  # (1, prompt_len, 1024)

    # Fuse all prompt positions
    with torch.no_grad():
        prompt_fused = mtp.fuse(prompt_hidden, shifted_embeds)  # (1, prompt_len, 1024)

    # ------------------------------------------------------------------
    # Phase 2: Autoregressive generation with MTP tracking
    # ------------------------------------------------------------------
    max_gen_tokens = 50
    eos_id = tokenizer.eos_token_id
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_end_id = im_end_ids[0] if im_end_ids else None

    generated_tokens = [first_token]
    mtp_predictions = []   # (mtp_pred_token_id, mtp_latency_ms)
    results = []           # (step, main_token_str, mtp_pred_str, hit, mtp_ms)

    # Accumulated fused sequence (starts with prompt positions)
    fused_seq = prompt_fused  # (1, prompt_len, 1024)

    # For the first MTP run, we use the prompt fused sequence.
    # The MTP prediction at the last position predicts token at prompt_len+1.
    # Compute position embeddings for the full fused sequence.
    def get_position_embeddings(seq_len):
        """Get (cos, sin) for positions 0..seq_len-1."""
        positions = torch.arange(seq_len, device=device)
        pos_3d = positions.unsqueeze(0).unsqueeze(0).expand(3, 1, -1)  # (3, 1, seq_len)
        dummy_x = torch.zeros(1, seq_len, 1024, device=device)
        return rotary_emb(dummy_x, pos_3d)

    print(f"\nGenerating (greedy, max {max_gen_tokens} tokens)...\n")

    with torch.no_grad():
        # First MTP prediction (from prefill)
        cos, sin = get_position_embeddings(fused_seq.shape[1])
        t0 = time.perf_counter()
        mtp_logits = mtp(fused_seq, (cos, sin), embed_weight)
        t1 = time.perf_counter()
        mtp_token = mtp_logits[0, -1].argmax().item()
        mtp_predictions.append((mtp_token, (t1 - t0) * 1000))

        # Extend main model input for generation
        ids = torch.cat([ids, torch.tensor([[first_token]], device=device)], dim=1)

        for step in range(1, max_gen_tokens):
            # --- Main model forward (full sequence, no KV cache for simplicity) ---
            out = model(ids, output_hidden_states=True)
            main_logits = out.logits[0, -1]
            main_token = main_logits.argmax().item()

            # Check if previous MTP prediction matched this token
            prev_mtp_pred, prev_mtp_ms = mtp_predictions[-1]
            hit = prev_mtp_pred == main_token
            prev_main_str = tokenizer.decode([generated_tokens[-1]])
            prev_mtp_str = tokenizer.decode([prev_mtp_pred])
            results.append((
                step - 1,
                prev_main_str,
                prev_mtp_str,
                hit,
                prev_mtp_ms,
            ))

            generated_tokens.append(main_token)

            # Check for EOS
            if main_token == eos_id or main_token == im_end_id:
                break

            # --- Extend MTP fused sequence ---
            # New position: hidden state from last position + embed of just-predicted token
            new_hidden = pre_norm_hidden["state"][:, -1:, :]  # (1, 1, 1024)
            new_embed = text_model.embed_tokens(
                torch.tensor([[main_token]], device=device)
            )
            new_fused = mtp.fuse(new_hidden, new_embed)  # (1, 1, 1024)
            fused_seq = torch.cat([fused_seq, new_fused], dim=1)

            # --- MTP forward on full fused sequence ---
            cos, sin = get_position_embeddings(fused_seq.shape[1])
            t0 = time.perf_counter()
            mtp_logits = mtp(fused_seq, (cos, sin), embed_weight)
            t1 = time.perf_counter()
            mtp_token = mtp_logits[0, -1].argmax().item()
            mtp_predictions.append((mtp_token, (t1 - t0) * 1000))

            # Extend main model input
            ids = torch.cat([ids, torch.tensor([[main_token]], device=device)], dim=1)

    handle.remove()

    # ------------------------------------------------------------------
    # Print results table
    # ------------------------------------------------------------------
    print(f"{'Step':>4} | {'Main token':<16} | {'MTP predicted':<16} | Hit | MTP ms")
    print(f"{'----':>4}-+-{'-'*16}-+-{'-'*16}-+-----+-------")

    hits = 0
    total = len(results)
    mtp_latencies = []

    for step, main_tok_str, mtp_tok_str, hit, mtp_ms in results:
        hit_str = "Y" if hit else "N"
        if hit:
            hits += 1
        mtp_latencies.append(mtp_ms)
        print(f"{step:>4} | {main_tok_str:<16} | {mtp_tok_str:<16} | {hit_str:>3} | {mtp_ms:5.1f}")

    # ------------------------------------------------------------------
    # Top-5 sanity check from last MTP prediction
    # ------------------------------------------------------------------
    if mtp_predictions:
        _, last_mtp_ms = mtp_predictions[-1]
        with torch.no_grad():
            cos, sin = get_position_embeddings(fused_seq.shape[1])
            logits = mtp(fused_seq, (cos, sin), embed_weight)
            probs = F.softmax(logits[0, -1], dim=-1)
            top5 = probs.topk(5)
        print("\nMTP top-5 tokens (last position, sanity check):")
        for i in range(5):
            tid = top5.indices[i].item()
            prob = top5.values[i].item()
            print(f"  {i+1}. {tokenizer.decode([tid])!r:20s} (id={tid:6d}, prob={prob:.4f})")

    # ------------------------------------------------------------------
    # Benchmark: incremental MTP cost (single new position with KV cache)
    # ------------------------------------------------------------------
    # The loop above recomputes full attention each step (no KV cache),
    # inflating latency.  Estimate the true incremental cost by timing
    # just fuse + single-token MTP forward (1 position, simulating KV
    # cache by running on the last fused token alone).
    print("\nBenchmarking incremental MTP cost (single-position forward)...")
    warmup = 3
    trials = 20
    with torch.no_grad():
        # Use the last step's data
        dummy_hidden = pre_norm_hidden.get("state", torch.randn(1, 1, 1024, device=device))
        if dummy_hidden.dim() == 3 and dummy_hidden.shape[1] > 1:
            dummy_hidden = dummy_hidden[:, -1:, :]
        dummy_embed = text_model.embed_tokens(
            torch.tensor([[generated_tokens[-1]]], device=device)
        )
        # Single position embedding
        single_pos = torch.tensor([fused_seq.shape[1] - 1], device=device)
        single_pos_3d = single_pos.unsqueeze(0).unsqueeze(0).expand(3, 1, -1)
        dummy_x = torch.zeros(1, 1, 1024, device=device)
        cos1, sin1 = rotary_emb(dummy_x, single_pos_3d)

        for _ in range(warmup):
            fused1 = mtp.fuse(dummy_hidden, dummy_embed)
            _ = mtp(fused1, (cos1, sin1), embed_weight)

        incr_times = []
        for _ in range(trials):
            t0 = time.perf_counter()
            fused1 = mtp.fuse(dummy_hidden, dummy_embed)
            _ = mtp(fused1, (cos1, sin1), embed_weight)
            t1 = time.perf_counter()
            incr_times.append((t1 - t0) * 1000)

    incr_avg_ms = sum(incr_times) / len(incr_times)
    print(f"  Incremental MTP (fuse + 1-position forward): {incr_avg_ms:.2f}ms "
          f"(over {trials} trials)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)

    generated_text = tokenizer.decode(generated_tokens)
    print(f"Generated text: {generated_text}")
    print()

    if total > 0:
        hit_rate = hits / total
        avg_mtp_ms_full = sum(mtp_latencies) / len(mtp_latencies)
        baseline_tok_s = 58.0  # from hypothesis doc (CoreML on ANE)
        baseline_step_ms = 1000.0 / baseline_tok_s  # ~17.2ms per step

        # MTP is 1 transformer layer vs 24 in the main model.
        # Estimate CoreML MTP overhead as ~1/24 * baseline + small fc/norm cost.
        # Conservative: ~2ms on CoreML (1 layer / 24 layers * 17.2ms + overhead)
        estimated_coreml_mtp_ms = baseline_step_ms / 24.0 * 2  # ~1.4ms

        def compute_speedup(mtp_ms):
            """Compute speculative decoding speedup.

            On each step we run main model + MTP.
            Hit  (prob=hit_rate): 2 tokens for (baseline + mtp) ms
            Miss (prob=1-hit_rate): 1 token for (baseline + mtp) ms
            """
            step_ms = baseline_step_ms + mtp_ms
            expected_ms_per_tok = step_ms * (1 - hit_rate / 2)
            eff_tok_s = 1000.0 / expected_ms_per_tok
            return eff_tok_s / baseline_tok_s, eff_tok_s

        speedup_cpu, eff_cpu = compute_speedup(incr_avg_ms)
        speedup_coreml, eff_coreml = compute_speedup(estimated_coreml_mtp_ms)

        print(f"Hit rate: {hits}/{total} = {hit_rate*100:.1f}%")
        print(f"Avg MTP latency (full recompute, PyTorch CPU): {avg_mtp_ms_full:.1f}ms")
        print(f"Incremental MTP latency (PyTorch CPU, no KV cache): {incr_avg_ms:.2f}ms")
        print(f"Estimated MTP latency (CoreML/ANE, 1 layer): ~{estimated_coreml_mtp_ms:.1f}ms")
        print()
        print(f"Projected speedup (PyTorch CPU baseline): {speedup_cpu:.2f}x "
              f"(effective {eff_cpu:.0f} tok/s vs {baseline_tok_s:.0f} tok/s)")
        print(f"Projected speedup (CoreML/ANE estimate): {speedup_coreml:.2f}x "
              f"(effective {eff_coreml:.0f} tok/s vs {baseline_tok_s:.0f} tok/s)")
    else:
        print("No MTP predictions to evaluate (generation too short)")


if __name__ == "__main__":
    main()
