"""Functional chunked DeltaNet kernel — no in-place ops.

Rewrites torch_chunk_gated_delta_rule for torch.jit.trace compatibility.
Three in-place ops eliminated:
  1. attn[..., i, :i] = ... → row-by-row torch.cat accumulation
  2. .masked_fill_() → .masked_fill() (non-in-place)
  3. core_attn_out[:, :, i] = ... → list accumulation + torch.stack

For N<=64 (chunk_size=64), the chunk loop runs exactly ONCE (single chunk).

Usage:
    /tmp/coreml-venv/bin/python scripts/experiments/chunked-prefill/functional_deltanet_chunked.py
"""

import torch
import torch.nn.functional as F


def l2norm(x, dim=-1, eps=1e-6):
    """L2 normalize along dim, matching HF's l2norm."""
    return F.normalize(x, p=2, dim=dim, eps=eps)


def torch_chunk_gated_delta_rule_functional(
    query, key, value, g, beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """Functional chunked DeltaNet — identical math, no in-place ops.

    Drop-in replacement for torch_chunk_gated_delta_rule.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask_upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay — FP16-safe: use large negative fill instead of tril() before exp()
    # Original: decay_diff.tril().exp() — coremltools FP16 may reorder to exp().tril(),
    # causing exp(+60) overflow. Fix: fill upper triangle with -1e9 before exp.
    g = g.cumsum(dim=-1)
    decay_diff = g.unsqueeze(-1) - g.unsqueeze(-2)  # [..., CS, CS]
    tril_mask = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device))
    decay_safe = torch.where(tril_mask, decay_diff, torch.tensor(-1e9, device=query.device))
    decay_exp = decay_safe.exp()  # lower tri: exp(real values), upper tri: exp(-1e9)≈0
    decay_mask = decay_exp.tril()  # zero upper tri (belt and suspenders)
    attn_raw = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask_upper, 0)

    # ── FIX 1: Resolution matrix via Neumann series with product factorization ──
    # Computes (I - L)^{-1} = I + L + L^2 + ... + L^63 where L = attn_raw (strictly lower triangular).
    # Since L^64 = 0 (strictly lower triangular), the series is exact.
    # Product factorization: (I+L)(I+L^2)(I+L^4)(I+L^8)(I+L^16)(I+L^32)
    # = 5 squarings + 5 matmuls = 10 ops (vs 64 sequential row ops)
    L = attn_raw  # strictly lower triangular [B, H, C, CS, CS]
    eye = torch.eye(chunk_size, dtype=L.dtype, device=L.device)
    L2 = L @ L
    L4 = L2 @ L2
    L8 = L4 @ L4
    L16 = L8 @ L8
    L32 = L16 @ L16
    attn = (eye + L) @ (eye + L2) @ (eye + L4) @ (eye + L8) @ (eye + L16) @ (eye + L32)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    mask_strict_upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # ── FIX 2 & 3: Chunk loop — non-in-place masked_fill + list accumulation ──
    chunk_outputs = []
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        # FIX 2: .masked_fill() instead of .masked_fill_()
        chunk_attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill(mask_strict_upper, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        # FIX 3: append to list instead of core_attn_out[:, :, i] = ...
        chunk_outputs.append(attn_inter + chunk_attn @ v_new)
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_attn_out = torch.stack(chunk_outputs, dim=2)  # [B, H, n_chunks, chunk_size, v_dim]

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# ── Standalone verification ──────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../coreml-convert'))

    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    print("Comparing functional chunked kernel vs original HF implementation...")

    torch.manual_seed(42)
    B, H, N, K, V = 1, 16, 30, 128, 128  # Match Qwen3.5-0.8B DeltaNet config

    query = torch.randn(B, N, H, K)
    key = torch.randn(B, N, H, K)
    value = torch.randn(B, N, H, V)
    g = torch.randn(B, N, H) * 0.1  # small gate values
    beta = torch.sigmoid(torch.randn(B, N, H))
    initial_state = torch.randn(B, H, K, V) * 0.01

    # Run original
    with torch.no_grad():
        out_orig, state_orig = torch_chunk_gated_delta_rule(
            query.clone(), key.clone(), value.clone(), g.clone(), beta.clone(),
            chunk_size=64, initial_state=initial_state.clone(), output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

    # Run functional
    with torch.no_grad():
        out_func, state_func = torch_chunk_gated_delta_rule_functional(
            query.clone(), key.clone(), value.clone(), g.clone(), beta.clone(),
            chunk_size=64, initial_state=initial_state.clone(), output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

    out_diff = (out_orig - out_func).abs().max().item()
    state_diff = (state_orig - state_func).abs().max().item()
    print(f"  Output diff:  {out_diff:.10f}")
    print(f"  State diff:   {state_diff:.10f}")

    if out_diff < 1e-5 and state_diff < 1e-5:
        print("  PASS — functional chunked kernel matches original")
    else:
        print("  FAIL — outputs differ!")

    # Also test without initial state (fresh start)
    with torch.no_grad():
        out_orig2, state_orig2 = torch_chunk_gated_delta_rule(
            query.clone(), key.clone(), value.clone(), g.clone(), beta.clone(),
            chunk_size=64, initial_state=None, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        out_func2, state_func2 = torch_chunk_gated_delta_rule_functional(
            query.clone(), key.clone(), value.clone(), g.clone(), beta.clone(),
            chunk_size=64, initial_state=None, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

    out_diff2 = (out_orig2 - out_func2).abs().max().item()
    state_diff2 = (state_orig2 - state_func2).abs().max().item()
    print(f"\n  (no initial state)")
    print(f"  Output diff:  {out_diff2:.10f}")
    print(f"  State diff:   {state_diff2:.10f}")

    if out_diff2 < 1e-5 and state_diff2 < 1e-5:
        print("  PASS")
    else:
        print("  FAIL")
