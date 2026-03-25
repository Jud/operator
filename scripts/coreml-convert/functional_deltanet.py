"""Functional DeltaNet recurrence — no in-place ops.

Replaces the HF implementation's in-place slice assignments with
functional list accumulation + torch.stack.

Strategy: Force the model to use the recurrent (per-token) version
instead of the chunked version for prefill. The recurrent version
has simpler math with only one in-place op to fix.
"""

import torch
import torch.nn.functional as F


def torch_recurrent_gated_delta_rule_functional(
    query, key, value, g, beta,
    initial_state=None, output_final_state=True,
    use_qk_l2norm_in_kernel=False,
):
    """Functional replacement for torch_recurrent_gated_delta_rule.

    Identical math to the original but accumulates outputs in a list
    instead of assigning into core_attn_out[:, :, i].
    """
    initial_dtype = query.dtype

    if use_qk_l2norm_in_kernel:
        query = F.normalize(query, p=2, dim=-1, eps=1e-6)
        key = F.normalize(key, p=2, dim=-1, eps=1e-6)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    # FUNCTIONAL: accumulate in list instead of core_attn_out[:, :, i] = ...
    outputs = []
    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out_t = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
        outputs.append(out_t)

    core_attn_out = torch.stack(outputs, dim=2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_causal_conv1d_update_functional(
    hidden_states, conv_state, weight, bias=None, activation=None,
):
    """Functional causal_conv1d_update — no conv_state.copy_()."""
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    # FUNCTIONAL: instead of conv_state.copy_(), return new state as output.
    # The caller must capture and propagate it.
    new_conv_state = hidden_states_new[:, :, -state_len:]
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out, new_conv_state


def patch_transformers():
    """Monkey-patch HF transformers to use functional DeltaNet.

    Forces the recurrent (per-token) version for both prefill and
    incremental modes. Slower than chunked but no in-place ops.
    Also patches the causal_conv1d_update to avoid in-place copy.
    """
    import transformers.models.qwen3_5.modeling_qwen3_5 as mod

    mod.torch_chunk_gated_delta_rule = torch_recurrent_gated_delta_rule_functional
    mod.torch_recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule_functional

    # Patch causal_conv1d_update to be functional
    mod.torch_causal_conv1d_update = torch_causal_conv1d_update_functional

    # Also need to patch the GatedDeltaNet class to handle the new return value
    original_forward = mod.Qwen3_5GatedDeltaNet.forward

    def patched_deltanet_forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        hidden_states = mod.apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_position is not None
        )

        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states:
            # Use our functional version that returns new conv state
            mixed_qkv, new_conv_state = torch_causal_conv1d_update_functional(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
            # Update cache with new state (no in-place copy)
            cache_params.conv_states[self.layer_idx] = new_conv_state
        else:
            if cache_params is not None:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.conv_states[self.layer_idx] = conv_state
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        output = self.out_proj(core_attn_out)
        return output

    mod.Qwen3_5GatedDeltaNet.forward = patched_deltanet_forward

    # Patch attention cache update to write at position instead of concatenating.
    # This keeps the KV cache at fixed size for CoreML compatibility.
    original_cache_update = mod.Qwen3_5DynamicCache.update

    def fixed_size_cache_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if (self.key_cache[layer_idx] is not None
            and self.key_cache[layer_idx].shape[2] > key_states.shape[2]
            and cache_kwargs is not None
            and "cache_position" in cache_kwargs
            and cache_kwargs["cache_position"][0] < self.key_cache[layer_idx].shape[2]):
            cache_position = cache_kwargs["cache_position"]
            # Use torch.where with position mask instead of scatter.
            # scatter is broken in CoreML's Swift runtime, and expand
            # fails in coremltools conversion. torch.where handles
            # broadcasting natively and works in both.
            seq_len = self.key_cache[layer_idx].shape[2]
            positions = torch.arange(seq_len, device=key_states.device)
            mask = (positions == cache_position[0]).view(1, 1, -1, 1)
            new_k = torch.where(mask, key_states, self.key_cache[layer_idx])
            new_v = torch.where(mask, value_states, self.value_cache[layer_idx])
            self.key_cache[layer_idx] = new_k
            self.value_cache[layer_idx] = new_v
            return new_k, new_v
        # Default: concat
        return original_cache_update(self, key_states, value_states, layer_idx, cache_kwargs)

    mod.Qwen3_5DynamicCache.update = fixed_size_cache_update
    print("Patched DeltaNet: functional recurrent + functional conv1d_update + fixed-size KV cache")
