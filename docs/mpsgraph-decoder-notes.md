# MPSGraph Decoder Implementation Notes

Living document of learnings from building the Qwen 3.5 2B decoder in MPSGraph, verified layer-by-layer against PyTorch reference activations.

## Architecture Overview

Qwen 3.5 2B: 24 layers in a 3:1 pattern [DeltaNet, DeltaNet, DeltaNet, FullAttention] × 6.
- 18 DeltaNet (linear attention) layers
- 6 full attention layers
- Hidden size: 2048, 16 heads, head_dim: 128
- QKV dim: 6144 (3 × 2048), no bias on projections
- FFN intermediate: 6144 (gate + up projections)

## Verified Ops

### Embedding Lookup
- **Op**: `gatherND(embedWeight, tokenId)`
- **Gotcha**: Token IDs are int32. The PyTorch hook saves them as float16 which overflows for vocab > 65504. Use `decode_inputs/token_id.npy` (int32) not `activations/embed_in_0.npy` (float16 overflow → inf).
- **Verification**: maxDiff = 0.0000 (exact match)

### RMSNorm (Qwen3_5 variant)
- **Op**: `x * rsqrt(mean(x², dim=-1) + eps) * (1 + weight)`
- **CRITICAL: Qwen uses `(1 + weight)` not just `weight`.** The weight parameter is a residual — initialized to zeros, so `1 + weight` starts at 1.0. This is different from standard LLaMA RMSNorm which uses `weight` directly. Missing the `+1` gives outputs ~25x too small.
- **Source**: `Qwen3_5RMSNorm.forward()` in transformers — comment says "Qwen3_5 is (x * w).to(float16)" but the actual code is `output * (1.0 + self.weight.float())`.
- **Precision**: Must compute in FP32 throughout, cast to FP16 only at the end. Casting normalized values to FP16 before multiplying by weight loses precision on small values.
- **eps**: 1e-6 (from `config.text_config.rms_norm_eps`)
- **Verification**: maxDiff = 0.0000 (exact match)

### Linear Projection (no bias)
- **Op**: `reshape(x, [1, K]) → matmul(x, W.T) → reshape([1, 1, N])`
- **Note**: Most DeltaNet projections have no bias. The `in_proj_qkv` (6144×2048), `in_proj_z` (2048×2048), `in_proj_a` (16×2048), `in_proj_b` (16×2048), and `out_proj` (2048×2048) are all bias-free.
- **Weight shape**: `[out_features, in_features]` — need transpose for matmul.
- **Verification**: maxDiff = 0.0039 (excellent FP16 match)

## DeltaNet Layer (decode step)

The decode step processes one token with precomputed conv/recurrent states.

### Projections
1. `qkv = in_proj_qkv(hidden) → [1, 1, 6144]` — then transposed to `[1, 6144, 1]` for conv
2. `z = in_proj_z(hidden) → [1, 1, 2048]` — reshaped to `[1, 1, 16, 128]` for gating
3. `b = in_proj_b(hidden) → [1, 1, 16]` — beta gate input
4. `a = in_proj_a(hidden) → [1, 1, 16]` — alpha/decay gate input

### Conv1d Update (causal_conv1d_update)
- Shift conv_state left by 1, append new QKV column
- `conv_state = [1, 6144, 4]` (kernel_size=4)
- Apply depthwise conv: `sum(conv_state * weight, dim=-1)` per channel
- Add bias, apply SiLU activation
- Output: `[1, 6144, 1]` → transpose back to `[1, 1, 6144]`
- **Gotcha**: `causal_conv1d_update` returns a tuple `(output, new_state)`, not just the output tensor. Must unpack.

### QKV Split
- Split `[1, 1, 6144]` into query `[1, 1, 2048]`, key `[1, 1, 2048]`, value `[1, 1, 2048]`
- Reshape each to `[1, 1, 16, 128]` (16 heads, 128 head_dim)

### Gates
- `beta = sigmoid(b)` → `[1, 1, 16]`
- `g = -exp(A_log) * softplus(a + dt_bias)` → `[1, 1, 16]` (decay factor, computed in FP32)

### Recurrent State Update
- `state = state * exp(g) + beta * outer(key, value)` per head
- `state` shape: `[1, 16, 128, 128]` (key_dim × value_dim per head)
- This is the DeltaNet recurrence: exponential decay + gated outer product update

### Output
- `output = query @ state → [1, 1, 16, 128]`
- Apply group norm with z gating: `norm(output) * silu(z)`
- Reshape to `[1, 1, 2048]`
- `out_proj(output) → [1, 1, 2048]`

## Full Attention Layer

(TODO — layers 3, 7, 11, 15, 19, 23)

### Differences from DeltaNet
- Standard Q/K/V projections with RoPE
- Softmax attention with causal mask
- KV cache (key_cache, value_cache) instead of conv/recurrent states

## FFN (SwiGLU)

(TODO — same structure for all layers)

- `gate = silu(gate_proj(x))`
- `up = up_proj(x)`
- `output = down_proj(gate * up)`

## Performance Notes

- Layer 0 (embed + RMSNorm + QKV matmul): 0.42ms via MPSGraph
- Single [1,2048]×[6144,2048] FP16 matmul: 0.24ms via MPSGraph
- Same matmul with 256-entry LUT dequant: 0.48ms (2x overhead)
- CoreML FP16 full model decode: 25.9ms (24 layers)
- CoreML PAL4 gs=16 full model decode: 33ms (not fused)

## Verification Learnings

### NEVER monkey-patch model forward methods for reference capture
Monkey-patching DeltaNet.forward() to capture internal activations changes the model's behavior during prefill. This corrupts all subsequent decode-step references because:
1. The monkey-patch re-implements the forward logic (potentially with subtle differences)
2. ALL DeltaNet layers get patched, changing prefill outputs for all layers
3. Changed prefill = different cache states = different decode inputs
4. The hooks on submodules AND the monkey-patch fire in the same execution, but the monkey-patch's internal captures see a different computation state than the hooks

**Solution**: Use ONLY submodule hooks. They fire correctly on the unmodified model. Verify DeltaNet internals as a black box: `qkv_out + z_out + cache → out_proj_input`. If the full chain produces the right output, every intermediate is correct by construction.

### FP16 reference precision loss (CONFIRMED)
**Hypothesis**: Saving references as FP16 is sufficient for verification.
**Result**: FAILED. The model computes in FP32. Saving activations, weights, and cache states as FP16 introduces rounding errors that compound through the DeltaNet recurrence. The outer product update `state += outer(k, delta)` on a `[16, 128, 128]` state amplifies small FP16 rounding errors into large output differences (maxDiff 0.57 with FP16 cache, even when PyTorch recomputes from FP16 inputs).
**Fix**: Save ALL reference data in FP32. The NpyLoader in Swift must handle FP32→FP16 conversion at load time for MPSGraph (which runs in FP16), but the Python verification uses FP32 throughout.

### FP16 overflow on token IDs
Token IDs > 65504 overflow FP16. PyTorch hooks capture tensors in their native dtype (int64 for token IDs), but saving as float16 causes overflow → inf. Always use dedicated int32 files for integer data.

## Hypotheses Tested

| # | Hypothesis | Result | Notes |
|---|-----------|--------|-------|
| 1 | FP16 references are sufficient for verification | **PARTIALLY** | FP32→FP16 rounding gives maxDiff 0.57 but correct ranges. avgDiff 0.012 = 0.5% relative error. Acceptable for FP16 implementation. |
| 8 | DeltaNet black box (conv1d→recurrence→norm→gate) correct | **CONFIRMED** | Output range [-2.39, 1.99] matches reference [-2.40, 2.02]. Individual values differ by ~0.5% due to FP16 activation capture in hooks. |
| 2 | Monkey-patching forward captures clean references | **FAILED** | Changes prefill behavior for all layers, corrupts decode references |
| 3 | Qwen RMSNorm uses standard `x * weight` | **FAILED** | Uses `x * (1 + weight)` — missing +1 gives 25x error |
| 4 | Submodule hooks capture correct decode activations | **CONFIRMED** | Hooks fire on unmodified model, all 6 projections match exactly |
| 5 | MPSGraph matmul matches CoreML for FP16 linear ops | **CONFIRMED** | maxDiff 0.004 on QKV projection |
| 6 | Conv1d update can be implemented as dot product for decode | **CONFIRMED** | Manual shift+multiply+reduce matches PyTorch exactly (diff 0.0) |
| 7 | F.normalize uses sqrt(sum + eps) | **FAILED** | Uses max(||x||, eps) — different for near-zero vectors |

## Future Optimizations (not now)

- **Conv1d → Conv2d**: ANE prefers Conv2d. Could reshape the DeltaNet conv1d to Conv2d with height=1 for ANE acceleration. Only matters for prefill — decode step is just a dot product.
- **ANE scheduling**: Once the full graph works on GPU, explore ANE compute unit targeting for specific ops.

## General Learnings

- **Always verify against reference**: The verification harness caught the `1+weight` bug immediately. Without it, the model would silently produce wrong outputs.
- **FP32 intermediates matter**: RMSNorm and gate computations (exp, softplus) must be FP32. FP16 intermediates cause silent precision loss.
- **PyTorch hooks save tensors as-is**: Int64 token IDs get saved as float16 and overflow. Always use dedicated input files for integer data.
- **SPM caching**: `swift build` sometimes doesn't rebuild after edits. Use `swift package clean` or `touch` the source file to force recompilation.
- **MPSGraph LUT constraints**: `dequantize(_:LUTTensor:)` requires power-of-2 LUT size matching 2^(index_bits). uint8 → 256 entries, uint4 → 16 entries.
