"""PyTorch-only correctness test: Hadamard rotation + simulated low-bit quantization on lm_head.

Compares top-1 token accuracy across configurations:
  1. FP16 baseline (no quantization)
  2. 2-bit quantization WITHOUT Hadamard rotation (control)
  3. 2-bit quantization WITH Hadamard rotation
  4. 4-bit quantization WITH Hadamard rotation

Usage:
    /tmp/coreml-venv/bin/python scripts/experiments/quip-hadamard-lmhead/benchmark.py \
        Qwen/Qwen3.5-0.8B
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import patches  # noqa: E402, F401

import torch
import numpy as np
from functional_deltanet import patch_transformers

patch_transformers()

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig  # noqa: E402


def hadamard_matrix(n: int) -> torch.Tensor:
    """Construct normalized Hadamard matrix of size n (must be power of 2)."""
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H / math.sqrt(n)


def lloyd_max_codebook(nbits: int, d: int) -> torch.Tensor:
    """Lloyd-Max optimal centroids for the post-rotation distribution.

    After Hadamard rotation, each coordinate is approximately N(0, sigma^2)
    where sigma^2 = ||w||^2 / d. For large d, the normalized coordinates
    w_rot / sigma are approximately N(0, 1).

    We use the well-known Lloyd-Max centroids for N(0,1):
    - 1-bit: [-0.7979, 0.7979]
    - 2-bit: [-1.5104, -0.4528, 0.4528, 1.5104]
    - 3-bit: [-2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520]
    - 4-bit: well-known 16 centroids

    Returns centroids in normalized form (caller must scale by sigma).
    """
    codebooks = {
        1: [-0.7979, 0.7979],
        2: [-1.5104, -0.4528, 0.4528, 1.5104],
        3: [-2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520],
        4: [-2.7326, -2.0690, -1.6180, -1.2562, -0.9424, -0.6568, -0.3881, -0.1284,
            0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6180, 2.0690, 2.7326],
    }
    return torch.tensor(codebooks[nbits])


def simulate_quantize(weight: torch.Tensor, nbits: int) -> torch.Tensor:
    """Simulate nbits scalar quantization using Lloyd-Max centroids.

    For each row, compute sigma = ||row|| / sqrt(d), normalize,
    quantize to nearest centroid, then denormalize.
    """
    d = weight.shape[1]
    centroids_norm = lloyd_max_codebook(nbits, d).to(weight.device)  # [2^nbits]

    # Per-row sigma
    sigma = weight.norm(dim=1, keepdim=True) / math.sqrt(d)  # [vocab, 1]
    sigma = sigma.clamp(min=1e-8)

    # Normalize
    w_norm = weight / sigma  # [vocab, d]

    # Quantize: find nearest centroid for each element
    # w_norm: [vocab, d], centroids: [2^nbits]
    dists = (w_norm.unsqueeze(-1) - centroids_norm.unsqueeze(0).unsqueeze(0)) ** 2  # [vocab, d, 2^nbits]
    indices = dists.argmin(dim=-1)  # [vocab, d]

    # Dequantize
    w_quant_norm = centroids_norm[indices]  # [vocab, d]
    w_quant = w_quant_norm * sigma

    return w_quant


def simulate_kmeans_quantize(weight: torch.Tensor, nbits: int) -> torch.Tensor:
    """Simulate nbits quantization using per-tensor k-means (what coremltools does)."""
    from sklearn.cluster import KMeans

    flat = weight.numpy().flatten()
    n_clusters = 2 ** nbits

    kmeans = KMeans(n_clusters=n_clusters, n_init=1, max_iter=50, random_state=42)
    kmeans.fit(flat.reshape(-1, 1))
    centroids = torch.tensor(kmeans.cluster_centers_.flatten())
    labels = torch.tensor(kmeans.labels_)

    w_quant_flat = centroids[labels]
    return w_quant_flat.reshape(weight.shape).float()


def generate_tokens(model, tokenizer, prompt: str, n_tokens: int) -> list[int]:
    """Generate n_tokens autoregressively, return token IDs."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    tokens = []
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        cache = out.past_key_values
        next_id = torch.argmax(out.logits[0, -1]).item()
        tokens.append(next_id)

        for _ in range(n_tokens - 1):
            out = model(torch.tensor([[next_id]]), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            next_id = torch.argmax(out.logits[0, -1]).item()
            tokens.append(next_id)

    return tokens


def generate_with_modified_head(model, tokenizer, prompt: str, n_tokens: int,
                                original_weight: torch.Tensor,
                                modified_weight: torch.Tensor) -> list[int]:
    """Generate tokens with a temporarily modified lm_head weight."""
    with torch.no_grad():
        model.lm_head.weight.copy_(modified_weight)
        tokens = generate_tokens(model, tokenizer, prompt, n_tokens)
        model.lm_head.weight.copy_(original_weight)
    return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", default="Qwen/Qwen3.5-0.8B", nargs="?")
    parser.add_argument("--n-tokens", type=int, default=15)
    parser.add_argument("--prompt", default="Clean up: um so I was like going to the uh store")
    args = parser.parse_args()

    print(f"Loading {args.model_id}...")
    config = AutoConfig.from_pretrained(args.model_id)
    config.text_config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32, config=config)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    tc = config.text_config if hasattr(config, "text_config") else config
    hidden_size = tc.hidden_size
    vocab_size = tc.vocab_size
    print(f"  hidden_size={hidden_size}, vocab_size={vocab_size}")
    print(f"  lm_head: [{vocab_size}, {hidden_size}]")

    W_orig = model.lm_head.weight.data.clone()

    # Build Hadamard matrix
    print(f"\nBuilding {hidden_size}x{hidden_size} Hadamard matrix...")
    H = hadamard_matrix(hidden_size)  # [1024, 1024]
    H_T = H.T.contiguous()

    # Rotate weights
    W_rot = W_orig @ H_T  # [vocab, hidden] @ [hidden, hidden]
    print(f"  W_orig stats: mean={W_orig.mean():.6f}, std={W_orig.std():.6f}")
    print(f"  W_rot stats:  mean={W_rot.mean():.6f}, std={W_rot.std():.6f}")

    # Check coordinate distribution after rotation
    print(f"\n  Per-coord std (original): {W_orig.std(dim=0).mean():.6f} (should vary)")
    print(f"  Per-coord std (rotated):  {W_rot.std(dim=0).mean():.6f} (should be uniform)")
    print(f"  Per-coord std range (orig): [{W_orig.std(dim=0).min():.6f}, {W_orig.std(dim=0).max():.6f}]")
    print(f"  Per-coord std range (rot):  [{W_rot.std(dim=0).min():.6f}, {W_rot.std(dim=0).max():.6f}]")

    # Generate reference tokens
    print(f"\nGenerating {args.n_tokens} reference tokens (FP16 baseline)...")
    ref_tokens = generate_tokens(model, tokenizer, args.prompt, args.n_tokens)
    ref_text = tokenizer.decode(ref_tokens)
    print(f"  Prompt: {repr(args.prompt)}")
    print(f"  Reference output: {repr(ref_text)}")

    # Test configurations
    configs = []

    for nbits in [2, 3, 4]:
        # Without rotation: quantize original weights directly
        print(f"\n--- {nbits}-bit WITHOUT Hadamard rotation (Lloyd-Max) ---")
        W_norot_q = simulate_quantize(W_orig, nbits)
        mse_norot = ((W_orig - W_norot_q) ** 2).mean().item()
        print(f"  MSE: {mse_norot:.8f}")
        tokens_norot = generate_with_modified_head(model, tokenizer, args.prompt, args.n_tokens, W_orig, W_norot_q)
        match_norot = sum(1 for a, b in zip(ref_tokens, tokens_norot) if a == b)
        text_norot = tokenizer.decode(tokens_norot)
        print(f"  Top-1 match: {match_norot}/{args.n_tokens}")
        print(f"  Output: {repr(text_norot)}")
        configs.append((f"{nbits}b no-rot Lloyd-Max", match_norot, mse_norot))

        # With rotation: rotate weights, quantize, then for generation we need
        # to also rotate the hidden state before the lm_head
        print(f"\n--- {nbits}-bit WITH Hadamard rotation (Lloyd-Max) ---")
        W_rot_q = simulate_quantize(W_rot, nbits)
        # The effective weight for generation: W_eff = W_rot_q @ H (undo rotation on dequantized)
        W_eff = W_rot_q @ H
        mse_rot = ((W_orig - W_eff) ** 2).mean().item()
        print(f"  MSE (in original space): {mse_rot:.8f}")
        tokens_rot = generate_with_modified_head(model, tokenizer, args.prompt, args.n_tokens, W_orig, W_eff)
        match_rot = sum(1 for a, b in zip(ref_tokens, tokens_rot) if a == b)
        text_rot = tokenizer.decode(tokens_rot)
        print(f"  Top-1 match: {match_rot}/{args.n_tokens}")
        print(f"  Output: {repr(text_rot)}")
        configs.append((f"{nbits}b Hadamard Lloyd-Max", match_rot, mse_rot))

    # Also test k-means (what coremltools would do)
    for nbits in [2, 4]:
        print(f"\n--- {nbits}-bit WITH Hadamard rotation (k-means, coremltools-style) ---")
        W_rot_q_km = simulate_kmeans_quantize(W_rot, nbits)
        W_eff_km = W_rot_q_km @ H
        mse_km = ((W_orig - W_eff_km) ** 2).mean().item()
        print(f"  MSE (in original space): {mse_km:.8f}")
        tokens_km = generate_with_modified_head(model, tokenizer, args.prompt, args.n_tokens, W_orig, W_eff_km)
        match_km = sum(1 for a, b in zip(ref_tokens, tokens_km) if a == b)
        text_km = tokenizer.decode(tokens_km)
        print(f"  Top-1 match: {match_km}/{args.n_tokens}")
        print(f"  Output: {repr(text_km)}")
        configs.append((f"{nbits}b Hadamard k-means", match_km, mse_km))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':<30} {'Match':>8} {'MSE':>12}")
    print("-" * 60)
    for name, match, mse in configs:
        print(f"{name:<30} {match:>5}/{args.n_tokens}   {mse:>12.8f}")
    print(f"\nReference: {repr(ref_text)}")


if __name__ == "__main__":
    main()
