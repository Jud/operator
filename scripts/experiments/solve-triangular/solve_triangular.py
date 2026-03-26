"""Triangular solve for CoreML: (I - L)^{-1} via traceable PyTorch ops.

Implements forward substitution using only ops that coremltools can convert:
matmul, slice, scatter, tril, eye, add, mul.

The key: blocked forward substitution with block size B.
- Within each block: Neumann series (exact for B×B, only B-1 terms)
- Between blocks: matmul propagation

For chunk_size=64 with block_size=8: 8 block solves + 7 propagations.
Each block solve is 3 matmuls (Neumann for 8×8). Total: ~31 matmuls.
But all WITHIN-block operations are parallel (no sequential dependency).

Compare:
- Row-by-row: 64 sequential ops (unrolled in trace = huge graph)
- Neumann product: 10 matmuls (FP16 NaN issue)
- Blocked solve: ~31 matmuls but better numerics (no high powers)
- This approach: direct column-by-column solve via matmul accumulation

Usage:
    /tmp/coreml-venv/bin/python scripts/experiments/solve-triangular/solve_triangular.py
"""

import torch
import torch.nn.functional as F
import numpy as np


def solve_triangular_lower(L, chunk_size=64):
    """Compute (I - L)^{-1} where L is strictly lower triangular.

    Uses column-by-column forward substitution expressed as matmul + accumulation.
    All ops are traceable and CoreML-convertible.

    Args:
        L: [..., chunk_size, chunk_size] strictly lower triangular matrix
    Returns:
        R: [..., chunk_size, chunk_size] where R = (I - L)^{-1}
    """
    eye = torch.eye(chunk_size, dtype=L.dtype, device=L.device)

    # R = (I - L)^{-1} means R @ (I - L) = I, so R - R @ L = I, so R = I + R @ L
    # Column j of R: R[:,j] = e_j + L @ R[:,j] ... no, forward sub is row-based.
    #
    # Actually: (I - L) @ R = I => R = I + L @ R
    # Row i: R[i,:] = e_i + sum_{k<i} L[i,k] * R[k,:]
    # This is sequential in rows.
    #
    # But we can compute R iteratively:
    # R_0 = I
    # R_1 = I + L @ R_0 = I + L
    # R_2 = I + L @ R_1 = I + L + L^2
    # R_k = I + L @ R_{k-1} = I + L + L^2 + ... + L^k
    #
    # After 64 iterations: R_63 = (I-L)^{-1} (exact since L^64 = 0)
    #
    # But 64 iterations is what we're trying to avoid!
    #
    # BLOCKED approach: split into blocks of size B.
    # Process block 0 (rows 0..B-1) independently — just Neumann within block.
    # Then propagate block 0's result to block 1, etc.
    #
    # For B=8, chunk_size=64: 8 blocks.
    # Within each 8×8 block: L_block^8 = 0, so Neumann is exact with 7 terms.
    # Use product factorization: (I+L)(I+L^2)(I+L^4) = 3 matmuls per block.
    # Between blocks: one matmul per propagation.
    # Total: 8 × 3 + 7 = 31 matmuls... but the between-block is sequential.

    # Simpler approach: iterative refinement with doubling.
    # R = I + L @ R can be solved by:
    # R_0 = I
    # R_{k+1} = I + L @ R_k
    # Each step doubles the number of correct rows (like iterative refinement).
    # Actually no, each step adds one more power of L.

    # BEST approach for CoreML: the same Neumann product factorization but
    # done in a numerically stable way that works in FP16.
    # The issue with FP16 was NOT the Neumann series (isolated test passed).
    # It was something else in the full model.
    #
    # So let me try: implement as a traced PyTorch function using
    # iterative R = I + L @ R (converges in ceil(log2(64)) = 6 steps if done
    # via doubling).
    #
    # Doubling trick:
    # If L has bandwidth b (nonzeros only within b diagonals of the main diagonal),
    # then after one step R_1 = I + L has bandwidth 1.
    # But L is general lower triangular, so bandwidth = 63.
    #
    # Actually, the right approach is simply:
    # R = I + L @ (I + L @ (I + L @ (I + L @ (I + L @ (I + L @ I)))))
    # = 6 nested matmuls (Horner form)
    # But this only gives I + L + L^2 + L^3 + L^4 + L^5 + L^6
    # We need up to L^63.

    # The PRODUCT FACTORIZATION is actually the right answer:
    # (I+L)(I+L^2)(I+L^4)(I+L^8)(I+L^16)(I+L^32) = I + L + L^2 + ... + L^63
    # This is 5 squarings + 5 multiplications of full matrices.
    # Each intermediate result has max value ~1 (as verified with real data).
    # This SHOULD work in FP16 — the isolated test proved it does.

    # Let me implement this cleanly and also add an alternative blocked approach.

    L2 = L @ L
    L4 = L2 @ L2
    L8 = L4 @ L4
    L16 = L8 @ L8
    L32 = L16 @ L16
    R = (eye + L) @ (eye + L2) @ (eye + L4) @ (eye + L8) @ (eye + L16) @ (eye + L32)
    return R


def solve_triangular_blocked(L, chunk_size=64, block_size=8):
    """Compute (I - L)^{-1} via blocked forward substitution.

    Splits into blocks of block_size. Within each block, uses small Neumann
    series (max power = block_size-1). Between blocks, propagates via matmul.

    More numerically stable than full Neumann since max power is block_size-1
    instead of chunk_size-1.

    For block_size=8: max L^7 (tiny), 3 matmuls per block Neumann,
    plus inter-block propagations.
    """
    n_blocks = chunk_size // block_size
    eye = torch.eye(chunk_size, dtype=L.dtype, device=L.device)
    eye_b = torch.eye(block_size, dtype=L.dtype, device=L.device)

    # Initialize R = I (will build up the inverse)
    R = eye.clone().expand_as(L).contiguous()  # [..., CS, CS]

    for bi in range(n_blocks):
        r0 = bi * block_size
        r1 = r0 + block_size

        # 1. Within-block solve: (I - L_diag)^{-1} for the diagonal block
        L_diag = L[..., r0:r1, r0:r1]  # [..., B, B] strictly lower tri
        # Small Neumann: (I+L)(I+L^2)(I+L^4) for B=8
        Ld2 = L_diag @ L_diag
        Ld4 = Ld2 @ Ld2
        R_diag = (eye_b + L_diag) @ (eye_b + Ld2) @ (eye_b + Ld4)  # exact for B=8

        # 2. Update block rows: R[r0:r1, :] using the diagonal inverse
        # For the first block (bi=0), R[0:B, :] = R_diag @ I[0:B, :] = R_diag[:, 0:B]
        # For subsequent blocks, include the contribution from previous blocks
        if bi > 0:
            # L_off = L[r0:r1, 0:r0]  — off-diagonal block (connects to previous blocks)
            L_off = L[..., r0:r1, :r0]  # [..., B, r0]
            # Contribution from previous rows: L_off @ R[0:r0, :]
            contrib = L_off @ R[..., :r0, :]  # [..., B, CS]
            # Apply diagonal inverse: R_diag @ contrib
            update = R_diag @ contrib  # [..., B, CS]
            # Build the full row block
            # R[r0:r1, :] = R_diag applied to (e_{r0..r1} + contribution)
            # More precisely: R[r0:r1, :] = R_diag @ (I[r0:r1, :] + L_off @ R[:r0, :])
            row_block = R_diag @ (eye[r0:r1, :].expand_as(R[..., r0:r1, :]) + contrib)
        else:
            # First block: just the diagonal inverse applied to identity rows
            row_block = R_diag @ eye[r0:r1, :].expand_as(R[..., r0:r1, :])

        # Write back (this is technically in-place, need to make functional for tracing)
        # Use scatter or concat approach
        R = torch.cat([
            R[..., :r0, :],
            row_block,
            R[..., r1:, :],
        ], dim=-2)

    return R


# ── Verification ──

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../coreml-convert'))

    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    print("=== Testing solve_triangular implementations ===\n")

    torch.manual_seed(42)
    CS = 64
    # Create a realistic L matrix
    L = torch.randn(1, 16, 1, CS, CS).tril(diagonal=-1) * 0.05  # small values like real data

    # Reference: numpy solve
    eye = torch.eye(CS)
    A = eye - L
    R_ref = torch.linalg.solve_triangular(A, eye.expand_as(A), upper=False)

    # Test Neumann product
    R_neumann = solve_triangular_lower(L, CS)
    diff_n = (R_ref - R_neumann).abs().max().item()
    print(f"Neumann product:  max_diff={diff_n:.10f}")

    # Test blocked
    R_blocked = solve_triangular_blocked(L, CS, block_size=8)
    diff_b = (R_ref - R_blocked).abs().max().item()
    print(f"Blocked (B=8):    max_diff={diff_b:.10f}")

    # Test FP16 stability
    L16 = L.half()
    R_ref16 = torch.linalg.solve_triangular((eye - L16).float(), eye.expand_as(L16).float(), upper=False)

    R_neumann16 = solve_triangular_lower(L16.float(), CS)
    diff_n16 = (R_ref16 - R_neumann16).abs().max().item()

    R_blocked16 = solve_triangular_blocked(L16.float(), CS, block_size=8)
    diff_b16 = (R_ref16 - R_blocked16).abs().max().item()
    print(f"\nFP16 input → FP32 compute:")
    print(f"  Neumann: max_diff={diff_n16:.10f}")
    print(f"  Blocked: max_diff={diff_b16:.10f}")

    # Test traceability
    print("\n=== Traceability tests ===")

    class NeumannModel(torch.nn.Module):
        def forward(self, L):
            return solve_triangular_lower(L, 64)

    class BlockedModel(torch.nn.Module):
        def forward(self, L):
            return solve_triangular_blocked(L, 64, 8)

    for name, model_cls in [("Neumann", NeumannModel), ("Blocked", BlockedModel)]:
        m = model_cls().eval()
        try:
            traced = torch.jit.trace(m, (L,))
            R_traced = traced(L)
            trace_diff = (R_ref - R_traced).abs().max().item()
            print(f"{name}: traces OK, diff={trace_diff:.10f}")
        except Exception as e:
            print(f"{name}: trace FAILED: {e}")

    # Test CoreML conversion
    print("\n=== CoreML conversion tests ===")
    import coremltools as ct

    for name, model_cls in [("Neumann", NeumannModel), ("Blocked", BlockedModel)]:
        m = model_cls().eval()
        traced = torch.jit.trace(m, (L,))
        ct_input = [ct.TensorType(name="L", shape=L.shape)]

        for precision_name, precision in [("FP32", ct.precision.FLOAT32), ("FP16", ct.precision.FLOAT16)]:
            try:
                ml = ct.convert(traced, inputs=ct_input, convert_to="mlprogram",
                               minimum_deployment_target=ct.target.iOS18,
                               compute_precision=precision)
                out = ml.predict({"L": L.numpy()})
                R_ml = list(out.values())[0]
                ml_diff = np.abs(R_ml - R_ref.numpy()).max()
                has_nan = np.isnan(R_ml).any()
                print(f"{name} {precision_name}: diff={ml_diff:.6f}, nan={has_nan}")
            except Exception as e:
                print(f"{name} {precision_name}: FAILED: {type(e).__name__}: {str(e)[:80]}")
