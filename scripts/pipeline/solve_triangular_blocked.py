"""Blocked forward substitution for (I - L)^{-1}.

True triangular solve — NOT a series expansion. Uses blocked forward
substitution with exact 8×8 block solves.

For L a 64×64 strictly lower triangular matrix, split into 8×8 blocks:
  - 8 diagonal blocks: each solved exactly via 3-matmul Neumann (L^8=0)
  - 7 propagation steps: matmul with already-computed rows

This is real forward substitution, just blocked for GPU efficiency.
Total: 8 × 3 (block solves) + 7 × 2 (propagations) = ~38 matmuls, 8 seq steps.

Usage:
    /tmp/coreml-venv/bin/python scripts/experiments/solve-triangular/solve_triangular_blocked.py
"""

import torch
import torch.nn.functional as F


def solve_block_8x8(L_block):
    """Exact inverse of (I - L) for 8×8 strictly lower triangular L.

    Uses Neumann: (I+L)(I+L^2)(I+L^4) — exact since L^8=0.
    """
    eye = torch.eye(8, dtype=L_block.dtype, device=L_block.device)
    L2 = L_block @ L_block
    L4 = L2 @ L2
    return (eye + L_block) @ (eye + L2) @ (eye + L4)


def solve_triangular_blocked(L, chunk_size=64, block_size=8):
    """Compute R = (I - L)^{-1} via blocked forward substitution.

    True triangular solve. For each block row b:
        R_b = D_b @ (I_b + L_{b,<b} @ R_{<b})
    where D_b = (I - L_bb)^{-1} is the diagonal block inverse.

    Args:
        L: [..., 64, 64] strictly lower triangular
    Returns:
        R: [..., 64, 64] = (I - L)^{-1}
    """
    n_blocks = chunk_size // block_size
    batch_shape = L.shape[:-2]
    eye_full = torch.eye(chunk_size, dtype=L.dtype, device=L.device)

    # Pre-compute all diagonal block inverses (independent, parallelizable)
    diag_inv = []
    for b in range(n_blocks):
        r0 = b * block_size
        r1 = r0 + block_size
        L_bb = L[..., r0:r1, r0:r1]
        diag_inv.append(solve_block_8x8(L_bb))

    # Forward substitution: build R block-row by block-row
    # R is built as a list of row-blocks, each [... , 8, 64]
    row_blocks = []

    for b in range(n_blocks):
        r0 = b * block_size
        r1 = r0 + block_size

        if b == 0:
            # First block: R_0 = D_0 @ I_0 = D_0 columns corresponding to rows 0..7
            # R[0:8, :] = D_0 @ eye[0:8, :] = D_0 padded with zeros
            block_row = F.pad(diag_inv[0], (0, chunk_size - block_size))  # [..., 8, 64]
        else:
            # R_b = D_b @ (I_b + L_{b,<b} @ R_{<b})
            # L_{b,<b}: the off-diagonal sub-row [..., 8, b*8]
            L_off = L[..., r0:r1, :r0]  # [..., 8, r0]

            # R_{<b}: all previously computed rows [..., r0, 64]
            R_prev = torch.cat(row_blocks, dim=-2)  # [..., r0, 64]

            # Contribution from previous blocks
            contrib = L_off @ R_prev  # [..., 8, 64]

            # Add identity rows for this block
            eye_block = eye_full[r0:r1, :]  # [8, 64]
            rhs = eye_block + contrib  # [..., 8, 64]

            # Apply diagonal block inverse
            block_row = diag_inv[b] @ rhs  # [..., 8, 64]

        row_blocks.append(block_row)

    # Assemble full R
    R = torch.cat(row_blocks, dim=-2)  # [..., 64, 64]
    return R


# ── Verification ──

if __name__ == "__main__":
    import sys
    import os
    import numpy as np

    print("=== Blocked Forward Substitution Test ===\n")

    torch.manual_seed(42)
    CS = 64

    # Test with random data
    L = torch.randn(1, 16, 1, CS, CS).tril(diagonal=-1) * 0.05
    eye = torch.eye(CS)
    R_ref = torch.linalg.solve_triangular(eye - L, eye.expand_as(L), upper=False)

    R_blocked = solve_triangular_blocked(L, CS, 8)
    diff = (R_ref - R_blocked).abs().max().item()
    print(f"Random data:  max_diff = {diff:.12f} {'PASS' if diff < 1e-5 else 'FAIL'}")

    # Test with realistic DeltaNet data (large negative g values)
    L_real = torch.randn(1, 16, 1, CS, CS).tril(diagonal=-1) * 0.3
    R_ref2 = torch.linalg.solve_triangular(eye - L_real, eye.expand_as(L_real), upper=False)
    R_blocked2 = solve_triangular_blocked(L_real, CS, 8)
    diff2 = (R_ref2 - R_blocked2).abs().max().item()
    print(f"Larger values: max_diff = {diff2:.12f} {'PASS' if diff2 < 1e-4 else 'FAIL'}")

    # Traceability
    print("\n=== Traceability ===")

    class BlockedSolveModel(torch.nn.Module):
        def forward(self, L):
            return solve_triangular_blocked(L, 64, 8)

    m = BlockedSolveModel().eval()
    traced = torch.jit.trace(m, (L,))
    R_traced = traced(L)
    trace_diff = (R_ref - R_traced).abs().max().item()
    print(f"Trace:  diff = {trace_diff:.12f} {'PASS' if trace_diff < 1e-5 else 'FAIL'}")

    # Count ops
    graph = traced.graph
    n_ops = sum(1 for _ in graph.nodes())
    print(f"Graph:  {n_ops} ops")

    # CoreML conversion
    print("\n=== CoreML Conversion ===")
    sys.path.insert(0, os.path.dirname(__file__))
    import patches  # noqa
    import coremltools as ct

    ct_in = [ct.TensorType(name="L", shape=L.shape)]

    for prec_name, prec in [("FP32", ct.precision.FLOAT32), ("FP16", ct.precision.FLOAT16)]:
        try:
            ml = ct.convert(traced, inputs=ct_in, convert_to="mlprogram",
                           minimum_deployment_target=ct.target.iOS18,
                           compute_precision=prec)
            out = ml.predict({"L": L.numpy()})
            R_ml = list(out.values())[0]
            ml_diff = np.abs(R_ml - R_ref.numpy()).max()
            has_nan = np.isnan(R_ml).any()
            status = "NAN" if has_nan else f"diff={ml_diff:.6f}"
            print(f"{prec_name}:  {status} {'PASS' if not has_nan and ml_diff < 0.01 else 'FAIL'}")
        except Exception as e:
            print(f"{prec_name}:  FAILED: {e}")
