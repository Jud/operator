"""Apply 6-bit palettization to the lm_head of an existing CoreML model.

Takes a converted .mlpackage, finds the lm_head weight (largest linear weight),
palettizes it to 6-bit using k-means, and saves a new model.

Usage:
    /tmp/coreml-venv/bin/python scripts/experiments/quip-hadamard-lmhead/palettize_lmhead.py \
        models/batch_prefill_fp16/model.mlpackage \
        --output models/batch_prefill_6bit/model.mlpackage \
        --nbits 6
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import coremltools as ct
import coremltools.optimize as cto


def find_lm_head_weight(mlmodel, verbose=True):
    """Find the lm_head weight — the largest 2D weight in the model."""
    metadata = cto.coreml.get_weights_metadata(mlmodel, weight_threshold=2048)

    largest_name = None
    largest_size = 0

    for name, meta in metadata.items():
        shape = tuple(meta.val.shape) if hasattr(meta, 'val') and meta.val is not None else None
        if shape and len(shape) == 2:
            size = shape[0] * shape[1]
            if verbose:
                print(f"  {name}: {shape} ({size:,} elements)")
            if size > largest_size:
                largest_size = size
                largest_name = name

    if largest_name is None:
        raise RuntimeError("Could not find lm_head weight")

    return largest_name


def main():
    parser = argparse.ArgumentParser(description="Palettize lm_head of a CoreML model")
    parser.add_argument("model_path", help="Path to .mlpackage")
    parser.add_argument("--output", required=True, help="Output .mlpackage path")
    parser.add_argument("--nbits", type=int, default=6, choices=[2, 3, 4, 6, 8])
    args = parser.parse_args()

    print(f"Loading {args.model_path}...")
    t0 = time.time()
    mlmodel = ct.models.MLModel(args.model_path)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print(f"\nFinding lm_head weight...")
    lm_head_name = find_lm_head_weight(mlmodel)
    print(f"  → {lm_head_name}")

    print(f"\nPalettizing to {args.nbits}-bit (k-means)...")
    t0 = time.time()

    op_config = cto.coreml.OpPalettizerConfig(mode="kmeans", nbits=args.nbits)
    opt_config = cto.coreml.OptimizationConfig()
    opt_config.set_op_name(lm_head_name, op_config)

    mlmodel = cto.coreml.palettize_weights(mlmodel, opt_config)
    print(f"  Palettized in {time.time() - t0:.1f}s")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    mlmodel.save(args.output)

    # Size comparison
    def dir_size(path):
        total = 0
        for dp, _, fns in os.walk(path):
            for f in fns:
                total += os.path.getsize(os.path.join(dp, f))
        return total

    orig_mb = dir_size(args.model_path) / 1e6
    new_mb = dir_size(args.output) / 1e6
    print(f"\nSaved {args.output}")
    print(f"  Original: {orig_mb:.1f} MB")
    print(f"  New:      {new_mb:.1f} MB")
    print(f"  Ratio:    {orig_mb / new_mb:.2f}x smaller")


if __name__ == "__main__":
    main()
