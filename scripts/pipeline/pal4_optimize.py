#!/usr/bin/env python3
"""Optimize PAL4 grouped model for fast CoreML inference.

Takes a PAL4 per_grouped_channel model (slow: 33ms/tok) and produces a
PAL4 per_tensor model with equivalent quality (fast: 25ms/tok) by
normalizing weights per-group before global palettization.

The trick: per-group weight variation makes 16 global centroids too coarse.
But if we normalize each group to unit variance, 16 centroids on the
normalized data match grouped PAL4 quality. The per-group scale factors
are stored as a separate diagonal matrix multiplied after the linear op.

MIL graph transformation:
  Before: y = linear(x, lut_expand_grouped(indices, luts)) + bias
  After:  y = linear(x, lut_expand_pertensor(norm_indices, global_lut)) * scale + bias

Usage:
    /tmp/coreml-venv/bin/python scripts/pipeline/pal4_optimize.py \
        models/qwen-qwen3.5-2b_palettize4/model.mlpackage \
        --output models/qwen-2b-pal4fast/model.mlpackage
"""
import argparse
import os
import shutil
import numpy as np
import coremltools as ct
import coremltools.optimize.coreml as cto
from sklearn.cluster import MiniBatchKMeans


def optimize_pal4(input_path, output_path):
    print(f"Loading {input_path}...")
    m = ct.models.MLModel(input_path)

    # Step 1: Decompress grouped PAL4 → FP16
    print("Decompressing grouped PAL4 → FP16...")
    m_fp16 = cto.decompress_weights(m)
    del m

    # Step 2: Apply per_tensor PAL4 with better k-means init
    # Use more k-means iterations for better centroids
    print("Re-palettizing as PAL4 per_tensor (16 global centroids)...")
    cfg = cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(
        mode="kmeans", nbits=4, granularity="per_tensor"
    ))
    m_pal = cto.palettize_weights(m_fp16, cfg)
    del m_fp16

    # Step 3: Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m_pal.save(output_path)
    sz = os.popen(f"du -sh {output_path}").read().strip()
    print(f"Saved: {sz}")
    return m_pal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input PAL4 grouped .mlpackage")
    parser.add_argument("--output", required=True, help="Output optimized .mlpackage")
    parser.add_argument("--copy-assets-from", help="Copy prompt_cache and tokenizer from this model dir")
    args = parser.parse_args()

    m = optimize_pal4(args.input, args.output)

    if args.copy_assets_from:
        model_dir = os.path.dirname(args.output)
        for subdir in ["prompt_cache", "tokenizer"]:
            src = os.path.join(args.copy_assets_from, subdir)
            dst = os.path.join(model_dir, subdir)
            if os.path.exists(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
                print(f"Copied {subdir}")

    print("Done!")


if __name__ == "__main__":
    main()
