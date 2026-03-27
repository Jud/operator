#!/usr/bin/env python3
"""Benchmark palettize4 decode speed across group sizes.

Tests different palettization granularities to find the fastest config
that still produces correct output.

Usage:
    /tmp/coreml-venv/bin/python scripts/experiments/palettize-bench/bench.py
"""
import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import time
import json
import os
import shutil

MODEL_PATH = "models/qwen-qwen3.5-2b_fp16/model.mlpackage"
PROMPT_CACHE = "models/qwen-qwen3.5-2b_fp16/prompt_cache"
OUTPUT_BASE = "models/pal4_bench"

# Get correct input shapes from the FP16 model
print("Loading FP16 model for reference...")
m_ref = ct.models.MLModel(MODEL_PATH)
shapes = {}
for inp in m_ref.get_spec().description.input:
    shapes[inp.name] = list(inp.type.multiArrayType.shape)


def make_sample():
    s = {}
    for name, shape in shapes.items():
        if "int" in name or "input_id" in name or "cache_position" in name:
            s[name] = np.zeros(shape, dtype=np.int32)
        elif any(x in name for x in ["conv_state", "rec_state", "key_cache", "value_cache"]):
            s[name] = np.zeros(shape, dtype=np.float16)
        else:
            s[name] = np.zeros(shape, dtype=np.float32)
    return s


sample = make_sample()


def bench_model(model, label, n=15):
    """Warm up and benchmark decode latency."""
    for _ in range(5):
        model.predict(sample)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(sample)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p50 = times[n // 2]
    mn = min(times)
    mx = max(times)
    print(f"  {label:40s}: p50={p50:.1f}ms  min={mn:.1f}ms  max={mx:.1f}ms")
    return p50


# Baseline: FP16
print("\n=== Baselines ===")
fp16_p50 = bench_model(m_ref, "FP16 (baseline)")

# INT4 baseline
m_int4 = ct.models.MLModel("models/qwen-qwen3.5-2b_int4/model.mlpackage")
int4_p50 = bench_model(m_int4, "INT4 block_size=32")
del m_int4

# Existing PAL4 (group_size=16)
m_pal_gs16 = ct.models.MLModel("models/qwen-qwen3.5-2b_palettize4/model.mlpackage")
pal16_p50 = bench_model(m_pal_gs16, "PAL4 per_grouped_channel gs=16")
del m_pal_gs16

# Test different palettization configs
configs = [
    ("PAL4 per_tensor", cto.OpPalettizerConfig(
        mode="kmeans", nbits=4, granularity="per_tensor"
    )),
    ("PAL4 per_grouped_channel gs=32", cto.OpPalettizerConfig(
        mode="kmeans", nbits=4, granularity="per_grouped_channel", group_size=32
    )),
    ("PAL4 per_grouped_channel gs=64", cto.OpPalettizerConfig(
        mode="kmeans", nbits=4, granularity="per_grouped_channel", group_size=64
    )),
    ("PAL4 per_grouped_channel gs=128", cto.OpPalettizerConfig(
        mode="kmeans", nbits=4, granularity="per_grouped_channel", group_size=128
    )),
    ("PAL4 per_grouped_channel gs=256", cto.OpPalettizerConfig(
        mode="kmeans", nbits=4, granularity="per_grouped_channel", group_size=256
    )),
]

print("\n=== Palettization group size sweep ===")
results = []

for label, cfg in configs:
    print(f"\nQuantizing: {label}...")
    # Reload fresh FP16 model each time
    m = ct.models.MLModel(MODEL_PATH)
    opt_config = cto.OptimizationConfig(global_config=cfg)

    try:
        m_pal = cto.palettize_weights(m, opt_config)
    except Exception as e:
        print(f"  FAILED: {e}")
        results.append((label, None, "FAILED"))
        continue

    # Save temporarily to load as compiled
    tmp_path = f"{OUTPUT_BASE}/{label.replace(' ', '_')}.mlpackage"
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    m_pal.save(tmp_path)
    del m, m_pal

    # Load and benchmark
    m_loaded = ct.models.MLModel(tmp_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    p50 = bench_model(m_loaded, label)
    ratio = p50 / fp16_p50

    # Quick quality check — get one output
    out = m_loaded.predict(sample)
    logits = out["logits"]
    top_token = int(np.argmax(logits))

    results.append((label, p50, f"ratio={ratio:.2f}x, top_tok={top_token}"))
    del m_loaded

# Summary
print("\n" + "=" * 70)
print("  SUMMARY — Decode step latency (single token)")
print("=" * 70)
print(f"  {'Config':40s} {'p50':>8s} {'vs FP16':>8s}")
print(f"  {'-'*40} {'-'*8} {'-'*8}")
print(f"  {'FP16 (baseline)':40s} {fp16_p50:7.1f}ms {'1.00x':>8s}")
print(f"  {'INT4 block_size=32':40s} {int4_p50:7.1f}ms {int4_p50/fp16_p50:7.2f}x")
print(f"  {'PAL4 gs=16 (current)':40s} {pal16_p50:7.1f}ms {pal16_p50/fp16_p50:7.2f}x")
for label, p50, note in results:
    if p50 is not None:
        print(f"  {label:40s} {p50:7.1f}ms {p50/fp16_p50:7.2f}x  {note}")
    else:
        print(f"  {label:40s}    {note}")
