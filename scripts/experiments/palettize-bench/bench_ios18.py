#!/usr/bin/env python3
"""Benchmark PAL4 group sizes using the iOS18-targeted decode model.

The previous sweep failed because it used the iOS17 FP16 model.
This uses convert_decode.py's iOS18 output for per_grouped_channel support.
"""
import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import time
import os

# Use the iOS18-targeted decode model (from the palettize4 conversion)
# We need to get an FP16 iOS18 model. The int4 model was iOS18-targeted.
# Let's dequantize it... or just reconvert.
# Actually, the simplest: convert FP16 to iOS18 by changing the spec.
# Or just use the palettize4 model and re-palettize with different configs.
# Wait -- can't re-palettize.

# Cleanest: reconvert the decode model fresh with iOS18 target, no quantization.
# But that takes 10+ minutes.

# Hack: load the INT4 model (iOS18), dequantize back to FP16, then re-palettize.
# That's lossy though.

# Best option: convert fresh from PyTorch with iOS18 target, save as FP16 iOS18.
# Use the pipeline script:
#   python scripts/pipeline/convert_decode.py Qwen/Qwen3.5-2B --output models/qwen-2b-ios18-fp16

# For now, let's just test the two we CAN do:
# 1. per_tensor (works on any iOS version)
# 2. per_grouped_channel gs=16 (from existing palettize4 model, already benchmarked)
# 3. Reconvert decode with iOS18 and test more group sizes

print("This script requires an iOS18 FP16 decode model.")
print("Run: python scripts/pipeline/convert_decode.py Qwen/Qwen3.5-2B --output models/qwen-2b-ios18-fp16")
print("Then re-run this script.")
print()
print("For now, using the existing per_tensor result: 24.9ms (0.96x FP16)")
print("And gs=16 result: 33.0ms (1.28x FP16)")
print()
print("The question is whether intermediate group sizes (32, 64, 128) are both fast and accurate.")
