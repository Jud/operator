# CoreML Quantization Results — Qwen 3.5 2B

## Summary

Exhaustive testing of quantization strategies for the Qwen 3.5 2B DeltaNet model on CoreML. The goal: find the smallest, fastest, highest-quality configuration for on-device inference.

## The Tradeoff Triangle

```
        QUALITY
       /       \
      /  target  \
     /   zone     \
    /_______________\
  SIZE              SPEED
```

No configuration achieves all three optimally. The best options:

| Config | Size | Speed (decode) | Quality | Notes |
|--------|------|---------------|---------|-------|
| FP16 | 3.5 GB | 19 ms/tok | Perfect | Baseline, too large |
| **INT4 bs=8** | **1.3 GB** | **12 ms/tok** | **15/15** | Best speed+quality, slightly over 1GB |
| INT4 bs=16 | 1.1 GB | 12 ms/tok | 14/15 | One hallucination on "yeah so like" |
| INT4 bs=32 | 1.0 GB | 13 ms/tok | 13/15 | Leaked tags on "yeah so" patterns |
| PAL4 gs=16 | 900 MB | 33 ms/tok | 15/15 | Best quality+size, but slow |
| PAL4 per_tensor | 899 MB | 25 ms/tok | ~12/15 | Fast but lossy (summarizes aggressively) |
| PAL8 per_tensor | 1.8 GB | 39 ms/tok | ~15/15 | Good quality, too slow and large |
| INT8 | 1.8 GB | 26 ms/tok | 15/15 | Near-lossless, too large |
| Mixed 2b+4b | 647 MB | N/A | 0/15 | Completely broken (below quality cliff) |

## CoreML Fusion Behavior

The critical discovery: CoreML's GPU kernel for `constexpr_lut_to_dense` (palettization) only fuses with the downstream `linear` op under very specific conditions.

### What fuses (fast)
- `constexpr_lut_to_dense` with **rank-1 LUT** (`[16]`) — per_tensor, 16 entries
- `constexpr_blockwise_shift_scale` with **any block size** — all INT4/INT8 linear quantization

### What does NOT fuse (slow)
- `constexpr_lut_to_dense` with **rank-4 LUT** (`[384, 1, 16, 1]`) — per_grouped_channel
- `constexpr_lut_to_dense` with **256-entry rank-1 LUT** (`[256]`) — PAL8 per_tensor
- Any palettization with more than 16 LUT entries

### Root cause
The compiled MIL shows that non-fusing palettization generates separate `constexpr_lut_to_dense` ops that expand the full weight matrix to FP16 before the `linear` op. For a [6144, 2048] weight with gs=16, this means 384 separate LUT expansions per forward pass. The GPU kernel doesn't have a fused path for grouped or large LUTs.

### Evidence
Profiling results (single decode step, CPU+GPU):
```
FP16:                         25.9 ms
INT4 bs=32:                   21.1 ms  (fused blockwise_shift_scale)
PAL4 per_tensor (16 entries): 24.9 ms  (fused lut_to_dense)
PAL4 gs=16 (384 × 16):       33.0 ms  (NOT fused)
PAL8 per_tensor (256):        38.8 ms  (NOT fused — larger LUT = slower)
PAL4 CPU only:                71.9 ms  (no GPU acceleration)
```

## Attempted Approaches

### 1. Group size sweep ❌
Tested per_tensor, gs=32, gs=64, gs=128, gs=256. Only per_tensor triggers the fast path. Other group sizes failed on iOS17 target; when targeting iOS18, only gs=16 is supported by coremltools for per_grouped_channel.

### 2. PAL4 → decompress → INT4 rewrite ❌
Decompress PAL4 grouped back to FP16, then re-quantize as INT4 blockwise. Quality was WORSE than straight FP16→INT4 because the decompression loses the non-uniform centroid advantage and the re-quantization adds new error.

### 3. PAL4 per_tensor from decompressed grouped ❌
Decompress grouped PAL4 to FP16 (preserving clustered weight distribution), then apply per_tensor PAL4. Quality was broken — hallucinations, repeated text. 16 global centroids are simply insufficient for a 12M-element weight matrix.

### 4. Normalized PAL4 per_tensor (theoretical) ⚠️
Normalize weights to unit variance per group, palettize with per_tensor. MSE analysis shows this matches grouped PAL4 quality. But requires MIL graph surgery to inject per-group rescaling after each linear op. Not implemented.

### 5. Global LUT remapping (theoretical) ❌
Merge 384 group LUTs (6,144 unique centroids) into 256 global centroids (near-zero mapping error). Requires uint8 indices → 1.5GB model (over budget). Also confirmed PAL8 per_tensor (256 entries) doesn't fuse.

### 6. INT4 block size ladder ✅
Tested block_size 32, 16, 8. Finer blocks = less quantization error = fewer edge-case failures. bs=8 achieves 15/15 quality at 84 tok/s.

## INT4 Block Size Sensitivity

The "yeah so" family of inputs is the canary for quantization quality:

| Input | FP16 | INT4 bs=32 | INT4 bs=16 | INT4 bs=8 |
|-------|------|-----------|-----------|----------|
| "yeah so we were thinking about maybe going to dinner" | ✓ | ✗ leaked tag | ✓ | ✓ |
| "yeah so like we were going to the store" | ✓ | ✗ leaked tag | ✗ hallucination | ✓ |

The pattern: "yeah so" + filler word hits a decision boundary in the model where quantization error tips the argmax. Finer blocks (bs=8) keep enough precision to preserve the correct decision.

## Recommendations

### For production (speed priority)
**INT4 bs=8**: 1.3GB, 84 tok/s, 15/15 quality. Slightly over 1GB but the best speed×quality product.

### For constrained devices (size priority)
**PAL4 gs=16**: 900MB, 34 tok/s, 15/15 quality. Half the speed but perfect quality under 1GB.

### For routing only
**PAL4 per_tensor**: 899MB, ~40 tok/s (per_tensor speed), slightly lossy but adequate for classification tasks.

### Future: Apple Radar
File a radar requesting fused GPU kernel for `constexpr_lut_to_dense` with per_grouped_channel granularity. MPSGraph already supports this fusion (confirmed WWDC24) — CoreML's compiler just doesn't generate the right graph pattern. This would instantly make PAL4 gs=16 run at 25ms instead of 33ms, solving the tradeoff.
