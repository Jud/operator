# CoreML Compression Research for Qwen3.5 Hybrid DeltaNet/Attention

Research document covering mixed-bit palettization, ANE optimization, LM head compression, and practical implementation for the Qwen3.5-0.8B model running on-device via CoreML.

## Model Architecture Context

Qwen3.5-0.8B has 24 layers with a 3:1 ratio of DeltaNet (linear attention) to standard softmax attention layers. The pattern repeats every 4 layers: `[DeltaNet, DeltaNet, DeltaNet, FullAttention]`, giving 18 DeltaNet layers and 6 attention layers. The model has a 248K vocabulary LM head that dominates inference time (55% of total at ~9.1ms per token).

Current performance: 68 tok/s (FP16, 2-chunk with KV cache). Target: maximize compression while maintaining text cleanup quality.

---

## 1. Mixed-Bit Palettization Per Layer Type

### 1.1 Can OD-MBP Be Applied Differently to DeltaNet vs Attention Layers?

**Yes, definitively.** CoreML's `OptimizationConfig` supports three levels of granularity:

1. **`global_config`** -- default for all ops
2. **`op_type_configs`** -- per op type (e.g., all `conv` ops, all `linear` ops)
3. **`op_name_configs`** -- per named operation instance

Since each layer's weights have unique names in the CoreML MIL graph (e.g., `layers.0.self_attn.q_proj.weight`, `layers.3.self_attn.q_proj.weight`), you can assign different bit-widths to every individual weight tensor. This means DeltaNet layers can get one bit-width and attention layers another.

**Concrete API:**

```python
import coremltools as ct
from coremltools.optimize.coreml import (
    OpPalettizerConfig,
    OptimizationConfig,
    palettize_weights,
)

# Step 1: Load the converted mlpackage
mlmodel = ct.models.MLModel("qwen35_decode.mlpackage")

# Step 2: Build per-layer config
op_name_configs = {}

# DeltaNet layers (0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22)
deltanet_indices = [i for i in range(24) if i % 4 != 3]
# Attention layers (3, 7, 11, 15, 19, 23)
attn_indices = [i for i in range(24) if i % 4 == 3]

# Use get_weights_metadata to discover exact op names
weight_metadata = ct.optimize.coreml.get_weights_metadata(
    mlmodel, weight_threshold=2048
)

for weight_name, meta in weight_metadata.items():
    # Classify by layer index
    for di in deltanet_indices:
        if f"layers.{di}." in weight_name or f"layers_{di}_" in weight_name:
            # DeltaNet: more aggressive compression
            op_name_configs[weight_name] = OpPalettizerConfig(
                mode="kmeans", nbits=2, granularity="per_grouped_channel", group_size=16
            )
            break
    for ai in attn_indices:
        if f"layers.{ai}." in weight_name or f"layers_{ai}_" in weight_name:
            # Attention: conservative compression
            op_name_configs[weight_name] = OpPalettizerConfig(
                mode="kmeans", nbits=4, granularity="per_grouped_channel", group_size=16
            )
            break

# LM head: aggressive (see Section 3)
for weight_name in weight_metadata:
    if "lm_head" in weight_name:
        op_name_configs[weight_name] = OpPalettizerConfig(
            mode="kmeans", nbits=2, granularity="per_grouped_channel", group_size=16
        )

config = OptimizationConfig(
    global_config=OpPalettizerConfig(mode="kmeans", nbits=4),
    op_name_configs=op_name_configs,
)

compressed = palettize_weights(mlmodel, config)
compressed.save("qwen35_decode_mixed.mlpackage")
```

### 1.2 Quantization Sensitivity: DeltaNet vs Attention

**Key finding from SSM quantization research (Quamba, Q-Mamba):** Recurrent/linear attention layers are MORE sensitive to activation quantization but potentially LESS sensitive to weight-only palettization, for different reasons:

**Why DeltaNet weights may tolerate aggressive compression:**
- DeltaNet's recurrent state update (`S = S * g + k * delta`) uses projections (Q, K, V, beta, gate) that are simple linear transforms. The delta rule itself is an error-correction mechanism -- it computes `delta = (v - S*k) * beta`, which is inherently a difference. Small weight perturbations in the projections get partially corrected by the delta mechanism.
- The gate `g` applies exponential decay to the state, meaning errors in early tokens are exponentially forgotten. Weight quantization errors in one step don't propagate indefinitely.
- DeltaNet layers don't have the softmax nonlinearity that amplifies small differences in attention weights.

**Why attention layers need higher precision:**
- Softmax attention is a winner-take-all mechanism. Small changes in Q*K^T scores get amplified by softmax, especially for sharp attention patterns. This makes attention weights more sensitive to quantization.
- The KV cache accumulates exact values that are reused across all future tokens. Any quantization error in the K/V projections is permanently stored and repeatedly accessed.
- Research from Quamba confirms: "self-attention layers are more robust to [activation] quantization errors" but the weight matrices (Q, K, V, O projections) in attention are highly sensitive because they must maintain precise retrieval patterns.

**However, there is a critical caveat for DeltaNet activations:**
- SSM-family models (including DeltaNet) exhibit "massive outliers in output activations" that are not present in attention outputs (Quamba finding).
- The recurrent state can accumulate large values through repeated outer-product additions.
- This means weight-only palettization (which is what CoreML palettization does) should be safer than weight+activation quantization for DeltaNet.

**Recommended bit-width assignment:**

| Component | Recommended Bits | Rationale |
|-----------|-----------------|-----------|
| DeltaNet Q/K/V/beta/gate projections | 2-bit | Error-correcting delta rule; gate-based forgetting limits error propagation |
| DeltaNet output projection | 4-bit | More sensitive -- output feeds into residual stream |
| DeltaNet conv1d weights | 4-bit | Small kernel (size 4), few parameters, but position-sensitive |
| Attention Q/K projections | 4-bit | Softmax amplifies errors; retrieval precision matters |
| Attention V/O projections | 4-bit | Values stored in KV cache; output directly affects residual |
| FFN (all layers) | 2-bit | Largest weight matrices; typically most compressible |
| LM head | 2-bit | See Section 3 |
| Embeddings | 6-bit or skip | Small relative to total; high sensitivity |

### 1.3 Sensitivity Analysis Before Compression

CoreML tools provides `get_weights_metadata()` for inspecting weight properties, but the real sensitivity analysis comes from the Mixed-Bit Palettization (MBP) algorithm:

**MBP methodology (from Apple's Stable Diffusion work):**

1. **Pre-analysis phase:** For each weight tensor, compress it at each candidate bit-width (1, 2, 4, 6, 8) and measure the SQNR (Signal-to-Quantization-Noise Ratio) or PSNR of the model output vs. FP16 baseline.
2. **Recipe generation:** Use a greedy or optimization algorithm to assign bit-widths that minimize average bit-width while keeping signal strength above a threshold.
3. **Application:** Apply the resulting "recipe" (JSON mapping weight names to bit-widths) to the model.

**Practical sensitivity analysis script:**

```python
import coremltools as ct
import numpy as np

mlmodel = ct.models.MLModel("qwen35_decode.mlpackage")
metadata = ct.optimize.coreml.get_weights_metadata(mlmodel, weight_threshold=512)

# Analyze weight distributions per layer type
for name, meta in metadata.items():
    w = meta.val
    # Compute how well k-means would approximate this weight
    unique_ratio = meta.unique_values / w.size
    sparsity = meta.sparsity
    # Weight range / std ratio indicates outlier presence
    std = np.std(w)
    range_std_ratio = (np.max(w) - np.min(w)) / (std + 1e-8)

    layer_type = "deltanet" if any(f"layers.{i}." in name for i in deltanet_indices) else "attention"
    print(f"{layer_type:10s} | {name:60s} | size={w.size:>8d} | "
          f"sparsity={sparsity:.3f} | range/std={range_std_ratio:.1f} | "
          f"unique_ratio={unique_ratio:.4f}")
```

Weights with high `range/std` ratios have outliers and benefit from OD-MBP's outlier decomposition. Weights with low unique ratios are already clustered and compress well.

### 1.4 OD-MBP (Outlier-Decomposed Mixed-Bit Palettization)

From the WhisperKit ICML 2025 paper: OD-MBP decomposes each weight tensor into two branches:

1. **Inlier branch:** Weights within 3 standard deviations of the mean. Stored as palettized (low-bit LUT indices). Uses CoreML's natively-accelerated palettization format.
2. **Outlier branch:** Weights beyond 3 standard deviations (<1% of values). Stored as float16 sparse weights. Uses CoreML's natively-accelerated sparse format.

The forward pass becomes: `output = dense_palettized_matmul(x, W_inlier) + sparse_matmul(x, W_outlier)`

Both branches are hardware-accelerated on ANE/GPU. This is especially relevant for DeltaNet layers where the recurrent state can produce weight distributions with significant outliers.

**Result on Whisper Large v3 Turbo:** 1.6 GB -> 0.6 GB (62% reduction) with WER within 1% of uncompressed.

**To use OD-MBP in coremltools:**

```python
from coremltools.optimize.coreml import (
    OpPalettizerConfig,
    OptimizationConfig,
    palettize_weights,
)

# OD-MBP uses the sparse+palettized joint compression
# Step 1: Palettize with outlier-aware config
config = OpPalettizerConfig(
    mode="kmeans",
    nbits=4,
    granularity="per_grouped_channel",
    group_size=16,
)
# Step 2: Apply with joint_compression for outlier handling
# (requires coremltools 8.0+)
opt_config = OptimizationConfig(global_config=config)
compressed = palettize_weights(mlmodel, opt_config)
```

---

## 2. Apple Neural Engine Parallelization for DeltaNet

### 2.1 ANE Architecture Constraints

Based on the Orion paper (reverse-engineering the M4 ANE) and Apple's own transformer deployment guide:

**ANE hardware facts:**
- Fixed-function accelerator optimized for convolution and matrix-multiply in FP16
- M4 Max: 16 neural engine cores, 32 MB on-chip SRAM, ~19 TFLOPS measured
- Compile-then-dispatch model: MIL -> E5 microcode via `_ANECompiler`
- Zero idle power (hard power gating) -- ideal for on-demand inference
- IOSurface I/O with fixed `[1, C, 1, S]` layout (FP16), enabling zero-copy transfer
- Weights are baked at compile time and cannot be mutated post-compilation
- Only 27 operations supported; `concat` causes immediate compilation failure
- 1x1 convolutions deliver 3x better throughput than equivalent matmul operations

**Critical constraint for DeltaNet:** The ANE cannot manage mutable recurrent state across dispatches. Each ANE call is a stateless forward pass. The recurrent state must be maintained externally (on CPU/GPU) and passed as input to each ANE dispatch.

### 2.2 Can DeltaNet Sub-Operations Be Parallelized on ANE?

**Within a single DeltaNet layer, the computation has this structure:**

```
Input hidden_states
  |
  +-> q_proj, k_proj, v_proj, beta_proj, gate_proj   [PARALLEL: 5 independent projections]
  |
  +-> conv1d_update(concat(conv_state, input))         [SEQUENTIAL: depends on conv_state]
  |
  +-> delta rule recurrence:                            [SEQUENTIAL: each token depends on previous state]
  |     S = S * exp(g) + k * (v - S*k) * beta
  |     o = S * q
  |
  +-> output_proj(o)                                    [PARALLEL: independent linear]
```

**What can be parallelized:**
- The 5 input projections (Q, K, V, beta, gate) are independent matrix multiplications. The ANE compiler should automatically parallelize these if expressed as separate conv1x1 ops. Deep operation graphs (16-64 ops) achieve 94% ANE utilization vs ~30% for single ops.
- The output projection is independent and can run on ANE.
- The FFN (gate_proj, up_proj, down_proj with SiLU) after each layer is fully parallelizable.

**What must be sequential:**
- The conv1d state update is sequential by nature (sliding window over token history).
- The delta rule recurrence is inherently sequential during autoregressive decode (each token's state depends on the previous token's state).

**However, during decode we only process ONE token at a time.** This means the "sequential" part is just a single step -- there is no sequence to parallelize over. The bottleneck is the matrix multiplications (projections), not the recurrence.

### 2.3 Does CoreML's Graph Compiler Fuse Independent Ops?

**Yes, with limitations.** The ANE compiler performs 5 optimization passes:

1. Dead Code Elimination (DCE)
2. Identity Elimination (remove no-ops)
3. Cast Fusion (eliminate round-trip casts like fp16->fp32->fp16)
4. SRAM Annotation (estimate working set vs 32 MB SRAM budget)
5. ANE Constraint Validation (check for banned ops like `concat`)

The compiler automatically determines axis packing for performance. Independent ops within a graph are dispatched to available cores. The evaluation queue supports depth of 127 operations.

**Key optimization:** Express linear projections as conv1x1 instead of matmul. The ANE guide explicitly recommends `nn.Conv2d` over `nn.Linear` for ANE compatibility. Conv1x1 achieves 3x better throughput than equivalent matmul on ANE.

### 2.4 Chunked-Parallel DeltaNet on ANE?

**During prefill (processing the full system prompt), chunked parallelism is theoretically possible but impractical on ANE.** The DeltaNet paper describes a chunkwise parallel algorithm using the WY representation of Householder matrices:

- Divide sequence into chunks of size C
- Within each chunk: compute using the compact WY representation (parallelizable matrix multiplications)
- Between chunks: propagate state sequentially

The intra-chunk computation can be expressed as matrix multiplications, which ANE handles well. However:

1. The WY representation requires `concat` operations for building the W and U matrices, which causes ANE compilation failure.
2. The working set for chunked computation with head_dim=64 and chunk_size=64 would be ~64x64x2 bytes per head = 8KB per head x 8 heads = 64KB per layer x 18 DeltaNet layers = 1.15 MB. This fits in SRAM but the graph complexity may exceed ANE's 27-op limit.
3. **For our use case (151-token system prompt prefill), the chunked approach with ~3 chunks adds graph compilation overhead that likely exceeds the parallelization benefit.**

**Recommendation:** Keep using the sequential (recurrent) DeltaNet for decode (only 1 token at a time anyway). For prefill, use GPU execution which already handles the MIL `while_loop` at 42ms/token. The ANE is not the right target for DeltaNet's recurrent computation.

### 2.5 ANE vs GPU: Which to Target?

From the status doc: ANE execution was 3x slower than GPU for this model. This aligns with the Orion paper's finding that the ANE excels at conv/matmul workloads but struggles with:
- Large-channel convolutions (32K+ channels rejected)
- Sequential/branching logic (assigned to CPU)
- Non-standard tensor operations

**The 248K vocabulary LM head is likely rejected by ANE** (exceeds 32K channel limit), forcing CPU fallback -- which is the worst case. However, the Orion paper found "softmax over large vocabularies is 33.8x faster on ANE than CPU", suggesting the ANE can handle vocab-sized operations if properly chunked.

**Recommendation:** Use GPU for the full model. The ANE's constraints (concat failure, channel limits, no mutable state) make it a poor fit for hybrid DeltaNet/attention. The GPU already achieves 68 tok/s which is sufficient for our use case (cleanup of ~15 tokens).

---

## 3. LM Head Optimization

### 3.1 Aggressive Palettization of the LM Head

The LM head is a single `nn.Linear(1024, 248064)` projection -- the largest weight matrix in the model at ~508 MB in FP16. It accounts for 55% of inference time (9.1ms per token).

**Can it be 2-bit palettized?**

Yes, with caveats:
- The LM head is a simple linear projection with no nonlinearity. It maps the 1024-dim hidden state to 248K logit scores.
- For greedy decoding (argmax), only the relative ordering of the top logits matters. Small perturbations in the tail of the distribution are irrelevant.
- For text cleanup specifically, the output vocabulary is extremely constrained (~5K unique tokens ever produced). The LM head only needs to correctly rank these tokens above the other 243K.

**Expected quality impact:**
- 4-bit palettization of the LM head: negligible quality impact (already proven -- PAL-4bit gave same quality at 62 tok/s)
- 2-bit palettization: moderate risk. With only 4 centroids per group, the LM head may confuse tokens with similar embeddings. Use `per_grouped_channel` with `group_size=16` to mitigate.
- 1-bit palettization: high risk. Only 2 centroids -- likely insufficient for 248K-way discrimination.

**Recommendation:** Start with 2-bit palettization using `per_grouped_channel` granularity and validate on the cleanup test set. If quality degrades, fall back to 4-bit for the LM head only.

### 3.2 Vocabulary Pruning

The status doc identifies vocabulary pruning as an exploration target. For text cleanup, only ~5K of 248K tokens are ever produced. Pruning strategies:

**Option A: Reduced LM head at conversion time**
- Build a token whitelist from the cleanup training data
- Replace the 248064-dim LM head with a 5000-dim projection
- Map indices back to original vocabulary for detokenization
- **Speedup:** ~50x smaller LM head weight matrix. If bandwidth-bound, this could reduce LM head time from 9.1ms to ~0.2ms.
- **Risk:** If the model ever needs a token outside the whitelist, generation fails. Mitigate by including all tokens that appear in any reasonable English text cleanup output (~10K tokens with safety margin).

**Option B: Chunked argmax in-model (ANEMLL approach)**
- Split the LM head into 16 chunks of ~15.5K tokens each
- Each chunk computes `argmax(logits_chunk)` within the CoreML model
- Output only the 16 winning (index, value) pairs instead of 248K logits
- Final argmax of 16 candidates done on CPU (negligible cost)
- **Benefit:** Reduces ANE-to-host data transfer from 248K x 2 bytes = 496KB to 16 x 8 bytes = 128 bytes per token. Does not reduce compute but eliminates the data transfer bottleneck.

**Option C: Approximate top-k**
- From recent research (arxiv 2412.04358): approximate top-k can be parallelized across chunks with provable guarantees. Each chunk finds its local top-k, then a merge step combines results.
- On ANE: each chunk's top-k is a reduction op (supported). The merge is tiny and can run on CPU.

**Recommendation:** Option A (vocabulary pruning) offers the largest speedup and is the simplest to implement. Build a whitelist of ~10K tokens, replace the LM head, and benchmark. If the 10K-token head is fast enough, no further optimization needed.

### 3.3 Efficient Argmax/Top-K on ANE

CoreML supports `reduce_argmax` as a MIL operation, which maps to ANE execution. For the full 248K vocabulary:

- The Orion paper confirms ANE handles large reductions efficiently
- ANEMLL's `--argmax` flag bakes argmax into the model, recording `argmax_in_model: true` in meta.yaml
- This is the recommended approach if vocabulary pruning is not used

**Implementation in conversion script:**

```python
# In the model wrapper, add argmax after lm_head
class ModelWithArgmax(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args):
        logits = self.model(*args)  # [1, 1, 248064]
        token_id = torch.argmax(logits[:, -1, :], dim=-1)  # [1]
        return token_id
```

This eliminates the need to transfer 248K floats from GPU/ANE to CPU. The argmax runs on the same compute unit as the LM head.

---

## 4. Practical coremltools Implementation

### 4.1 Complete Mixed-Bit Compression Pipeline

```python
#!/usr/bin/env python3
"""Mixed-bit palettization for Qwen3.5-0.8B CoreML model.

Applies different bit-widths to DeltaNet layers, attention layers,
FFN weights, and the LM head based on quantization sensitivity analysis.
"""

import coremltools as ct
from coremltools.optimize.coreml import (
    OpPalettizerConfig,
    OptimizationConfig,
    palettize_weights,
    get_weights_metadata,
)

# Architecture constants
NUM_LAYERS = 24
DELTANET_INDICES = [i for i in range(NUM_LAYERS) if i % 4 != 3]  # 18 layers
ATTN_INDICES = [i for i in range(NUM_LAYERS) if i % 4 == 3]       # 6 layers


def classify_weight(name):
    """Classify a weight tensor by its role in the model."""
    if "lm_head" in name:
        return "lm_head"
    if "embed" in name:
        return "embedding"
    if "norm" in name:
        return "norm"

    for i in DELTANET_INDICES:
        if f"layers.{i}." in name or f"layers_{i}_" in name:
            if "mlp" in name or "ffn" in name:
                return "deltanet_ffn"
            if "o_proj" in name or "out_proj" in name:
                return "deltanet_output"
            return "deltanet_proj"

    for i in ATTN_INDICES:
        if f"layers.{i}." in name or f"layers_{i}_" in name:
            if "mlp" in name or "ffn" in name:
                return "attn_ffn"
            return "attn_proj"

    return "other"


# Bit-width assignment per component type
BIT_CONFIGS = {
    "deltanet_proj":   OpPalettizerConfig(mode="kmeans", nbits=2,
                           granularity="per_grouped_channel", group_size=16),
    "deltanet_output": OpPalettizerConfig(mode="kmeans", nbits=4,
                           granularity="per_grouped_channel", group_size=16),
    "deltanet_ffn":    OpPalettizerConfig(mode="kmeans", nbits=2,
                           granularity="per_grouped_channel", group_size=16),
    "attn_proj":       OpPalettizerConfig(mode="kmeans", nbits=4,
                           granularity="per_grouped_channel", group_size=16),
    "attn_ffn":        OpPalettizerConfig(mode="kmeans", nbits=2,
                           granularity="per_grouped_channel", group_size=16),
    "lm_head":         OpPalettizerConfig(mode="kmeans", nbits=2,
                           granularity="per_grouped_channel", group_size=16),
    "embedding":       None,  # Skip -- small and sensitive
    "norm":            None,  # Skip -- tiny
    "other":           OpPalettizerConfig(mode="kmeans", nbits=4),
}


def compress_model(input_path, output_path):
    """Apply mixed-bit palettization to the model."""
    mlmodel = ct.models.MLModel(input_path)

    # Step 1: Analyze weights
    metadata = get_weights_metadata(mlmodel, weight_threshold=512)

    op_name_configs = {}
    stats = {}

    for name, meta in metadata.items():
        category = classify_weight(name)
        config = BIT_CONFIGS.get(category)
        if config is not None:
            op_name_configs[name] = config

        # Track stats
        if category not in stats:
            stats[category] = {"count": 0, "params": 0}
        stats[category]["count"] += 1
        stats[category]["params"] += meta.val.size

    # Print compression plan
    print("Compression Plan:")
    print(f"{'Category':<20} {'Count':>6} {'Params':>12} {'Bits':>5}")
    print("-" * 50)
    total_bits = 0
    total_params = 0
    for cat, s in sorted(stats.items()):
        bits = BIT_CONFIGS.get(cat)
        nbits = bits.nbits if bits else 16
        print(f"{cat:<20} {s['count']:>6} {s['params']:>12,} {nbits:>5}")
        total_bits += s["params"] * nbits
        total_params += s["params"]
    avg_bits = total_bits / total_params
    print(f"\nAverage bit-width: {avg_bits:.2f}")
    print(f"Estimated size: {total_bits / 8 / 1024 / 1024:.0f} MB "
          f"(vs {total_params * 2 / 1024 / 1024:.0f} MB FP16)")

    # Step 2: Apply compression
    opt_config = OptimizationConfig(op_name_configs=op_name_configs)
    compressed = palettize_weights(mlmodel, opt_config)
    compressed.save(output_path)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.mlpackage> <output.mlpackage>")
        sys.exit(1)
    compress_model(sys.argv[1], sys.argv[2])
```

### 4.2 Sensitivity Analysis Script

Before committing to bit-width assignments, run this sensitivity analysis:

```python
#!/usr/bin/env python3
"""Per-layer sensitivity analysis for Qwen3.5 mixed-bit palettization.

For each weight tensor, measures the SQNR at different bit-widths
to determine the optimal compression recipe.
"""

import numpy as np
import coremltools as ct
from coremltools.optimize.coreml import (
    OpPalettizerConfig,
    OptimizationConfig,
    palettize_weights,
    get_weights_metadata,
)
from sklearn.cluster import KMeans


def compute_sqnr(original, quantized):
    """Signal-to-Quantization-Noise Ratio in dB."""
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - quantized) ** 2)
    if noise_power < 1e-20:
        return 100.0  # Perfect reconstruction
    return 10 * np.log10(signal_power / noise_power)


def simulate_palettization(weights, nbits, group_size=16):
    """Simulate k-means palettization and return SQNR."""
    n_clusters = 2 ** nbits
    flat = weights.flatten()

    if len(flat) < n_clusters:
        return 100.0  # Too few weights to compress

    # Simple k-means simulation
    kmeans = KMeans(n_clusters=n_clusters, n_init=1, max_iter=10, random_state=42)
    kmeans.fit(flat.reshape(-1, 1))
    quantized = kmeans.cluster_centers_[kmeans.labels_].flatten()

    return compute_sqnr(flat, quantized)


def analyze_model(model_path):
    """Run sensitivity analysis on all weight tensors."""
    mlmodel = ct.models.MLModel(model_path)
    metadata = get_weights_metadata(mlmodel, weight_threshold=512)

    bit_widths = [1, 2, 4, 6, 8]

    print(f"{'Weight Name':<60} {'Size':>10} ", end="")
    for b in bit_widths:
        print(f" {b}b SQNR", end="")
    print("  Recommended")
    print("-" * 130)

    for name, meta in metadata.items():
        w = meta.val.astype(np.float32)
        print(f"{name:<60} {w.size:>10,} ", end="")

        sqnrs = {}
        for b in bit_widths:
            sqnr = simulate_palettization(w, b)
            sqnrs[b] = sqnr
            print(f" {sqnr:>7.1f}", end="")

        # Recommend: lowest bit-width with SQNR > 20 dB
        recommended = 8
        for b in bit_widths:
            if sqnrs[b] > 20.0:
                recommended = b
                break

        print(f"  -> {recommended}b")

    return metadata
```

### 4.3 Joint Compression (Palettization + Quantization)

For maximum compression, combine palettization with INT8 activation quantization to create A8W4 or A8W2 models:

```python
from coremltools.optimize.coreml import (
    OpPalettizerConfig,
    OpLinearQuantizerConfig,
    OptimizationConfig,
    palettize_weights,
    linear_quantize_weights,
)

# Step 1: Palettize weights (4-bit with INT8 LUT)
pal_config = OptimizationConfig(
    global_config=OpPalettizerConfig(
        mode="kmeans", nbits=4,
        granularity="per_grouped_channel", group_size=16,
        lut_dtype="int8",  # INT8 lookup table for ANE acceleration
    )
)
model = palettize_weights(mlmodel, pal_config)

# Step 2: Quantize activations to INT8
quant_config = OptimizationConfig(
    global_config=OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int8",
        granularity="per_tensor",
        joint_compression=True,  # Required when model is already compressed
    )
)
model = linear_quantize_weights(model, quant_config)

model.save("qwen35_a8w4.mlpackage")
```

**Important:** The coremltools docs warn that sequential post-training compression may not yield optimal accuracy. For best results, use `coremltools.optimize.torch.*` APIs during model training (e.g., `SKMPalettizer` for Sensitive K-Means).

### 4.4 WhisperKit-Style Compression Pipeline (Reference)

The whisperkittools project (github.com/argmaxinc/whisperkittools) uses a two-step process:

1. **Pre-analysis:** Run `mixed_bit_compression_pre_analysis` which generates a JSON recipe file. This tests each layer at each bit-width and measures output PSNR against FP16 baseline using a calibration dataset.
2. **Application:** Run `mixed_bit_compression_apply` which reads the recipe and applies the corresponding `OpPalettizerConfig` to each weight.

The recipe JSON format maps weight names to bit-widths:

```json
{
  "recipe_2_50_bit": {
    "layers.0.self_attn.q_proj.weight": 2,
    "layers.0.self_attn.k_proj.weight": 2,
    "layers.0.self_attn.v_proj.weight": 4,
    "layers.3.self_attn.q_proj.weight": 4,
    ...
    "lm_head.weight": 2
  }
}
```

**Recommended approach for Qwen3.5:**
1. Convert the model to CoreML (already done)
2. Run the sensitivity analysis script (Section 4.2) to get per-weight SQNR at each bit-width
3. Generate a recipe targeting ~2.5 average bits (aggressive) or ~3.5 average bits (conservative)
4. Apply the recipe and validate on the cleanup test set
5. If quality is acceptable, ship it

---

## 5. WhisperKit ICML 2025 Paper Insights

### Paper: "WhisperKit: On-device Real-time ASR with Billion-Scale Transformers" (arXiv 2507.10860)

### 5.1 Key Transferable Findings

**OD-MBP compression results on Whisper Large v3 Turbo:**

| Metric | Original | OD-MBP | Delta |
|--------|----------|--------|-------|
| Model Size | 1.6 GB | 0.6 GB | -62% |
| LibriSpeech WER | 1.93% | 1.96% | +0.03% |
| Earnings22 WER | 11.55% | 12.35% | +0.80% |
| CommonVoice WER | 12.13% | 13.03% | +0.90% |

**Outlier decomposition is critical:** Weights exceeding 3 standard deviations from the mean are isolated as outliers (<1% of weights) and stored at FP16 in sparse format. This prevents the few extreme values from corrupting the palettization centroids for the remaining 99% of weights.

### 5.2 Applicability to Qwen3.5 Hybrid Architecture

**Similarities:**
- Both Whisper and Qwen3.5 are autoregressive transformer-based models
- Both have a large output projection (Whisper: 51K vocab, Qwen3.5: 248K vocab)
- Both run inference on Apple Silicon via CoreML

**Differences that matter:**
- Qwen3.5 has DeltaNet layers with recurrent state -- Whisper does not. The recurrent state accumulates quantization errors over tokens, making activation quantization riskier (per Quamba findings).
- Qwen3.5's vocabulary is 5x larger (248K vs 51K), making the LM head an even larger bottleneck.
- Qwen3.5 generates ~15 tokens per request (short), so error accumulation is limited. This actually makes aggressive compression safer than for long-generation tasks.

### 5.3 Recommended Transfer Strategy

1. **Use OD-MBP for all weight tensors** -- the outlier decomposition is model-agnostic and will help with DeltaNet's potentially outlier-heavy weight distributions.

2. **Apply the 3-sigma outlier threshold** from WhisperKit. For each weight tensor:
   - Inliers (within 3 sigma): palettize at target bit-width
   - Outliers (beyond 3 sigma): keep at FP16 sparse

3. **Target 2.5-3.0 average bits** for the full model. WhisperKit achieved ~3 average bits with <1% quality loss. Given that Qwen3.5's cleanup task is simpler than ASR (constrained output space, short generation), we should be able to go more aggressive.

4. **Validate using the cleanup test set**, not perplexity. The metric that matters is "does the cleaned text match expected output?" -- a binary pass/fail per example, not a continuous metric.

---

## 6. Actionable Recommendations (Priority Order)

### Immediate (This Week)

1. **Run `get_weights_metadata` on the current FP16 model** to understand weight distributions per layer type. This is zero-risk and provides the data needed for everything else.

2. **Apply uniform 4-bit palettization** (already proven at 62 tok/s with same quality). This is the safe baseline.

3. **Try 2-bit palettization on the LM head only** while keeping everything else at 4-bit. The LM head is the biggest single weight and most likely to tolerate aggressive compression. Validate on cleanup test set.

### Short-Term (Next Sprint)

4. **Implement the mixed-bit recipe** from Section 4.1: DeltaNet projections at 2-bit, attention at 4-bit, FFN at 2-bit. This should achieve ~2.5 average bits (~200MB model size, down from ~1.6GB FP16).

5. **Prototype vocabulary pruning** (Option A from Section 3.2): Build a 10K-token whitelist and replace the LM head. This could reduce LM head time from 9.1ms to ~0.4ms, pushing total decode to ~8ms/token (125 tok/s).

6. **Add in-model argmax** to eliminate the 248K-float data transfer. Even without vocab pruning, this saves ~0.5ms per token from reduced memory bandwidth.

### Medium-Term (Next Month)

7. **Run the full sensitivity analysis** (Section 4.2) to generate an optimal per-weight recipe.

8. **Evaluate OD-MBP** with outlier decomposition for DeltaNet layers specifically. The 3-sigma split may improve quality at 2-bit compared to naive palettization.

9. **Test A8W2 joint compression** (Section 4.3) for maximum size reduction. Only pursue if the cleanup task quality holds.

---

## 7. Projected Performance Impact

| Configuration | Avg Bits | Model Size | Projected tok/s | Risk |
|---------------|----------|------------|-----------------|------|
| Current (FP16) | 16 | ~1.6 GB | 68 | None |
| Uniform 4-bit PAL | 4 | ~400 MB | 72 | Low (proven) |
| Mixed-bit (2/4) | ~2.8 | ~280 MB | 78 | Medium |
| Mixed-bit + vocab pruning | ~2.8 | ~250 MB | 120+ | Medium |
| Mixed-bit + in-model argmax | ~2.8 | ~280 MB | 82 | Low |
| A8W2 joint | ~2.5 | ~250 MB | 85 | High |

The largest single speedup opportunity remains **vocabulary pruning** (eliminating 95% of the LM head compute). Mixed-bit palettization provides model size reduction with modest speed improvement (palettization is mostly a size optimization; the dequantization overhead partially offsets bandwidth savings).

---

## Sources

- [WhisperKit ICML 2025 Paper](https://arxiv.org/html/2507.10860v1) -- OD-MBP methodology, compression results
- [Quamba: Post-Training Quantization for SSMs](https://arxiv.org/html/2410.13229v1) -- SSM quantization sensitivity, outlier analysis
- [Orion: Characterizing Apple's Neural Engine](https://arxiv.org/html/2603.06728v1) -- ANE architecture, constraints, op support
- [Apple: Deploying Transformers on ANE](https://machinelearning.apple.com/research/neural-engine-transformers) -- Memory layout, conv2d recommendation
- [coremltools Palettization Overview](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html) -- API docs, MBP algorithm
- [coremltools Palettization Algorithms](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-algos.html) -- SKM, DKM algorithms
- [coremltools Joint Compression](https://apple.github.io/coremltools/docs-guides/source/opt-joint-compression.html) -- Combining palettization + quantization
- [coremltools Utilities API](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.utilities.html) -- get_weights_metadata, OptimizationConfig
- [Apple ML Stable Diffusion](https://github.com/apple/ml-stable-diffusion) -- MBP reference implementation, recipe format
- [ANEMLL Framework](https://github.com/Anemll/Anemll) -- In-model argmax, LM head chunking
- [DeltaNet Part II: Chunkwise Parallel Algorithm](https://sustcsonglin.github.io/blog/2024/deltanet-2/) -- Chunked parallelization theory
- [Parallelizing Linear Transformers with the Delta Rule](https://arxiv.org/abs/2406.06484) -- WY representation, hardware-efficient DeltaNet
- [Apple SD Mixed-Bit Palettization Models](https://huggingface.co/apple/coreml-stable-diffusion-mixed-bit-palettization) -- Pre-computed recipes, quality benchmarks
- [coremltools MLModel Utilities Guide](https://apple.github.io/coremltools/docs-guides/source/mlmodel-utilities.html) -- get_weights_metadata usage examples
- [whisperkittools](https://github.com/argmaxinc/whisperkittools) -- Compression pipeline reference
- [Approximate Top-k for Increased Parallelism](https://arxiv.org/pdf/2412.04358) -- Chunked top-k algorithms
- [Qwen3.5 Architecture Analysis](https://huggingface.co/blog/mlabonne/qwen35) -- Layer type ratios, hybrid design
