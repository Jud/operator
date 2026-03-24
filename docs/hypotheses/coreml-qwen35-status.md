# CoreML Qwen3.5 for On-Device Speech Cleanup — Status

## The Goal

Clean up raw speech-to-text output (remove fillers, fix grammar, handle self-corrections) on-device using Qwen3.5, running natively on Apple Silicon via CoreML.

## What We Built

### 1. Direct HF → CoreML Conversion Pipeline

**First ever direct conversion of Qwen3.5's Gated DeltaNet architecture to CoreML.**

No custom model reimplementation needed. We trace the HuggingFace model directly and convert with coremltools, using patches we wrote:

- `functional_deltanet.py` — Rewrites DeltaNet recurrence to be functional (no in-place ops). Also patches conv1d_update and KV cache update for traceability.
- `patches.py` — 7 coremltools fixes: `_cast`, `diff`, `new_ones`, `chunk`, `pad`, `reshape`, `slice`
- `convert.py` — Prefill model (full sequence → logits)
- `convert_decode.py` — Decode model (single token + DeltaNet states → logits)
- `convert_split2.py` — 2-chunk split (layers 0-11, layers 12-23+head)
- `convert_disaggregated.py` — 14-model split (per-block)

Scripts: `scripts/coreml-convert/`

### 2. Prompt Engineering

The v7-rich few-shot prompt with `<transcription>` tags produces excellent cleanup. With KV-cached system prompt, per-request latency is ~250ms in MLX Python.

Key findings:
- Few-shot examples are essential — rules alone don't work
- `<transcription>` tags prevent the model from answering questions in the text
- System prompt caching eliminates the 300-token prompt cost

Script: `scripts/cleanup-preview.py` (live STT cleanup preview via log tailing)
Prompt iteration: `docs/hypotheses/prompt-iteration/`

### 3. Swift Inference Pipeline

Validated CoreML inference from Swift:
- 2-chunk model: **58 tok/s realistic generation** (17.2ms/tok) — **but without KV cache, so output is broken after ~4 tokens**
- Single predict correctness: top-1 matches PyTorch for single decode step
- GPU concurrent execution works: 1.44x for 2 models, 2.7x for 3 models
- **KV cache not yet in CoreML** — this is the #1 blocker (see "CRITICAL" section below)

Scripts: `scripts/coreml-bench/`

## Key Findings

### Correctness

| Configuration | Top-1 Match | Notes |
|---|---|---|
| All-at-once prefill (no KV) | ✓ | Correct for prompt processing |
| Decode with KV cache | ✓ **10/10 tokens** | Exact match with PyTorch |
| Decode without KV cache | ✗ 4/10 tokens | Degenerates into repetition |
| Streaming one-at-a-time (no KV) | ✗ | Position-blind attention |

**KV cache is required for correct generation.** Without it, attention layers are position-blind and the model breaks after a few tokens.

### Performance (0.8B model, Apple Silicon GPU)

| Metric | Value |
|---|---|
| Realistic generation (Swift, 2-chunk) | **58 tok/s** |
| Transformer layers (A+B) | ~8ms |
| lm_head (248K vocab) | ~9.1ms |
| KV cache overhead (6 attn layers) | +3.1ms |
| With KV cache (projected) | ~49 tok/s |
| MLX Python baseline | 74 tok/s |

### Architecture

The lm_head (55% of total time) is **bandwidth-bound** — parallelizing it via sharding gives zero speedup (confirmed experimentally). The transformer layers are compute-bound and benefit from concurrent execution.

### What Didn't Work

| Approach | Result | Why |
|---|---|---|
| ANE execution | 3x slower than GPU | DeltaNet ops not ANE-optimized |
| Head sharding (2-way, 3-way) | 0% speedup | Bandwidth-bound, single memory bus |
| Fake concurrent pipeline | 82 tok/s (invalid) | Used same token — no real dependencies |
| MTP (multi-token prediction) | 0.88x on 0.8B | Head dominates; MTP adds head cost |
| Streaming without KV cache | Broken | Position-blind attention |

### What Worked

| Approach | Result |
|---|---|
| Direct HF→CoreML conversion | First working Qwen3.5 DeltaNet conversion |
| Functional DeltaNet rewrite | Correct output, traceable |
| 2-chunk split | Correct, 58 tok/s, minimal overhead |
| PAL-4bit quantization | 62 tok/s (monolithic), same quality |
| Swift concurrent execution | 1.44x speedup (2 models on GPU) |
| KV cache for attention | 10/10 token match with PyTorch |
| Few-shot cleanup prompt | Excellent cleanup quality on real speech |

## Current State of the Art

**NOT yet production-ready.** The conversion pipeline, Swift inference, and prompt engineering are done. The critical missing piece is KV cache in CoreML — without it, generation produces garbage after a few tokens.

**Projected performance once KV cache works:**
- ~49 tok/s with KV cache (58 tok/s current minus 3.1ms overhead)
- 15 tokens in ~306ms generation + ~15ms prefill = **~321ms from key release to clean text**

## CRITICAL: KV Cache Is Not In CoreML Yet

**The 58 tok/s Swift benchmark runs WITHOUT KV cache.** It produces degenerate output (repeating tokens) after ~4 generation steps. The 10/10 token-match correctness proof was done in **pure PyTorch only**.

Adding KV cache to the CoreML model is the #1 blocker before any integration.

### What we tried and why it failed

1. **Padded fixed-size KV cache (`convert_decode.py` with `--max-seq-len 256`):**
   - Allocated [1, 2, 256, 256] zero-padded KV tensors
   - `Qwen3_5DynamicCache.get_seq_length()` returned 256 (the pad size) instead of the real filled count
   - This corrupted internal position calculations and causal mask construction
   - Result: max logit diff 6.0, wrong top-1 token
   - The attention computation was correct in isolation (we proved a single attention layer with padded cache + mask gives diff=0.0), but the model-level code uses `get_seq_length()` in ways we couldn't override through tracing

2. **Variable-length KV cache (`ct.RangeDim`):**
   - CoreML's `RangeDim` for the sequence dimension caused symbolic shape errors during conversion
   - The `slice` op inside attention couldn't handle symbolic dimensions
   - Never got past the conversion step

3. **Scatter-based fixed-size update:**
   - Used `torch.scatter` to write new KV at `cache_position` instead of concatenating
   - The scatter worked but `get_seq_length()` still returned the wrong value
   - Overriding `get_seq_length` with a lambda didn't help because trace bakes it as a constant

### The correct PyTorch implementation (reference)

This code produces 10/10 token match. The key: use the REAL HuggingFace cache object with naturally growing KV tensors, passed to ALL layers:

```python
# Prefill → get cache
pt_out = model(prompt_ids, use_cache=True)
cache = pt_out.past_key_values

# Decode one token at a time
for i in range(num_tokens):
    hidden = model.model.embed_tokens(torch.tensor([[current_token]]))
    cos, sin = all_cos[:, pos:pos+1, :], all_sin[:, pos:pos+1, :]

    for layer in model.model.layers:
        result = layer(hidden, position_embeddings=(cos, sin),
                      past_key_values=cache, cache_position=torch.tensor([pos]))
        hidden = result[0] if isinstance(result, tuple) else result

    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden)
    current_token = torch.argmax(logits[0, -1]).item()
    pos += 1
```

The cache object grows by 1 each step (attention layers call `cache.update()` which concatenates). This is what CoreML needs to replicate.

### Suggested approach for next session

The most promising path is to **separate attention layers into their own CoreML models** that each manage a small, growing KV cache:

1. Split into 3 model types:
   - **DeltaNet chunks** (3 layers each, 6 models): fixed-size state, no KV issue
   - **Attention models** (1 layer each, 6 models): each has its own KV cache [1, 2, N, 256] where N grows
   - **Head model**: norm + lm_head

2. For each attention model, use `EnumeratedShapes` to support specific cache sizes (e.g., 1, 5, 10, 20, 50, 100 entries) instead of `RangeDim`. CoreML supports enumerated shapes well.

3. Or: keep attention layers in PyTorch (they're only 1.2ms each × 6 = 7.2ms) and only convert DeltaNet layers + head to CoreML. The DeltaNet layers are the compute-heavy part (8ms) and have no KV cache issue.

## Next Steps

### High Priority
1. **Add KV cache to the CoreML pipeline** — Required for correctness. See "Suggested approach" above. Without this, the model cannot generate coherent text.
2. **Convert the 4B model** — Better cleanup quality. The conversion pipeline is model-agnostic. The lm_head bottleneck is proportionally smaller on 4B (transformer layers dominate).
3. **Integrate into Operator** — Wire up the Swift inference pipeline into `finishAndTranscribe()`.

### Medium Priority
4. **Streaming prefill during speech** — Feed confirmed WhisperKit tokens through the model while the user is still speaking. Requires fixing the conv_state handoff between batches.
5. **PAL-4bit the 2-chunk model** — Free 1.04x speedup + 4x model size reduction.
6. **Pre-compute system prompt** — Cache DeltaNet states + KV cache for the 300-token system prompt at app startup.

### Exploration
7. **4B model with MTP** — MTP is viable on 4B where transformer layers (not head) are the bottleneck. 77.6% hit rate already proven.
8. **Vocabulary pruning** — The 248K vocab head is the bottleneck. For text cleanup, only ~5K tokens are ever produced. A pruned head projection could be 50x smaller.
9. **ANEMLL-style Conv2d optimization** — Rewrite layers as Conv2d for ANE acceleration.

## File Map

```
scripts/
├── coreml-convert/
│   ├── patches.py              # coremltools op fixes
│   ├── functional_deltanet.py  # Functional DeltaNet + KV cache patches
│   ├── convert.py              # Prefill model conversion
│   ├── convert_decode.py       # Decode model conversion
│   ├── convert_split2.py       # 2-chunk split conversion
│   └── convert_disaggregated.py # 14-model disaggregated conversion
├── coreml-bench/
│   ├── main.swift              # Concurrent execution test
│   ├── inference.swift         # Correctness verification
│   ├── realistic_gen.swift     # Realistic generation benchmark
│   └── concurrent_pipeline.swift # Pipeline benchmark
├── cleanup-preview.py          # Live STT cleanup via log tailing
docs/hypotheses/
├── coreml-qwen35-status.md     # This document
├── prompt-iteration/           # Prompt engineering experiments
├── report.md                   # Model comparison report
├── qwen35-stt-cleanup/         # Qwen3.5-2B hypothesis (disproven)
├── qwen35-4b-cleanup/          # Qwen3.5-4B hypothesis (disproven)
├── bert-disfluency/            # BERT classifier hypothesis (disproven)
├── coedit-cleanup/             # CoEdit hypothesis (disproven)
└── qwen35-mtp-speedup/         # MTP hypothesis (disproven on 0.8B)
```
