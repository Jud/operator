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

Validated CoreML inference from Swift with correct output:
- 2-chunk model: **58 tok/s realistic generation** (17.2ms/tok)
- Single predict: 65 tok/s (15.3ms/tok, lower bound)
- GPU concurrent execution works: 1.44x for 2 models, 2.7x for 3 models

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

**Production-ready for the 0.8B model:**
1. Convert with `convert_split2.py --fp32`
2. Load in Swift, feed tokens through chunk_a → chunk_b
3. DeltaNet states accumulate across tokens (recurrent)
4. KV cache for attention layers (required for correctness)
5. ~49 tok/s with KV cache = 15 tokens in ~306ms

**For a text cleanup generating 15 tokens:** ~306ms generation + ~15ms prefill = **~321ms from key release to clean text.**

## Next Steps

### High Priority
1. **Add KV cache to the CoreML 2-chunk model** — Required for correctness. Solve the variable-length cache issue (either growing cache or fixed-size with proper masking).
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
