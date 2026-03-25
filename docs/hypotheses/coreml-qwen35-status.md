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

**End-to-end prefill + decode pipeline working. Correct output, prefill + decode in Swift.**

| Config | tok/s | ms/tok | 15 tokens | Correct |
|---|---|---|---|---|
| 2-chunk, no KV cache, FP16 | 58 | 17.2 | N/A | Degenerates after 4 tokens |
| **2-chunk, with KV cache, FP16** | **68** | **14.6** | **219ms** | **10/10 token match** |
| 2-chunk, with KV cache, FP32 | 56 | 17.7 | 266ms | 10/10 token match |
| **Prefill (MIL while_loop, FP32)** | — | **~42** | — | **Exact match** (0.0003 logit diff) |
| **Decode after prefill** | **40** | **25.1** | **376ms** | **Exact match** with PT |

### Prefill + Decode Pipeline

MIL while_loop prefill (151 tokens) → feed states → monolith KV decode (15 tokens):
- Prefill: 6.4s (151 tokens, FP32, includes first-run compilation)
- Decode: 376ms (15 tokens, 40 tok/s)
- Generated text matches PyTorch greedy output exactly
- Script: `scripts/coreml-bench/prefill_decode.swift`

Remaining before production: optimize prefill speed (FP16, quantization) and Operator integration.

## KV Cache: How It Works

**Padded fixed-size KV cache + external attention mask + torch.where update.**

Key insights:
1. `get_seq_length()` is NOT called inside individual decoder layers — only at the model level. Since we call layers directly, we bypass it entirely.
2. `apply_mask_to_padding_states()` is a no-op when `mask.shape[1] == 1` (single-token decode). So we safely pass the 4D attention mask `[1, 1, 1, max_seq_len]` to ALL layers — DeltaNet ignores it, attention uses it.
3. The attention mask is an explicit model INPUT (not constructed inside), avoiding any dependence on cache state. Swift constructs it per-step: 0 for positions 0..pos, -1e4 for padding.
4. KV cache update uses `torch.where(pos == arange, new_kv, old_cache)` — a comparison mask that works reliably in CoreML. (`torch.scatter` also works but `torch.where` avoids the `expand` op issue in coremltools conversion.)

### Pitfalls we hit

1. **PyTorch conv_states have non-contiguous (Fortran) memory order.** When saved via `tensor.numpy()` → `np.save()`, the Fortran order is preserved in the .npy file. Swift's `loadNpy` copies bytes assuming C-order, scrambling the data. **Fix:** `np.ascontiguousarray()` before saving. This caused all the "CoreML produces wrong values" symptoms.

2. **FP16 model outputs are Float16 MLMultiArray.** Reading with `dataPointer.bindMemory(to: Float.self)` reinterprets 2-byte float16 as 4-byte float32 — garbage. **Fix:** Check `arr.dataType == .float16` and read with `Float16.self`. This caused all the "inf" symptoms.

3. **Previous KV cache attempts failed for real reasons:**
   - Padded KV + `get_seq_length()`: returned pad size (256) instead of filled count, corrupting position calculations inside the HF model forward. Solved by calling layers directly.
   - `RangeDim` variable-length: symbolic shape errors in coremltools conversion. Not fixable.
   - These were genuine blockers. The external-mask approach bypasses both.

### Verification

All three decode approaches produce identical tokens in PyTorch (10/10):
1. HF native `model.forward()` with growing cache
2. Manual layer-by-layer with growing cache
3. Manual layer-by-layer with padded cache + external mask

CoreML (both FP16 and FP32) matches PyTorch exactly on first decode step and full 15-token generation.

### Scripts

- `verify_kv_decode.py` — PyTorch 3-way correctness proof
- `convert_split2_kv.py` — 2-chunk conversion with KV cache
- `kv_gen.swift` — Swift KV-cached generation benchmark (handles float16 outputs)

## Next Steps

### High Priority
1. **Optimize prefill speed** — Current 42ms/token is 3x slower than decode. Try FP16 prefill, or pre-compile the mlpackage.
2. **Integrate into Operator** — Wire the Swift inference pipeline into `finishAndTranscribe()`. Prefill at app startup, decode per-request.

### Medium Priority
3. **Convert the 4B model** — Better cleanup quality. The conversion pipeline is model-agnostic. The lm_head bottleneck is proportionally smaller on 4B (transformer layers dominate).
4. **PAL-4bit quantization** — Free 1.04x speedup + 4x model size reduction. Proven on the old model.
5. **Streaming prefill during speech** — Feed confirmed WhisperKit tokens through the model while the user is still speaking.

### Exploration
6. **4B model with MTP** — MTP is viable on 4B where transformer layers (not head) are the bottleneck. 77.6% hit rate already proven.
7. **Vocabulary pruning** — The 248K vocab head is 55% of inference time. For text cleanup, only ~5K tokens are ever produced.

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
