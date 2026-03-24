# Hypothesis: Qwen3.5-0.8B's built-in MTP head can enable speculative decoding for >30% generation speedup

**Status:** PROVEN
**Date:** 2026-03-24
**Slug:** qwen35-mtp-speedup

## What We're Testing

Qwen3.5-0.8B ships with a Multi-Token Prediction (MTP) head -- a single transformer layer that predicts the *next-next* token from the main model's hidden state and the embedding of the current predicted token. If the MTP head's predictions match the main model frequently enough, we can use speculative decoding: on each step, accept both the main model's token and the MTP's speculative token when they agree, effectively generating 2 tokens per step on a hit.

We are testing whether the MTP head produces correct, high-hit-rate predictions on our real workload (STT cleanup) with low enough overhead to yield a meaningful generation speedup over the baseline 58 tok/s CoreML inference.

## Why This Matters

Operator's on-device transcription cleanup runs Qwen3.5-0.8B via CoreML on the ANE at ~58 tok/s. For dictation mode, lower latency directly improves the user experience -- every 100ms shaved off the cleanup step is 100ms less delay before pasted text appears. MTP speculative decoding could boost effective throughput by 30-50% with no model quality degradation, since the main model's output is unchanged (we just skip re-running it when the speculation is correct).

The MTP weights are already in the checkpoint but are ignored by the standard transformers inference path. This hypothesis tests whether they are usable and valuable.

## Success Criteria

- [x] **MTP correctness** -- MTP head produces correct logits on manually-verified examples (top-5 tokens are plausible English words, not random IDs)
- [x] **Hit rate > 50%** -- MTP predictions match the main model's actual next token more than half the time on the real cleanup task
- [x] **Generation speedup > 30%** -- Projected effective throughput exceeds 75.4 tok/s (1.3x baseline 58 tok/s), accounting for MTP overhead
- [x] **MTP overhead < 5ms per prediction** -- Incremental MTP cost (fuse + single-position forward) is under 5ms on CoreML/ANE (estimated from layer count ratio)

## Failure Criteria

- [ ] **Garbage logits** -- MTP top-5 predictions are random token IDs, not coherent words
- [ ] **Hit rate < 30%** -- Predictions are too inaccurate for speculative decoding to help
- [ ] **MTP overhead > 10ms** -- The extra compute per step eats all the gains from speculation
- [ ] **Speedup < 10%** -- Even with decent hit rate, the overhead makes it not worth the complexity

## Constraints

- **PyTorch CPU probe only.** This tests the MTP head's accuracy and overhead characteristics, not the full CoreML integration. CoreML MTP latency is estimated from the layer count ratio (1/24 of main model).
- **Single cleanup task.** Hit rate is measured on one representative cleanup prompt. Production hit rate may vary across different input types.
- **Greedy decoding only.** No sampling -- argmax at each step for deterministic comparison.

## Prototype Approach

A standalone Python script (`docs/hypotheses/qwen35-mtp-speedup/working/mtp_probe.py`) that:

1. Loads Qwen3.5-0.8B and extracts the MTP weights from the checkpoint
2. Builds the MTP module (RMSNorm fusion + 1 transformer layer + tied lm_head)
3. Runs greedy autoregressive generation on a cleanup prompt
4. At each step, runs the MTP head and records whether its prediction matches the main model's next token
5. Reports per-token hit/miss table, hit rate, MTP latency, and projected speedup

## Results

**Verdict: PROVEN** -- See `results.md` for full analysis.

Hit rate of 77.6% (38/49) far exceeds the 50% threshold. Projected CoreML speedup is 1.51x (87 tok/s vs 58 tok/s baseline), exceeding the 30% target. Estimated CoreML MTP overhead of ~1.4ms is well under the 5ms ceiling. All 4 success criteria met.
