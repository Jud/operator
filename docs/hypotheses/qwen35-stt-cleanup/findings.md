# Findings: Qwen3.5-2B via MLX can clean raw STT output into polished dictation text with sub-800ms latency on Apple Silicon

**Date:** 2026-03-23
**Verdict:** Disproven

## Hypothesis

Qwen3.5-2B, running locally via mlx-lm (Python, 8-bit quantized), can transform raw WhisperKit speech transcriptions into clean, polished written text -- removing fillers, resolving self-corrections, fixing grammar, and adding proper punctuation -- with quality and latency acceptable for real-time dictation on Apple Silicon.

## Verdict Summary

The hypothesis is clearly disproven. Qwen3.5-2B lacks the reasoning capacity to perform meaningful speech cleanup. The model's dominant failure mode is echoing input verbatim -- it returns the raw transcription with minimal or no changes in 6 of 12 test cases. Self-correction resolution (the highest-value cleanup task) fails completely: 0/2. Filler removal succeeds only for the most obvious cases (bare "um" and "uh") but leaves discourse markers ("So", "Like", "Yeah") untouched. Only 3 of 7 success criteria pass, 1 of 4 failure criteria is triggered, and the results are perfectly deterministic across three runs, ruling out bad luck.

## Evidence

### Success Criteria Results

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | Filler removal >= 9/10 | **Fail** | 2/5 (40%). Model removes "um" and "uh" but leaves "So", discourse "Like", and "you know" intact. The ramble cases return virtually unchanged. |
| 2 | Self-correction handling >= 8/10 | **Fail** | 0/2 (0%). Both self-correction inputs returned verbatim. "Tell him Tuesday -- actually no, Wednesday works better" comes back unchanged. The model does not recognize self-correction patterns at all. |
| 3 | Grammar & punctuation >= 9/10 | Pass | 12/12 (100%). Format compliance is perfect -- but this is trivially achieved by echoing input, which inherits WhisperKit's punctuation. |
| 4 | Meaning preservation >= 9/10 | **Fail** | 8/12 (67%). 4 cases have meaning_sim < 0.80 because the model echoes verbose input instead of producing the expected concise version. This is a measurement artifact (the model preserves meaning by not changing anything -- but the metric penalizes divergence from expected output). |
| 5 | Latency < 800ms p95 (<=30w) | Pass | p95 = 472ms. Comfortable margin. Short inputs complete in 300-515ms. |
| 6 | Decode speed >= 100 tok/s | **Fail** | 49.1 tok/s mean. Less than half the target. Consistent across all three runs (48.9, 49.0, 49.1). Only the longest input (ramble-research-guardrails) hits 107 tok/s due to higher output token count amortizing prefill. |
| 7 | Memory < 4GB | Pass | 2.55 GB peak. Well within budget. |

### Failure Criteria Results

| # | Criterion | Triggered? | Evidence |
|---|-----------|------------|----------|
| 1 | Meaning corruption >= 3/10 | **Yes** | 3/12 cases have meaning_sim < 0.70: correction-day (0.545), correction-team (0.400), ramble-research-guardrails (0.587). However, this is not "corruption" in the intended sense -- the model is not changing meaning, it is failing to change anything. The low similarity scores reflect the gap between verbose echoed output and the concise expected cleanup. |
| 2 | Latency >= 1.5s typical | No | 411ms median for typical (15-30 word) inputs. Only the 150-word ramble exceeds 1.5s (1709ms). |
| 3 | Instruction non-compliance >= 3/10 | No | 0/12. The model never outputs meta-commentary, markdown, or refusals. Format compliance is the one thing it does well. |
| 4 | Quality regression on clean >= 2/10 | No | 0/2. Both clean inputs returned unchanged. The model does not degrade clean text. |

### Raw Output

```
Run 3 of 3 (evaluator run), 2026-03-23:

Loading model: mlx-community/Qwen3.5-2B-8bit
Model loaded in 1.3s, peak memory: 2.00 GB
Warmup complete.

====================================================================================================
Name                           Category            Lat(ms)   tok/s  MeanSim  Pass Notes
----------------------------------------------------------------------------------------------------
filler-simple                  filler                318.3    25.1    0.971  PASS
filler-heavy                   filler                352.8    34.0    0.886  FAIL fillers remain
filler-with-correction         filler                513.4    60.4    1.000  PASS
correction-day                 self-correction       334.6    29.9    0.545  FAIL meaning_sim=0.55; hallucination
correction-team                self-correction       410.8    48.7    0.400  FAIL meaning_sim=0.40; hallucination
ramble-research-guardrails     ramble               1709.4   106.5    0.587  FAIL fillers remain; meaning_sim=0.59; hallucination
ramble-codex-background        ramble                845.6    88.7    0.805  FAIL fillers remain
clean-meeting                  clean-passthrough     354.1    33.9    1.000  PASS
clean-pr                       clean-passthrough     365.3    30.1    1.000  PASS
command-timer                  short-command         336.7    29.7    0.714  FAIL meaning_sim=0.71; hallucination
command-message                short-command         298.5    16.7    0.957  PASS
fixture-afk-mode               real-fixture          806.8    85.5    0.830  PASS
====================================================================================================

SUMMARY
============================================================
Filler removal rate:       2/5 (40%)
Self-correction rate:      0/2 (0%)
Meaning preservation rate: 8/12 (67%)
Format compliance rate:    12/12 (100%)
Passthrough accuracy:      2/2 (100%)

Latency (all):   p50=360ms  p95=1234ms  p99=1614ms
Latency (<=30w): p95=472ms
Token throughput: mean=49.1 tok/s
Peak memory: 2.55 GB

HYPOTHESIS CRITERIA SCORING
============================================================
  [     FAIL] Filler removal >= 9/10: 2/5
  [     FAIL] Self-correction >= 8/10: 0/2
  [     PASS] Grammar & punctuation >= 9/10: 12/12
  [     FAIL] Meaning preservation >= 9/10: 8/12
  [     PASS] Latency < 800ms p95 (<=30w): 472ms
  [     FAIL] Decode speed >= 100 tok/s: 49.1 tok/s
  [     PASS] Memory < 4GB: 2.55 GB
  [TRIGGERED] Meaning corruption >= 3/10: 3/12
  [       OK] Latency >= 1.5s typical: 411ms median
  [       OK] Instruction non-compliance >= 3/10: 0/12
  [       OK] Quality regression on clean >= 2/10: 0/2

Results written to: reviews/run-20260323T052503Z.json
```

Three runs produced byte-identical model outputs. Latency varied by <1%.

## Analysis

### The model defaults to "do nothing"

The defining pattern across all failures is that Qwen3.5-2B interprets the cleanup instruction conservatively: when in doubt, return the input unchanged. This is visible in:

- **correction-day**: Returned "Tell him Tuesday -- actually no, Wednesday works better" verbatim. The model sees the dashes and the self-correction language but does not edit.
- **correction-team**: Returned "I'll send it to the product team, or wait, the engineering team, yeah the engineering team" verbatim. Despite the explicit "or wait" hedge, no editing occurs.
- **ramble-research-guardrails**: A 150-word ramble dense with "like", "you know", "I guess", "I don't know" returned with only one change (a capitalization fix: "For" to "for"). No filler removal, no condensation.
- **command-timer**: "Hey Siri set a timer for 10 minutes" returned unchanged. The model does not recognize "Hey Siri" as a wake word to strip.

The 2B model appears to treat the system prompt as a suggestion rather than a binding instruction when the required edits are non-trivial. It can handle the easiest case (removing isolated "um"/"uh" from simple sentences) but lacks the reasoning to identify discourse structure, self-corrections, or wake words.

### What the model actually does well

- **Simple filler removal**: "Um, I think we should, uh, go with option B" becomes "I think we should go with option B" -- correct.
- **Self-correction within repetition**: "instead of the, um, instead of the Anthropic API" becomes "instead of the Anthropic API" -- the model handles the inline repetition + filler. This is the one case where it edits competently.
- **Clean passthrough**: Perfect. Does not degrade clean input.
- **Format compliance**: Never wraps in quotes, never adds commentary. This is genuinely good.
- **AFK fixture**: Removes the "make make" duplication and some verbal padding, achieving 0.83 similarity to expected. Partial success -- it is making some edits, just not enough.

### The decode speed gap

At 49 tok/s, the model runs at less than half the 100 tok/s target. This is likely inherent to the Gated DeltaNet architecture in Qwen3.5, which uses a different attention mechanism than standard transformers. The only case that hit >100 tok/s was the long ramble (107 tok/s), where the large number of output tokens amortized the prefill cost. For short inputs (5-12 output tokens), throughput drops to 16-34 tok/s because prefill dominates.

This is somewhat academic given the quality failures, but it means that even if quality were acceptable, the tok/s criterion would need to be reconsidered or the model would need to be benchmarked differently (measuring wall-clock latency rather than throughput).

### Measurement caveats

The benchmark's "meaning corruption" trigger (3/12 cases with meaning_sim < 0.70) is technically correct but semantically misleading. The model is not corrupting meaning -- it is failing to clean up the text. The low similarity scores reflect the gap between verbose echoed output and concise expected output. A model that echoes input verbatim preserves meaning perfectly; it just does not do the job.

Similarly, the "hallucination" flag on correction-day, correction-team, and command-timer fires because the echoed output has more words than the expected condensed output. This is the opposite of hallucination -- the model is being too conservative, not too creative.

## Implications

### This model size cannot do the job

The failure is fundamental. The 2B parameter model does not have the capacity to:
1. Identify self-correction patterns ("actually no", "or wait", "I mean")
2. Determine which parts of a ramble are substantive vs. verbal noise
3. Recognize wake words ("Hey Siri") as removable prefixes
4. Remove discourse markers that are technically meaningful words ("So", "Like", "Yeah")

These are not prompt engineering problems. The model produces identical output across three runs with no variation -- there is no stochastic space to exploit with better prompts.

### What to try next

1. **Hybrid approach (most promising)**: Use deterministic rules for the easy stuff (regex filler removal for "um", "uh", "you know"; wake word stripping) and only invoke an LLM for inputs that heuristically appear to contain self-corrections. This dramatically reduces the LLM's job scope and latency cost for common cases.

2. **Larger model test (7B-8B)**: A Qwen3.5-7B or Llama-3.2-8B would likely handle self-corrections and condensation, but the memory/latency tradeoffs need validation. Expected memory: ~6-8GB. Expected latency: likely 1-2s for typical inputs, which may exceed the 800ms budget.

3. **Fine-tuned 2B model**: A 2B model trained specifically on (messy-transcript, clean-text) pairs might succeed where the general-purpose model fails. The consistent echo-back behavior suggests the model's pretraining strongly biases toward copying input, which a targeted fine-tune could override. This requires building a training dataset.

4. **Cloud API fallback**: Claude Haiku or a similar fast cloud model would handle all cases reliably, but trades offline capability for quality. Latency budget of ~500ms network round-trip is feasible but would need graceful degradation for offline use.

### New questions raised

- Would the hybrid approach (regex + LLM-for-corrections-only) achieve acceptable quality while keeping the LLM latency budget small?
- Is 7B feasible within the latency/memory constraints, or does it push past the 800ms wall?
- Could the echo-back behavior be overcome with in-context examples (few-shot prompting) rather than just system instructions?

## Prototype Location

The working prototype is at: `docs/hypotheses/qwen35-stt-cleanup/working/`
To run it:
```bash
cd docs/hypotheses/qwen35-stt-cleanup/working
bash setup.sh  # first time only
.venv/bin/python cleanup_bench.py
```

JSON results from all runs are in `docs/hypotheses/qwen35-stt-cleanup/working/reviews/`.
