# Hypothesis: Qwen3.5-4B can clean raw STT output with acceptable quality for on-device dictation, trading higher latency (~1-1.5s) for dramatically better instruction following

**Status:** Disproven
**Date:** 2026-03-23
**Slug:** qwen35-4b-cleanup
**Prior art:** qwen35-stt-cleanup (disproven -- 2B model echoed input verbatim, 0% self-correction, 40% filler removal)

## What We're Testing

The previous test (qwen35-stt-cleanup) proved that Qwen3.5-2B lacks the reasoning capacity for speech cleanup. The model echoed input verbatim in 6/12 cases, achieved 0% self-correction resolution, and only removed fillers 40% of the time. The failure was fundamental: the 2B model treated cleanup instructions as suggestions and defaulted to "do nothing" for any non-trivial edit.

We are testing whether Qwen3.5-4B, with its dramatically better instruction following (IFEval 89.8% vs 61.2%/78.6% for 2B), can actually perform the cleanup that the 2B model could not: removing fillers, resolving self-corrections, condensing rambles, and stripping wake words. The 4B model is ~2x larger, so we expect higher latency and memory, but the core question is whether it crosses the quality threshold that the 2B model missed entirely.

## Why This Matters

This is a direct retest of the most promising "next step" from the 2B findings. The 2B test established that:
- The benchmark infrastructure works and produces deterministic, reproducible results
- The quality failures were not prompt engineering problems (identical output across 3 runs)
- The model size was the bottleneck, not the approach

If the 4B model succeeds on quality, it validates the local-LLM cleanup architecture for Operator's dictation pipeline. The latency and memory costs need to be acceptable but can be higher than the 2B targets. If it fails on quality despite 89.8% IFEval, it suggests this task requires either a 7B+ model or a fundamentally different approach (hybrid rules + LLM, fine-tuning, or cloud API).

## Success Criteria

- [ ] **Quality: filler removal** -- Fillers (um, uh, like, you know, discourse "so"/"like"/"yeah") are removed from output in >= 4/5 filler+ramble test cases -- FAIL: 3/5 (60%)
- [ ] **Quality: self-correction handling** -- When the speaker corrects themselves ("actually no, I meant X"), the output contains only the corrected version in >= 2/2 cases -- FAIL: 0/2 (0%)
- [ ] **Quality: meaning preservation** -- Meaning similarity >= 0.80 against expected output in >= 10/12 test cases -- FAIL: 8/12 (67%)
- [x] **Quality: format compliance** -- Output is raw text (no markdown, no meta-commentary, no quotes) in >= 11/12 test cases -- PASS: 12/12 (100%)
- [x] **Quality: clean passthrough** -- Already-clean inputs returned unchanged in 2/2 cases -- PASS: 2/2 (100%)
- [x] **Latency: acceptable for dictation** -- End-to-end time < 1500ms for inputs <= 30 words (p95) -- PASS: 899ms
- [ ] **Throughput: decode speed** -- >= 30 tok/s mean (relaxed from 100 for 2B) -- FAIL: 25.3 tok/s
- [x] **Memory: on-device feasible** -- Peak memory < 6GB -- PASS: 5.08 GB

## Failure Criteria

- [x] **Echo-back persistence** -- Model echoes input verbatim (meaning_sim to raw input > 0.95) in >= 4/12 cases, indicating the same fundamental failure as the 2B -- TRIGGERED: 4/12
- [x] **Meaning corruption** -- The model changes the meaning of what was said (adding content, inverting intent, dropping critical information) in >= 3/12 cases -- TRIGGERED: 3/12
- [ ] **Unacceptable latency** -- Cleanup takes >= 3s for typical utterances (15-30 words) -- OK: 791ms median
- [ ] **Instruction non-compliance** -- Model outputs meta-commentary, asks questions, refuses, or wraps output in markdown in >= 3/12 cases -- OK: 0/12
- [ ] **Quality regression on clean input** -- When given already-clean transcriptions, the model makes them worse in >= 1/2 cases -- OK: 0/2

## What's Essential vs Incidental

### Essential to the hypothesis
- Must use **Qwen3.5-4B** specifically -- we are testing whether doubling the parameter count from the failed 2B test crosses the quality threshold
- Must use the **same 12 test cases** from the 2B test for direct comparison
- Must use the **same system prompt** for fair comparison
- Must use **8-bit quantization** (the practical deployment format for on-device)

### Incidental (can be substituted)
- Python via mlx-lm is fine (same as 2B test). Swift binding not needed for this validation.
- The exact mlx-community model slug may vary -- use whatever 8-bit quantized Qwen3.5-4B is available

### Language/Framework Flexibility
Not language-specific. The hypothesis is about the model's capability, not the language binding. Python + mlx-lm is the fastest path since the infrastructure already exists from the 2B test.

## Constraints

- **Reuses existing benchmark infrastructure.** The test cases, evaluation logic, and reporting from qwen35-stt-cleanup are directly reusable. The only change is the model ID.
- **Does NOT need to integrate with the app.** Standalone Python benchmark, same as the 2B test.
- **Scope:** Swap model ID in cleanup_bench.py, adjust success/failure thresholds to match the relaxed latency/throughput/memory targets, run the benchmark, report results.

## Prototype Approach

Copy the existing benchmark from `qwen35-stt-cleanup/working/` with these changes:

1. **Model ID:** `mlx-community/Qwen3.5-4B-MLX-8bit` (verified on HuggingFace; note the `-MLX-` infix differs from the 2B's `Qwen3.5-2B-8bit` naming). Qwen3.5-4B is a unified VLM with early fusion -- the MLX variant may require `mlx_vlm` rather than `mlx_lm`. The benchmark tries `mlx_lm` first and falls back to `mlx_vlm`.
2. **Threshold adjustments in scoring:**
   - Latency target: < 1500ms p95 for <= 30 word inputs (was 800ms)
   - Decode speed target: >= 30 tok/s (was 100)
   - Memory target: < 6GB (was 4GB)
3. **Add echo-back detection:** New failure criterion that checks if the model is echoing input verbatim (meaning_sim between output and raw input > 0.95), since this was the dominant failure mode of the 2B
4. **Same system prompt, same test cases, same quality evaluation logic**

**What "running" the prototype looks like:**
```bash
cd docs/hypotheses/qwen35-4b-cleanup/working
bash setup.sh
.venv/bin/python cleanup_bench.py
```

**How we evaluate:**
- Direct comparison against 2B results on the same 12 test cases
- Focus on the categories the 2B failed: self-correction (was 0/2), filler removal (was 2/5), ramble condensation
- Quality is the primary question; latency/memory are secondary (must be "acceptable" not "fast")

## Results

**Verdict: Disproven.** Qwen3.5-4B does not cross the quality threshold for on-device STT cleanup. The 4B model shows only marginal improvement over the 2B on the quality dimensions that matter most, while doubling memory and halving throughput. 4/8 success criteria pass, 2/5 failure criteria triggered.

### 2B vs 4B Comparison

| Metric | 2B Result | 4B Result | Change |
|---|---|---|---|
| Filler removal | 2/5 (40%) | 3/5 (60%) | +1 case |
| Self-correction | 0/2 (0%) | 0/2 (0%) | No change |
| Meaning preservation | 8/12 (67%) | 8/12 (67%) | No change |
| Format compliance | 12/12 (100%) | 12/12 (100%) | No change |
| Passthrough accuracy | 2/2 (100%) | 2/2 (100%) | No change |
| Echo-back count | ~6/12 | 4/12 | -2 cases |
| Latency p95 (<=30w) | 472ms | 899ms | +90% slower |
| Mean tok/s | 49.1 | 25.3 | -49% slower |
| Peak memory | 2.55 GB | 5.08 GB | +99% more |

### Key Findings

**1. Self-correction remains completely broken (0/2).** This was the most important failure mode to fix. Both self-correction cases failed identically to the 2B: the model preserved the original incorrect statement alongside the correction instead of resolving to the final intent. "Tell him Tuesday -- actually no, Wednesday works better" became "Tell him Tuesday; actually, no, Wednesday works better" instead of "Tell him Wednesday works better." The 4B model treats self-corrections as punctuation cleanup, not semantic edits.

**2. Filler removal improved modestly (40% to 60%) but still fails on longer inputs.** The 3 simple filler cases (filler-simple, filler-heavy, filler-with-correction) all pass perfectly. But both ramble cases fail: the model leaves fillers like "you know", "like", "I guess" in the output for longer, more complex inputs. The 4B model can handle filler removal when that is the only task, but fails when fillers are embedded in structurally complex speech that also needs condensation.

**3. Echo-back improved but still triggers the failure criterion (4/12).** Down from ~6/12 on the 2B. However, 2 of the 4 echo-backs are correct behavior (clean-meeting, clean-pr are clean passthrough cases where echoing IS the right answer). The problematic echo-backs are correction-team (should have resolved the correction) and command-message (trivially close to expected, just missing a period). The real echo-back failure rate for cases that need editing is 1/10, a significant improvement over the 2B. But the failure criterion as written counts clean passthrough echo-backs, which inflates the number.

**4. The 4B model is a better punctuation/formatting tool but not a semantic editor.** It excels at format compliance (12/12), clean passthrough (2/2), and simple filler removal. It fails at tasks requiring semantic reasoning: self-correction resolution, ramble condensation, and wake-word stripping ("Hey Siri" retained in command-timer). The IFEval improvement (89.8% vs 61.2%) translates to better instruction following for surface-level tasks but does not unlock the deeper language understanding needed for speech cleanup.

**5. Performance costs are significant.** 2x memory (5.08 vs 2.55 GB), 2x slower throughput (25.3 vs 49.1 tok/s), 2x higher latency (899 vs 472ms p95). These are within the relaxed targets but represent a poor trade for the minimal quality improvement.

### Per-Case Analysis

| Case | Category | 2B | 4B | Notes |
|---|---|---|---|---|
| filler-simple | filler | PASS | PASS | Both handle simple fillers |
| filler-heavy | filler | FAIL | PASS | 4B gained this one |
| filler-with-correction | filler | PASS | PASS | Both handle filler+repetition |
| correction-day | self-correction | FAIL | FAIL | Both keep "Tuesday" alongside "Wednesday" |
| correction-team | self-correction | FAIL | FAIL | Both echo the full correction sequence |
| ramble-research-guardrails | ramble | FAIL | FAIL | Both leave fillers in long rambles |
| ramble-codex-background | ramble | FAIL | FAIL | 4B keeps "Yeah", "Like", doesn't condense |
| clean-meeting | passthrough | PASS | PASS | Correct passthrough |
| clean-pr | passthrough | PASS | PASS | Correct passthrough |
| command-timer | command | FAIL | FAIL | Both keep "Hey Siri" |
| command-message | command | PASS | PASS | Near-identical output |
| fixture-afk-mode | real-fixture | PASS | PASS | Both handle real fixture well |

### Next Steps

The 4B model does not validate the local-LLM cleanup architecture. The quality failures are in the same categories as the 2B, with the same patterns. This suggests the problem is not simply model capacity -- it is that instruction-tuned generalist LLMs at this scale treat speech cleanup as a conservative copy-editing task rather than a semantic transformation task.

Options to explore:
1. **7B+ model** -- Qwen3.5-7B or similar. But the pattern (self-correction 0/2 on both 2B and 4B) suggests this may be a fundamental limitation of the general instruction-tuning approach, not just a scale issue. Diminishing returns likely.
2. **Hybrid rules + LLM** -- Use regex/rules for deterministic patterns (filler removal, wake-word stripping) and only route complex cases (self-correction, ramble condensation) to the LLM. This could fix 3-4 of the failing cases immediately without a bigger model.
3. **Fine-tuning** -- Fine-tune a small model specifically on speech-cleanup pairs. The 12 test cases plus more examples could create a specialized model that outperforms the 4B generalist.
4. **Cloud API fallback** -- Use Claude or similar for complex cleanup. Higher latency but reliable quality. Could be gated behind a complexity detector.
