# Hypothesis: Qwen3.5-4B can clean raw STT output with acceptable quality for on-device dictation, trading higher latency (~1-1.5s) for dramatically better instruction following

**Status:** Testing
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

- [ ] **Quality: filler removal** -- Fillers (um, uh, like, you know, discourse "so"/"like"/"yeah") are removed from output in >= 4/5 filler+ramble test cases
- [ ] **Quality: self-correction handling** -- When the speaker corrects themselves ("actually no, I meant X"), the output contains only the corrected version in >= 2/2 cases
- [ ] **Quality: meaning preservation** -- Meaning similarity >= 0.80 against expected output in >= 10/12 test cases
- [ ] **Quality: format compliance** -- Output is raw text (no markdown, no meta-commentary, no quotes) in >= 11/12 test cases
- [ ] **Quality: clean passthrough** -- Already-clean inputs returned unchanged in 2/2 cases
- [ ] **Latency: acceptable for dictation** -- End-to-end time < 1500ms for inputs <= 30 words (p95)
- [ ] **Throughput: decode speed** -- >= 30 tok/s mean (relaxed from 100 for 2B)
- [ ] **Memory: on-device feasible** -- Peak memory < 6GB

## Failure Criteria

- [ ] **Echo-back persistence** -- Model echoes input verbatim (meaning_sim to raw input > 0.95) in >= 4/12 cases, indicating the same fundamental failure as the 2B
- [ ] **Meaning corruption** -- The model changes the meaning of what was said (adding content, inverting intent, dropping critical information) in >= 3/12 cases
- [ ] **Unacceptable latency** -- Cleanup takes >= 3s for typical utterances (15-30 words)
- [ ] **Instruction non-compliance** -- Model outputs meta-commentary, asks questions, refuses, or wraps output in markdown in >= 3/12 cases
- [ ] **Quality regression on clean input** -- When given already-clean transcriptions, the model makes them worse in >= 1/2 cases

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
