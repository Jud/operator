# Hypothesis: Qwen3.5-2B via MLX can clean raw STT output into polished dictation text with sub-800ms latency on Apple Silicon

**Status:** Testing
**Date:** 2026-03-22
**Slug:** qwen35-stt-cleanup

## What We're Testing

Raw WhisperKit transcriptions faithfully capture how people actually talk: fillers ("um", "uh", "like"), self-corrections ("actually no, let me rephrase that"), repetitions, missing punctuation, and stream-of-consciousness grammar. This is fine when the output goes to an LLM (Claude interprets intent), but it is unacceptable for dictation mode, where the text is pasted directly into a text message, email, or document.

We are testing whether Qwen3.5-2B, running locally via mlx-lm (Python), can transform raw speech transcriptions into clean, polished written text -- removing fillers, resolving self-corrections, fixing grammar, and adding proper punctuation -- with quality and latency acceptable for real-time dictation on Apple Silicon.

## Why This Matters

Operator's bimodal design routes voice input to either agent sessions (terminal) or dictation (cursor insertion via pasteboard). Dictation is the default path for all non-terminal apps. Today, raw WhisperKit output goes directly into the text field with no cleanup. The difference between "um you know like tell him that actually no wait let me say it this way I'll be there at like around 8" and "I'll be there around 8" is the difference between a usable dictation system and a party trick.

If Qwen3.5-2B can do this well and fast enough, it becomes a post-processing stage in the transcription pipeline, sitting between `finishAndTranscribe()` and `DictationDelivery.deliver()`. If not, we need to explore alternatives (larger models, cloud APIs, or a different architecture).

## Success Criteria

How do we know the hypothesis is proven or disproven? Be concrete:

- [ ] **Quality: filler removal** -- Fillers (um, uh, like, you know) are removed from output in >= 9/10 test cases
- [ ] **Quality: self-correction handling** -- When the speaker corrects themselves ("actually no, I meant X"), the output contains only the corrected version in >= 8/10 cases
- [ ] **Quality: grammar and punctuation** -- Output has correct capitalization, punctuation, and grammar for >= 9/10 test cases
- [ ] **Quality: meaning preservation** -- The semantic content of the original speech is preserved (no hallucinated additions, no dropped meaning) in >= 9/10 cases
- [ ] **Latency: total post-release** -- End-to-end time from raw transcription text to cleaned text is under 800ms for utterances up to 30 words (p95)
- [ ] **Latency: token throughput** -- Decode speed is >= 100 tok/s on the test machine (confirms published benchmarks)
- [ ] **Memory: acceptable footprint** -- Model loaded memory stays under 4GB (8-bit quantized)

## Failure Criteria

What would disprove the hypothesis?

- [ ] **Meaning corruption** -- The model changes the meaning of what was said in >= 3/10 cases (adding content that wasn't spoken, inverting intent, dropping critical information)
- [ ] **Unacceptable latency** -- Cleanup takes >= 1.5s for typical utterances (15-30 words), making dictation feel laggy
- [ ] **Instruction non-compliance** -- The model fails to follow the cleanup prompt reliably (outputs meta-commentary, asks questions, refuses, or wraps output in markdown) in >= 3/10 cases
- [ ] **Quality regression on clean input** -- When given already-clean transcriptions (no fillers, good grammar), the model makes them worse in >= 2/10 cases

## Constraints

- **Python prototype using mlx-lm.** MLX Swift does not support the Gated DeltaNet architecture in Qwen3.5 yet. The prototype uses Python mlx-lm, which already works. This is explicitly NOT a Swift integration -- it validates quality and latency before any integration investment.
- **Does NOT need to integrate with the app.** The prototype is a standalone Python CLI that takes raw transcription text and outputs cleaned text. No HTTP server, no pipeline integration, no Swift bridge.
- **Scope:** Minimum viable is a Python script that loads Qwen3.5-2B 8-bit, runs a cleanup prompt against a set of test inputs, and measures quality + latency. We skip: streaming token output, prompt optimization, model fine-tuning, Swift/Python bridge design.

## Prototype Approach

Build a Python CLI prototype (`docs/hypotheses/qwen35-stt-cleanup/working/cleanup_bench.py`) that:

1. **Loads Qwen3.5-2B 8-bit** via `mlx_lm.load()` from Hugging Face (or local cache)
2. **Defines a cleanup system prompt** instructing the model to remove fillers, resolve self-corrections, fix grammar/punctuation, and preserve meaning. Output raw text only, no formatting.
3. **Runs against a test suite** of raw transcription inputs with expected cleaned outputs. Test cases should include:
   - Simple filler removal ("Um, I think we should, uh, go with option B")
   - Self-correction ("Tell him Tuesday -- actually no, Wednesday works better")
   - Stream-of-consciousness ramble (long input with multiple fillers, repetitions, corrections)
   - Already-clean input (should pass through unchanged)
   - Short commands ("Hey Siri set a timer" -- should not over-edit)
   - Real WhisperKit output from existing test fixtures (the `passing-very-long` and `passing-medium-long` fixtures contain natural messy speech)
4. **Measures per-input:**
   - Wall-clock latency (ms)
   - Token count (input + output)
   - Decode speed (tok/s)
   - Output quality (manual review against expected output)
5. **Reports results** in a summary table: input, expected, actual, latency, pass/fail

**What "running" the prototype looks like:**
```bash
cd docs/hypotheses/qwen35-stt-cleanup/working
python cleanup_bench.py
```
Outputs a table of test cases with pass/fail per criterion, plus aggregate latency stats.

**How we evaluate:**
- Quality is judged against hand-written expected outputs for each test case
- Latency is measured wall-clock, not just decode time
- Results are written to `docs/hypotheses/qwen35-stt-cleanup/working/reviews/` as timestamped run logs
