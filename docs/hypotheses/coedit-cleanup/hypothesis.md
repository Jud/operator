# Hypothesis: CoEdit (grammarly/coedit-large), an instruction-tuned T5 text editing model, can clean raw STT output with higher quality than Qwen generative LLMs and BERT classifiers because its encoder-decoder architecture is trained to TRANSFORM text rather than echo or label it

**Status:** Testing
**Date:** 2026-03-23
**Slug:** coedit-cleanup
**Prior art:** qwen35-stt-cleanup (disproven -- 2B echoed input, 0% self-correction, 40% filler removal), qwen35-4b-cleanup (disproven -- marginal improvement, 60% filler, 0% self-correction, 25 tok/s, 5GB), bert-disfluency (disproven -- 13.2% FP rate, destroyed content words, wrong training domain)

## What We're Testing

Three previous approaches to cleaning raw WhisperKit transcriptions for dictation all failed:
- **Generalist LLMs (Qwen 2B/4B)**: Default to echoing input verbatim. 0% self-correction across both models. The generative paradigm treats cleanup as conservative copy-editing.
- **BERT token classifiers**: Wrong training domain (stuttering corpus, not conversational speech). 13.2% false positive rate -- labeled content words like "engineering" as repetitions and deleted them.

We are testing whether CoEdit (`grammarly/coedit-large`, 770M params, FLAN-T5-large backbone) represents a fundamentally better architecture for this task. CoEdit is purpose-built for text editing: it was trained on 82K instruction-tuned editing examples spanning grammar correction, fluency improvement, simplification, paraphrase, and style transfer. Unlike Qwen, it is an encoder-decoder model trained to TRANSFORM input text, not a chat model that defaults to echoing. Unlike BERT classifiers, it generates new text rather than labeling tokens, so it can handle self-corrections that require understanding intent, not just tagging individual words.

The critical unknown: CoEdit was trained on WRITTEN text editing tasks, not speech disfluency cleanup. Its "fluency" training data is about making written prose flow better, not about removing "um" and resolving "actually no, I meant X" self-corrections. Will instruction prompts like "Make this text fluent" or "Fix the grammar and remove filler words" generalize to speech cleanup? Or will the model only fix written-style errors and leave speech artifacts intact?

## Why This Matters

Operator needs a post-processing stage between `finishAndTranscribe()` and `DictationDelivery.deliver()` that turns raw speech into polished written text. Four models have now failed. The pattern across failures suggests the problem is not model size or prompt engineering -- it is architectural mismatch. Generative LLMs are too conservative. Token classifiers lack semantic understanding.

CoEdit occupies a unique middle ground: it generates output text (so it can restructure sentences and resolve self-corrections) but it was trained specifically to EDIT input text (so it should not default to echoing). If CoEdit succeeds on filler removal and self-correction where all four previous models failed, it validates instruction-tuned text editing as the right paradigm for speech cleanup. If it fails, the written-text training domain is a fundamental limitation and we need either a speech-specific solution (CrisperWhisper disfluency-aware ASR) or a cloud API.

## Success Criteria

How do we know the hypothesis is proven or disproven? Be concrete:

- [ ] **Filler removal >= 4/5 filler+ramble cases** -- fillers (um, uh, like, you know) are removed from output. Must beat Qwen 4B's 3/5.
- [ ] **Self-correction >= 1/2 cases** -- when the speaker corrects themselves, the output contains only the corrected version. Every previous model scored 0/2. Even 1/2 would be a breakthrough.
- [ ] **Meaning preservation >= 8/12 cases** -- Levenshtein similarity >= 0.75 to expected output. Must match or beat Qwen 4B's 9/12.
- [ ] **Clean passthrough 2/2** -- already-clean inputs returned unchanged (or minimally altered). No over-editing.
- [ ] **No content hallucination** -- model does not add information that was not in the input in any of the 12 cases.
- [ ] **Latency < 1500ms p95** for inputs up to 30 words. T5-large encoder-decoder should be faster than autoregressive Qwen for short text.

## Failure Criteria

What would disprove the hypothesis?

- [ ] **Self-correction still 0/2** -- if CoEdit also cannot resolve self-corrections, the task fundamentally requires understanding conversational speech intent that no off-the-shelf written-text model has. This would confirm we need a speech-specific solution.
- [ ] **Filler removal <= 2/5** -- if CoEdit cannot remove fillers better than a regex, the "fluency" instruction does not generalize from written to spoken text.
- [ ] **Meaning corruption >= 3/12** -- model rewrites/paraphrases too aggressively, changing what the speaker said. This would be a different failure mode than Qwen (too conservative) or BERT (wrong labels) -- over-editing.
- [ ] **Output format problems** -- model adds meta-commentary, explanations, or instruction-format artifacts (e.g., repeating the instruction prefix) in >= 3/12 cases.
- [ ] **Latency >= 3s** -- model is too slow for interactive dictation.

## What's Essential vs Incidental

### Essential to the hypothesis
- Must use **`grammarly/coedit-large`** specifically (770M params, FLAN-T5-large). This is the core model being evaluated. If it fails, coedit-xl (3B) and coedit-xxl (11B) could be tested as follow-ups, but they are NOT part of this hypothesis.
- Must use **instruction-driven generation** -- CoEdit's interface is "Task prefix: input text" -> generated output. The choice of instruction prefix is critical to test.
- Must use the **same 12 test cases** from all prior hypotheses for direct comparison.
- Must test **multiple instruction prompts** to find what works best and understand how sensitive the model is to prompt wording. At minimum:
  - "Make this text fluent: {input}"
  - "Fix the grammar in this text: {input}"
  - "Remove disfluencies from this text: {input}" (out-of-distribution instruction)
  - "Fix the grammar and remove filler words: {input}" (compound instruction)

### Incidental (can be substituted)
- Python via HuggingFace transformers is fine. No Swift needed for validation.
- CPU or MPS inference -- either is acceptable for measuring quality. Latency is secondary to quality for this test.
- The exact generation parameters (max_length, num_beams, etc.) can be tuned. CoEdit defaults to beam search with num_beams=5.

### Language/Framework Flexibility
Not language-specific. The hypothesis is about the CoEdit model's capability on speech disfluency cleanup. Python + HuggingFace transformers is the fastest path since both the model and the benchmark infrastructure already exist. If the model proves out, CoreML conversion for Swift integration would be a separate follow-up.

## Constraints

- **Does NOT need to integrate with the app.** Standalone Python prototype that validates quality and latency.
- **Scope:** Load coedit-large, run 4 instruction prompts x 12 test cases, measure quality and latency, compare against all prior results. Skip: coedit-xl/xxl, CoreML conversion, Swift integration, fine-tuning, streaming generation.
- **Uses existing `.venv-poc/`** environment with transformers and torch already installed. CoEdit uses the standard T5 model class, no additional dependencies needed.

## Prototype Approach

Build a Python benchmark (`docs/hypotheses/coedit-cleanup/working/cleanup_bench.py`) that:

1. **Loads `grammarly/coedit-large`** via HuggingFace transformers (`AutoTokenizer`, `T5ForConditionalGeneration`). CoEdit uses the T5 tokenizer and encoder-decoder architecture -- NOT a chat template.
2. **Defines 4 instruction prompts** to test how instruction wording affects quality:
   - `"Make this text fluent: "` (CoEdit's native fluency task)
   - `"Fix the grammar in this text: "` (CoEdit's native GEC task)
   - `"Remove disfluencies from this text: "` (out-of-distribution -- tests generalization)
   - `"Fix the grammar and remove filler words: "` (compound instruction)
3. **For each prompt x test case combination**, generates cleaned text using beam search (num_beams=5, CoEdit default). Records the output, latency, and input/output token counts.
4. **Evaluates quality** per output using the same metrics as prior tests: Levenshtein similarity to expected output, filler presence check, passthrough check, content preservation check.
5. **Finds the best prompt** by aggregating quality metrics per prompt across all 12 cases.
6. **Reports results** in a summary table with per-prompt breakdowns and a comparison against Qwen 2B, Qwen 4B, and BERT baselines.
7. **Writes JSON results** to `reviews/` directory for archival.

**What "running" the prototype looks like:**
```bash
cd docs/hypotheses/coedit-cleanup/working
python cleanup_bench.py  # uses .venv-poc from project root
```
Outputs per-prompt results tables, best-prompt summary, cross-model comparison, and writes timestamped JSON to `reviews/`.

**How we evaluate:**
- Quality is the primary question. If CoEdit achieves >= 1/2 on self-correction (beating all 4 prior models at 0%), that alone would be a significant finding regardless of other metrics.
- Best-prompt analysis shows how sensitive the model is to instruction wording and whether "fluency" or "grammar" framing works better for speech cleanup.
- Direct comparison table against Qwen 2B, 4B, and BERT on the same 12 cases shows whether the architectural difference matters.
- Latency is measured wall-clock. T5-large with beam search should complete in 200-1500ms for short inputs on Apple Silicon.
