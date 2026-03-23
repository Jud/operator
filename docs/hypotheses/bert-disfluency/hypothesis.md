# Hypothesis: A BERT-based token classifier (4i-ai/BERT_disfluency_cls) can detect and strip speech disfluencies from WhisperKit output with >=90% F1 and sub-50ms latency on Apple Silicon

**Status:** Testing
**Date:** 2026-03-23
**Slug:** bert-disfluency
**Prior art:** qwen35-stt-cleanup (disproven -- 2B echoed input, 0% self-correction, 40% filler removal), qwen35-4b-cleanup (disproven -- marginal improvement, 60% filler, 0% self-correction, 25 tok/s, 5GB)

## What We're Testing

Two attempts at using instruction-tuned generalist LLMs (Qwen3.5 2B and 4B) for speech cleanup both failed. The core finding: generalist LLMs at 2-4B scale treat disfluency removal as conservative copy-editing and default to echoing input verbatim rather than making semantic edits. Self-correction handling was 0% across both models. This is a fundamental architectural mismatch -- generative models optimized for instruction following are not the right tool for what is essentially a sequence labeling task.

We are testing whether a fundamentally different approach works: a BERT-based token classifier (`4i-ai/BERT_disfluency_cls`) that was specifically fine-tuned for disfluency detection. Instead of generating cleaned text, this model labels each input token as fluent or disfluent. Disfluent tokens are stripped and the remaining tokens are reassembled into clean text. This is the dominant approach in production speech systems (AssemblyAI, Google, Apple) and achieves 92% F1 on the Switchboard benchmark with augmented training data. The model is ~400MB (BERT-base), versus 2-5GB for the Qwen models.

## Why This Matters

This hypothesis tests a fundamentally different architecture for Operator's dictation cleanup pipeline. The prior failures established that the problem is not about model size or prompt engineering -- it is about using the wrong class of model. Token classification is how the industry solves this problem: a 2024 STIL paper showed that even GPT-4o, Claude 3.5, Gemini 1.5, and LLaMA 3 72B all over-correct or hallucinate on disfluency removal, while BERT taggers achieve 92% F1. Google showed a 1.3 MiB quantized BERT retains near-SOTA performance.

If this works, the integration path is clear: the BERT classifier becomes a lightweight post-processing step between `finishAndTranscribe()` and delivery, running in <50ms with minimal memory. If it fails on WhisperKit-style output (which may differ from Switchboard training data), we know to look at CrisperWhisper (disfluency-aware ASR) or a hybrid pipeline approach instead.

## Success Criteria

- [ ] **Filler detection**: Model correctly tags fillers (um, uh, you know, like-as-filler) as disfluent in >= 4/5 filler+ramble test cases
- [ ] **Self-correction detection**: Model correctly tags the abandoned portion of self-corrections as disfluent in >= 1/2 self-correction cases
- [ ] **Content preservation**: Model does NOT tag content words as disfluent (false positive rate < 5% of content tokens across all test cases)
- [ ] **Clean passthrough**: Already-clean inputs have zero tokens tagged as disfluent in 2/2 clean passthrough cases
- [ ] **Reassembly quality**: After stripping disfluent tokens, the reassembled text preserves the speaker's intended meaning (Levenshtein similarity >= 0.75 to expected output) in >= 8/12 cases
- [ ] **Latency**: Inference completes in < 50ms per sentence on Apple Silicon (CPU, no CoreML)
- [ ] **Memory**: Model loaded memory < 1GB

## Failure Criteria

- [ ] **Massive false positives**: Model tags >= 10% of content words as disfluent across all test cases (would corrupt output)
- [ ] **Filler blindness**: Model fails to detect fillers in >= 3/5 filler+ramble cases (worse than a regex)
- [ ] **Complete self-correction failure**: Model tags 0/2 self-correction cases correctly (same failure as Qwen models)
- [ ] **Unacceptable latency**: Inference takes >= 200ms per sentence on CPU (would negate the latency advantage over generative models)
- [ ] **Tokenizer mismatch**: BERT WordPiece tokenization produces unusable subword splits on WhisperKit output, making reassembly unreliable

## What's Essential vs Incidental

### Essential to the hypothesis
- Must use **`4i-ai/BERT_disfluency_cls`** specifically -- we are testing this particular fine-tuned model's capability on WhisperKit output
- Must use **token classification** (per-token fluent/disfluent labels), not generative text output -- the hypothesis is that the token-classification approach is fundamentally better than generative approaches for this task
- Must use the **same 12 test cases** from the Qwen tests for direct comparison
- Must measure **per-token predictions** to understand exactly what the model tags, not just the final output

### Incidental (can be substituted)
- Python via HuggingFace transformers is fine. No Swift needed for validation.
- CPU inference is acceptable for the PoC. CoreML/ANE conversion is a future optimization step, not part of this hypothesis.
- The exact reassembly strategy (simple concatenation vs. grammar-aware join) is incidental -- the hypothesis is about the classifier's accuracy, not the reassembly polish.

### Language/Framework Flexibility
Not language-specific. The hypothesis is about the BERT model's classification accuracy and latency. Python + HuggingFace transformers is the fastest path. If the model proves out, CoreML conversion for Swift integration would be a separate follow-up task.

## Constraints

- **Does NOT need to integrate with the app.** Standalone Python prototype that proves classification quality and latency.
- **Scope:** Load the model, run token classification on 12 test cases, inspect per-token labels, strip disfluent tokens, measure quality and latency. Skip: CoreML conversion, Swift integration, prompt engineering (there is no prompt -- it is a classifier), streaming inference.

## Prototype Approach

Build a Python benchmark (`docs/hypotheses/bert-disfluency/working/classify_bench.py`) that:

1. **Loads `4i-ai/BERT_disfluency_cls`** via HuggingFace transformers (`AutoModelForTokenClassification`, `AutoTokenizer`)
2. **For each of the 12 test cases**, runs token classification and collects per-token fluent/disfluent labels
3. **Inspects the raw labels** -- prints each token with its predicted label so we can see exactly what the model thinks is disfluent
4. **Strips disfluent tokens** and reassembles the remaining tokens into cleaned text
5. **Evaluates quality** using the same metrics as the Qwen benchmarks: Levenshtein similarity to expected output, filler presence check, content preservation check
6. **Measures latency** per-sentence (wall clock, CPU inference) and peak memory
7. **Reports results** in a summary table with pass/fail per criterion, plus a detailed per-token label dump for debugging

The benchmark should also output a **per-token label grid** for each test case so we can visually inspect what the model is tagging. This is critical for understanding failure modes -- is the model missing fillers? Tagging content words? Getting confused by punctuation?

**What "running" the prototype looks like:**
```bash
cd docs/hypotheses/bert-disfluency/working
pip install transformers torch
python classify_bench.py
```
Outputs a results table, per-token label grids, and aggregate stats. Results written to `reviews/` as timestamped JSON.

**How we evaluate:**
- Per-token labels are manually inspected for correctness on each test case
- Quality metrics are compared directly against the Qwen 2B and 4B results on the same 12 cases
- Latency is compared against the 50ms target and the Qwen models' latency (472ms p95 for 2B, higher for 4B)
- The key question is not "does it produce perfect output" but "does it correctly identify disfluencies" -- reassembly polish is a separate problem
