# Hypothesis Test Tasks: ModernBERT Disfluency Token Classification

**Status:** Planning
**Worktree:** .claude/worktrees/hypothesis-bert-disfluency
**Branch:** hypothesis/bert-disfluency
**Total Tasks:** 3
**Completed:** 0

## Context

Previous plan was marked infeasible because `4i-ai/BERT_disfluency_cls` is a sentence-level classifier. We are pivoting to two real token classifiers:

- **arielcerdap/modernbert-base-multiclass-disfluency-v2** (149M params, ModernBERT-base, 0.754 F1)
- **arielcerdap/modernbert-disfluency-expC-large-realonly** (400M params, ModernBERT-large, 0.833 macro-F1)

Both use `ModernBertForTokenClassification` with 5-class labels: O (fluent), FP (filled pause), RP (repetition), RV (revision), PW (partial word). Both confirmed on HuggingFace.

Existing `test_cases.py` (12 cases) is reusable as-is. The benchmark script and setup need updates.

## Task Graph

```
T1 (setup.sh) ──┐
                 ├── T3 (run + evaluate)
T2 (classify_bench.py) ─┘
```

Batch 1: T1, T2 (parallel -- no dependencies)
Batch 2: T3 (depends on T1 + T2)

## Tasks

### T1. Update setup.sh for ModernBERT compatibility
**Status:** pending
**Depends on:** --
**Files:** `docs/hypotheses/bert-disfluency/working/setup.sh`
**Review:** --
**Spec:**

ModernBERT requires `transformers>=4.48.0` (that's when `ModernBertForTokenClassification` was added). Update setup.sh to pin minimum version.

Edit the pip install line to:
```
.venv/bin/pip install --quiet "transformers>=4.48.0" torch
```

No other changes needed. The venv creation logic is fine.

**Acceptance:** `setup.sh` installs transformers with ModernBERT support. Running `bash setup.sh` completes without error.

---

### T2. Rewrite classify_bench.py for dual-model ModernBERT comparison
**Status:** pending
**Depends on:** --
**Files:** `docs/hypotheses/bert-disfluency/working/classify_bench.py`
**Review:** --
**Spec:**

The existing script is well-structured but needs three changes. Keep everything that works (Levenshtein, filler check, test harness, reporting, JSON output, criteria scoring). Change only what's broken.

**Change 1: Model IDs and dual-model loop.**

Replace the single `MODEL_ID` constant with a list:
```python
MODELS = [
    ("base", "arielcerdap/modernbert-base-multiclass-disfluency-v2"),
    ("large", "arielcerdap/modernbert-disfluency-expC-large-realonly"),
]
```

Wrap the main logic in a loop over both models. Each model gets its own load, warmup, and test run. Print results for each model separately, then print a side-by-side comparison table at the end.

**Change 2: Replace heuristic label detection with explicit 5-class scheme.**

The current `reassemble_text()` uses a heuristic to guess which labels mean "disfluent" (checking for substrings like "disf", "remove", etc.). This is fragile and wrong for the 5-class scheme.

Replace with explicit knowledge: `O` is the only fluent label. `FP`, `RP`, `RV`, `PW` are all disfluent (should be stripped).

```python
DISFLUENT_LABELS = {"FP", "RP", "RV", "PW"}
FLUENT_LABELS = {"O"}
```

Remove the heuristic label-detection logic from `reassemble_text()`. The function should accept the label sets as parameters or use the module constants directly.

Update the per-token grid display to show the actual label (O/FP/RP/RV/PW) instead of just F/D. This is more informative for debugging -- we want to see what *type* of disfluency the model predicts.

**Change 3: Fix tokenizer handling for ModernBERT.**

ModernBERT does NOT use BERT's WordPiece tokenizer. It uses a BPE tokenizer (GPT-NeoX style). This means:
- No `##` continuation prefixes. Subword tokens use different conventions.
- Special tokens are `[CLS]`/`[SEP]` (same IDs but different vocab IDs).

The safest reassembly strategy for BPE: use **offset mapping** instead of token string inspection. The existing `classify_tokens()` already extracts `offset_mapping` from the tokenizer. Use these character offsets to reconstruct text from the original input string:

```python
def reassemble_text(tokens, labels, offsets, original_text):
    """Strip disfluent tokens using character offsets from original text."""
    kept_spans = []
    for label, (start, end) in zip(labels, offsets):
        if label in DISFLUENT_LABELS:
            continue
        if start == end:  # skip empty spans (special tokens)
            continue
        kept_spans.append((start, end))

    # Merge adjacent/overlapping spans and extract from original
    if not kept_spans:
        return ""
    pieces = []
    for start, end in kept_spans:
        pieces.append(original_text[start:end])
    result = " ".join(pieces)

    # Cleanup: collapse whitespace, fix orphaned punctuation
    result = re.sub(r"\s+", " ", result)
    result = re.sub(r"\s+([,.\-!?;:])", r"\1", result)
    return result.strip()
```

This approach is tokenizer-agnostic -- works for WordPiece, BPE, or any tokenizer that provides offset mappings.

Update the call sites to pass `offsets` through.

**Also update:**
- The `compute_false_positive_rate()` function to use `DISFLUENT_LABELS` instead of the passed-in set.
- The JSON output to include results for both models, keyed by model name.
- The comparison table at the end: add columns for both ModernBERT models alongside the Qwen results.
- The summary `MODEL_ID` references.

**Do NOT change:**
- `test_cases.py` -- reuse as-is
- The Levenshtein similarity function
- The filler check logic
- The hypothesis criteria scoring logic (thresholds are fine)
- The overall structure of the reporting

**Acceptance:** Script loads both models, runs all 12 test cases on each, prints per-token label grids with 5-class labels, prints per-model results tables, prints a comparison table (base vs large vs Qwen 2B vs Qwen 4B), writes JSON to reviews/, and exits cleanly. No references to `4i-ai/BERT_disfluency_cls` remain.

---

### T3. Run benchmark and capture results
**Status:** pending
**Depends on:** T1, T2
**Files:** `docs/hypotheses/bert-disfluency/working/reviews/*.json`
**Review:** --
**Spec:**

Execute the benchmark end-to-end:

```bash
cd docs/hypotheses/bert-disfluency/working
bash setup.sh
.venv/bin/python classify_bench.py
```

Verify:
1. Both models load without error
2. All 12 test cases run on each model
3. Per-token label grids show O/FP/RP/RV/PW labels (not generic F/D)
4. JSON results are written to `reviews/`
5. Hypothesis criteria scoring runs and produces pass/fail verdicts
6. No Python tracebacks or model loading errors

If either model fails to load (e.g., transformers version too old, network error), debug and fix. If inference produces all-O labels (model not working), investigate and report.

Capture the full terminal output. The JSON file in `reviews/` is the deliverable.

**Acceptance:** A timestamped JSON file exists in `reviews/` containing results for both models across all 12 test cases, with hypothesis criteria scoring.
