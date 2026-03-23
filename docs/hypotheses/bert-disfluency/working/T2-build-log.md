### T2: Rewrite classify_bench.py for dual-model ModernBERT comparison
**Date:** 2026-03-23
**Status:** complete
**Files changed:**
- `docs/hypotheses/bert-disfluency/working/classify_bench.py` -- complete rewrite for dual-model ModernBERT token classification

**Notes:**

All three changes from the spec implemented:

1. **Change 1 (dual-model loop):** Replaced single `MODEL_ID` with `MODELS` list containing both base (149M) and large (400M) model IDs. Main logic wrapped in a per-model loop with separate load, warmup, and test runs. Per-model results printed separately, then a side-by-side comparison table at the end (base vs large vs Qwen 2B vs Qwen 4B). Also added per-case comparison between base and large.

2. **Change 2 (explicit 5-class scheme):** Replaced heuristic label detection with explicit `DISFLUENT_LABELS = {"FP", "RP", "RV", "PW"}` and `FLUENT_LABELS = {"O"}`. Removed all the guessing logic from `reassemble_text()`. Updated per-token grid to show actual 5-class labels (e.g., `word[O]`, `um[FP]`, `the[RV]`) instead of generic F/D tags. Updated `compute_false_positive_rate()` to use `DISFLUENT_LABELS` directly instead of accepting a passed-in set.

3. **Change 3 (BPE tokenizer handling):** Replaced WordPiece `##` prefix inspection with offset-mapping + `word_ids()` approach. The `classify_tokens()` function now groups subtokens by `word_id`, takes the label of the FIRST subtoken for each word, and maps character offsets back to original text. The `reassemble_text()` function uses character offsets from the original string to extract kept spans, making it tokenizer-agnostic.

Preserved unchanged: test_cases.py, Levenshtein similarity, filler check, passthrough check, hypothesis criteria scoring (thresholds), overall reporting structure.

No references to `4i-ai/BERT_disfluency_cls` remain.

**Baseline results (before changes):**
- Python syntax check: OK (original script parsed successfully)
- Script could not actually run because it referenced the wrong model (`4i-ai/BERT_disfluency_cls`)

**Post-change results (after changes):**
- Python syntax check: OK (659 lines, parses cleanly)
- No references to old model ID
- Both model IDs present: `arielcerdap/modernbert-base-multiclass-disfluency-v2`, `arielcerdap/modernbert-disfluency-expC-large-realonly`
- `DISFLUENT_LABELS` used in 4 locations (module constant, reassemble_text, compute_false_positive_rate, run_model)
- `word_ids()` used for proper subtoken-to-word mapping
- Cannot run full inference without `setup.sh` + model downloads (that's T3's job)
