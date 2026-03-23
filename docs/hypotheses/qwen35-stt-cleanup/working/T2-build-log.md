### T2: Benchmark script (cleanup_bench.py)
**Date:** 2026-03-23
**Status:** complete
**Files changed:**
- `docs/hypotheses/qwen35-stt-cleanup/working/cleanup_bench.py` — Fixed filler detection to normalize punctuation before word boundary matching

**Notes:**
- Rework round 1: `check_filler_clean()` was using space-padded word boundary matching (`f" {f} " in f" {lower} "`) which failed when fillers appeared adjacent to punctuation (e.g., "Um," or "I, uh,"). These are the most common filler patterns in real speech transcription.
- Fix: Added `_normalize_for_filler_check()` helper that strips punctuation (`[,.\-!?;:"'()]+`) via regex before doing word boundary matching. This converts "Um, I think" -> "Um  I think" so " um " matches correctly.
- Added `import re` to support the regex-based normalization.
- The `_LIKE_LEGIT` allowlist still works correctly after normalization since punctuation is stripped from both the text and the legitimate phrases are plain words.
- Verified fix catches: "Um," "uh," "So," "Actually," "Basically," at sentence boundaries and inline positions.
- Verified fix preserves: legitimate "like" in "seems like", "would like", and clean text without fillers.

**Baseline results (before changes):**
- Swift build: pass (Build complete! 43.49s)
- Python syntax: valid
- Filler detection bug confirmed: `check_filler_clean("Um, I think we should go with option B.")` returned `True` (false pass)

**Post-change results (after changes):**
- Swift build: pass (Build complete! 4.32s, incremental)
- Python syntax: valid
- Filler detection fix confirmed:
  - `check_filler_clean("Um, I think we should go with option B.")` returns `False` (correctly detected)
  - `check_filler_clean("I, uh, think we should go with option B.")` returns `False` (correctly detected)
  - `check_filler_clean("So, the thing is we need to figure this out.")` returns `False` (correctly detected)
  - `check_filler_clean("It seems like a good idea.")` returns `True` (legitimate use preserved)
  - `check_filler_clean("I think we should go with option B.")` returns `True` (clean text preserved)
