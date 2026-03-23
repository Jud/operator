### T1: Python venv + test case definitions
**Date:** 2026-03-22
**Status:** complete
**Files changed:**
- `docs/hypotheses/qwen35-stt-cleanup/working/setup.sh` — Created idempotent venv setup script that installs mlx-lm
- `docs/hypotheses/qwen35-stt-cleanup/working/test_cases.py` — Created 12 test cases across 6 categories

**Notes:**
- setup.sh is idempotent: skips venv creation if it already exists, always runs pip install (quick no-op if already installed)
- Hand-wrote cleaned expected outputs for 3 real fixtures: passing-very-long (research guardrails ramble), no-background-confirmation (CodexExec/background process), passing-medium-long (AFK mode feedback)
- The filler-with-correction case (3rd filler) also appears as the no-background-confirmation fixture content, but the filler category version uses a simpler expected output that just strips fillers, while the ramble version does deeper cleanup
- All 12 test cases have name, raw, expected, and category fields as specified

**Test results:**
- `setup.sh` runs successfully, creates venv, installs mlx-lm, smoke test prints "ok"
- Second run confirms idempotency (skips venv creation)
- `from test_cases import TEST_CASES` succeeds, `len(TEST_CASES) == 12`
- Categories: filler=3, self-correction=2, ramble=2, clean-passthrough=2, short-command=2, real-fixture=1

### T2: Benchmark script (cleanup_bench.py)
**Date:** 2026-03-22
**Status:** complete
**Files changed:**
- `docs/hypotheses/qwen35-stt-cleanup/working/cleanup_bench.py` — Created benchmark script (276 lines)
- `docs/hypotheses/qwen35-stt-cleanup/working/reviews/` — Created directory for JSON result files

**Notes:**
- Script uses `mlx_lm.generate()` as specified, with `time.perf_counter()` for wall-clock timing
- Implemented Levenshtein distance/similarity inline (no external dependency required)
- Smart filler word detection: "like" is only flagged as filler when not part of legitimate phrases (e.g., "seems like", "looks like", "would like")
- Quality checks: filler_clean, meaning_preserved (Levenshtein >= 0.80), no_hallucination (output words <= 1.3x expected), format_clean, passthrough_ok
- Includes model warmup before measurements
- Produces readable table to stdout with per-case results and actual outputs
- Prints aggregate summary with all hypothesis criteria explicitly scored
- Writes timestamped JSON to reviews/ directory with full per-case data and summary stats
- All helper functions verified with unit tests

**Baseline results (before changes):**
- Build: pass (swift build clean)
- Tests: 394 passed, 0 failed
- Lint: N/A (Python script, not Swift)

**Post-change results (after changes):**
- Build: pass (swift build clean)
- Tests: 394 passed, 0 failed
- Python syntax: valid
- Python helper function tests: all pass (Levenshtein, filler check, hallucination check, format check, passthrough check, percentile)
