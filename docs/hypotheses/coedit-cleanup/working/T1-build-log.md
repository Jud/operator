### T1: Build CoEdit cleanup benchmark script
**Date:** 2026-03-23
**Status:** complete
**Files changed:**
- `docs/hypotheses/coedit-cleanup/working/test_cases.py` -- copied verbatim from qwen35-stt-cleanup
- `docs/hypotheses/coedit-cleanup/working/cleanup_bench.py` -- new benchmark script
- `docs/hypotheses/coedit-cleanup/working/reviews/coedit-large-results.json` -- benchmark output

**Notes:**
Script built and run in single pass. All 4 prompts x 12 cases = 48 generations completed.
Model loaded on MPS in 4.4s. No additional dependencies needed beyond transformers/torch.

The first generation per prompt is significantly slower (7.3s for "Make this text fluent" on filler-simple)
likely due to MPS kernel compilation. Subsequent calls for the same prompt are 400-900ms for short inputs.

The ramble cases (long inputs) trigger extremely long latencies (13-31s) with beam search on long
sequences. This is a fundamental limitation of beam search with num_beams=5 on T5-large for long inputs.

**Baseline results (before changes):**
- Syntax check: OK
- test_cases.py import: 12 cases loaded, all categories present
- No pre-existing code to break

**Post-change results (after changes):**
- Script runs without errors: PASS
- All 4 result tables printed: PASS
- Per-prompt summary table: PASS
- Cross-model comparison: PASS
- Criteria check: PASS (prints correctly)
- JSON written to reviews/: PASS (40KB file)
- test_cases match 12 cases from prior hypotheses: PASS
