### T3: Run benchmark and record results
**Date:** 2026-03-23
**Status:** complete
**Files changed:**
- `docs/hypotheses/qwen35-stt-cleanup/results.md` -- Created with DISPROVEN verdict and detailed analysis
- `docs/hypotheses/qwen35-stt-cleanup/working/reviews/run-20260323T051736Z.json` -- First benchmark run results
- `docs/hypotheses/qwen35-stt-cleanup/working/reviews/run-20260323T051756Z.json` -- Second benchmark run results (confirms consistency)
- `docs/hypotheses/qwen35-stt-cleanup/working/reviews/run-20260323T051736Z.txt` -- Stdout capture from benchmark

**Notes:**
- Ran the benchmark twice to confirm results are consistent (they are -- identical pass/fail per case, similar latencies)
- Model downloaded successfully from HuggingFace (~1.5GB, mlx-community/Qwen3.5-2B-8bit)
- Model loads in ~1.3s from cache, peak memory 2.55 GB
- The dominant failure mode is the model echoing input verbatim instead of cleaning it
- Self-correction resolution completely absent (0/2)
- Filler removal works for obvious cases (um, uh) but not discourse markers (so, like, you know)
- Ramble condensation does not happen at all -- long inputs returned nearly unchanged
- Decode speed 49 tok/s is well below the 100 tok/s target
- Verdict: DISPROVEN -- 4/7 success criteria failed, 1/4 failure criteria triggered

**Baseline results (before changes):**
- No source code changes to verify (T3 is a run-and-evaluate task)
- Setup.sh: runs successfully, creates venv with mlx-lm
- cleanup_bench.py: syntactically valid, imports resolve

**Post-change results (after changes):**
- Benchmark run 1: completed without errors, 6/12 cases passed, JSON written
- Benchmark run 2: completed without errors, 6/12 cases passed, JSON written (consistent)
- results.md: written with verdict, per-criterion scoring, and alternatives analysis
