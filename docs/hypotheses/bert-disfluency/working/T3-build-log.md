### T3: Run benchmark and capture results
**Date:** 2026-03-23
**Status:** complete
**Files changed:**
- `docs/hypotheses/bert-disfluency/working/setup.sh` -- fixed to detect/reuse project-root `.venv-poc`, pin `transformers>=4.48.0`
- `docs/hypotheses/bert-disfluency/working/reviews/run-20260323T143103Z.json` -- benchmark results

**Notes:**

Setup.sh in the hypothesis branch still had the original un-updated version (T1 changes were committed to a different worktree branch). Fixed setup.sh to properly detect `.venv-poc` at the project root and pin `transformers>=4.48.0`.

Both models loaded successfully on MPS (Apple Silicon GPU):
- **base** (149M params, 598 MB at fp32): loaded in 15.78s
- **large** (396M params, 1583 MB at fp32): loaded in 31.73s

Both use the 5-class label scheme: `{0: 'O', 1: 'FP', 2: 'RP', 3: 'RV', 4: 'PW'}`

**Key findings:**

Both models **fail** the hypothesis decisively. Every failure criterion is triggered for at least one model:

| Metric | Base | Large | Target |
|--------|------|-------|--------|
| Filler removal | 2/5 (40%) | 2/5 (40%) | >=4/5 |
| Self-correction | 0/2 (0%) | 0/2 (0%) | >=1/2 |
| Meaning preservation | 6/12 (50%) | 7/12 (58%) | >=8/12 |
| Passthrough | 1/2 (50%) | 1/2 (50%) | 2/2 |
| FP rate | 13.2% | 4.9% | <5% |
| Latency p95 | 522ms | 220ms | <50ms |
| Memory | 598MB | 1583MB | <1GB |

**Failure criteria triggered:**
- Base: all 4 failure criteria triggered (massive FP, filler blindness, self-correction failure, unacceptable latency)
- Large: 3 of 4 failure criteria triggered (filler blindness, self-correction failure, unacceptable latency)

The models fail to detect common fillers like "like", "you know", "so" -- they only reliably catch "um" and "uh" as FP. Self-correction is completely unhandled (0/2 for both models). The base model also has severe false positive issues (13.2% of content tokens mislabeled as disfluent). Latency is 4-10x over the 50ms target even on MPS.

Compared to the Qwen models, ModernBERT performs **worse** on meaning preservation (50-58% vs 67-75%) and equally badly on self-correction (0% across all four models tested). The only advantage is lower memory usage for the base model (598MB vs 2.5-5GB).

**Baseline results (before changes):**
- setup.sh was not yet updated for `.venv-poc` reuse or transformers version pin
- `.venv-poc` already existed with transformers 5.3.0 and torch 2.10.0

**Post-change results (after changes):**
```
$ bash setup.sh
Reusing existing venv at /Users/jud/Projects/operator/.venv-poc
Installing dependencies...
Setup complete.

$ .venv-poc/bin/python classify_bench.py
Both models loaded, all 12 test cases ran, JSON written to reviews/run-20260323T143103Z.json
No tracebacks, no model loading errors.
```
