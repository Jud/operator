# Hypothesis Test Tasks: CoEdit STT Cleanup

**Status:** Planning
**Worktree:** .claude/worktrees/hypothesis-coedit-cleanup
**Branch:** hypothesis/coedit-cleanup
**Total Tasks:** 2
**Completed:** 0

## Task Graph

```
T1 (build benchmark) -> T2 (run and evaluate)
```

Batch 1: [T1]
Batch 2: [T2]

## Tasks

### T1. Build CoEdit cleanup benchmark script
**Status:** in-progress
**Depends on:** —
**Files:** `docs/hypotheses/coedit-cleanup/working/cleanup_bench.py`, `docs/hypotheses/coedit-cleanup/working/test_cases.py`
**Review:** —
**Spec:**

Create two files in the worktree at `docs/hypotheses/coedit-cleanup/working/`:

**1. `test_cases.py`** — Copy from `docs/hypotheses/qwen35-stt-cleanup/working/test_cases.py` verbatim. Same 12 test cases, same format.

**2. `cleanup_bench.py`** — Python benchmark script. Requirements:

**Model loading:**
- Load `grammarly/coedit-large` via `AutoTokenizer` and `T5ForConditionalGeneration` from `transformers`.
- Use MPS device if available, else CPU.
- Print model load time.

**Instruction prompts (4):**
```python
PROMPTS = [
    "Make this text fluent: ",
    "Fix the grammar in this text: ",
    "Remove disfluencies from this text: ",
    "Fix the grammar and remove filler words: ",
]
```

**Generation:**
- For each prompt x test case, concatenate `prompt + raw_text` as the input string.
- Tokenize, generate with `model.generate(num_beams=5, max_new_tokens=256)`.
- Decode output, strip whitespace.
- Measure wall-clock latency per generation (time just the `model.generate()` call).

**Evaluation per output:**
- `levenshtein_similarity`: Levenshtein ratio between output and expected (use `difflib.SequenceMatcher`). Pass if >= 0.75.
- `filler_removed`: For filler/ramble categories, check that common fillers (um, uh, like, you know) are removed or reduced. Simple regex check: count filler words in input vs output.
- `self_correction_resolved`: For self-correction category, check that the corrected version appears and the pre-correction version does not. Manual spot-check in output.
- `clean_passthrough`: For clean-passthrough category, check levenshtein_similarity >= 0.95.
- `no_hallucination`: Check that output word count is not > 1.5x input word count (crude but catches additions).

**Output (print to stdout):**
1. Per-prompt results table: for each of the 4 prompts, show all 12 test cases with columns: `name | category | output (truncated to 80 chars) | lev_sim | latency_ms | pass/fail`.
2. Per-prompt summary: `prompt | filler_score (x/5) | correction_score (x/2) | passthrough_score (x/2) | meaning_score (x/12) | hallucination_count | avg_latency_ms`.
3. Best prompt selection: which prompt scored highest overall.
4. Cross-model comparison table against prior results (hardcode from hypothesis.md): Qwen 2B (0/2 correction, 2/5 filler), Qwen 4B (0/2 correction, 3/5 filler, 9/12 meaning), BERT (0/2 correction, 13.2% FP).

**JSON output:**
- Write full results (all prompts, all cases, all metrics) to `reviews/coedit-large-results.json`.

**Running:**
```bash
cd /Users/jud/Projects/operator
.venv-poc/bin/python docs/hypotheses/coedit-cleanup/working/cleanup_bench.py
```

The script must be self-contained (no external deps beyond transformers, torch, difflib, json, time, re, os). No argparse needed — just run it.

**Acceptance criteria:**
- Script runs without errors using `.venv-poc/bin/python`.
- Produces readable stdout output with all 4 tables described above.
- Writes JSON results file.
- Test cases match the 12 cases from prior hypotheses exactly.

---

### T2. Run benchmark and evaluate results
**Status:** pending
**Depends on:** T1
**Files:** `docs/hypotheses/coedit-cleanup/working/reviews/coedit-large-results.json`
**Review:** —
**Spec:**

Run the benchmark:
```bash
cd /Users/jud/Projects/operator
.venv-poc/bin/python docs/hypotheses/coedit-cleanup/working/cleanup_bench.py
```

Expected runtime: 2-5 minutes (model download if not cached + 48 generations with beam search).

After the run completes:

1. **Capture the full stdout output** — paste it into the build log.
2. **Evaluate against success criteria from hypothesis.md:**
   - Filler removal >= 4/5?
   - Self-correction >= 1/2?
   - Meaning preservation >= 8/12?
   - Clean passthrough 2/2?
   - No content hallucination in any case?
   - Latency < 1500ms p95?
3. **Check failure criteria:**
   - Self-correction still 0/2?
   - Filler removal <= 2/5?
   - Meaning corruption >= 3/12?
   - Output format problems >= 3/12?
   - Latency >= 3s?
4. **Write verdict** to `build-log.md`: which criteria passed, which failed, overall proven/disproven/mixed.
5. **Verify JSON results file** was written to `reviews/`.

**Acceptance criteria:**
- Benchmark ran to completion.
- Build log contains full output and clear verdict.
- JSON results archived.
