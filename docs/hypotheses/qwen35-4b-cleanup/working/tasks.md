# Hypothesis Test Tasks: Qwen3.5-4B STT Cleanup

**Status:** Planning
**Worktree:** .claude/worktrees/hypothesis-qwen35-4b-cleanup
**Branch:** hypothesis/qwen35-4b-cleanup
**Total Tasks:** 2
**Completed:** 0

## Task Graph

```
T1 (setup + run benchmark) --> T2 (record results + verdict)
```

Batch 1: [T1]
Batch 2: [T2]

## Tasks

### T1. Run the Qwen3.5-4B cleanup benchmark
**Status:** pending
**Depends on:** --
**Files:** docs/hypotheses/qwen35-4b-cleanup/working/cleanup_bench.py, docs/hypotheses/qwen35-4b-cleanup/working/setup.sh
**Review:** --
**Spec:**

The benchmark infrastructure (cleanup_bench.py, test_cases.py, setup.sh) already exists and is adapted from the 2B hypothesis. The task is to run it and capture output.

Steps:
1. `cd docs/hypotheses/qwen35-4b-cleanup/working && bash setup.sh` -- creates venv, installs mlx-lm and mlx-vlm.
2. Run the benchmark: `.venv/bin/python cleanup_bench.py`
3. Capture the full terminal output (copy-paste into the build log). The script writes a JSON results file to `working/reviews/run-<timestamp>.json` automatically.

**CRITICAL: mlx_vlm double-template bug.** The HuggingFace model page confirms Qwen3.5-4B-MLX-8bit requires `mlx_vlm` (not `mlx_lm`). The `mlx_vlm.generate()` function applies `apply_chat_template` internally. But the current `cleanup_bench.py` also applies the chat template before calling generate (lines 178-179). This will double-template the prompt when using the mlx_vlm backend, producing garbage.

**Fix before running:** When `backend == "mlx_vlm"`, pass the raw user message to `generate()` instead of the pre-templated prompt. Specifically, update `main()` so that:
- For `mlx_lm` backend: keep current behavior (pre-template, pass full prompt string)
- For `mlx_vlm` backend: pass `tc["raw"]` as the prompt to `do_generate`, and ensure the system prompt is passed through `mlx_vlm`'s own chat template mechanism. The `mlx_vlm.generate` function accepts a plain text prompt and handles templating. You may need to check if `mlx_vlm.generate` supports a `system_prompt` kwarg or if you need to use `mlx_vlm.apply_chat_template` with the messages array.

Also fix the warmup call similarly (lines 165-168).

**Acceptance criteria:**
- Benchmark completes all 12 test cases without error
- JSON results file is written to `working/reviews/`
- Full terminal output is captured in the build log
- If model download is needed, that completes successfully (model is ~5GB)

---

### T2. Record results and write verdict
**Status:** pending
**Depends on:** T1
**Files:** docs/hypotheses/qwen35-4b-cleanup/hypothesis.md, docs/hypotheses/qwen35-4b-cleanup/working/build-log.md
**Review:** --
**Spec:**

After T1 completes, analyze the results and update the hypothesis:

1. Read the JSON results file from `working/reviews/run-*.json`
2. Check each success criterion checkbox in hypothesis.md (mark with [x] or leave [ ])
3. Check each failure criterion checkbox in hypothesis.md
4. Update the hypothesis status from "Testing" to either "Proven" or "Disproven"
5. Add a "## Results" section to hypothesis.md with:
   - Summary table: 2B vs 4B comparison (the script prints this)
   - Overall verdict and reasoning
   - Key observations (did the 4B model fix the echo-back problem? Did self-correction improve?)
   - Next steps recommendation (if proven: integration path; if disproven: what to try next)
6. Write a build-log.md summarizing the full run

**Acceptance criteria:**
- All checkboxes in hypothesis.md are scored against actual results
- Status is updated to Proven or Disproven
- Results section includes the 2B vs 4B comparison
- build-log.md has the full terminal output and timestamps
