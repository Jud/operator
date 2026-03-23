# Hypothesis Test Tasks: Qwen3.5-2B STT Cleanup

**Status:** Planning
**Worktree:** .claude/worktrees/hypothesis-qwen35-stt-cleanup
**Branch:** hypothesis/qwen35-stt-cleanup
**Total Tasks:** 3
**Completed:** 0

## Task Graph

```
T1 (setup + test cases) ──► T2 (bench script) ──► T3 (run + evaluate)
```

All tasks are sequential. T2 depends on T1 for the venv and test case definitions. T3 depends on T2 for the runnable script.

## Tasks

### T1. Python venv + test case definitions
**Status:** built
**Depends on:** —
**Files:** `docs/hypotheses/qwen35-stt-cleanup/working/setup.sh`, `docs/hypotheses/qwen35-stt-cleanup/working/test_cases.py`
**Review:** —
**Spec:**

Create two files:

1. **`setup.sh`** — A shell script that:
   - Creates a Python venv at `docs/hypotheses/qwen35-stt-cleanup/working/.venv`
   - Installs `mlx-lm` (latest) into it
   - Runs a quick smoke test: `python -c "from mlx_lm import load; print('ok')"`
   - Is idempotent (skips creation if venv already exists)

2. **`test_cases.py`** — A Python module defining the test suite as a list of dicts. Each entry has:
   - `name`: short identifier
   - `raw`: the raw transcription input (as if from WhisperKit)
   - `expected`: hand-written cleaned output
   - `category`: one of `filler`, `self-correction`, `ramble`, `clean-passthrough`, `short-command`, `real-fixture`

   Include these test cases (minimum 12):

   **Filler removal (3):**
   - "Um, I think we should, uh, go with option B" -> "I think we should go with option B."
   - "So like, you know, the thing is, uh, we need to like figure this out" -> "The thing is, we need to figure this out."
   - "Yeah, I definitely wanted to use CodexExec instead of the, um, instead of the Anthropic API. Um, and now I'm kind of thinking, should we be doing this?" -> "Yeah, I definitely wanted to use CodexExec instead of the Anthropic API. And now I'm kind of thinking, should we be doing this?"

   **Self-correction (2):**
   - "Tell him Tuesday -- actually no, Wednesday works better" -> "Tell him Wednesday works better."
   - "I'll send it to the product team, or wait, the engineering team, yeah the engineering team" -> "I'll send it to the engineering team."

   **Stream-of-consciousness ramble (2):**
   - The `passing-very-long` fixture expected text (from `~/Library/Application Support/Operator/test-fixtures/passing-very-long.expected.txt`) -> Hand-written cleaned version that preserves the core question about research findings and guardrails but strips filler
   - The `no-background-confirmation` fixture expected text -> Hand-written cleaned version

   **Clean passthrough (2):**
   - "The meeting is at 3pm in conference room B." -> Same (unchanged)
   - "Please review the pull request and merge it when ready." -> Same (unchanged)

   **Short commands (2):**
   - "Hey Siri set a timer for 10 minutes" -> "Set a timer for 10 minutes."
   - "Send a message to John" -> "Send a message to John."

   **Real fixture (1):**
   - The `passing-medium-long` fixture expected text -> Hand-written cleaned version

**Acceptance criteria:**
- `./setup.sh` succeeds and produces a working venv with mlx-lm importable
- `test_cases.py` is importable and `len(TEST_CASES) >= 12`
- Each expected output is a realistic, manually verified cleanup of the raw input

---

### T2. Benchmark script (cleanup_bench.py)
**Status:** pending
**Depends on:** T1
**Files:** `docs/hypotheses/qwen35-stt-cleanup/working/cleanup_bench.py`
**Review:** —
**Spec:**

Create `cleanup_bench.py` — a self-contained Python script that:

1. **Loads model:** Uses `mlx_lm.load()` to load `mlx-community/Qwen3.5-2B-8bit` (or the exact 8-bit model ID from HuggingFace). Prints model load time and memory estimate.

2. **Defines system prompt:** A concise system prompt instructing the model to:
   - Remove filler words (um, uh, like, you know, so, actually, basically)
   - Resolve self-corrections: keep only the final intended version
   - Fix grammar, capitalization, and punctuation
   - Preserve the speaker's meaning exactly — never add content
   - Output ONLY the cleaned text, no explanations, no markdown, no quotes
   - If the input is already clean, return it unchanged

3. **Runs all test cases:** Imports `TEST_CASES` from `test_cases.py`. For each:
   - Constructs chat messages (system + user with the raw text)
   - Calls `mlx_lm.generate()` with `max_tokens=512`, measuring wall-clock time
   - Records: input tokens, output tokens, latency_ms, tok/s, raw output

4. **Evaluates quality:** For each test case, computes:
   - `filler_clean`: no filler words remain in output (for filler/ramble categories)
   - `meaning_preserved`: Levenshtein similarity between output and expected >= 0.80
   - `no_hallucination`: output word count <= expected word count * 1.3 (no added content)
   - `format_clean`: output does not start with quotes, backticks, "Here", "Sure", or contain markdown
   - `passthrough_ok`: for clean-passthrough category, output matches expected exactly (ignoring trailing punctuation)

5. **Prints results table:** Columns: name, category, latency_ms, tok/s, meaning_sim, pass/fail, notes. Then a summary section:
   - Aggregate: filler removal rate, self-correction rate, meaning preservation rate, format compliance rate
   - Latency: p50, p95, p99 for all cases; p95 for cases <= 30 words
   - Token throughput: mean tok/s
   - Maps each metric to the success/failure criteria from hypothesis.md

6. **Writes JSON results** to `docs/hypotheses/qwen35-stt-cleanup/working/reviews/run-TIMESTAMP.json` with all per-case data and summary stats.

**Implementation notes:**
- Use `mlx_lm.generate()` NOT `mlx_lm.stream_generate()` — we want total latency, not streaming
- Warm up the model with one throwaway generation before measuring
- Use `time.perf_counter()` for timing
- Keep the script under 300 lines
- Shebang: `#!/usr/bin/env python3`
- The script should work when run as: `cd docs/hypotheses/qwen35-stt-cleanup/working && .venv/bin/python cleanup_bench.py`

**Acceptance criteria:**
- Script runs without errors (with model downloaded)
- Produces a readable table to stdout
- Writes JSON results file
- All hypothesis criteria are explicitly scored in the summary

---

### T3. Run benchmark and record results
**Status:** pending
**Depends on:** T2
**Files:** `docs/hypotheses/qwen35-stt-cleanup/working/reviews/run-*.json`, `docs/hypotheses/qwen35-stt-cleanup/results.md`
**Review:** —
**Spec:**

Execute the prototype and evaluate results:

1. **Run setup:** `cd docs/hypotheses/qwen35-stt-cleanup/working && bash setup.sh`
2. **Run benchmark:** `.venv/bin/python cleanup_bench.py` — let it download the model if needed (first run will be slow)
3. **Capture output:** Save full stdout to `reviews/run-TIMESTAMP.txt`
4. **Evaluate against hypothesis criteria:** Read the JSON results and score each criterion from hypothesis.md:

   For each success criterion, mark pass/fail based on the data:
   - Filler removal >= 9/10
   - Self-correction handling >= 8/10
   - Grammar and punctuation >= 9/10
   - Meaning preservation >= 9/10
   - Latency < 800ms p95 for <= 30 word inputs
   - Decode speed >= 100 tok/s
   - Memory < 4GB

   For each failure criterion, check if triggered:
   - Meaning corruption >= 3/10
   - Latency >= 1.5s typical
   - Instruction non-compliance >= 3/10
   - Quality regression on clean input >= 2/10

5. **Write `results.md`** at `docs/hypotheses/qwen35-stt-cleanup/results.md` with:
   - Verdict: PROVEN / DISPROVEN / MIXED
   - Per-criterion scoring with actual numbers
   - Notable observations (quality surprises, prompt issues, edge cases)
   - If DISPROVEN or MIXED: what specifically failed and what alternatives to consider
   - Raw data reference (path to JSON)

**Acceptance criteria:**
- Benchmark completes without crashes
- Every success/failure criterion from hypothesis.md has an explicit score
- results.md contains a clear verdict with supporting data
- If the model needs prompt tuning to pass, note what was tried and what worked/didn't
