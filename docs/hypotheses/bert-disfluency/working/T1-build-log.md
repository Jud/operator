### T1: Update setup.sh for ModernBERT compatibility
**Date:** 2026-03-23
**Status:** complete
**Files changed:**
- `docs/hypotheses/bert-disfluency/working/setup.sh` -- pinned `transformers>=4.48.0`, added `.venv-poc` reuse logic

**Notes:**
- ModernBERT requires transformers >= 4.48.0 (when `ModernBertForTokenClassification` was added)
- Script now detects project-root `.venv-poc` via `git rev-parse --git-common-dir` (works in both main checkout and worktrees)
- Falls back to creating a local `.venv` if `.venv-poc` doesn't exist
- Verified `ModernBertForTokenClassification` imports successfully with transformers 5.3.0

**Baseline results (before changes):**
Original `setup.sh` installed `transformers torch` without version pin. No version constraint for ModernBERT.

**Post-change results (after changes):**
```
$ bash setup.sh
Reusing existing venv at /Users/jud/Projects/operator/.venv-poc
Installing dependencies...

Setup complete. Run the benchmark with:
  cd .../docs/hypotheses/bert-disfluency/working
  /Users/jud/Projects/operator/.venv-poc/bin/python classify_bench.py

$ python -c "import transformers; print(transformers.__version__)"
5.3.0

$ python -c "from transformers import ModernBertForTokenClassification; print('OK')"
OK

$ python -c "import torch; print(torch.__version__)"
2.10.0
```
