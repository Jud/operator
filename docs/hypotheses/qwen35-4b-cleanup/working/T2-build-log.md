### T2: Record results and write verdict
**Date:** 2026-03-23
**Status:** complete
**Files changed:**
- docs/hypotheses/qwen35-4b-cleanup/hypothesis.md -- Updated status to "Disproven", scored all success/failure criteria checkboxes, added Results section with 2B vs 4B comparison, per-case analysis, key findings, and next steps
- docs/hypotheses/qwen35-4b-cleanup/working/T2-build-log.md -- This file
- docs/hypotheses/qwen35-4b-cleanup/working/T2-status.md -- Task status

**Notes:**

Analyzed results from `working/reviews/run-20260323T130738Z.json` (produced by T1).

Scoring summary:
- Success criteria: 4/8 pass (format compliance, clean passthrough, latency, memory)
- Success criteria: 4/8 fail (filler removal 3/5, self-correction 0/2, meaning preservation 8/12, decode speed 25.3 tok/s)
- Failure criteria: 2/5 triggered (echo-back 4/12, meaning corruption 3/12)
- Failure criteria: 3/5 OK (latency, instruction compliance, quality regression on clean)

Key decision: Marked as **Disproven** because:
1. The three most important quality criteria all fail (filler removal, self-correction, meaning preservation)
2. Two failure criteria are triggered (echo-back persistence, meaning corruption)
3. Self-correction -- the single most important metric to improve from the 2B -- remained at 0/2
4. The improvements over 2B are marginal (filler removal 40%->60%, echo-back ~6->4) while costs doubled (memory, latency, throughput all 2x worse)

Note on echo-back scoring: The echo-back failure criterion counts 4/12 cases, but 2 of those (clean-meeting, clean-pr) are clean passthrough cases where echoing is correct behavior. The criterion as written triggers at >=4/12, which means the 2 legitimate echo-backs push the count to exactly the threshold. A fairer count would be 2/10 (non-passthrough cases only), which would not trigger. This is worth noting but does not change the overall verdict -- the hypothesis fails on quality criteria regardless.

**Baseline results (before changes):**
N/A -- T2 is a documentation/analysis task, not a code change. No build or test commands apply.

**Post-change results (after changes):**
N/A -- documentation only. Verified hypothesis.md renders correctly by reading the final file.
