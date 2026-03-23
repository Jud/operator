### T1: Run the Qwen3.5-4B cleanup benchmark
**Date:** 2026-03-23
**Status:** complete
**Files changed:**
- docs/hypotheses/qwen35-4b-cleanup/working/cleanup_bench.py -- added `enable_thinking=False` to `apply_chat_template` calls (warmup + main loop) to disable Qwen3.5 thinking mode
- docs/hypotheses/qwen35-4b-cleanup/working/reviews/run-20260323T130738Z.json -- benchmark results

**Notes:**

Two issues discovered and resolved during execution:

1. **Qwen3.5-4B thinking mode (critical).** Qwen3.5-4B defaults to "thinking mode" where it outputs verbose chain-of-thought reasoning (`Thinking Process: 1. Analyze the Request...`) before the actual answer. The chat template supports an `enable_thinking` parameter -- when set to `False`, it inserts an empty `<think>\n\n</think>` block in the generation prompt, signaling the model to skip reasoning. Without this fix, all 12 test cases failed with 0% quality across every metric. With the fix, results are meaningful.

2. **mlx_vlm double-template bug (not triggered).** The task spec warned about `mlx_vlm.generate()` applying `apply_chat_template` internally. Investigation of the `mlx_vlm` source code (v0.x) showed that `mlx_vlm.generate` does NOT apply chat template for text-only inputs -- it calls `prepare_inputs` which just tokenizes. Furthermore, `mlx_lm` loaded the model successfully, so the `mlx_vlm` fallback path was never reached. No double-template fix was needed.

**Model loading:** `mlx_lm` loaded `mlx-community/Qwen3.5-4B-MLX-8bit` successfully (2.1s load time, 4.47 GB initial memory). The `mlx_vlm` fallback was not needed.

**Baseline results (before changes):**
First run without `enable_thinking=False` -- all 12 cases FAIL. Model output verbose markdown reasoning for every input. 0/12 pass, 0% meaning preservation, 0% format compliance. Not a valid baseline since the thinking mode bug made results meaningless.

**Post-change results (after changes):**

```
==============================================================================================================
Name                           Category            Lat(ms)   tok/s  MeanSim  Echo  Pass Notes
--------------------------------------------------------------------------------------------------------------
filler-simple                  filler                601.7    15.0    1.000    no  PASS
filler-heavy                   filler                632.9    17.4    1.000    no  PASS
filler-with-correction         filler                970.8    31.9    1.000    no  PASS
correction-day                 self-correction       652.4    18.4    0.582    no  FAIL meaning_sim=0.58; hallucination
correction-team                self-correction       790.6    27.8    0.402   YES  FAIL ECHO-BACK; meaning_sim=0.40; hallucination
ramble-research-guardrails     ramble               3111.8    50.8    0.659    no  FAIL fillers remain; meaning_sim=0.66; hallucination
ramble-codex-background        ramble               1722.1    43.6    0.805    no  FAIL fillers remain
clean-meeting                  clean-passthrough     738.1    16.3    1.000   YES  PASS ECHO-BACK
clean-pr                       clean-passthrough     661.9    16.6    1.000   YES  PASS ECHO-BACK
command-timer                  short-command         662.1    18.1    0.730    no  FAIL meaning_sim=0.73; hallucination
command-message                short-command         545.8     9.2    0.957   YES  PASS ECHO-BACK
fixture-afk-mode               real-fixture         1774.0    38.9    0.824    no  PASS
==============================================================================================================

--- Actual outputs ---

[PASS] filler-simple:
       Raw: Um, I think we should, uh, go with option B
  Expected: I think we should go with option B.
    Actual: I think we should go with option B.

[PASS] filler-heavy:
       Raw: So like, you know, the thing is, uh, we need to like figure this out
  Expected: The thing is, we need to figure this out.
    Actual: The thing is, we need to figure this out.

[PASS] filler-with-correction:
       Raw: Yeah, I definitely wanted to use CodexExec instead of the, um, instead of the Anthropic AP...
  Expected: Yeah, I definitely wanted to use CodexExec instead of the Anthropic API. And now I'm kind ...
    Actual: Yeah, I definitely wanted to use CodexExec instead of the Anthropic API. And now I'm kind ...

[FAIL] correction-day:
       Raw: Tell him Tuesday -- actually no, Wednesday works better
  Expected: Tell him Wednesday works better.
    Actual: Tell him Tuesday; actually, no, Wednesday works better.

[FAIL] [ECHO] correction-team:
       Raw: I'll send it to the product team, or wait, the engineering team, yeah the engineering team
  Expected: I'll send it to the engineering team.
    Actual: I'll send it to the product team, or wait, the engineering team, yeah, the engineering tea...

[FAIL] ramble-research-guardrails:
       Raw: Well, now when we say that, should it update all docs? Or I guess the question is, like, y...
  Expected: Should it update all docs? We'll end up with research findings, and some of those may be p...
    Actual: Well, now when we say that, should it update all docs? Or, I guess the question is, we wil...

[FAIL] ramble-codex-background:
       Raw: Yeah, I definitely wanted to use CodexExec instead of the, um, instead of the Anthropic AP...
  Expected: I definitely wanted to use CodexExec instead of the Anthropic API. Now I'm thinking, shoul...
    Actual: Yeah, I definitely wanted to use CodexExec instead of the Anthropic API. And now I'm kind ...

[PASS] [ECHO] clean-meeting:
       Raw: The meeting is at 3pm in conference room B.
  Expected: The meeting is at 3pm in conference room B.
    Actual: The meeting is at 3pm in conference room B.

[PASS] [ECHO] clean-pr:
       Raw: Please review the pull request and merge it when ready.
  Expected: Please review the pull request and merge it when ready.
    Actual: Please review the pull request and merge it when ready.

[FAIL] command-timer:
       Raw: Hey Siri set a timer for 10 minutes
  Expected: Set a timer for 10 minutes.
    Actual: Hey Siri, set a timer for 10 minutes.

[PASS] [ECHO] command-message:
       Raw: Send a message to John
  Expected: Send a message to John.
    Actual: Send a message to John

[PASS] fixture-afk-mode:
       Raw: This seems right. Useful addition seems correct. I would say that the only thing that I wo...
  Expected: This seems right. Useful addition. The only thing I would say is that if we're in AFK mode...
    Actual: This seems right. Useful addition seems correct. I would say that the only thing I would a...

============================================================
SUMMARY
============================================================
Filler removal rate:       3/5 (60%)
Self-correction rate:      0/2 (0%)
Meaning preservation rate: 8/12 (67%)
Format compliance rate:    12/12 (100%)
Passthrough accuracy:      2/2 (100%)
Echo-back count:           4/12

Latency (all):   p50=700ms  p95=2376ms  p99=2965ms
Latency (<=30w): p95=899ms
Token throughput: mean=25.3 tok/s
Peak memory: 5.08 GB

============================================================
HYPOTHESIS CRITERIA SCORING
============================================================

  --- Success Criteria ---
  [     FAIL] Filler removal >= 4/5: 3/5
  [     FAIL] Self-correction >= 2/2: 0/2
  [     FAIL] Meaning preservation >= 10/12: 8/12
  [     PASS] Format compliance >= 11/12: 12/12
  [     PASS] Clean passthrough 2/2: 2/2
  [     PASS] Latency < 1500ms p95 (<=30w): 899ms
  [     FAIL] Decode speed >= 30 tok/s: 25.3 tok/s
  [     PASS] Memory < 6GB: 5.08 GB

  --- Failure Criteria ---
  [TRIGGERED] Echo-back persistence >= 4/12: 4/12
  [TRIGGERED] Meaning corruption >= 3/12: 3/12
  [       OK] Latency >= 3s typical: 791ms median
  [       OK] Instruction non-compliance >= 3/12: 0/12
  [       OK] Quality regression on clean >= 1/2: 0/2

============================================================
2B vs 4B COMPARISON
============================================================
Metric                              2B Result            4B Result
---------------------------------------------------------------------------
  Filler removal                      2/5 (40%)            3/5 (60%)
  Self-correction                     0/2 (0%)             0/2 (0%)
  Meaning preservation                8/12 (67%)           8/12 (67%)
  Format compliance                   12/12 (100%)         12/12 (100%)
  Passthrough accuracy                2/2 (100%)           2/2 (100%)
  Echo-back count                     ~6/12                4/12
  Latency p95 (<=30w)                 472ms                899ms
  Mean tok/s                          49.1                 25.3
  Peak memory                         2.55 GB              5.08 GB

Results written to: working/reviews/run-20260323T130738Z.json
```
