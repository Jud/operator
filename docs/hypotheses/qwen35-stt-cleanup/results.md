# Results: Qwen3.5-2B STT Cleanup

**Verdict: DISPROVEN**

**Date:** 2026-03-23
**Model:** mlx-community/Qwen3.5-2B-8bit via mlx-lm (Python)
**Machine:** Apple Silicon (Darwin 24.6.0)

## Summary

Qwen3.5-2B cannot reliably clean raw STT transcriptions. The model overwhelmingly echoes input verbatim rather than transforming it. Self-corrections are never resolved (0/2), filler removal works only 40% of the time, and the model fails to condense rambling speech at all. While latency and memory are acceptable, and the model follows format instructions (no markdown, no meta-commentary), the core cleanup quality is far below the bar needed for dictation.

## Success Criteria Scoring

| Criterion | Required | Actual | Result |
|---|---|---|---|
| Filler removal | >= 9/10 | 2/5 (40%) | **FAIL** |
| Self-correction handling | >= 8/10 | 0/2 (0%) | **FAIL** |
| Grammar & punctuation | >= 9/10 | 12/12 (100%) | PASS |
| Meaning preservation | >= 9/10 | 8/12 (67%) | **FAIL** |
| Latency < 800ms p95 (<=30w) | < 800ms | 471ms | PASS |
| Decode speed >= 100 tok/s | >= 100 | 49.0 tok/s | **FAIL** |
| Memory < 4GB | < 4GB | 2.55 GB | PASS |

**3/7 success criteria passed. 4/7 failed.**

## Failure Criteria Scoring

| Criterion | Threshold | Actual | Result |
|---|---|---|---|
| Meaning corruption >= 3/10 | >= 3 cases | 3/12 | **TRIGGERED** |
| Latency >= 1.5s typical | >= 1.5s median | 409ms | OK |
| Instruction non-compliance >= 3/10 | >= 3 cases | 0/12 | OK |
| Quality regression on clean >= 2/10 | >= 2 cases | 0/2 | OK |

**1/4 failure criteria triggered (meaning corruption).**

## Detailed Observations

### The model echoes input rather than cleaning it

The dominant failure mode is the model returning the input essentially unchanged. In 5 of 12 test cases, the output was nearly identical to the raw input:

- **correction-day:** Input "Tell him Tuesday -- actually no, Wednesday works better" was returned verbatim. The model did not resolve the self-correction at all.
- **correction-team:** Input echoed verbatim. No self-correction resolution.
- **ramble-research-guardrails:** The entire 150-word ramble with dense filler words ("like", "you know", "I guess", "I don't know") was returned almost unchanged. Only minor fixes (removed a capitalization error "For" -> "for").
- **command-timer:** "Hey Siri set a timer for 10 minutes" returned unchanged -- the model should have stripped the wake word.
- **ramble-codex-background:** Fillers partially removed ("um") but discourse markers ("Yeah", "Like", "And then") left intact. The output was still 75 tokens vs the 42-token expected cleanup.

### What worked

- **Clean passthrough:** Both already-clean inputs were returned unchanged (2/2). The model does not degrade clean text.
- **Format compliance:** 12/12 outputs were raw text, no markdown, no meta-commentary, no quotes. The model follows output format instructions well.
- **Simple filler removal:** When fillers were straightforward ("Um, I think we should, uh, go with option B"), the model handled them correctly. The filler-with-correction case also worked well.
- **Memory:** At 2.55 GB peak, well within the 4 GB budget.
- **Latency for short inputs:** p95 of 471ms for inputs <= 30 words is well under the 800ms target.

### What failed

- **Self-correction is entirely absent.** The model has no ability to identify "actually no" or "or wait" as self-corrections and keep only the final version. This is the most complex semantic task and the 2B model simply cannot do it.
- **Filler removal is inconsistent.** "So" at the start of a sentence was preserved. Discourse "like" in "Like, I'm obviously doing..." was preserved. The model only removes the most obvious fillers (um, uh) but leaves discourse markers intact.
- **Ramble condensation does not happen.** The model appears to treat long inputs as "already clean enough" and returns them with minimal changes. It lacks the ability to identify and remove redundant phrasing.
- **Decode speed is 49 tok/s**, less than half the 100 tok/s target. This was consistent across both runs. The published benchmarks for Qwen3.5-2B may assume different hardware or batch sizes.

### Throughput note

The 49 tok/s decode speed is notably low for a 2B parameter model on Apple Silicon. This may be due to:
- The Gated DeltaNet architecture in Qwen3.5 having different compute characteristics than standard transformers
- mlx-lm overhead vs native MLX inference
- Single-request inference (no batching)

Even with optimized inference, the quality failures make throughput optimization irrelevant for this use case.

## Conclusion

Qwen3.5-2B at 8-bit quantization lacks the reasoning capability to perform meaningful speech cleanup. The model can handle trivial filler word removal but fails at the semantic tasks that make cleanup valuable: self-correction resolution, ramble condensation, and discourse marker identification. This is likely a model size limitation -- a 2B model does not have enough capacity for the nuanced text understanding required.

### Alternatives to consider

1. **Larger local model (7B-8B):** A Qwen3.5-7B or similar may have the reasoning capacity for self-correction and condensation, but will have higher memory (~6-8 GB) and latency costs. Worth testing if latency can stay under 800ms.
2. **Rule-based filler removal + LLM for self-correction only:** Use regex/simple rules for the easy cases (um, uh, "you know"), and only invoke the LLM for inputs that appear to contain self-corrections. This reduces latency for common cases.
3. **Cloud API (Claude Haiku):** If latency budget allows ~500ms network round-trip, a capable cloud model would handle all cleanup cases reliably. This trades offline capability for quality.
4. **Fine-tuned small model:** A 2B model fine-tuned specifically on speech-to-clean-text pairs might succeed where the general-purpose model fails. Requires training data collection.

## Raw Data

- Run 1: `docs/hypotheses/qwen35-stt-cleanup/working/reviews/run-20260323T051736Z.json`
- Run 2: `docs/hypotheses/qwen35-stt-cleanup/working/reviews/run-20260323T051756Z.json`
- Stdout capture: `docs/hypotheses/qwen35-stt-cleanup/working/reviews/run-20260323T051736Z.txt`
