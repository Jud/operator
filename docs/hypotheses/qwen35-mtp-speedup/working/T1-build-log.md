### T1: Build the MTP probe script
**Date:** 2026-03-24
**Status:** complete
**Files changed:**
- `docs/hypotheses/qwen35-mtp-speedup/working/mtp_probe.py` -- MTP probe script (full implementation)

**Notes:**

Critical bug found and fixed during development: the fc layer input ordering
must be `cat(e_norm, h_norm)` (embed first, hidden second), matching the
Deepseek-V3 `eh_proj` naming convention. The task spec and hypothesis doc
both described it as `cat(h_norm, e_norm)` (hidden first), which produced 0%
hit rate because the lm_head (tied to embed_tokens) naturally reconstructed
the embedding token through the residual path. Reversing the ordering to
match the checkpoint's actual training convention yielded 77.6% hit rate.

The MTP module requires full-sequence context (not just the last position) --
its attention layer is causal over the accumulated sequence of fused
(hidden, embedding) representations. The probe accumulates these during
both prefill and generation.

The model generates in "thinking mode" (<think>...</think>) before answering,
so the generated text doesn't match the meta.json expected tokens exactly.
This is fine per the spec ("minor wording variations are fine") and the MTP
hit rate is measured on whatever tokens the model actually generates.

**Baseline results (before changes):**
No pre-existing Python script to verify. The Swift project builds/tests are
not applicable to this standalone Python prototype.

**Post-change results (after changes):**
```
/tmp/coreml-venv/bin/python mtp_probe.py
```
Script runs without errors. Output:
- Hit rate: 38/49 = 77.6%
- MTP top-5 tokens are plausible English words (not random IDs)
- Per-token hit/miss table printed
- Incremental MTP latency (PyTorch CPU): ~22ms
- Estimated CoreML/ANE MTP latency: ~1.4ms
- Projected speedup (CoreML/ANE): 1.51x (87 tok/s vs 58 tok/s baseline)
