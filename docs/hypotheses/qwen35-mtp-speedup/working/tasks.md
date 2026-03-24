# Hypothesis Test Tasks: Qwen3.5-0.8B MTP Speculative Decoding

**Status:** Planning
**Worktree:** .claude/worktrees/hypothesis-qwen35-mtp-speedup
**Branch:** hypothesis/qwen35-mtp-speedup
**Total Tasks:** 2
**Completed:** 0

## Task Graph

```
T1 (build mtp_probe.py) --> T2 (run and evaluate)
```

## Tasks

### T1. Build the MTP probe script
**Status:** pending
**Depends on:** --
**Files:** `docs/hypotheses/qwen35-mtp-speedup/working/mtp_probe.py`
**Review:** --
**Spec:**

Build `docs/hypotheses/qwen35-mtp-speedup/working/mtp_probe.py` that loads Qwen3.5-0.8B,
extracts MTP weights, and measures hit rate on the cleanup task.

**Use the `/tmp/coreml-venv` venv** (has torch 2.11.0, transformers 5.3.0, safetensors).
Run with: `/tmp/coreml-venv/bin/python mtp_probe.py`

**Model path:** `Qwen/Qwen3.5-0.8B` (cached at `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B`)

**Architecture details the builder MUST get right:**

1. **Load the main model** via `transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", torch_dtype=torch.float32)`. The model class is `Qwen3_5ForConditionalGeneration` but the text backbone is at `model.language_model`. Use `model.eval()` and `torch.no_grad()` throughout.

2. **The MTP module** has this forward pass (matching the training architecture):
   ```
   h_norm = rms_norm(hidden_state, mtp.pre_fc_norm_hidden.weight)
   e_norm = rms_norm(embedding,    mtp.pre_fc_norm_embedding.weight)
   fused  = mtp.fc(cat([h_norm, e_norm], dim=-1))   # Linear(2048 -> 1024)
   x      = transformer_block(fused)                  # 1 full-attention layer
   x      = rms_norm(x, mtp.norm.weight)              # final norm
   logits = x @ embed_tokens.weight.T                 # tied lm_head
   ```

3. **RMSNorm is the Qwen3.5 variant:** `(1.0 + weight) * (x / rms(x))`, where weight is initialized to 0 (so `1+weight` starts at 1). eps=1e-6.

4. **The full-attention layer (mtp.layers.0) has an output gate baked into q_proj:**
   - `q_proj` shape is `[4096, 1024]` = 8 heads * 256 head_dim * 2 (query + gate)
   - After projecting, split the last dim in half: `query_states, gate = chunk(q_proj_out.view(..., num_heads, head_dim*2), 2, dim=-1)`
   - `gate = gate.reshape(batch, seq, -1)` (flatten heads)
   - After attention: `attn_output = attn_output * sigmoid(gate)`
   - Then: `attn_output = o_proj(attn_output)`
   - `o_proj` shape is `[1024, 2048]` because input is 8*256=2048

5. **GQA config:** 8 query heads, 2 KV heads (4 groups). `k_proj` [512,1024], `v_proj` [512,1024].

6. **QK norms:** Both q and k get per-head RMSNorm (`q_norm`, `k_norm`, dim=256, eps=1e-6).

7. **RoPE:** Partial rotary with `partial_rotary_factor=0.25` means only the first 64 dims (of 256 head_dim) get rotated. The rest pass through unchanged. Position IDs for text-only are `pos.unsqueeze(0).expand(3, -1, -1)` (3 copies for MRoPE sections [11,11,10]). Use interleaved MRoPE layout. **Simplest approach:** reuse the model's own `rotary_emb` module to compute cos/sin for the MTP layer's positions. The model stores it at `model.language_model.rotary_emb`.

8. **Residual connection:** The transformer block is pre-norm with residual:
   ```
   residual = fused
   x = input_layernorm(fused)
   x = self_attn(x, ...)
   x = residual + x
   residual = x
   x = post_attention_layernorm(x)
   x = mlp(x)
   x = residual + x
   ```

9. **MLP:** SwiGLU: `down_proj(silu(gate_proj(x)) * up_proj(x))`. Intermediate size 3584.

**Generation loop:**
- Use a real STT cleanup prompt. The cleanup task input is messy speech, output is: `"I think we should delete the local copy because it's going to be symlinked anyway."` (18 tokens).
- Use a simple prompt like: `<|im_start|>user\nClean up this transcription: "Um, I think we should, like, you know, delete the local copy, um, because it's, it's going to be, uh, symlinked anyway."\n<|im_end|>\n<|im_start|>assistant\n`
- Run greedy autoregressive generation (argmax at each step, no sampling).
- At each generation step `t`:
  1. Run the main model forward pass, get hidden state from last layer and predicted token
  2. Feed hidden state + embedding of predicted token into MTP module
  3. MTP predicts the token at position `t+2`
  4. At step `t+1`, compare what the main model actually predicts with what MTP predicted
  5. Record hit/miss
- Time the MTP forward pass separately (use `time.perf_counter`, not cuda events -- this is CPU/MPS).

**Output format:**
```
Step | Main token      | MTP predicted   | Hit | MTP ms
-----|-----------------|-----------------|-----|-------
  0  | I               | think           |  Y  |  1.2
  1  | think           | should          |  N  |  1.1
...
Hit rate: 14/18 = 77.8%
Avg MTP latency: 1.3ms
Projected speedup: 1.78x (effective 103 tok/s vs 58 tok/s baseline)
```

**Weight loading:** Load weights directly from the safetensors file. All MTP keys start with `mtp.` and map directly to the module structure:
- `mtp.pre_fc_norm_hidden.weight` -> RMSNorm weight (dim 1024)
- `mtp.pre_fc_norm_embedding.weight` -> RMSNorm weight (dim 1024)
- `mtp.fc.weight` -> Linear weight (2048 -> 1024)
- `mtp.layers.0.self_attn.{q,k,v,o}_proj.weight` -> attention projections
- `mtp.layers.0.self_attn.{q,k}_norm.weight` -> QK norms (dim 256)
- `mtp.layers.0.input_layernorm.weight` -> pre-attention RMSNorm
- `mtp.layers.0.post_attention_layernorm.weight` -> pre-MLP RMSNorm
- `mtp.layers.0.mlp.{gate_proj,up_proj,down_proj}.weight` -> SwiGLU MLP
- `mtp.norm.weight` -> final RMSNorm before lm_head

**Tied lm_head:** No separate `lm_head` weight. Use `model.language_model.embed_tokens.weight` as the lm_head (transposed matmul: `x @ embed.T`). Shape: [248320, 1024].

**Acceptance criteria:**
- Script runs without errors
- MTP module produces non-garbage logits (top-5 tokens are plausible English words, not random IDs)
- Per-token hit/miss table is printed
- Summary shows hit rate, avg MTP latency, projected speedup
- Generation produces the expected clean output (or close to it -- minor wording variations are fine)

---

### T2. Run the probe and evaluate results
**Status:** in-progress
**Depends on:** T1
**Files:** `docs/hypotheses/qwen35-mtp-speedup/working/results.md`
**Review:** --
**Spec:**

Run the probe script and evaluate the results against the hypothesis criteria.

```bash
cd /Users/jud/Projects/operator/docs/hypotheses/qwen35-mtp-speedup/working
/tmp/coreml-venv/bin/python mtp_probe.py
```

If it crashes, debug and fix `mtp_probe.py` (this is expected -- getting the wiring right may take iteration).

Once it runs successfully, write `results.md` with:

1. **Raw output** from the script (copy-paste the full terminal output)
2. **Evaluation against criteria:**
   - MTP correctness: Do the MTP logits produce sensible tokens? (check top-5)
   - Hit rate: Is it >50%? >30%? What is the exact number?
   - MTP latency: How many ms per step?
   - Projected speedup: Based on hit rate and MTP overhead, what's the effective tok/s?
3. **Verdict:** PROVEN, DISPROVEN, or INCONCLUSIVE with one-line justification
4. **Next steps** (if proven): What would it take to integrate MTP speculative decoding into the CoreML pipeline?

**Acceptance criteria:**
- `mtp_probe.py` runs to completion with no errors
- `results.md` contains the raw output, evaluation, and verdict
- The verdict clearly maps to the success/failure criteria in hypothesis.md
