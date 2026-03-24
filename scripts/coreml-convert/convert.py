"""Convert Qwen3.5 directly from HuggingFace to CoreML.

No custom model reimplementation — traces the original HF model and
converts with coremltools + our op patches.

Usage:
    cd /Users/jud/Projects/operator
    /tmp/coreml-venv/bin/python scripts/coreml-convert/convert.py \
        Qwen/Qwen3.5-0.8B --seq-len 64 --output models/
"""

import argparse
import sys
import os

# Apply patches before any coremltools import
sys.path.insert(0, os.path.dirname(__file__))
import patches  # noqa: F401, E402

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np

# Patch DeltaNet BEFORE importing the model
from functional_deltanet import patch_transformers
patch_transformers()

from transformers import AutoModelForCausalLM, AutoTokenizer


class PrefillWrapper(nn.Module):
    """Wraps HF model for tracing — returns only last-token logits.

    Pre-computes attention mask, position IDs, and RoPE cos/sin as
    buffers so the trace doesn't hit dynamic construction code with
    in-place ops that coremltools can't handle.
    """

    def __init__(self, model, seq_len):
        super().__init__()
        self.model = model
        self.seq_len = seq_len
        self.register_buffer("position_ids", torch.arange(seq_len).unsqueeze(0))
        self.register_buffer("cache_position", torch.arange(seq_len))

        # Pre-compute causal attention mask (0=attend, -inf=mask)
        causal = torch.tril(torch.ones(1, 1, seq_len, seq_len))
        attn_mask = torch.where(causal.bool(), 0.0, float("-inf"))
        self.register_buffer("attention_mask", attn_mask)

        # Pre-compute RoPE embeddings to avoid in-place ops during trace.
        # Run the rotary embedding once to capture cos/sin caches.
        with torch.no_grad():
            rotary = model.model.rotary_emb
            # Force the rotary embedding to compute for our seq_len
            dummy_ids = torch.zeros(1, seq_len, dtype=torch.long)
            dummy_pos = torch.arange(seq_len).unsqueeze(0)
            # The rotary emb forward computes cos/sin from position_ids
            cos, sin = rotary(
                torch.zeros(1, seq_len, model.config.hidden_size),
                dummy_pos,
            )
            self.register_buffer("rope_cos", cos)
            self.register_buffer("rope_sin", sin)

        # Monkey-patch the rotary embedding to return our pre-computed values
        self._patch_rope()

    def _patch_rope(self):
        """Replace the rotary embedding forward with our pre-computed buffers."""
        wrapper = self
        original_forward = self.model.model.rotary_emb.forward

        def patched_forward(x, position_ids):
            return wrapper.rope_cos, wrapper.rope_sin

        self.model.model.rotary_emb.forward = patched_forward

    def forward(self, input_ids):
        out = self.model(
            input_ids=input_ids,
            position_ids=self.position_ids,
            cache_position=self.cache_position,
            attention_mask=self.attention_mask,
            use_cache=False,
        )
        return out.logits[:, -1:, :]


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3.5 to CoreML")
    parser.add_argument("model_id", help="HuggingFace model ID (e.g. Qwen/Qwen3.5-0.8B)")
    parser.add_argument("--seq-len", type=int, default=64, help="Fixed sequence length")
    parser.add_argument("--output", default="models/", help="Output directory")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 instead of FP16")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 1. Load
    print(f"Loading {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # 2. Wrap and trace
    wrapper = PrefillWrapper(model, args.seq_len)
    wrapper.eval()
    sample = torch.zeros((1, args.seq_len), dtype=torch.long)

    print(f"Tracing (seq_len={args.seq_len})...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (sample,), strict=False)

    # 3. Verify trace matches
    test_text = "Hello, how are you?"
    test_ids = tokenizer(test_text, return_tensors="pt")["input_ids"]
    padded = torch.zeros(1, args.seq_len, dtype=torch.long)
    padded[0, -test_ids.shape[1]:] = test_ids[0]

    with torch.no_grad():
        ref = wrapper(padded)
        tr = traced(padded)
        ref_top = torch.topk(ref[0, 0], 5).indices.tolist()
        tr_top = torch.topk(tr[0, 0], 5).indices.tolist()
        assert ref_top == tr_top, f"Trace mismatch: {ref_top} vs {tr_top}"
        print(f"Trace verified (top 5: {ref_top})")

    # 4. Convert
    precision = ct.precision.FLOAT32 if args.fp32 else ct.precision.FLOAT16
    print(f"Converting to CoreML ({precision})...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_ids", shape=(1, args.seq_len), dtype=np.int32)],
        outputs=[ct.TensorType(name="logits")],
        compute_precision=precision,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )

    # 5. Save
    slug = args.model_id.replace("/", "_").replace(".", "_")
    out_path = os.path.join(args.output, f"{slug}_seq{args.seq_len}.mlpackage")
    mlmodel.save(out_path)
    print(f"Saved to {out_path}")

    # 6. Verify CoreML output
    print("Verifying CoreML model...")
    coreml_out = mlmodel.predict({"input_ids": padded.numpy().astype(np.int32)})
    coreml_logits = coreml_out["logits"]
    coreml_top = np.argsort(coreml_logits[0, 0])[-5:][::-1].tolist()
    print(f"CoreML top 5: {coreml_top}")
    if coreml_top == ref_top:
        print("CoreML output matches PyTorch!")
    else:
        print(f"WARNING: CoreML output differs from PyTorch (ref: {ref_top})")


if __name__ == "__main__":
    main()
