#!/usr/bin/env python3
"""Tokenizer helper for CleanupCLI.

Commands (one per line on stdin):
  encode <text>     → prints space-separated token IDs
  decode <id> ...   → prints decoded text
  suffix            → prints the suffix token IDs (closing tags + assistant turn)

Uses the real HuggingFace tokenizer. Zero ambiguity.
"""
import sys
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-0.8B"
SUFFIX = "</transcription>\n<|im_end|>\n<|im_start|>assistant\n"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
suffix_ids = tokenizer.encode(SUFFIX)

# Print ready signal
print("ready", flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    if line.startswith("encode "):
        text = line[7:]
        ids = tokenizer.encode(text)
        print(" ".join(str(i) for i in ids), flush=True)

    elif line.startswith("decode "):
        ids = [int(x) for x in line[7:].split()]
        text = tokenizer.decode(ids, skip_special_tokens=False)
        # Use repr-safe output (newlines as \n) so Swift can parse
        print(text, flush=True)

    elif line == "suffix":
        print(" ".join(str(i) for i in suffix_ids), flush=True)

    elif line == "quit":
        break

    else:
        print(f"ERROR unknown command: {line}", flush=True)
