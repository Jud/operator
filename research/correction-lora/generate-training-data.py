#!/usr/bin/env python3
"""Generate correction training data from Operator routing traces.

Reads transcriptions from the routing traces JSONL, sends them to Claude
for correction, and outputs JSONL training pairs for MLX LoRA fine-tuning.

Usage:
    # Generate training data (requires ANTHROPIC_API_KEY)
    python generate-training-data.py

    # Dry run — just extract and preview transcriptions
    python generate-training-data.py --dry-run

    # Limit number of examples
    python generate-training-data.py --limit 50

    # Use a specific model
    python generate-training-data.py --model claude-sonnet-4-20250514
"""

import argparse
import json
import os
import sys
from pathlib import Path

TRACES_PATH = Path.home() / "Library/Application Support/Operator/routing-traces/traces.jsonl"
OUTPUT_DIR = Path(__file__).parent / "data"

SYSTEM_PROMPT = """You are a speech-to-text correction assistant. You will receive raw transcriptions
from a voice dictation system. Your job is to produce a minimally corrected version.

Rules:
- Fix obvious transcription errors (misheard words, missing punctuation)
- Remove filler words (um, uh, like when used as filler, you know)
- Fix capitalization and basic punctuation
- Preserve the speaker's exact meaning and tone
- Do NOT rephrase, restructure, or add content
- Do NOT make the text more formal than the speaker intended
- Keep contractions if the speaker would naturally use them
- If the transcription looks correct, return it unchanged

Output ONLY the corrected text, nothing else."""


def load_transcriptions(path: Path, min_words: int = 3) -> list[str]:
    """Load unique transcriptions from routing traces."""
    texts = set()
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            t = d.get("transcribedText", "").strip()
            if t and len(t.split()) >= min_words:
                texts.add(t)
    return sorted(texts)


def generate_correction(client, text: str, model: str) -> str | None:
    """Send a transcription to Claude for correction."""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return None


def make_training_example(raw: str, corrected: str) -> dict:
    """Format a training example in MLX chat format."""
    return {
        "messages": [
            {"role": "system", "content": "Correct the speech transcription. Output only the corrected text."},
            {"role": "user", "content": raw},
            {"role": "assistant", "content": corrected},
        ]
    }


def is_meaningful_correction(raw: str, corrected: str) -> bool:
    """Filter out cases where correction is identical or drastically different."""
    if not corrected:
        return False
    # Skip if correction is way longer (hallucination)
    if len(corrected) > len(raw) * 2:
        return False
    # Skip if correction is way shorter (lost content)
    if len(corrected) < len(raw) * 0.3 and len(raw) > 20:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate correction training data")
    parser.add_argument("--dry-run", action="store_true", help="Just preview transcriptions")
    parser.add_argument("--limit", type=int, default=200, help="Max examples to generate")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model to use")
    parser.add_argument("--min-words", type=int, default=3, help="Min words per transcription")
    args = parser.parse_args()

    if not TRACES_PATH.exists():
        print(f"No traces file at {TRACES_PATH}")
        sys.exit(1)

    texts = load_transcriptions(TRACES_PATH, min_words=args.min_words)
    print(f"Found {len(texts)} unique transcriptions (>= {args.min_words} words)")

    if args.dry_run:
        for t in texts[:20]:
            print(f"  \"{t[:100]}\"")
        print(f"\n  ... and {len(texts) - 20} more")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print("pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    texts = texts[: args.limit]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "all.jsonl"
    train_path = OUTPUT_DIR / "train.jsonl"
    valid_path = OUTPUT_DIR / "valid.jsonl"

    examples = []
    skipped = 0

    for i, raw in enumerate(texts):
        print(f"[{i+1}/{len(texts)}] \"{raw[:60]}...\"" if len(raw) > 60 else f"[{i+1}/{len(texts)}] \"{raw}\"")
        corrected = generate_correction(client, raw, args.model)

        if not is_meaningful_correction(raw, corrected):
            print(f"  SKIP (filter)")
            skipped += 1
            continue

        if corrected == raw:
            print(f"  IDENTICAL (keeping as-is example)")
        else:
            print(f"  -> \"{corrected[:60]}\"")

        example = make_training_example(raw, corrected)
        examples.append(example)

    # Write all examples
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    # Split 80/20
    split = int(len(examples) * 0.8)
    train = examples[:split]
    valid = examples[split:]

    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    with open(valid_path, "w") as f:
        for ex in valid:
            f.write(json.dumps(ex) + "\n")

    print(f"\nDone: {len(examples)} examples ({skipped} skipped)")
    print(f"  {train_path}: {len(train)} training examples")
    print(f"  {valid_path}: {len(valid)} validation examples")


if __name__ == "__main__":
    main()
