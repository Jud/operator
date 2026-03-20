---
description: Run transcription fixtures, investigate failures, and fix them one by one
---

# Fix Transcription Fixtures

Work through accumulated transcription test fixtures, find failures, investigate root causes, and fix them.

## Step 1: List Fixtures

List all fixtures in the test-fixtures directory:

```bash
ls ~/Library/Application\ Support/Operator/test-fixtures/*.wav 2>/dev/null | while read f; do
    name=$(basename "$f" .wav)
    expected="$(dirname "$f")/$name.expected.txt"
    if [ -f "$expected" ]; then
        words=$(wc -w < "$expected" | tr -d ' ')
        dur=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$f" 2>/dev/null || echo "?")
        echo "  $name (${dur}s, ${words} words)"
    else
        echo "  $name (NO EXPECTED TEXT)"
    fi
done
```

Report the count: total fixtures, how many have `auto-` prefix (auto-captured), how many are manually captured.

## Step 2: Run Tests

Run the full transcription test suite:

```bash
cd Operator && swift run TranscriptionTests
```

Capture the output. Identify which fixtures pass and which fail, with their similarity percentages.

## Step 3: For Each Failure

Work through failures one at a time, starting with the lowest similarity (worst regression).

For each failing fixture:

### 3a: Understand the failure

- What's the expected text vs actual text?
- What kind of content loss? (truncation, missing sentences, wrong words)
- How long is the audio? Short (<3s), medium (3-15s), long (>30s)?
- Is it a streaming pipeline issue or a batch decode issue? (Batch should be 100% — if not, it's a WhisperKit/model issue, not our code)

### 3b: Trace the root cause

Read the TranscriptionSession code at `Operator/Sources/Voice/TranscriptionSession.swift` and the WhisperKitEngine coordinator at `Operator/Sources/Voice/WhisperKitEngine.swift`.

Consider these common failure patterns:
- **Confirmation stall**: frontier stops advancing, zero-progress cycling
- **Strategy transition**: word-level → segment-level too early or too late
- **Tail decode failure**: final decode returns empty or truncated
- **Filter pipeline**: hallucination filter removing valid words
- **Empty guard**: valid transcriptions being skipped as "empty after filtering"

### 3c: Write a failing unit test first

Before fixing, write a Level 1 algorithmic test in `Tests/OperatorTests/TranscriptionSessionTests.swift` that reproduces the failure pattern with synthetic data. The test should fail before the fix and pass after.

If the failure is in the coordinator (not the session), document the gap and fix it anyway — but note that a coordinator-level unit test isn't possible without mockable WhisperKit.

### 3d: Apply the fix

Make the minimal code change to fix the issue. Only change what's necessary.

### 3e: Verify

Run the full test suite:
```bash
cd Operator && swift test && swift run TranscriptionTests
```

ALL unit tests must pass. The previously-failing fixture must now pass. No other fixtures should regress.

If a fix causes another fixture to regress, investigate the interaction before proceeding.

## Step 4: Summary

After working through all failures, report:
- How many fixtures were fixed
- What root causes were found
- Any remaining failures and why they can't be fixed (e.g., WhisperKit model limitation)
- Any new unit tests added

## Step 5: Speak

Use the `speak` MCP tool to summarize what was fixed.
