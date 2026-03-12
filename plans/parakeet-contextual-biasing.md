# Task: Add Contextual Vocabulary Biasing to Parakeet ASR

## Objective

Add contextual biasing (hotword boosting) to Parakeet TDT so that domain-specific vocabulary from Claude Code sessions and visible UI text improves transcription accuracy. Terms like session names ("sudo", "devbox"), file paths, function names, and technical jargon should be transcribed correctly when the acoustic signal is ambiguous.

## Background

Operator is a voice-first orchestration layer for Claude Code sessions on macOS. Users dictate into text fields or speak commands to agents. The STT engine (Parakeet TDT 0.6B CoreML) uses pure greedy argmax decoding with no vocabulary biasing. This causes frequent misrecognitions of technical terms: "pseudo" instead of "sudo", "reddis" instead of "redis", "pie test" instead of "pytest", etc.

## Available Vocabulary Sources

Operator has two sources of contextual vocabulary at runtime:

### 1. Claude Code Session State (from Operator's SessionRegistry)

Each registered session provides:
- `name: String` — session name (e.g. "sudo", "devbox")
- `cwd: String` — working directory path (e.g. "/Users/jud/Projects/operator")
- `context: String` — summary of what the session is working on
- `recentMessages: [SessionMessage]` — recent conversation history (role + text)

File: `Operator/Sources/State/SessionTypes.swift`

### 2. Visible UI Vocabulary (from HarnessCore)

`HarnessSession.dictationContext()` returns:
- `visibleVocabulary: [String]` — up to 200 words from AX labels + OCR of visible windows
- `focusedElement.value: String?` — current text in the focused field
- `windowTitle: String?`

File: `harness/Sources/HarnessCore/HarnessSession.swift:965`

## Biasing Insertion Point

The biasing must happen in speech-swift's `TDTGreedyDecoder.decode()`:

```
File: speech-swift/Sources/ParakeetASR/TDTGreedyDecoder.swift
Lines 96-99:

let tokenLogits = jointOut.featureValue(for: "token_logits")!.multiArrayValue!
let durationLogits = jointOut.featureValue(for: "duration_logits")!.multiArrayValue!
let tokenId = argmax(tokenLogits, count: config.vocabSize + 1, floatBuf: argmaxBuf)
```

The `tokenLogits` is an `MLMultiArray` of Float16 with `config.vocabSize + 1` (8194) elements. The pointer is accessed as `array.dataPointer.assumingMemoryBound(to: Float16.self)`. Before the `argmax` call, we can add a bias value to specific token indices to boost their probability.

## Vocabulary → Token ID Mapping

Parakeet uses SentencePiece tokenization. The vocabulary is loaded from `vocab.json`:

```
File: speech-swift/Sources/ParakeetASR/Vocabulary.swift

Format: {"0": "▁the", "1": "▁a", "274": "▁sudo", ...}
- Token IDs 0-273 are special tokens (language tags, control tokens)
- Token IDs 274+ are text tokens
- "▁" (U+2581) prefix indicates word boundary (start of word)
```

`ParakeetVocabulary` currently only has `idToToken: [Int: String]`. You'll need a reverse index `tokenToId` or a method to look up token IDs by word/subword.

SentencePiece is subword: "kubernetes" might be tokenized as ["▁kub", "ern", "etes"]. Biasing needs to handle multi-token words — boost the first subword token, and when it's emitted, boost the next expected subword, etc. This is called **prefix tree biasing** or **trie-based biasing**.

## Implementation Plan

### Phase 1: Simple Single-Token Biasing (speech-swift)

Modify speech-swift to accept and apply bias tokens:

1. **Add reverse vocab index** to `ParakeetVocabulary`:
   - `tokenToId: [String: Int]` — reverse of `idToToken`
   - `func tokenIds(for word: String) -> [Int]` — returns token IDs whose text matches or starts with the word

2. **Add bias parameter** to `TDTGreedyDecoder.decode()`:
   ```swift
   func decode(encoded: MLMultiArray, encodedLength: Int, biasTokenIds: Set<Int>? = nil, biasWeight: Float16 = 3.0) throws -> [Int]
   ```

3. **Apply bias before argmax** (in the decode loop):
   ```swift
   // After getting tokenLogits, before argmax:
   if let biasIds = biasTokenIds {
       let ptr = tokenLogits.dataPointer.assumingMemoryBound(to: Float16.self)
       for id in biasIds {
           ptr[id] += biasWeight
       }
   }
   ```

4. **Thread bias through the API**:
   - `ParakeetASRModel.transcribeAudio()` gains `biasTokenIds: Set<Int>?` parameter
   - `ParakeetASRModel.tokenIds(forWords:)` public method to convert words → token IDs using the vocabulary

### Phase 2: Multi-Token (Trie) Biasing (speech-swift)

For multi-subword terms, implement prefix tree tracking:

1. Build a trie from bias vocabulary where each path is a sequence of token IDs
2. Track active trie positions during decoding
3. When a biased token is emitted, advance the trie pointer and boost the next expected tokens
4. Reset trie branches on mismatch

This is the approach used by NVIDIA NeMo's contextual biasing and Google's LODR. It's more complex but handles compound terms correctly.

### Phase 3: Operator Integration

Wire vocabulary collection into the transcription pipeline:

1. **Collect vocabulary** in `ParakeetEngine` or `ParakeetSegmentTranscriber`:
   - Session names, cwd path components, context words from `SessionRegistry.allSessions()`
   - Visible vocabulary from `HarnessSession.dictationContext()`
   - Deduplicate, filter common English words (don't bias "the", "and", etc.)

2. **Convert to token IDs** once per session (not per segment):
   - Call `model.tokenIds(forWords: vocabularyWords)` after model load
   - Cache the `Set<Int>` until vocabulary changes

3. **Pass to decoder** on each `transcribeAudio()` call

Integration files:
- `Operator/Sources/Voice/ParakeetSegmentTranscriber.swift` — pass bias tokens to `transcribeAudio()`
- `Operator/Sources/Voice/ParakeetEngine.swift` — collect vocabulary, convert to token IDs
- `Operator/Sources/State/AccessibilityQueryService.swift` — already has HarnessCore integrated

## Key Files

| File | Role |
|------|------|
| `speech-swift/Sources/ParakeetASR/TDTGreedyDecoder.swift` | Decode loop — bias insertion point (line 99) |
| `speech-swift/Sources/ParakeetASR/Vocabulary.swift` | Token ↔ word mapping |
| `speech-swift/Sources/ParakeetASR/ParakeetASR.swift` | Public API — `transcribeAudio()` |
| `speech-swift/Sources/ParakeetASR/Configuration.swift` | `vocabSize`, `blankTokenId` |
| `Operator/Sources/Voice/ParakeetSegmentTranscriber.swift` | Calls `transcribeAudio()` per segment |
| `Operator/Sources/Voice/ParakeetEngine.swift` | Streaming STT engine, has access to `SessionRegistry` |
| `Operator/Sources/State/SessionTypes.swift` | Session state with name/cwd/context/recentMessages |
| `harness/Sources/HarnessCore/HarnessSession.swift` | `dictationContext()` with visible vocabulary |

## Constraints

- Bias weight must be tunable. Too high → hallucinated vocabulary words inserted where they don't belong. Too low → no effect. Start with +3.0 in logit space, test empirically.
- Logits are Float16 — bias arithmetic must use Float16.
- The decode loop runs per-frame and is latency-critical (~0.03 RTF). The bias application must be O(|biasTokenIds|) per frame, not O(vocabSize). For typical vocabulary of 50-200 words mapping to maybe 500 tokens, iterating the bias set is fine.
- Don't bias blank token (config.blankTokenId) or special tokens (IDs < 274).
- Vocabulary should be refreshed between PTT sessions (not between segments within a session).
- Phase 1 (single-token biasing) is valuable on its own — many technical terms ("sudo", "redis", "nginx", "pytest", "webpack") are single SentencePiece tokens. Ship phase 1 first.

## Testing

- Unit test in speech-swift: transcribe audio with and without bias, verify biased term appears
- Benchmark: verify no measurable latency regression with 500-token bias set
- Integration test in Operator: register a session named "sudo", speak "tell sudo to run the tests", verify transcription contains "sudo" not "pseudo"
