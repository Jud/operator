# Task: Post-Transcription Vocabulary Correction

## Objective

After Parakeet produces a transcript, automatically correct misrecognized words by matching them against known vocabulary from session context and visible UI. Uses Double Metaphone phonetic encoding computed at runtime — no bundled dictionaries, no hand-crafted rules.

## Problem

Parakeet TDT uses greedy decoding with no vocabulary awareness. When the user says "sudo" (their session name), Parakeet outputs "pseudo" because it's a more common English word. We know the target vocabulary at runtime (session names, visible code identifiers) but we don't know in advance what Parakeet will mis-transcribe them as.

## Approach: Phonetic Matching with Double Metaphone

Double Metaphone is an algorithm (not a dictionary) that computes phonetic codes from any word at runtime. It handles silent letters, non-English origins, and produces two codes (primary + secondary) for ambiguous pronunciations.

Key property: `DoubleMetaphone("pseudo")` and `DoubleMetaphone("sudo")` produce matching codes because the algorithm drops the silent P in "ps-" clusters — both encode starting with S.

### Why not other approaches

| Approach | Problem |
|----------|---------|
| Soundex | Keeps first letter literally — pseudo=P230, sudo=S300, never matches |
| CMUdict + manual phonemes | Static dictionary, can't handle dynamic vocabulary |
| Neural G2P | Heavyweight dependency (ONNX or MLX model), overkill for this |
| Hand-crafted replacement rules | Can't predict what ASR will misrecognize |
| Character edit distance alone | "pseudo" and "sudo" are 3 edits apart — too far to confidently match without phonetic signal |

## Architecture

```
PTT Press
    │
    ├─ Parakeet transcribes audio → raw transcript
    │
    ├─ VocabularyIndex built (once per PTT session)
    │   ├─ SessionRegistry.allSessions() → session names, cwd components
    │   ├─ DictationContext → visible identifiers from UI
    │   └─ Each term → Double Metaphone codes computed on the fly
    │
    └─ PhoneticCorrector.correct(transcript, vocabularyIndex)
        ├─ For each word: compute metaphone code, look up matching vocab entries
        ├─ Confirm match with character edit distance (hybrid scoring)
        ├─ Sliding window for multi-word matches (identifiers)
        └─ Return corrected transcript
```

## Components

### 1. VocabularyIndex

Built once per PTT session from runtime context. Maps phonetic codes to vocabulary entries.

```swift
struct VocabularyEntry {
    let term: String                    // "sudo", "AVAudioUnitTimePitch"
    let metaphonePrimary: String        // "ST"
    let metaphoneSecondary: String?     // "ST"
    let expandedSpokenForm: String?     // nil for simple words, "a v audio unit time pitch" for identifiers
}

struct VocabularyIndex {
    /// Metaphone code → entries with that code
    private var primaryIndex: [String: [VocabularyEntry]]
    private var secondaryIndex: [String: [VocabularyEntry]]

    /// Multi-word spoken forms for identifier matching (longest first)
    private var expandedForms: [(spoken: String, entry: VocabularyEntry)]

    init(terms: [String]) {
        // For each term:
        //   1. Compute Double Metaphone codes
        //   2. Index by primary and secondary codes
        //   3. If identifier (contains uppercase mid-word), expand to spoken form
        //      and compute metaphone codes for each component word
    }

    func candidates(forMetaphoneCode code: String) -> [VocabularyEntry]
}
```

**Vocabulary sources:**

| Source | Terms extracted |
|--------|---------------|
| `SessionRegistry.allSessions()` | `.name` (e.g. "sudo", "devbox") |
| `SessionRegistry.allSessions()` | `.cwd` path components (e.g. "operator", "harness") |
| `DictationContext.visibleVocabulary` | Visible text from AX labels + OCR (up to 200 words) |
| `AccessibilityContext.textBeforeCursor` | Identifiers near cursor |

### 2. PhoneticCorrector

Runs on the joined transcript after all segments are assembled.

```swift
struct PhoneticCorrector {
    let index: VocabularyIndex

    func correct(_ transcript: String) -> String {
        var words = transcript.split(separator: " ").map(String.init)

        // Pass 1: Multi-word identifier matching (longest first)
        // Check sliding windows of 2-6 consecutive words against expanded forms
        replaceIdentifierSpans(&words)

        // Pass 2: Single-word phonetic correction
        for i in words.indices {
            let word = words[i]
            let (primary, secondary) = DoubleMetaphone.encode(word)

            guard let candidates = index.candidates(forMetaphoneCode: primary)
                                   ?? index.candidates(forMetaphoneCode: secondary) else {
                continue
            }

            // Hybrid scoring: metaphone match + character edit distance confirmation
            if let best = bestMatch(word: word, candidates: candidates) {
                words[i] = preserveCase(original: word, replacement: best.term)
            }
        }

        return words.joined(separator: " ")
    }
}
```

**Hybrid scoring** (from Microsoft Research, 87% accuracy):
1. Double Metaphone code matches → candidate
2. Normalized character edit distance as confirmation: `editDistance(a, b) / max(a.count, b.count)`
3. Accept if character distance < 0.5 (metaphone already confirmed phonetic similarity, character distance filters outliers)
4. Short words (≤3 chars) need stricter character threshold (< 0.3) to avoid false positives

### 3. Identifier Expansion

For code identifiers found in visible context, auto-expand to their spoken form:

```swift
func expandIdentifier(_ id: String) -> String {
    // Split on camelCase/PascalCase boundaries
    // Expand uppercase runs letter-by-letter
    // Lowercase everything

    // "AVAudioUnitTimePitch" → "a v audio unit time pitch"
    // "getElementById"       → "get element by id"
    // "URLSession"           → "u r l session"
    // "HTTPSConnection"      → "h t t p s connection"
    // "useState"             → "use state"
}
```

Rules:
- Split before uppercase letter preceded by lowercase: `getElement` → `get Element`
- Split before uppercase letter followed by lowercase when preceded by uppercase: `URLSession` → `URL Session`
- Expand consecutive uppercase as individual letters: `URL` → `U R L`, `AV` → `A V`, `HTTPS` → `H T T P S`
- Exception: don't expand if the uppercase run is the entire word (likely an acronym that's spoken as a word, like "NASA")

Multi-word matching uses case-insensitive substring matching on the transcript against these expanded forms, sorted longest-first to prevent partial matches.

## Integration

### Where vocabulary is collected

`ParakeetEngine.prepare()` — called once when PTT is pressed. Fires an async task to collect vocabulary and build the index. The index is stored on the engine and reused for all segments in that PTT session.

```swift
// ParakeetEngine gains:
private var corrector: PhoneticCorrector?

public func prepare() throws {
    // ... existing reset logic ...
    Task {
        let terms = await vocabularyProvider.collectTerms()
        let index = VocabularyIndex(terms: terms)
        self.corrector = PhoneticCorrector(index: index)
    }
}
```

### Where correction is applied

`ParakeetEngine.finishAndTranscribe()` — after all segments are joined, before returning.

```swift
public func finishAndTranscribe() async -> String? {
    // ... existing segment joining ...
    let text = await transcriber.finishSession(
        sessionID: finalizePayload.sessionID,
        fallbackSamples: finalizePayload.fallback
    )
    // Apply correction
    return text.map { corrector?.correct($0) ?? $0 }
}
```

This is after segment joining so multi-word identifier matches work across segment boundaries, and before the BimodalDecisionEngine sees the text.

### Vocabulary provider

```swift
protocol VocabularyProviding: Sendable {
    func collectTerms() async -> [String]
}
```

Concrete implementation pulls from SessionRegistry + AccessibilityQueryService. Filters out common English stopwords and very short words (≤2 chars) that would cause false positives.

## Dependency

One new SPM dependency: [DoubleMetaphoneSwift](https://github.com/ZebulonRouseFrantzich/DoubleMetaphoneSwift) (MIT license, wraps PostgreSQL C implementation). Alternatively, the Double Metaphone algorithm is well-documented and could be implemented directly in ~300 lines of Swift to avoid the dependency.

## Relationship to Logit Biasing (parakeet-contextual-biasing.md)

These are complementary, not competing:

| Layer | When | What it does | Handles |
|-------|------|-------------|---------|
| Logit biasing (Phase 1-2) | During decoding | Boosts token probabilities for known vocab | Prevents misrecognition at the source |
| Post-transcription correction | After decoding | Phonetic matching on output text | Catches residual errors, handles identifier reassembly |

Ship post-transcription correction first — it's entirely in Operator, no speech-swift changes. Add logit biasing later as defense in depth.

## Key Files

| File | Change |
|------|--------|
| `Sources/Voice/PhoneticCorrector.swift` | **New** — core matching logic |
| `Sources/Voice/VocabularyIndex.swift` | **New** — phonetic index from runtime vocab |
| `Sources/Voice/IdentifierExpander.swift` | **New** — camelCase → spoken form expansion |
| `Sources/Voice/ParakeetEngine.swift` | Add corrector field, wire into prepare() and finishAndTranscribe() |
| `Sources/State/OperatorVocabularyProvider.swift` | **New** — collects terms from SessionRegistry + AccessibilityQueryService |
| `Package.swift` | Add DoubleMetaphoneSwift dependency (or inline implementation) |

## Testing

- Unit test: `DoubleMetaphone.encode("pseudo")` matches `DoubleMetaphone.encode("sudo")`
- Unit test: `expandIdentifier("AVAudioUnitTimePitch")` → `"a v audio unit time pitch"`
- Unit test: corrector with vocabulary `["sudo"]` transforms `"tell pseudo to run tests"` → `"tell sudo to run tests"`
- Unit test: corrector does NOT replace `"the super method"` when vocabulary contains `["sudo"]` (character distance too high despite metaphone similarity)
- Unit test: multi-word identifier matching replaces `"a v audio unit time pitch"` → `"AVAudioUnitTimePitch"`
- Integration test: register session "sudo", speak into Parakeet, verify correction in output
- Benchmark: correction pass on 50-word transcript with 200-term vocabulary index < 1ms

## Constraints

- Vocabulary index must be rebuilt per PTT session (not per segment) — terms don't change mid-utterance
- Keep vocabulary under ~500 terms to limit false positives from metaphone collisions
- Skip correction for words ≤ 2 characters (too many collisions)
- The corrector must handle punctuation attached to words (strip before matching, reattach after)
- Identifier expansion should not expand all-uppercase words shorter than 5 chars (might be spoken as acronyms: "NASA", "ASAP")
