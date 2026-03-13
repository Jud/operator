/// Post-transcription correction using known vocabulary.
///
/// For each word in the transcription output, if a vocabulary term is within
/// edit distance 2 (case-insensitive), the vocabulary term is substituted.
/// Words shorter than 4 characters are skipped to avoid false positives on
/// common short words like "the", "is", "go".
public enum VocabularyCorrector {
    /// Correct transcribed text against a known vocabulary.
    ///
    /// - Parameters:
    ///   - text: The raw transcription output.
    ///   - vocabulary: Domain-specific terms to match against.
    /// - Returns: Corrected text with vocabulary substitutions applied.
    public static func correct(text: String, vocabulary: [String]) -> String {
        guard !vocabulary.isEmpty else {
            return text
        }

        let lowercaseVocab = vocabulary.map { ($0, $0.lowercased()) }
        let words = text.split(separator: " ", omittingEmptySubsequences: false)

        let corrected = words.map { word -> String in
            let wordStr = String(word)
            let wordLower = wordStr.lowercased()

            guard wordLower.count >= 4 else {
                return wordStr
            }

            var bestMatch: String?
            var bestDistance = Int.max

            for (original, vocabLower) in lowercaseVocab {
                if abs(wordLower.count - vocabLower.count) > 2 { continue }

                let dist = editDistance(wordLower, vocabLower)
                if dist < bestDistance && dist <= 2 && dist > 0 {
                    bestDistance = dist
                    bestMatch = original
                }
            }

            return bestMatch ?? wordStr
        }

        return corrected.joined(separator: " ")
    }

    /// Compute Levenshtein edit distance between two strings.
    public static func editDistance(_ lhs: String, _ rhs: String) -> Int {
        let aChars = Array(lhs)
        let bChars = Array(rhs)
        let lenA = aChars.count
        let lenB = bChars.count

        if lenA == 0 {
            return lenB
        }
        if lenB == 0 {
            return lenA
        }

        var prev = Array(0...lenB)
        var curr = [Int](repeating: 0, count: lenB + 1)

        for idx in 1...lenA {
            curr[0] = idx
            for jdx in 1...lenB {
                if aChars[idx - 1] == bChars[jdx - 1] {
                    curr[jdx] = prev[jdx - 1]
                } else {
                    curr[jdx] = 1 + min(prev[jdx], curr[jdx - 1], prev[jdx - 1])
                }
            }
            swap(&prev, &curr)
        }

        return prev[lenB]
    }
}
