import Foundation

/// Morphological suffix rules for -s, -ed, -ing.
///
/// Strips suffixes, looks up the stem in the dictionary, and applies
/// phonological rules to derive the correct pronunciation.
enum SuffixRules {
    /// Try all suffix rules. Returns phoneme string if a stem was found.
    static func stemAndLookup(_ word: String, lookup: (String) -> String?) -> String? {
        if let result = stemS(word, lookup: lookup) { return result }
        if let result = stemEd(word, lookup: lookup) { return result }
        if let result = stemIng(word, lookup: lookup) { return result }
        return nil
    }

    // MARK: - Plural/3rd-person -s

    private static func stemS(_ word: String, lookup: (String) -> String?) -> String? {
        guard word.hasSuffix("s") && word.count > 2 else { return nil }

        // -ies → -y (carries → carry)
        if word.hasSuffix("ies") {
            let stem = String(word.dropLast(3)) + "y"
            if let phonemes = lookup(stem) { return phonemes + "z" }
        }

        // -es (boxes → box)
        if word.hasSuffix("es") && word.count > 3 {
            let stem = String(word.dropLast(2))
            if let phonemes = lookup(stem) {
                let last = phonemes.last
                if last == "s" || last == "z" || last == "ʃ" || last == "ʒ" {
                    return phonemes + "ɪz"
                }
                return phonemes + "z"
            }
        }

        // Simple -s (cats → cat)
        let stem = String(word.dropLast(1))
        if let phonemes = lookup(stem) {
            let voiceless: Set<Character> = ["p", "t", "k", "f", "θ"]
            if let last = phonemes.last, voiceless.contains(last) {
                return phonemes + "s"
            }
            return phonemes + "z"
        }

        return nil
    }

    // MARK: - Past tense -ed

    private static func stemEd(_ word: String, lookup: (String) -> String?) -> String? {
        guard word.hasSuffix("ed") && word.count > 3 else { return nil }

        // -ied → -y (carried → carry)
        if word.hasSuffix("ied") {
            let stem = String(word.dropLast(3)) + "y"
            if let phonemes = lookup(stem) { return phonemes + "d" }
        }

        // Doubled consonant (stopped → stop)
        let stemEd = String(word.dropLast(2))
        if stemEd.count >= 2 {
            let chars = Array(stemEd)
            if chars[chars.count - 1] == chars[chars.count - 2] {
                let dedoubled = String(stemEd.dropLast(1))
                if let phonemes = lookup(dedoubled) {
                    return phonemes + edSuffix(phonemes)
                }
            }
        }

        // Direct stem (walked → walk)
        if let phonemes = lookup(stemEd) {
            return phonemes + edSuffix(phonemes)
        }

        return nil
    }

    private static func edSuffix(_ phonemes: String) -> String {
        let last = phonemes.last
        if last == "t" || last == "d" { return "ɪd" }
        let voiceless: Set<Character> = ["p", "k", "f", "θ", "s", "ʃ"]
        if let l = last, voiceless.contains(l) { return "t" }
        return "d"
    }

    // MARK: - Gerund -ing

    private static func stemIng(_ word: String, lookup: (String) -> String?) -> String? {
        guard word.hasSuffix("ing") && word.count > 4 else { return nil }

        let stem = String(word.dropLast(3))

        // Doubled consonant (running → run)
        if stem.count >= 2 {
            let chars = Array(stem)
            if chars[chars.count - 1] == chars[chars.count - 2] {
                let dedoubled = String(stem.dropLast(1))
                if let phonemes = lookup(dedoubled) { return phonemes + "ɪŋ" }
            }
        }

        // Direct stem (walking → walk)
        if let phonemes = lookup(stem) { return phonemes + "ɪŋ" }

        // Silent-e stem (loving → love)
        let stemE = stem + "e"
        if let phonemes = lookup(stemE) { return phonemes + "ɪŋ" }

        return nil
    }
}
