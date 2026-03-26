import Foundation

/// Generates short, speakable nicknames from project/session names.
///
/// Long directory names like "kokoro-tts-swift" become "Kokoro" for TTS.
/// Single short words like "operator" stay as-is. Duplicate nicknames are
/// disambiguated with NATO phonetic alphabet suffixes: "Operator Alpha",
/// "Operator Bravo", etc.
internal enum NicknameGenerator {
    // MARK: - NATO Phonetic Alphabet

    /// NATO phonetic alphabet for duplicate disambiguation.
    static let natoAlphabet: [String] = [
        "Alpha", "Bravo", "Charlie", "Delta", "Echo",
        "Foxtrot", "Golf", "Hotel", "India", "Juliet",
        "Kilo", "Lima", "Mike", "November", "Oscar",
        "Papa", "Quebec", "Romeo", "Sierra", "Tango",
        "Uniform", "Victor", "Whiskey", "X-ray", "Yankee",
        "Zulu"
    ]

    /// Words too generic to use as a standalone nickname.
    private static let genericWords: Set<String> = [
        "my", "the", "a", "an", "app", "project", "new", "old", "test", "tmp", "temp"
    ]

    // MARK: - Nickname Generation

    /// Generate a short speakable nickname from a full session/project name.
    ///
    /// Rules:
    /// 1. Split on hyphens, underscores, and spaces.
    /// 2. If single word and ≤ 12 chars, use it as-is (capitalized).
    /// 3. If multi-word, pick the first non-generic word (capitalized).
    /// 4. If all words are generic, use the first two joined.
    ///
    /// - Parameter name: The full session or project name.
    /// - Returns: A short, capitalized nickname suitable for TTS.
    static func nickname(from name: String) -> String {
        let words =
            name
            .split { $0 == "-" || $0 == "_" || $0 == " " }
            .map(String.init)
            .filter { !$0.isEmpty }

        guard !words.isEmpty else {
            return name.capitalizingFirst()
        }

        // Single word: use as-is if short enough.
        if words.count == 1 {
            return words[0].capitalizingFirst()
        }

        // Multi-word: find the first non-generic word.
        if let distinctive = words.first(where: { !genericWords.contains($0.lowercased()) }) {
            return distinctive.capitalizingFirst()
        }

        // All generic: join first two.
        let joined = words.prefix(2).map { $0.capitalizingFirst() }.joined()
        return joined
    }

    // MARK: - NATO Deduplication

    /// Deduplicate a nickname against existing nicknames using NATO suffixes.
    ///
    /// - If the nickname is unique, returns it unchanged.
    /// - If duplicates exist without NATO suffixes, the first gets "Alpha"
    ///   and the new one gets the next available letter.
    ///
    /// - Parameters:
    ///   - nickname: The base nickname to deduplicate.
    ///   - existing: Nicknames already in use by other sessions.
    /// - Returns: A unique nickname, possibly with a NATO suffix.
    static func deduplicate(_ nickname: String, existing: [String]) -> String {
        let lowerNick = nickname.lowercased()

        // Check if this exact nickname or any NATO variant is already taken.
        let conflicts = existing.filter { existingNick in
            let lower = existingNick.lowercased()
            if lower == lowerNick {
                return true
            }
            // Check if it's a NATO variant of the same base.
            return natoAlphabet.contains { nato in
                lower == "\(lowerNick) \(nato.lowercased())"
            }
        }

        if conflicts.isEmpty {
            return nickname
        }

        // Find the next available NATO suffix.
        for nato in natoAlphabet {
            let candidate = "\(nickname) \(nato)"
            if !existing.contains(where: { $0.lowercased() == candidate.lowercased() }) {
                return candidate
            }
        }

        // Exhausted NATO alphabet (26 agents with same name) — fall back to number.
        var suffix = 27
        while existing.contains(where: { $0.lowercased() == "\(lowerNick) \(suffix)" }) {
            suffix += 1
        }
        return "\(nickname) \(suffix)"
    }
}

// MARK: - String Extension

extension String {
    /// Capitalize only the first character, leaving the rest unchanged.
    fileprivate func capitalizingFirst() -> String {
        guard let first else {
            return self
        }
        return first.uppercased() + dropFirst()
    }
}
