import Foundation

/// GPL-free phonemizer for Kokoro TTS.
///
/// Three-tier approach (all Apache-2.0 compatible):
/// 1. **Dictionary lookup** — gold + silver IPA dictionaries from misaki
/// 2. **Suffix stemming** — strips -s/-ed/-ing, looks up stem, applies phonological rules
/// 3. **CoreML BART G2P** — encoder-decoder neural model for OOV words
///
/// No eSpeak-NG or NLTagger dependency.
final class Phonemizer: @unchecked Sendable {
    /// Gold dictionary (high-confidence).
    private var goldDict: [String: DictEntry] = [:]
    /// Silver dictionary (lower-confidence).
    private var silverDict: [String: DictEntry] = [:]
    /// G2P model for out-of-vocabulary words.
    private var g2pModel: G2PModel?

    enum DictEntry {
        case simple(String)
        case heteronym([String: String])
    }

    init() {}

    /// Load dictionaries from SPM bundle (non-throwing; missing resources are fine).
    func loadDictionariesFromBundle() {
        if let goldURL = Bundle.module.url(forResource: "us_gold", withExtension: "json"),
           let dict = try? parseDictionary(from: goldURL) {
            goldDict = dict
            growDictionary(&goldDict)
        }
        if let silverURL = Bundle.module.url(forResource: "us_silver", withExtension: "json"),
           let dict = try? parseDictionary(from: silverURL) {
            silverDict = dict
            growDictionary(&silverDict)
        }
    }

    /// Load dictionaries from an external directory (for runtime-downloaded models).
    func loadDictionaries(from directory: URL) throws {
        let goldURL = directory.appendingPathComponent("us_gold.json")
        let silverURL = directory.appendingPathComponent("us_silver.json")
        let fm = FileManager.default

        if fm.fileExists(atPath: goldURL.path) {
            goldDict = try parseDictionary(from: goldURL)
            growDictionary(&goldDict)
        }
        if fm.fileExists(atPath: silverURL.path) {
            silverDict = try parseDictionary(from: silverURL)
            growDictionary(&silverDict)
        }
    }

    /// Attach a G2P model for OOV fallback.
    func attachG2P(_ model: G2PModel) {
        self.g2pModel = model
    }

    // MARK: - Text-to-Phoneme Pipeline

    /// Convert text to IPA phoneme string.
    func textToPhonemes(_ text: String) -> String {
        let normalized = normalizeText(text)
        let words = splitWords(normalized)

        var result = ""
        for word in words {
            if word.allSatisfy({ $0.isWhitespace }) {
                result += " "
                continue
            }
            if word.allSatisfy({ $0.isPunctuation || $0.isSymbol }) {
                if let mapped = punctuationToPhoneme(word) {
                    result += mapped
                }
                continue
            }
            if let phonemes = resolveWord(word) {
                result += phonemes
            }
        }
        return result
    }

    // MARK: - Word Resolution

    private func resolveWord(_ word: String) -> String? {
        let lower = word.lowercased()
        if let special = specialCase(lower) { return special }
        if let entry = lookupDict(lower) { return entry }
        if let stemmed = SuffixRules.stemAndLookup(lower, lookup: lookupDict) { return stemmed }
        if let g2p = g2pModel?.phonemize(lower) { return g2p }
        return lower
    }

    private func lookupDict(_ word: String) -> String? {
        if let entry = goldDict[word] { return resolveEntry(entry) }
        if let entry = silverDict[word] { return resolveEntry(entry) }
        return nil
    }

    private func resolveEntry(_ entry: DictEntry) -> String {
        switch entry {
        case .simple(let phonemes):
            return phonemes
        case .heteronym(let posMap):
            // Without NLTagger, use DEFAULT or first available
            return posMap["DEFAULT"] ?? posMap.values.first ?? ""
        }
    }

    // MARK: - Special Cases

    private func specialCase(_ word: String) -> String? {
        switch word {
        case "the": return "ðə"
        case "a": return "ɐ"
        case "an": return "ən"
        case "to": return "tʊ"
        case "of": return "ʌv"
        case "i": return "aɪ"
        default: return nil
        }
    }

    // MARK: - Text Normalization

    private func normalizeText(_ text: String) -> String {
        var result = text
        let contractions: [(String, String)] = [
            ("can't", "can not"), ("won't", "will not"), ("don't", "do not"),
            ("doesn't", "does not"), ("didn't", "did not"), ("isn't", "is not"),
            ("aren't", "are not"), ("wasn't", "was not"), ("weren't", "were not"),
            ("couldn't", "could not"), ("wouldn't", "would not"), ("shouldn't", "should not"),
            ("haven't", "have not"), ("hasn't", "has not"), ("hadn't", "had not"),
            ("i'm", "i am"), ("i've", "i have"), ("i'll", "i will"), ("i'd", "i would"),
            ("you're", "you are"), ("you've", "you have"), ("you'll", "you will"),
            ("he's", "he is"), ("she's", "she is"), ("it's", "it is"),
            ("we're", "we are"), ("we've", "we have"), ("we'll", "we will"),
            ("they're", "they are"), ("they've", "they have"), ("they'll", "they will"),
            ("that's", "that is"), ("there's", "there is"), ("let's", "let us"),
        ]
        for (contraction, expansion) in contractions {
            result = result.replacingOccurrences(
                of: contraction, with: expansion, options: .caseInsensitive)
        }
        while result.contains("  ") {
            result = result.replacingOccurrences(of: "  ", with: " ")
        }
        return result.trimmingCharacters(in: .whitespaces)
    }

    private func splitWords(_ text: String) -> [String] {
        var words: [String] = []
        var current = ""
        for char in text {
            if char.isWhitespace {
                if !current.isEmpty { words.append(current); current = "" }
                words.append(" ")
            } else if char.isPunctuation || char.isSymbol {
                if !current.isEmpty { words.append(current); current = "" }
                words.append(String(char))
            } else {
                current.append(char)
            }
        }
        if !current.isEmpty { words.append(current) }
        return words
    }

    private func punctuationToPhoneme(_ text: String) -> String? {
        switch text {
        case ",": return ","
        case ".": return "."
        case "!": return "!"
        case "?": return "?"
        case ";": return ";"
        case ":": return ":"
        case "-": return "-"
        case "'": return "'"
        default: return nil
        }
    }

    // MARK: - Dictionary Parsing

    private func parseDictionary(from url: URL) throws -> [String: DictEntry] {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [:] }
        var dict = [String: DictEntry]()
        for (key, value) in json {
            if let phonemes = value as? String {
                dict[key] = .simple(phonemes)
            } else if let posMap = value as? [String: String?] {
                var resolved = [String: String]()
                for (pos, pron) in posMap {
                    if let p = pron { resolved[pos] = p }
                }
                if !resolved.isEmpty { dict[key] = .heteronym(resolved) }
            }
        }
        return dict
    }

    private func growDictionary(_ dict: inout [String: DictEntry]) {
        var additions = [String: DictEntry]()
        for (key, entry) in dict {
            if key == key.lowercased() && !key.isEmpty {
                let capitalized = key.prefix(1).uppercased() + key.dropFirst()
                if dict[capitalized] == nil { additions[capitalized] = entry }
            }
            if key.first?.isUppercase == true {
                let lower = key.lowercased()
                if dict[lower] == nil { additions[lower] = entry }
            }
        }
        for (key, entry) in additions { dict[key] = entry }
    }
}
