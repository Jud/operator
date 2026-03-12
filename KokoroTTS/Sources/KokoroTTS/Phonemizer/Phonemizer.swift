import Foundation
import NaturalLanguage

/// GPL-free phonemizer for Kokoro TTS.
///
/// Three-tier approach (all Apache-2.0 compatible):
/// 1. **Dictionary lookup** — gold + silver IPA dictionaries from misaki
/// 2. **Suffix stemming** — strips -s/-ed/-ing, looks up stem, applies phonological rules
/// 3. **CoreML BART G2P** — encoder-decoder neural model for OOV words
///
/// Uses NLTagger for POS-based heteronym resolution.
final class Phonemizer: @unchecked Sendable {
    /// Gold dictionary (high-confidence).
    private var goldDict: [String: DictEntry] = [:]
    /// Silver dictionary (lower-confidence).
    private var silverDict: [String: DictEntry] = [:]
    /// G2P model for out-of-vocabulary words.
    private var g2pModel: G2PModel?
    /// NL tagger for POS tagging (heteronym resolution).
    private let tagger = NLTagger(tagSchemes: [.lexicalClass])

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
        let posTagged = tagPOS(normalized)

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
            let pos = posTagged[word.lowercased()]
            if let phonemes = resolveWord(word, pos: pos) {
                result += phonemes
            }
        }
        return result
    }

    // MARK: - Word Resolution

    private func resolveWord(_ word: String, pos: String?) -> String? {
        let lower = word.lowercased()
        if let special = specialCase(lower, pos: pos) { return special }
        if let entry = lookupDict(lower, pos: pos) { return entry }
        if let stemmed = SuffixRules.stemAndLookup(lower, lookup: { self.lookupDict($0, pos: nil) }) {
            return stemmed
        }
        if let g2p = g2pModel?.phonemize(lower) { return g2p }
        return spellOutWord(lower)
    }

    private func spellOutWord(_ word: String) -> String? {
        let spelled = word.compactMap { char -> String? in
            lookupDict(String(char), pos: nil)
        }
        if spelled.isEmpty { return nil }
        return spelled.joined(separator: " ")
    }

    private func lookupDict(_ word: String, pos: String?) -> String? {
        if let entry = goldDict[word] { return resolveEntry(entry, pos: pos) }
        if let entry = silverDict[word] { return resolveEntry(entry, pos: pos) }
        return nil
    }

    private func resolveEntry(_ entry: DictEntry, pos: String?) -> String {
        switch entry {
        case .simple(let phonemes):
            return phonemes
        case .heteronym(let posMap):
            if let pos, let phonemes = posMap[pos] { return phonemes }
            return posMap["DEFAULT"] ?? posMap.values.first ?? ""
        }
    }

    // MARK: - Special Cases

    private func specialCase(_ word: String, pos: String?) -> String? {
        switch word {
        case "the": return "ðə"
        case "a":
            if pos == "Determiner" { return "ɐ" }
            return "eɪ"
        case "an": return "ən"
        case "to": return "tʊ"
        case "of": return "ʌv"
        case "i": return "aɪ"
        default: return nil
        }
    }

    // MARK: - POS Tagging

    private func tagPOS(_ text: String) -> [String: String] {
        var result = [String: String]()
        tagger.string = text
        let range = text.startIndex..<text.endIndex
        tagger.enumerateTags(in: range, unit: .word, scheme: .lexicalClass) { tag, tokenRange in
            let word = String(text[tokenRange]).lowercased()
            if let tag { result[word] = tag.rawValue }
            return true
        }
        return result
    }

    // MARK: - Text Normalization

    private func normalizeText(_ text: String) -> String {
        var result = text
        result = splitCamelCase(result)
        result = expandNumbers(result)
        result = spellAcronyms(result)
        while result.contains("  ") {
            result = result.replacingOccurrences(of: "  ", with: " ")
        }
        return result.trimmingCharacters(in: .whitespaces)
    }

    // MARK: - CamelCase Splitting

    private func splitCamelCase(_ text: String) -> String {
        // Split "NLTagger" → "NL Tagger", "CoreML" → "Core ML", "AVAudioEngine" → "AV Audio Engine"
        let pattern = try! NSRegularExpression(
            pattern: "([A-Z]{2,})([A-Z][a-z])|([a-z])([A-Z])")
        let nsText = text as NSString
        var result = text
        // Apply replacements from end to preserve ranges
        let matches = pattern.matches(in: text, range: NSRange(location: 0, length: nsText.length))
        for match in matches.reversed() {
            let range = match.range
            // Insert space before the last uppercase-lowercase transition
            if match.range(at: 1).location != NSNotFound {
                // "NLT" + "a" → "NL" + " " + "Ta"
                let acronym = nsText.substring(with: match.range(at: 1))
                let rest = nsText.substring(with: match.range(at: 2))
                result = (result as NSString).replacingCharacters(
                    in: range, with: "\(acronym) \(rest)")
            } else if match.range(at: 3).location != NSNotFound {
                // "eM" → "e" + " " + "M"
                let lower = nsText.substring(with: match.range(at: 3))
                let upper = nsText.substring(with: match.range(at: 4))
                result = (result as NSString).replacingCharacters(
                    in: range, with: "\(lower) \(upper)")
            }
        }
        return result
    }

    // MARK: - Number Expansion

    private func expandNumbers(_ text: String) -> String {
        let pattern = try! NSRegularExpression(pattern: "\\b\\d+\\.?\\d*\\b")
        let nsText = text as NSString
        let matches = pattern.matches(in: text, range: NSRange(location: 0, length: nsText.length))
        if matches.isEmpty { return text }

        var result = ""
        var lastEnd = 0
        for match in matches {
            let range = match.range
            result += nsText.substring(with: NSRange(location: lastEnd, length: range.location - lastEnd))
            let numStr = nsText.substring(with: range)
            result += numberToWords(numStr)
            lastEnd = range.location + range.length
        }
        result += nsText.substring(from: lastEnd)
        return result
    }

    private static let spellOutFormatter: NumberFormatter = {
        let f = NumberFormatter()
        f.numberStyle = .spellOut
        f.locale = Locale(identifier: "en_US")
        return f
    }()

    private func numberToWords(_ str: String) -> String {
        if str.contains(".") {
            let parts = str.split(separator: ".", maxSplits: 1)
            let whole = numberToWords(String(parts[0]))
            let decimal = String(parts[1]).map { char -> String in
                if let d = Int(String(char)) {
                    return Self.spellOutFormatter.string(from: NSNumber(value: d)) ?? String(char)
                }
                return String(char)
            }.joined(separator: " ")
            return "\(whole) point \(decimal)"
        }
        guard let n = Int(str) else { return str }
        return Self.spellOutFormatter.string(from: NSNumber(value: n)) ?? str
    }

    // MARK: - Acronym Spelling

    private static let letterNames: [Character: String] = [
        "A": "ay", "B": "bee", "C": "see", "D": "dee", "E": "ee",
        "F": "ef", "G": "gee", "H": "aitch", "I": "eye", "J": "jay",
        "K": "kay", "L": "el", "M": "em", "N": "en", "O": "oh",
        "P": "pee", "Q": "cue", "R": "ar", "S": "es", "T": "tee",
        "U": "you", "V": "vee", "W": "double you", "X": "ex", "Y": "why",
        "Z": "zee",
    ]

    private func spellAcronyms(_ text: String) -> String {
        let pattern = try! NSRegularExpression(pattern: "\\b[A-Z]{2,}\\b")
        let nsText = text as NSString
        let matches = pattern.matches(in: text, range: NSRange(location: 0, length: nsText.length))
        if matches.isEmpty { return text }

        var result = ""
        var lastEnd = 0
        for match in matches {
            let range = match.range
            result += nsText.substring(with: NSRange(location: lastEnd, length: range.location - lastEnd))
            let acronym = nsText.substring(with: range)
            let spelled = acronym.compactMap { Self.letterNames[$0] }.joined(separator: " ")
            result += spelled
            lastEnd = range.location + range.length
        }
        result += nsText.substring(from: lastEnd)
        return result
    }

    private func splitWords(_ text: String) -> [String] {
        var words: [String] = []
        var current = ""
        let chars = Array(text)
        for (i, char) in chars.enumerated() {
            if char.isWhitespace {
                if !current.isEmpty { words.append(current); current = "" }
                words.append(" ")
            } else if char == "'" || char == "\u{2019}" {
                // Keep apostrophes within words (o'clock, it's)
                let hasLetterBefore = !current.isEmpty && current.last?.isLetter == true
                let hasLetterAfter = i + 1 < chars.count && chars[i + 1].isLetter
                if hasLetterBefore && hasLetterAfter {
                    current.append(char)
                } else {
                    if !current.isEmpty { words.append(current); current = "" }
                    words.append(String(char))
                }
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
