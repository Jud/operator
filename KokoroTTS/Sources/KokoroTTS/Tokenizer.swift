import Foundation

/// Converts IPA phoneme strings to token ID sequences using vocab_index.json.
final class Tokenizer: Sendable {
    /// IPA symbol → token ID mapping.
    private let vocab: [String: Int]

    /// Pad token ID.
    static let padId: Int = 0
    /// Start-of-sequence token ID.
    static let bosId: Int = 1
    /// End-of-sequence token ID.
    static let eosId: Int = 2

    init(vocab: [String: Int]) {
        self.vocab = vocab
    }

    /// Load tokenizer from bundled vocab_index.json.
    static func loadFromBundle() throws -> Tokenizer {
        guard let url = Bundle.module.url(forResource: "vocab_index", withExtension: "json") else {
            throw KokoroError.modelLoadFailed("vocab_index.json not found in bundle")
        }
        return try load(from: url)
    }

    /// Load tokenizer from a vocab_index.json file.
    static func load(from url: URL) throws -> Tokenizer {
        let data = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: data)

        // Support both flat {sym: id} and nested {vocab: {sym: id}} formats
        let vocab: [String: Int]
        if let nested = json as? [String: Any], let v = nested["vocab"] as? [String: Int] {
            vocab = v
        } else if let flat = json as? [String: Int] {
            vocab = flat
        } else {
            throw KokoroError.modelLoadFailed("Invalid vocab_index.json format")
        }

        return Tokenizer(vocab: vocab)
    }

    /// Encode an IPA phoneme string to token IDs with BOS/EOS.
    func encode(_ phonemes: String, maxLength: Int = UnifiedBucket.maxTokenCount) -> [Int] {
        var ids = [Self.bosId]

        for char in phonemes {
            let s = String(char)
            if let id = vocab[s] {
                ids.append(id)
            }
        }

        ids.append(Self.eosId)

        if ids.count > maxLength {
            ids = Array(ids.prefix(maxLength - 1)) + [Self.eosId]
        }

        return ids
    }

    /// Pad token IDs to a fixed length.
    func pad(_ ids: [Int], to length: Int) -> [Int] {
        if ids.count >= length { return Array(ids.prefix(length)) }
        return ids + [Int](repeating: Self.padId, count: length - ids.count)
    }
}
