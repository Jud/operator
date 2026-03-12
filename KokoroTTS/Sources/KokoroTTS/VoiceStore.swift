import Foundation

/// Loads and caches 256-dimensional voice style embeddings from JSON files.
final class VoiceStore: Sendable {
    /// Voice name → 256-dim float embedding.
    private let embeddings: [String: [Float]]

    /// Style embedding dimension.
    static let styleDim = 256

    /// Load all voice embeddings from a directory of JSON files.
    ///
    /// Each file should be named `<voice_name>.json` and contain:
    /// ```json
    /// {"embedding": [256 floats], ...}
    /// ```
    init(directory: URL) throws {
        let fm = FileManager.default
        var loaded = [String: [Float]]()

        guard fm.fileExists(atPath: directory.path) else {
            throw KokoroError.modelsNotAvailable(directory)
        }

        let files = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        for file in files where file.pathExtension == "json" {
            let voiceName = file.deletingPathExtension().lastPathComponent
            if let embedding = try? Self.loadEmbedding(from: file) {
                loaded[voiceName] = embedding
            }
        }

        self.embeddings = loaded
    }

    /// Get the embedding for a specific voice.
    func embedding(for voice: String) throws -> [Float] {
        guard let emb = embeddings[voice] else {
            let available = Array(embeddings.keys).sorted().prefix(5)
            throw KokoroError.voiceNotFound(
                "\(voice) — available: \(available.joined(separator: ", "))...")
        }
        return emb
    }

    /// Available voice preset names.
    var availableVoices: [String] {
        Array(embeddings.keys).sorted()
    }

    /// Number of loaded voices.
    var count: Int { embeddings.count }

    // MARK: - Private

    private static func loadEmbedding(from url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let embedding = json["embedding"] as? [Double],
              !embedding.isEmpty else {
            throw KokoroError.modelLoadFailed(
                "Invalid voice embedding: \(url.lastPathComponent)")
        }
        return embedding.prefix(styleDim).map { Float($0) }
    }
}
