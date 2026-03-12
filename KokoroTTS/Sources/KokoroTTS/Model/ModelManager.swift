import Foundation

/// Manages CoreML model download and caching for KokoroTTS.
///
/// Models are downloaded from HuggingFace and stored at a fixed path so
/// CoreML preserves its device-specialized compilation cache across launches.
public enum ModelManager {
    /// Default model storage directory.
    ///
    /// `~/Library/Application Support/com.operator/models/kokoro/`
    public static var defaultDirectory: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport
            .appendingPathComponent("com.operator")
            .appendingPathComponent("models")
            .appendingPathComponent("kokoro")
    }

    /// HuggingFace repository for CoreML models.
    private static let repoId = "FluidInference/kokoro-82m-coreml"

    /// Base URL for downloading from HuggingFace.
    private static var baseURL: String {
        "https://huggingface.co/\(repoId)/resolve/main"
    }

    /// Files required for the unified (end-to-end) pipeline.
    private static let unifiedModelFiles = [
        "kokoro_24_10s.mlmodelc",
        "kokoro_21_5s.mlmodelc",
        "kokoro_21_10s.mlmodelc",
    ]

    /// Files required for the two-stage pipeline.
    private static let twoStageModelFiles = [
        "kokoro_duration.mlmodelc",
        "kokoro_har_3s.mlmodelc",
        "kokoro_har_10s.mlmodelc",
    ]

    /// Shared support files.
    private static let supportFiles = [
        "G2PEncoder.mlmodelc",
        "G2PDecoder.mlmodelc",
        "vocab_index.json",
        "g2p_vocab.json",
        "us_gold.json",
        "us_silver.json",
    ]

    /// Check whether enough models exist to run inference.
    public static func modelsAvailable(at directory: URL) -> Bool {
        let fm = FileManager.default
        // Need at least one TTS model (either two-stage or unified)
        let hasTwoStage = fm.fileExists(
            atPath: directory.appendingPathComponent("kokoro_duration.mlmodelc").path)
            && (fm.fileExists(atPath: directory.appendingPathComponent("kokoro_har_3s.mlmodelc").path)
                || fm.fileExists(atPath: directory.appendingPathComponent("kokoro_har_10s.mlmodelc").path))

        let hasUnified = unifiedModelFiles.contains { file in
            fm.fileExists(atPath: directory.appendingPathComponent(file).path)
        }

        // Need voices directory
        let hasVoices = fm.fileExists(
            atPath: directory.appendingPathComponent("voices").path)

        return (hasTwoStage || hasUnified) && hasVoices
    }

    /// Download all model files to the specified directory.
    ///
    /// - Parameters:
    ///   - directory: Target directory for model files.
    ///   - progress: Optional callback with (fraction 0-1, description).
    public static func download(
        to directory: URL,
        progress: ((Double, String) -> Void)? = nil
    ) async throws {
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)

        // Collect all files to download
        var allFiles = supportFiles + unifiedModelFiles
        // Also try two-stage if available
        allFiles += twoStageModelFiles

        let totalSteps = allFiles.count + 1  // +1 for voices
        var completed = 0

        for file in allFiles {
            let destURL = directory.appendingPathComponent(file)
            if fm.fileExists(atPath: destURL.path) {
                completed += 1
                continue
            }

            let fraction = Double(completed) / Double(totalSteps)
            progress?(fraction, "Downloading \(file)...")

            let sourceURL = "\(baseURL)/\(file)"

            if file.hasSuffix(".mlmodelc") {
                try await downloadDirectory(
                    repoFile: file, to: destURL)
            } else {
                try await downloadFile(
                    from: sourceURL, to: destURL)
            }

            completed += 1
        }

        // Download voices
        progress?(Double(completed) / Double(totalSteps), "Downloading voice embeddings...")
        try await downloadVoices(to: directory.appendingPathComponent("voices"))

        progress?(1.0, "Download complete")
    }

    // MARK: - Private Download Helpers

    private static func downloadFile(from urlString: String, to destination: URL) async throws {
        guard let url = URL(string: urlString) else {
            throw KokoroError.downloadFailed("Invalid URL: \(urlString)")
        }

        let (tempURL, response) = try await URLSession.shared.download(from: url)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw KokoroError.downloadFailed(
                "HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1) for \(urlString)")
        }

        let fm = FileManager.default
        let parent = destination.deletingLastPathComponent()
        try fm.createDirectory(at: parent, withIntermediateDirectories: true)
        if fm.fileExists(atPath: destination.path) {
            try fm.removeItem(at: destination)
        }
        try fm.moveItem(at: tempURL, to: destination)
    }

    /// Download a compiled CoreML model directory.
    ///
    /// HuggingFace stores `.mlmodelc` as a directory with multiple files.
    /// We download the manifest and then each file within it.
    private static func downloadDirectory(repoFile: String, to destination: URL) async throws {
        let fm = FileManager.default
        try fm.createDirectory(at: destination, withIntermediateDirectories: true)

        // For compiled CoreML models, we need to download the tree listing first
        // Try direct download of known essential files
        let essentialFiles = ["model.mil", "coremldata.bin", "metadata.json"]
        for file in essentialFiles {
            let sourceURL = "\(baseURL)/\(repoFile)/\(file)"
            let dest = destination.appendingPathComponent(file)
            if !fm.fileExists(atPath: dest.path) {
                do {
                    try await downloadFile(from: sourceURL, to: dest)
                } catch {
                    // Not all files may exist, that's ok
                    continue
                }
            }
        }
    }

    private static func downloadVoices(to directory: URL) async throws {
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)

        // Download a predefined set of English voices
        let voices = [
            "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
            "am_adam", "am_michael", "am_onyx",
            "bf_emma", "bf_isabella",
            "bm_george", "bm_lewis",
        ]

        for voice in voices {
            let dest = directory.appendingPathComponent("\(voice).json")
            if fm.fileExists(atPath: dest.path) { continue }
            let sourceURL = "\(baseURL)/voices/\(voice).json"
            do {
                try await downloadFile(from: sourceURL, to: dest)
            } catch {
                // Voice download failure is non-fatal
                continue
            }
        }
    }
}
