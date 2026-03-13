import Foundation
import WhisperKit
import os

/// Manages WhisperKit model download and presence checking.
///
/// Models are stored at `~/Library/Application Support/com.operator/models/whisperkit/`.
/// Uses WhisperKit's built-in HuggingFace download. No auto-download — the user
/// must explicitly trigger download from Settings.
public enum WhisperKitModelManager {
    private static let logger = Log.logger(for: "WhisperKitModelManager")

    /// Available model variants ordered by size/quality.
    public static let availableModels = [
        "openai_whisper-tiny",
        "openai_whisper-base",
        "openai_whisper-small",
        "openai_whisper-large-v3_turbo"
    ]

    /// Default model variant.
    public static let defaultModel = "openai_whisper-large-v3_turbo"

    /// Root directory for WhisperKit models.
    public static let modelDirectory: URL = {
        let appSupport =
            FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support")
        return appSupport.appendingPathComponent("com.operator/models/whisperkit", isDirectory: true)
    }()

    /// Path to a specific downloaded model.
    ///
    /// - Parameter variant: The model variant name.
    /// - Returns: The folder path, or nil if the model isn't downloaded.
    public static func modelPath(_ variant: String) -> String? {
        let folder = modelDirectory.appendingPathComponent(variant)
        guard FileManager.default.fileExists(atPath: folder.path) else {
            return nil
        }
        let contents = (try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? []
        return contents.isEmpty ? nil : folder.path
    }

    /// Check if a model variant has been downloaded.
    ///
    /// - Parameter variant: The model variant name (e.g. "openai_whisper-large-v3_turbo").
    /// - Returns: True if the model folder exists and contains at least one file.
    public static func modelAvailable(_ variant: String) -> Bool {
        modelPath(variant) != nil
    }

    /// Download a model variant using WhisperKit's built-in downloader.
    ///
    /// - Parameters:
    ///   - variant: The model variant to download.
    ///   - progressCallback: Optional callback for download progress.
    /// - Returns: The local folder path where the model was saved.
    /// - Throws: If the directory cannot be created or the download fails.
    @discardableResult
    public static func download(
        variant: String,
        progressCallback: ((Progress) -> Void)? = nil
    ) async throws -> URL {
        logger.info("Downloading WhisperKit model: \(variant)")

        try FileManager.default.createDirectory(
            at: modelDirectory,
            withIntermediateDirectories: true
        )

        let folder = try await WhisperKit.download(
            variant: variant,
            downloadBase: modelDirectory,
            progressCallback: progressCallback
        )

        logger.info("Model downloaded to: \(folder)")
        return folder
    }
}
