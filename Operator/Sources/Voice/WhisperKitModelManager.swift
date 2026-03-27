import Foundation
import WhisperKit
import os

/// Metadata for a WhisperKit model variant.
public struct WhisperKitModel: Identifiable, Sendable {
    /// HuggingFace variant name used for download.
    public let variant: String
    /// Human-readable display name.
    public let displayName: String
    /// Approximate download size.
    public let size: String
    /// Approximate RAM usage during inference.
    public let memory: String
    /// Tags describing model characteristics.
    public let tags: [Tag]
    /// Whether this is the recommended default.
    public let isDefault: Bool

    /// Unique identifier for SwiftUI (uses the HuggingFace variant name).
    public var id: String { variant }

    /// Model characteristic tags shown as badges in the UI.
    public enum Tag: String, Sendable {
        case turbo = "Turbo"
        case quantized = "Quantized"
        case distilled = "Distilled"
        case english = "EN-only"
        case multilingual = "Multi"
    }
}

/// Manages WhisperKit model download and presence checking.
///
/// Models are stored at `~/Library/Application Support/com.operator/models/whisperkit/`.
/// Uses WhisperKit's built-in HuggingFace download. No auto-download — the user
/// must explicitly trigger download from Settings.
public enum WhisperKitModelManager {
    private static let logger = Log.logger(for: "WhisperKitModelManager")

    /// Available model variants ordered by size/quality.
    ///
    /// Curated from [argmaxinc/whisperkit-coreml](https://huggingface.co/argmaxinc/whisperkit-coreml).
    public static let availableModels: [WhisperKitModel] = [
        WhisperKitModel(
            variant: "openai_whisper-tiny",
            displayName: "Tiny",
            size: "~40 MB",
            memory: "~100 MB",
            tags: [.multilingual],
            isDefault: false
        ),
        WhisperKitModel(
            variant: "openai_whisper-base",
            displayName: "Base",
            size: "~75 MB",
            memory: "~200 MB",
            tags: [.multilingual],
            isDefault: true
        ),
        WhisperKitModel(
            variant: "openai_whisper-small",
            displayName: "Small",
            size: "~250 MB",
            memory: "~500 MB",
            tags: [.multilingual],
            isDefault: false
        ),
        WhisperKitModel(
            variant: "openai_whisper-small_216MB",
            displayName: "Small (Quantized)",
            size: "~216 MB",
            memory: "~400 MB",
            tags: [.multilingual, .quantized],
            isDefault: false
        ),
        WhisperKitModel(
            variant: "openai_whisper-small.en",
            displayName: "Small (English)",
            size: "~250 MB",
            memory: "~500 MB",
            tags: [.english],
            isDefault: false
        ),
        WhisperKitModel(
            variant: "distil-whisper_distil-large-v3_turbo_600MB",
            displayName: "Distil Large v3 Turbo",
            size: "~600 MB",
            memory: "~800 MB",
            tags: [.turbo, .distilled, .quantized, .multilingual],
            isDefault: false
        ),
        WhisperKitModel(
            variant: "openai_whisper-large-v3-v20240930_turbo_632MB",
            displayName: "Large v3.1 Turbo (Quantized)",
            size: "~632 MB",
            memory: "~900 MB",
            tags: [.turbo, .quantized, .multilingual],
            isDefault: false
        ),
        WhisperKitModel(
            variant: "openai_whisper-large-v3_turbo",
            displayName: "Large v3 Turbo",
            size: "~950 MB",
            memory: "~1.5 GB",
            tags: [.turbo, .multilingual],
            isDefault: false
        ),
        WhisperKitModel(
            variant: "openai_whisper-large-v3-v20240930_turbo",
            displayName: "Large v3.1 Turbo",
            size: "~950 MB",
            memory: "~1.5 GB",
            tags: [.turbo, .multilingual],
            isDefault: false
        ),
        WhisperKitModel(
            variant: "openai_whisper-large-v3",
            displayName: "Large v3 (Full)",
            size: "~950 MB",
            memory: "~2 GB",
            tags: [.multilingual],
            isDefault: false
        )
    ]

    /// Variant names for backward compatibility.
    public static let availableModelVariants: [String] = availableModels.map(\.variant)

    /// Default model variant.
    ///
    /// Base offers fast load (~1s) and transcription while maintaining
    /// reasonable accuracy for voice commands.
    public static let defaultModel = "openai_whisper-small"

    /// Look up model metadata by variant name.
    public static func model(for variant: String) -> WhisperKitModel? {
        availableModels.first { $0.variant == variant }
    }

    /// Root directory for WhisperKit models.
    public static let modelDirectory: URL = {
        let appSupport =
            FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support")
        return appSupport.appendingPathComponent("com.operator/models/whisperkit", isDirectory: true)
    }()

    /// Path to a specific downloaded model under the given base directory.
    ///
    /// WhisperKit downloads to `<base>/models/argmaxinc/whisperkit-coreml/<variant>/`,
    /// so we check both the nested path and a flat `<base>/<variant>/` for flexibility.
    internal static func modelPath(_ variant: String, baseDirectory: URL) -> String? {
        let nested = baseDirectory.appendingPathComponent(
            "models/argmaxinc/whisperkit-coreml/\(variant)"
        )
        let nestedContents = (try? FileManager.default.contentsOfDirectory(atPath: nested.path)) ?? []
        if !nestedContents.isEmpty {
            return nested.path
        }

        let flat = baseDirectory.appendingPathComponent(variant)
        let flatContents = (try? FileManager.default.contentsOfDirectory(atPath: flat.path)) ?? []
        return flatContents.isEmpty ? nil : flat.path
    }

    /// Path to a specific downloaded model.
    ///
    /// - Parameter variant: The model variant name.
    /// - Returns: The folder path, or nil if the model isn't downloaded.
    public static func modelPath(_ variant: String) -> String? {
        modelPath(variant, baseDirectory: modelDirectory)
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
        progressCallback: (@Sendable (Progress) -> Void)? = nil
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
