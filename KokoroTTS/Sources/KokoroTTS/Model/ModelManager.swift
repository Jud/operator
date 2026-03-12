import Foundation

/// Manages CoreML model location and validation for KokoroTTS.
///
/// Models are stored at a fixed path so CoreML preserves its
/// device-specialized compilation cache across launches.
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

    /// Model files for the unified (end-to-end) pipeline.
    private static let modelFiles = [
        "kokoro_24_10s.mlmodelc",
        "kokoro_21_5s.mlmodelc",
        "kokoro_21_10s.mlmodelc",
    ]

    /// Check whether enough models exist to run inference.
    public static func modelsAvailable(at directory: URL) -> Bool {
        let fm = FileManager.default
        let hasModel = modelFiles.contains { file in
            fm.fileExists(atPath: directory.appendingPathComponent(file).path)
        }
        let hasVoices = fm.fileExists(
            atPath: directory.appendingPathComponent("voices").path)
        return hasModel && hasVoices
    }
}
