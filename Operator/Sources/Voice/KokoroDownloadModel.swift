import Foundation
import KokoroCoreML

/// Observable model for Kokoro CoreML model download state.
///
/// Shared singleton so Settings UI can display download progress while
/// the AppDelegate drives the background download.
@MainActor
public final class KokoroDownloadModel: ObservableObject {
    /// Shared instance.
    public static let shared = KokoroDownloadModel()

    /// Whether models are downloaded and ready to use.
    @Published public var isDownloaded: Bool

    /// Whether a download is currently in progress.
    @Published public var isDownloading = false

    /// Download progress (0.0–1.0).
    @Published public var progress: Double = 0

    /// Error message from the last failed download attempt.
    @Published public var error: String?

    private static let logger = Log.logger(for: "KokoroDownload")

    private init() {
        isDownloaded = KokoroEngine.isDownloaded
    }

    /// Start downloading Kokoro models in the background.
    ///
    /// Safe to call if models are already downloaded — returns immediately.
    /// Progress updates are published to `progress` for UI observation.
    ///
    /// - Parameter completion: Called on MainActor when download finishes
    ///   successfully, passing the loaded KokoroEngine.
    public func downloadIfNeeded(
        completion: (@MainActor (KokoroEngine) -> Void)? = nil
    ) {
        guard !isDownloaded, !isDownloading else {
            if isDownloaded, let completion {
                do {
                    let engine = try KokoroEngine(
                        modelDirectory: KokoroEngine.defaultModelDirectory
                    )
                    completion(engine)
                } catch {
                    Self.logger.error("Failed to load Kokoro engine: \(error)")
                }
            }
            return
        }

        isDownloading = true
        error = nil
        progress = 0

        Task.detached {
            do {
                try KokoroEngine.download { progressValue in
                    Task { @MainActor in
                        self.progress = progressValue
                    }
                }
                let engine = try KokoroEngine(
                    modelDirectory: KokoroEngine.defaultModelDirectory
                )
                await MainActor.run {
                    self.isDownloading = false
                    self.isDownloaded = true
                    Self.logger.info("Kokoro models downloaded and loaded")
                    completion?(engine)
                }
            } catch {
                await MainActor.run {
                    self.error = error.localizedDescription
                    self.isDownloading = false
                    Self.logger.error("Kokoro download failed: \(error)")
                }
            }
        }
    }
}
