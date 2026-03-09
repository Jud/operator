import Foundation
import MLXLLM
import MLXLMCommon

extension MLXRoutingEngine {
    func ensureModelLoaded() async throws {
        guard modelContainer == nil else {
            return
        }
        guard !isLoading else {
            while isLoading {
                try await Task.sleep(nanoseconds: 50_000_000)
            }
            return
        }

        isLoading = true
        await modelManager.markLoading(.routing)
        Self.logger.info("Loading routing model: \(Self.modelID)")

        do {
            let container = try await LLMModelFactory.shared.loadContainer(
                configuration: modelConfiguration
            ) { progress in
                Self.logger.debug(
                    "Routing model download: \(Int(progress.fractionCompleted * 100))%"
                )
            }
            modelContainer = container
            isLoading = false
            await modelManager.markLoaded(.routing)
            Self.logger.info("Routing model loaded and ready")
        } catch {
            isLoading = false
            await modelManager.markError(.routing, message: error.localizedDescription)
            throw MLXRoutingError.modelLoadFailed(error.localizedDescription)
        }
    }

    func registerDownloadHandler() {
        let manager = modelManager
        let modelID = Self.modelID
        // swiftlint:disable:next unhandled_throwing_task
        Task {
            await manager.registerDownloadHandler(for: .routing) { _, progressCallback in
                await progressCallback(0.0)
                let configuration = ModelConfiguration(id: modelID)
                _ = try await LLMModelFactory.shared.loadContainer(configuration: configuration)
                await progressCallback(1.0)
                let cacheDir = FileManager.default
                    .urls(for: .cachesDirectory, in: .userDomainMask)[0]
                    .appendingPathComponent("huggingface/hub", isDirectory: true)
                return cacheDir
            }
        }
    }
}
