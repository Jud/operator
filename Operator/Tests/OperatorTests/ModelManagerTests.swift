import Foundation
import Testing

@testable import OperatorCore

/// Namespace matching the file name to satisfy the file_name lint rule.
private enum ModelManagerTests {}

private actor DownloadCounter {
    // MARK: - Instance Properties

    var value = 0

    // MARK: - Instance Methods

    func increment() { value += 1 }
}

private actor StateChangeCollector {
    // MARK: - Instance Properties

    private(set) var received: [(ModelManager.ModelType, ModelManager.ModelState)] = []

    var count: Int { received.count }

    // MARK: - Instance Methods

    func append(_ item: (ModelManager.ModelType, ModelManager.ModelState)) { received.append(item) }

    func get(_ index: Int) -> (ModelManager.ModelType, ModelManager.ModelState) { received[index] }
}

// MARK: - State Transitions

@Suite("ModelManager - State Transitions")
internal struct ModelManagerStateTests {
    private func makeTempCacheDir() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mm-test-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func cleanup(_ dir: URL) { try? FileManager.default.removeItem(at: dir) }

    @Test("initial states are all notDownloaded")
    func initialStates() async {
        let cacheDir = makeTempCacheDir()
        defer { cleanup(cacheDir) }

        let manager = ModelManager(cacheDirectory: cacheDir)
        for type in ModelManager.ModelType.allCases {
            #expect(await manager.state(for: type) == .notDownloaded)
        }
    }

    @Test("markLoaded transitions state to loaded")
    func markLoadedTransition() async {
        let cacheDir = makeTempCacheDir()
        defer { cleanup(cacheDir) }

        let manager = ModelManager(cacheDirectory: cacheDir)
        await manager.markLoaded(.stt)
        #expect(await manager.state(for: .stt) == .loaded)
    }

    @Test("markUnloaded transitions state to unloaded")
    func markUnloadedTransition() async {
        let cacheDir = makeTempCacheDir()
        defer { cleanup(cacheDir) }

        let manager = ModelManager(cacheDirectory: cacheDir)
        await manager.markLoaded(.stt)
        await manager.markUnloaded(.stt)
        #expect(await manager.state(for: .stt) == .unloaded)
    }

    @Test("markLoading transitions state to loading")
    func markLoadingTransition() async {
        let cacheDir = makeTempCacheDir()
        defer { cleanup(cacheDir) }

        let manager = ModelManager(cacheDirectory: cacheDir)
        await manager.markLoading(.routing)
        #expect(await manager.state(for: .routing) == .loading)
    }

    @Test("markError transitions state to error with message")
    func markErrorTransition() async {
        let cacheDir = makeTempCacheDir()
        defer { cleanup(cacheDir) }

        let manager = ModelManager(cacheDirectory: cacheDir)
        await manager.markError(.tts, message: "Disk full")
        #expect(await manager.state(for: .tts) == .error("Disk full"))
    }

    @Test("full lifecycle: notDownloaded -> downloading -> ready -> loading -> loaded -> unloaded")
    func fullLifecycle() async throws {
        let cacheDir = makeTempCacheDir()
        defer { cleanup(cacheDir) }

        let manager = ModelManager(cacheDirectory: cacheDir)
        #expect(await manager.state(for: .stt) == .notDownloaded)

        await manager.registerDownloadHandler(for: .stt) { cacheDirectory, progressCallback in
            await progressCallback(0.5)
            await progressCallback(1.0)
            try FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)

            let marker = cacheDirectory.appendingPathComponent("model.bin")
            FileManager.default.createFile(atPath: marker.path, contents: Data("test".utf8))
            return cacheDirectory
        }

        let url = try await manager.ensureDownloaded(.stt)
        #expect(FileManager.default.fileExists(atPath: url.path))
        #expect(await manager.state(for: .stt) == .ready)

        await manager.markLoading(.stt)
        #expect(await manager.state(for: .stt) == .loading)

        await manager.markLoaded(.stt)
        #expect(await manager.state(for: .stt) == .loaded)

        await manager.markUnloaded(.stt)
        #expect(await manager.state(for: .stt) == .unloaded)
    }

    @Test("states dictionary tracks all model types independently")
    func independentStateTracking() async {
        let cacheDir = makeTempCacheDir()
        defer { cleanup(cacheDir) }

        let manager = ModelManager(cacheDirectory: cacheDir)
        await manager.markLoaded(.stt)
        await manager.markError(.tts, message: "failed")

        let states = await manager.states
        #expect(states[.stt] == .loaded)
        #expect(states[.tts] == .error("failed"))
        #expect(states[.routing] == .notDownloaded)
    }

    @Test("stateChanged stream emits on state transitions")
    func stateChangedStream() async {
        let cacheDir = makeTempCacheDir()
        defer { cleanup(cacheDir) }

        let manager = ModelManager(cacheDirectory: cacheDir)
        let collector = StateChangeCollector()

        let listener = Task {
            var count = 0
            for await change in await manager.stateChanged {
                await collector.append(change)
                count += 1
                if count >= 2 { break }
            }
        }

        await manager.markLoading(.routing)
        await manager.markLoaded(.routing)
        try? await Task.sleep(for: .milliseconds(100))
        listener.cancel()

        #expect(await collector.count >= 2)

        let first = await collector.get(0)
        let second = await collector.get(1)
        #expect(first.0 == .routing)
        #expect(first.1 == .loading)
        #expect(second.0 == .routing)
        #expect(second.1 == .loaded)
    }

    @Test("ModelState equality works for all variants")
    func modelStateEquality() {
        #expect(ModelManager.ModelState.notDownloaded == .notDownloaded)
        #expect(ModelManager.ModelState.ready == .ready)
        #expect(ModelManager.ModelState.loaded != .unloaded)
        #expect(ModelManager.ModelState.downloading(progress: 0.5) == .downloading(progress: 0.5))
        #expect(ModelManager.ModelState.downloading(progress: 0.5) != .downloading(progress: 0.7))
        #expect(ModelManager.ModelState.error("a") == .error("a"))
        #expect(ModelManager.ModelState.error("a") != .error("b"))
    }
}

// MARK: - Download & Cache

@Suite("ModelManager - Download & Cache")
internal struct ModelManagerDownloadTests {
    private func makeTempCacheDir() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mm-test-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func cleanup(_ dir: URL) { try? FileManager.default.removeItem(at: dir) }

    @Test("ensureDownloaded returns immediately when already ready")
    func ensureDownloadedWhenReady() async throws {
        let cacheDir = makeTempCacheDir()
        defer { cleanup(cacheDir) }

        let manager = ModelManager(cacheDirectory: cacheDir)
        let counter = DownloadCounter()

        await manager.registerDownloadHandler(for: .routing) { cacheDirectory, progressCallback in
            await counter.increment()
            await progressCallback(1.0)
            try FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
            return cacheDirectory
        }

        _ = try await manager.ensureDownloaded(.routing)
        #expect(await counter.value == 1)

        _ = try await manager.ensureDownloaded(.routing)
        #expect(await counter.value == 1)
    }

    @Test("ensureDownloaded returns immediately when loaded")
    func ensureDownloadedWhenLoaded() async throws {
        let cacheDir = makeTempCacheDir()
        defer { cleanup(cacheDir) }

        let manager = ModelManager(cacheDirectory: cacheDir)
        await manager.registerDownloadHandler(for: .tts) { cacheDirectory, progressCallback in
            await progressCallback(1.0)
            try FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
            return cacheDirectory
        }

        _ = try await manager.ensureDownloaded(.tts)
        await manager.markLoaded(.tts)

        let url = try await manager.ensureDownloaded(.tts)

        let expected = cacheDir.appendingPathComponent("qwen3-tts", isDirectory: true).path
        #expect(url.path == expected)
    }

    @Test("ensureDownloaded returns cache path when state is unloaded")
    func ensureDownloadedWhenUnloaded() async throws {
        let cacheDir = makeTempCacheDir()
        defer { cleanup(cacheDir) }

        let manager = ModelManager(cacheDirectory: cacheDir)
        await manager.registerDownloadHandler(for: .stt) { cacheDirectory, progressCallback in
            await progressCallback(1.0)
            try FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
            return cacheDirectory
        }

        _ = try await manager.ensureDownloaded(.stt)
        await manager.markLoaded(.stt)
        await manager.markUnloaded(.stt)

        let url = try await manager.ensureDownloaded(.stt)

        let expected = cacheDir.appendingPathComponent("parakeet-coreml", isDirectory: true).path
        #expect(url.path == expected)
    }

    @Test("ensureDownloaded throws when no handler registered")
    func ensureDownloadedNoHandler() async {
        let cacheDir = makeTempCacheDir()
        defer { cleanup(cacheDir) }

        let manager = ModelManager(cacheDirectory: cacheDir)
        do {
            _ = try await manager.ensureDownloaded(.stt)
            Issue.record("Expected error but succeeded")
        } catch {
            #expect(error is ModelManagerError)
        }
    }

    @Test("ensureDownloaded transitions to error state on download failure")
    func ensureDownloadedFailure() async {
        let cacheDir = makeTempCacheDir()
        defer { cleanup(cacheDir) }

        let manager = ModelManager(cacheDirectory: cacheDir)
        await manager.registerDownloadHandler(for: .routing) { _, _ in
            throw ModelManagerError.downloadFailed(.routing, "Network error")
        }

        do {
            _ = try await manager.ensureDownloaded(.routing)
            Issue.record("Expected error but succeeded")
        } catch {
            if case .error = await manager.state(for: .routing) {
                // Error state reached as expected
            } else {
                Issue.record("Expected error state")
            }
        }
    }

    @Test("scanCachedModels detects previously downloaded models")
    func scanCachedModels() async {
        let cacheDir = makeTempCacheDir()
        defer { cleanup(cacheDir) }

        let sttDir = cacheDir.appendingPathComponent("parakeet-coreml", isDirectory: true)
        try? FileManager.default.createDirectory(at: sttDir, withIntermediateDirectories: true)

        let manager = ModelManager(cacheDirectory: cacheDir)
        await manager.scanCachedModels()
        #expect(await manager.state(for: .stt) == .ready)
        #expect(await manager.state(for: .tts) == .notDownloaded)
    }

    @Test("isCached returns correct values")
    func isCachedCheck() async {
        let cacheDir = makeTempCacheDir()
        defer { cleanup(cacheDir) }

        let routingDir = cacheDir.appendingPathComponent("qwen3.5-0.8b-mlx-4bit", isDirectory: true)
        try? FileManager.default.createDirectory(at: routingDir, withIntermediateDirectories: true)

        let manager = ModelManager(cacheDirectory: cacheDir)
        #expect(await manager.isCached(.routing) == true)
        #expect(await manager.isCached(.tts) == false)
    }

    @Test("modelCacheDirectory returns correct path per model type")
    func modelCacheDirectoryPaths() async {
        let cacheDir = makeTempCacheDir()
        defer { cleanup(cacheDir) }

        let manager = ModelManager(cacheDirectory: cacheDir)
        #expect(await manager.modelCacheDirectory(for: .stt).lastPathComponent == "parakeet-coreml")
        #expect(await manager.modelCacheDirectory(for: .tts).lastPathComponent == "qwen3-tts")
        #expect(await manager.modelCacheDirectory(for: .routing).lastPathComponent == "qwen3.5-0.8b-mlx-4bit")
    }

    @Test("totalCacheSize returns size of cached model files")
    func totalCacheSizeCalculation() async {
        let cacheDir = makeTempCacheDir()
        defer { cleanup(cacheDir) }

        let modelDir = cacheDir.appendingPathComponent("parakeet-coreml", isDirectory: true)
        try? FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        let testData = Data(repeating: 0x42, count: 1_024)
        FileManager.default.createFile(
            atPath: modelDir.appendingPathComponent("model.bin").path,
            contents: testData
        )

        let manager = ModelManager(cacheDirectory: cacheDir)
        #expect(await manager.totalCacheSize() == 1_024)
    }

    @Test("default cache directory uses ~/Library/Caches/com.operator.app/models")
    func defaultCacheDirectory() async {
        let manager = ModelManager()

        let expected = FileManager.default
            .urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("com.operator.app/models", isDirectory: true)
        #expect(await manager.cacheDirectory == expected)
    }

    @Test("ModelType displayName and rawValue are correct")
    func modelTypeProperties() {
        #expect(ModelManager.ModelType.stt.displayName == "Parakeet STT")
        #expect(ModelManager.ModelType.tts.displayName == "Qwen3-TTS")
        #expect(ModelManager.ModelType.routing.displayName == "Qwen 3.5 Routing")
        #expect(ModelManager.ModelType.stt.rawValue == "parakeet-coreml")
        #expect(ModelManager.ModelType.tts.rawValue == "qwen3-tts")
        #expect(ModelManager.ModelType.routing.rawValue == "qwen3.5-0.8b-mlx-4bit")
    }

    @Test("ModelManagerError descriptions are informative")
    func errorDescriptions() {
        let noHandler = ModelManagerError.noDownloadHandler(.stt)
        #expect(noHandler.description.contains("Parakeet STT"))

        let failed = ModelManagerError.downloadFailed(.routing, "timeout")
        #expect(failed.description.contains("timeout"))
        #expect(failed.description.contains("Qwen 3.5 Routing"))
    }
}
