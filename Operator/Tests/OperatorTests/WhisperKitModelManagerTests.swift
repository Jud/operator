import Foundation
import Testing

@testable import OperatorCore

@Suite("WhisperKitModelManager")
internal struct WhisperKitModelManagerTests {
    private func makeTempDir() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("WhisperKitModelManagerTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func cleanup(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    @Test("modelPath returns nil for non-existent variant")
    func modelPathNonExistent() throws {
        let base = try makeTempDir()
        defer { cleanup(base) }

        #expect(WhisperKitModelManager.modelPath("no-such-model", baseDirectory: base) == nil)
    }

    @Test("modelPath returns nil for empty directory")
    func modelPathEmptyDirectory() throws {
        let base = try makeTempDir()
        defer { cleanup(base) }

        let variantDir = base.appendingPathComponent("empty-model")
        try FileManager.default.createDirectory(at: variantDir, withIntermediateDirectories: true)

        #expect(WhisperKitModelManager.modelPath("empty-model", baseDirectory: base) == nil)
    }

    @Test("modelPath returns path when directory has files")
    func modelPathWithFiles() throws {
        let base = try makeTempDir()
        defer { cleanup(base) }

        let variant = "test-model"
        let variantDir = base.appendingPathComponent(variant)
        try FileManager.default.createDirectory(at: variantDir, withIntermediateDirectories: true)
        FileManager.default.createFile(
            atPath: variantDir.appendingPathComponent("weights.bin").path,
            contents: Data([0x00])
        )

        let result = WhisperKitModelManager.modelPath(variant, baseDirectory: base)
        #expect(result == variantDir.path)
    }

    @Test("defaultModel is in availableModels")
    func defaultModelInAvailable() {
        #expect(WhisperKitModelManager.availableModelVariants.contains(WhisperKitModelManager.defaultModel))
    }
}
