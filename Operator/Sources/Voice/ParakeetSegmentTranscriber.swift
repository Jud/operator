import Foundation
import ParakeetASR

// swiftlint:disable type_contents_order

internal actor ParakeetSegmentTranscriber {
    private static let logger = Log.logger(for: "ParakeetSegmentTranscriber")
    private static let sampleRate = 16_000

    private let modelManager: ModelManager
    private var activeSessionID = 0
    private var model: ParakeetASRModel?
    private var didWarmUp = false
    private var transcripts: [Int: String] = [:]

    init(modelManager: ModelManager) {
        self.modelManager = modelManager
    }

    func resetSession(_ sessionID: Int) {
        activeSessionID = sessionID
        transcripts.removeAll(keepingCapacity: true)
    }

    func prewarm() async {
        do {
            try await ensureModelLoaded()
            guard !didWarmUp, let model else {
                return
            }
            try model.warmUp()
            didWarmUp = true
            Self.logger.info("Parakeet warmup complete")
        } catch {
            Self.logger.error("Parakeet warmup failed: \(error.localizedDescription)")
        }
    }

    func enqueueSegment(sessionID: Int, id: Int, samples: [Float]) async {
        if sessionID > activeSessionID {
            resetSession(sessionID)
        }
        guard sessionID == activeSessionID, !samples.isEmpty, !Task.isCancelled else {
            return
        }

        do {
            try await ensureModelLoaded()
            guard let model else {
                return
            }
            let text = try model.transcribeAudio(samples, sampleRate: Self.sampleRate)
            if let normalized = Self.normalizeTranscript(text) {
                transcripts[id] = normalized
            }
        } catch {
            Self.logger.error("Parakeet segment transcription failed: \(error.localizedDescription)")
        }
    }

    func finishSession(sessionID: Int, fallbackSamples: [Float]?) async -> String? {
        if sessionID > activeSessionID {
            resetSession(sessionID)
        }
        guard sessionID == activeSessionID else {
            return nil
        }
        if transcripts.isEmpty, let fallbackSamples, !fallbackSamples.isEmpty {
            await enqueueSegment(sessionID: sessionID, id: 0, samples: fallbackSamples)
        }

        let orderedText = transcripts.keys.sorted().compactMap { transcripts[$0] }
        let joined = orderedText.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
        return joined.isEmpty ? nil : joined
    }

    func unloadModel() async {
        model?.unload()
        model = nil
        didWarmUp = false
        transcripts.removeAll(keepingCapacity: false)
        await modelManager.markUnloaded(.stt)
        Self.logger.info("Parakeet model unloaded")
    }

    private func ensureModelLoaded() async throws {
        guard model == nil else {
            return
        }

        await modelManager.markLoading(.stt)
        Self.logger.info("Loading Parakeet model...")

        let loaded = try await ParakeetASRModel.fromPretrained()
        model = loaded

        await modelManager.markLoaded(.stt)
        Self.logger.info("Parakeet model loaded")
    }

    private static func normalizeTranscript(_ text: String) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}

// swiftlint:enable type_contents_order
