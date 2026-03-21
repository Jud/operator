import Foundation

/// Test scheduler that triggers transcriptions at exact sample counts.
///
/// Uses a recorded session manifest to enable deterministic replay of
/// production sessions.
public final class ManifestReplayScheduler: TranscriptionScheduler, @unchecked Sendable {
    /// A single background transcription trigger point.
    public struct Entry: Sendable {
        /// Number of audio samples accumulated when this transcription should fire.
        public let sampleCount: Int

        /// Creates a new entry.
        public init(sampleCount: Int) {
            self.sampleCount = sampleCount
        }
    }

    private let entries: [Entry]

    /// Creates a scheduler that fires transcriptions at the given sample counts.
    public init(entries: [Entry]) {
        self.entries = entries
    }

    /// Start replaying the manifest.
    ///
    /// Waits for each entry's sample count to be reached, then triggers a transcription.
    public func start(engine: SchedulableEngine) async {
        for entry in entries {
            // Poll until enough samples have accumulated
            while engine.sampleCount < entry.sampleCount {
                try? await Task.sleep(nanoseconds: 10_000_000)  // 10ms poll
            }
            await engine.transcribeWorkingWindow(maxSampleCount: entry.sampleCount)
        }
    }

    /// No-op for manifest replay — the start() method exits when all entries are replayed.
    public func stop() async {}
}
