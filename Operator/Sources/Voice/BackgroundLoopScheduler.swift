import Foundation
import os

/// Production scheduler that runs a background loop to trigger
/// periodic transcriptions of the working window.
///
/// Behavioral contracts:
/// 1. Initial delay: 1 second before first transcription attempt.
/// 2. Threshold: only transcribes when >= 16,000 new samples since last.
/// 3. Pacing: 500ms after first transcription, 1000ms after subsequent.
/// 4. Interruptible: stop() interrupts sleep, returns after current transcription completes.
/// 5. Serial: never calls transcribeWorkingWindow() concurrently.
public final class BackgroundLoopScheduler: TranscriptionScheduler, @unchecked Sendable {
    /// Minimum new samples before a transcription is triggered (1s at 16kHz).
    private static let sampleThreshold = 16_000

    private let stopped = OSAllocatedUnfairLock(initialState: false)
    private let sleepContinuation = OSAllocatedUnfairLock<UnsafeContinuation<Void, Never>?>(initialState: nil)
    private let loopTask = OSAllocatedUnfairLock<Task<Void, Never>?>(initialState: nil)

    public init() {}

    public func start(engine: SchedulableEngine) async {
        stopped.withLock { $0 = false }

        let task = Task { [weak self] in
            guard let self else { return }
            await self.interruptibleSleep(nanoseconds: 1_000_000_000)

            var lastSampleCount = 0
            var transcriptionCount = 0

            while !self.stopped.withLock({ $0 }) {
                let current = engine.sampleCount
                if current - lastSampleCount >= Self.sampleThreshold {
                    await engine.transcribeWorkingWindow(maxSampleCount: nil)
                    lastSampleCount = current
                    transcriptionCount += 1
                }

                guard !self.stopped.withLock({ $0 }) else { return }
                let intervalNs: UInt64 =
                    transcriptionCount <= 1
                    ? 500_000_000
                    : 1_000_000_000
                await self.interruptibleSleep(nanoseconds: intervalNs)
            }
        }
        loopTask.withLock { $0 = task }
        await task.value
    }

    public func stop() async {
        stopped.withLock { $0 = true }
        wakeSleep()
        if let task = loopTask.withLock({ $0 }) {
            await task.value
        }
        loopTask.withLock { $0 = nil }
    }

    // MARK: - Interruptible Sleep

    private func interruptibleSleep(nanoseconds: UInt64) async {
        await withUnsafeContinuation { continuation in
            sleepContinuation.withLock { $0 = continuation }
            if stopped.withLock({ $0 }) {
                sleepContinuation.withLock { cont in
                    cont?.resume()
                    cont = nil
                }
                return
            }
            Task { [weak self] in
                try? await Task.sleep(nanoseconds: nanoseconds)
                self?.wakeSleep()
            }
        }
    }

    private func wakeSleep() {
        sleepContinuation.withLock { cont in
            cont?.resume()
            cont = nil
        }
    }
}
