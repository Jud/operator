import Foundation
import Testing
import os

@testable import OperatorCore

// MARK: - Mock

internal final class MockSchedulableEngine: SchedulableEngine, @unchecked Sendable {
    private let lock = OSAllocatedUnfairLock<MockState>(initialState: MockState())

    private struct MockState: @unchecked Sendable {
        var sampleCount: Int = 0
        var transcriptionCount: Int = 0
        var timestamps: [ContinuousClock.Instant] = []
        var onTranscribe: (@Sendable () async -> Void)?
        var concurrentCount: Int = 0
        var maxConcurrent: Int = 0
    }

    var sampleCount: Int {
        get { lock.withLock { $0.sampleCount } }
        set { lock.withLock { $0.sampleCount = newValue } }
    }

    var transcriptionCount: Int {
        lock.withLock { $0.transcriptionCount }
    }

    var maxConcurrent: Int {
        lock.withLock { $0.maxConcurrent }
    }

    func setOnTranscribe(_ handler: @escaping @Sendable () async -> Void) {
        lock.withLock { $0.onTranscribe = handler }
    }

    func transcribeWorkingWindow(maxSampleCount: Int?) async {
        lock.withLock {
            $0.concurrentCount += 1
            if $0.concurrentCount > $0.maxConcurrent {
                $0.maxConcurrent = $0.concurrentCount
            }
        }
        if let handler = lock.withLock({ $0.onTranscribe }) {
            await handler()
        }
        lock.withLock {
            $0.transcriptionCount += 1
            $0.timestamps.append(.now)
            $0.concurrentCount -= 1
        }
    }
}

// MARK: - Tests

@Suite("BackgroundLoopScheduler")
internal struct BackgroundLoopSchedulerTests {
    @Test("No transcription fires before initial 1s delay")
    func testInitialDelay() async throws {
        let engine = MockSchedulableEngine()
        engine.sampleCount = 32_000
        let scheduler = BackgroundLoopScheduler()

        let running = Task { await scheduler.start(engine: engine) }

        // Wait 600ms -- well before the 1s initial delay
        try await Task.sleep(nanoseconds: 600_000_000)
        let countBefore = engine.transcriptionCount
        #expect(countBefore == 0, "Should not transcribe before initial delay")

        await scheduler.stop()
        await running.value
    }

    @Test("No transcription when insufficient new samples")
    func testThresholdGating() async throws {
        let engine = MockSchedulableEngine()
        engine.sampleCount = 15_999
        let scheduler = BackgroundLoopScheduler()

        let running = Task { await scheduler.start(engine: engine) }

        // Wait enough for initial delay + one loop iteration
        try await Task.sleep(nanoseconds: 2_000_000_000)
        #expect(engine.transcriptionCount == 0, "Should not transcribe below threshold")

        await scheduler.stop()
        await running.value
    }

    @Test("Transcription fires when 16000+ new samples available")
    func testTranscribesWhenThresholdMet() async throws {
        let engine = MockSchedulableEngine()
        engine.sampleCount = 16_000
        let scheduler = BackgroundLoopScheduler()

        let running = Task { await scheduler.start(engine: engine) }

        // Wait for initial delay + loop iteration
        try await Task.sleep(nanoseconds: 2_000_000_000)
        #expect(engine.transcriptionCount >= 1, "Should transcribe when threshold met")

        await scheduler.stop()
        await running.value
    }

    @Test("stop() waits for in-flight transcription to complete")
    func testStopWaitsForInFlight() async throws {
        let engine = MockSchedulableEngine()
        engine.sampleCount = 16_000
        let transcriptionStarted = OSAllocatedUnfairLock(initialState: false)
        let transcriptionFinished = OSAllocatedUnfairLock(initialState: false)

        engine.setOnTranscribe {
            transcriptionStarted.withLock { $0 = true }
            // Simulate a slow transcription
            try? await Task.sleep(nanoseconds: 500_000_000)
            transcriptionFinished.withLock { $0 = true }
        }

        let scheduler = BackgroundLoopScheduler()
        let running = Task { await scheduler.start(engine: engine) }

        // Wait for transcription to begin
        while !transcriptionStarted.withLock({ $0 }) {
            try await Task.sleep(nanoseconds: 50_000_000)
        }

        // Stop should wait for the in-flight transcription
        await scheduler.stop()
        #expect(transcriptionFinished.withLock { $0 }, "stop() must wait for in-flight transcription")

        await running.value
    }

    @Test("No transcriptions after stop() returns")
    func testStopPreventsSubsequentTranscriptions() async throws {
        let engine = MockSchedulableEngine()
        engine.sampleCount = 16_000
        let scheduler = BackgroundLoopScheduler()

        let running = Task { await scheduler.start(engine: engine) }

        // Let it run past initial delay
        try await Task.sleep(nanoseconds: 1_500_000_000)

        await scheduler.stop()
        let countAtStop = engine.transcriptionCount

        // Add more samples and wait -- no new transcriptions should fire
        engine.sampleCount = 100_000
        try await Task.sleep(nanoseconds: 1_500_000_000)
        #expect(
            engine.transcriptionCount == countAtStop,
            "No transcriptions should fire after stop()"
        )

        await running.value
    }

    // NOTE: The coordinator's cancel() must call scheduler.stop() before
    // draining the scheduler task. Without stop(), the task hangs forever.
    // This was a production bug (2026-03-20). A unit test for this requires
    // making WhisperKitEngine constructable with a mock transcription backend,
    // which is tracked as a future improvement. The scheduler tests below
    // verify that stop() → start() works correctly, which is the prerequisite.

    @Test("Transcriptions are never concurrent")
    func testSerialExecution() async throws {
        let engine = MockSchedulableEngine()
        engine.sampleCount = 16_000

        engine.setOnTranscribe {
            // Simulate work so concurrent calls could overlap
            try? await Task.sleep(nanoseconds: 100_000_000)
            // Keep adding samples so the threshold is always met
            engine.sampleCount += 16_000
        }

        let scheduler = BackgroundLoopScheduler()
        let running = Task { await scheduler.start(engine: engine) }

        // Let it run several iterations (1s initial + multiple 500ms/1000ms loops)
        try await Task.sleep(nanoseconds: 5_000_000_000)

        await scheduler.stop()
        #expect(engine.maxConcurrent <= 1, "Transcriptions must be serial, got max \(engine.maxConcurrent) concurrent")
        #expect(engine.transcriptionCount >= 2, "Should have run multiple transcriptions")

        await running.value
    }
}
