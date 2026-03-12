import Testing
@testable import KokoroTTS

@Suite("Alignment")
struct AlignmentTests {
    @Test("Duration normalization sums to target frames")
    func normalizeDurations() {
        let durations: [Float] = [3.0, 5.0, 2.0]
        let targetFrames = 72
        let normalized = Alignment.normalizeDurations(
            durations, tokenCount: 3, targetFrames: targetFrames)

        let sum = normalized.reduce(0, +)
        #expect(abs(sum - Float(targetFrames)) < 1.0)
    }

    @Test("Alignment matrix has correct shape")
    func alignmentMatrixShape() {
        let durations: [Float] = [10, 20, 42]
        let matrix = Alignment.buildAlignmentMatrix(
            durations: durations, tokenCount: 3, frames: 72)

        #expect(matrix.count == 3 * 72)
    }

    @Test("Alignment matrix rows are contiguous")
    func alignmentMatrixContiguous() {
        let durations: [Float] = [10, 20, 42]
        let frames = 72
        let matrix = Alignment.buildAlignmentMatrix(
            durations: durations, tokenCount: 3, frames: frames)

        // Each token should own a contiguous span
        for t in 0..<3 {
            var firstOne = -1
            var lastOne = -1
            for f in 0..<frames {
                if matrix[t * frames + f] == 1.0 {
                    if firstOne == -1 { firstOne = f }
                    lastOne = f
                }
            }
            if firstOne >= 0 {
                // All frames between first and last should be 1.0
                for f in firstOne...lastOne {
                    #expect(matrix[t * frames + f] == 1.0)
                }
            }
        }
    }

    @Test("Zero durations handled gracefully")
    func zeroDurations() {
        let durations: [Float] = [0, 0, 0]
        let normalized = Alignment.normalizeDurations(
            durations, tokenCount: 3, targetFrames: 72)

        let sum = normalized.reduce(0, +)
        #expect(abs(sum - 72.0) < 1.0)
    }

    @Test("Negative durations clamped to zero")
    func negativeDurations() {
        let durations: [Float] = [-5, 10, 5]
        let normalized = Alignment.normalizeDurations(
            durations, tokenCount: 3, targetFrames: 72)

        // After clamping, [-5,10,5] becomes [0,10,5], total=15, scale to 72
        let sum = normalized.reduce(0, +)
        #expect(abs(sum - 72.0) < 1.0)
        #expect(normalized[0] >= 0)
    }

    @Test("F0 curve has correct length")
    func f0Length() {
        let durations: [Float] = [10, 20, 42]
        let frames = 72
        let f0 = Alignment.deriveF0(durations: durations, tokenCount: 3, frames: frames)

        #expect(f0.count == frames * 2)
    }

    @Test("Noise curve has correct length")
    func noiseLength() {
        let durations: [Float] = [10, 20, 42]
        let frames = 72
        let noise = Alignment.deriveNoise(durations: durations, tokenCount: 3, frames: frames)

        #expect(noise.count == frames * 2)
    }

    @Test("Full alignment compute doesn't crash")
    func fullCompute() throws {
        let predDur: [Float] = [3.0, 5.0, 2.0, 4.0, 6.0]
        let hiddenDim = 512
        let tokenCount = 5
        let tEn = (0..<hiddenDim).map { _ in
            (0..<tokenCount).map { _ in Float.random(in: -1...1) }
        }

        let output = try Alignment.compute(
            predDur: predDur, tEn: tEn, tokenCount: tokenCount, targetFrames: 72)

        #expect(output.asr.count == hiddenDim)
        #expect(output.asr[0].count == 72)
        #expect(output.f0Pred.count == 144)
        #expect(output.nPred.count == 144)
    }
}
