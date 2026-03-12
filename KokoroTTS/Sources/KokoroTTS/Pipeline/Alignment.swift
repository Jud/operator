import Accelerate
import Foundation

/// Converts duration model output to HAR decoder input via alignment matrix construction.
///
/// Pipeline: pred_dur → alignment matrix → asr = t_en @ alignment, derive F0 and N curves.
///
/// Uses Accelerate vDSP for matrix operations.
enum Alignment {
    struct Output {
        /// Acoustic sequence representation [512][frames].
        let asr: [[Float]]
        /// Fundamental frequency curve [frames * 2].
        let f0Pred: [Float]
        /// Noise curve [frames * 2].
        let nPred: [Float]
    }

    /// Build alignment matrix from predicted durations and compute decoder inputs.
    ///
    /// - Parameters:
    ///   - predDur: Per-token predicted durations in frames [N].
    ///   - tEn: Text encoder hidden states [512][N].
    ///   - tokenCount: Number of valid tokens.
    ///   - targetFrames: Target frame count for the selected bucket.
    /// - Returns: Decoder inputs (asr, F0, N).
    static func compute(
        predDur: [Float],
        tEn: [[Float]],
        tokenCount: Int,
        targetFrames: Int
    ) throws -> Output {
        let n = min(predDur.count, tokenCount)
        guard n > 0 else {
            throw KokoroError.alignmentFailed("Empty duration prediction")
        }

        // Normalize durations to sum to targetFrames
        let normalizedDur = normalizeDurations(predDur, tokenCount: n, targetFrames: targetFrames)

        // Build alignment matrix [N, targetFrames]
        let alignmentMatrix = buildAlignmentMatrix(
            durations: normalizedDur, tokenCount: n, frames: targetFrames)

        // Compute asr = t_en @ alignment: [512, N] @ [N, frames] → [512, frames]
        let hiddenDim = tEn.count
        var asr = [[Float]](repeating: [Float](repeating: 0, count: targetFrames), count: hiddenDim)

        for h in 0..<hiddenDim {
            let row = tEn[h]
            guard row.count >= n else { continue }
            // Matrix-vector multiply using vDSP for each hidden dim
            for f in 0..<targetFrames {
                var sum: Float = 0
                for t in 0..<n {
                    sum += row[t] * alignmentMatrix[t * targetFrames + f]
                }
                asr[h][f] = sum
            }
        }

        // Derive F0 and N curves from durations
        let f0Pred = deriveF0(durations: normalizedDur, tokenCount: n, frames: targetFrames)
        let nPred = deriveNoise(durations: normalizedDur, tokenCount: n, frames: targetFrames)

        return Output(asr: asr, f0Pred: f0Pred, nPred: nPred)
    }

    /// Normalize durations so they sum to targetFrames using proportional scaling.
    static func normalizeDurations(
        _ durations: [Float], tokenCount: Int, targetFrames: Int
    ) -> [Float] {
        let n = min(durations.count, tokenCount)
        var dur = Array(durations.prefix(n))

        // Clamp negatives to 0
        for i in 0..<n { dur[i] = max(dur[i], 0) }

        var total: Float = 0
        vDSP_sve(dur, 1, &total, vDSP_Length(n))

        guard total > 0 else {
            // Uniform distribution if all zeros
            let uniform = Float(targetFrames) / Float(n)
            return [Float](repeating: uniform, count: n)
        }

        let scale = Float(targetFrames) / total
        var scaled = [Float](repeating: 0, count: n)
        var s = scale
        vDSP_vsmul(dur, 1, &s, &scaled, 1, vDSP_Length(n))

        // Round and adjust to hit exact frame count
        var rounded = scaled.map { roundf($0) }
        var roundedTotal: Float = 0
        vDSP_sve(rounded, 1, &roundedTotal, vDSP_Length(n))
        let diff = Float(targetFrames) - roundedTotal

        if abs(diff) > 0 {
            // Distribute rounding error to longest duration tokens
            let adjustIdx = rounded.enumerated()
                .sorted { $0.element > $1.element }
                .first?.offset ?? 0
            rounded[adjustIdx] += diff
        }

        return rounded
    }

    /// Build alignment matrix: each token gets a contiguous span of frames.
    ///
    /// Result is a flat array [N * frames] where matrix[t][f] = 1.0 if frame f belongs to token t.
    static func buildAlignmentMatrix(
        durations: [Float], tokenCount: Int, frames: Int
    ) -> [Float] {
        let n = min(durations.count, tokenCount)
        var matrix = [Float](repeating: 0, count: n * frames)

        var cumStart: Float = 0
        for t in 0..<n {
            let dur = max(durations[t], 0)
            let startFrame = Int(cumStart)
            let endFrame = min(Int(cumStart + dur), frames)
            for f in startFrame..<endFrame {
                matrix[t * frames + f] = 1.0
            }
            cumStart += dur
        }

        return matrix
    }

    /// Derive F0 (fundamental frequency) curve from durations.
    ///
    /// Generates a smooth pitch contour at 2x frame rate. Each token's span gets
    /// a slight pitch curve centered on a base frequency.
    static func deriveF0(
        durations: [Float], tokenCount: Int, frames: Int
    ) -> [Float] {
        let outputLen = frames * 2
        var f0 = [Float](repeating: 0, count: outputLen)
        let baseF0: Float = 200.0  // Base frequency in Hz

        var cumStart: Float = 0
        let n = min(durations.count, tokenCount)
        for t in 0..<n {
            let dur = max(durations[t], 0)
            let startIdx = Int(cumStart * 2)
            let endIdx = min(Int((cumStart + dur) * 2), outputLen)
            let span = endIdx - startIdx
            guard span > 0 else { cumStart += dur; continue }

            for i in startIdx..<endIdx {
                let progress = Float(i - startIdx) / Float(max(span - 1, 1))
                // Slight downward intonation within each phoneme
                f0[i] = baseF0 * (1.0 - 0.1 * progress)
            }
            cumStart += dur
        }

        return f0
    }

    /// Derive noise curve from durations at 2x frame rate.
    static func deriveNoise(
        durations: [Float], tokenCount: Int, frames: Int
    ) -> [Float] {
        let outputLen = frames * 2
        var noise = [Float](repeating: 0, count: outputLen)
        let baseNoise: Float = 0.003

        var cumStart: Float = 0
        let n = min(durations.count, tokenCount)
        for t in 0..<n {
            let dur = max(durations[t], 0)
            let startIdx = Int(cumStart * 2)
            let endIdx = min(Int((cumStart + dur) * 2), outputLen)
            for i in startIdx..<endIdx {
                noise[i] = baseNoise
            }
            cumStart += dur
        }

        return noise
    }
}
