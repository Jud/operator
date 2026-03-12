import CoreML
import Foundation

/// Stage 1 of the two-stage pipeline: predicts phoneme durations and text encoder features.
///
/// Runs on CPU/GPU (LSTM layers lack ANE support).
///
/// Inputs:
/// - `input_ids` [1, N] int32: phoneme token IDs
/// - `attention_mask` [1, N] int32: 1 for real tokens, 0 for padding
/// - `ref_s` [1, 256] float32: voice style embedding
/// - `speed` [1] float32: speaking rate multiplier
///
/// Outputs:
/// - `pred_dur` [1, N] float32: predicted phoneme durations in frames
/// - `t_en` [1, 512, N] float32: text encoder hidden states
final class DurationModel: @unchecked Sendable {
    private let model: MLModel

    /// Fixed input length for the duration model.
    static let maxInputTokens = 128

    init(modelURL: URL) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU
        self.model = try MLModel(contentsOf: modelURL, configuration: config)
    }

    struct Output {
        /// Predicted duration per token in frames.
        let predDur: [Float]
        /// Text encoder hidden states [512, N].
        let tEn: [[Float]]
        /// Number of valid tokens.
        let tokenCount: Int
    }

    func predict(tokenIds: [Int], styleVector: [Float], speed: Float) throws -> Output {
        let n = min(tokenIds.count, Self.maxInputTokens)

        let (inputIds, mask) = try MLArrayHelpers.makeTokenInputs(
            tokenIds: tokenIds, maxLength: Self.maxInputTokens)

        let refS = try MLArrayHelpers.makeStyleArray(styleVector)

        let speedArray = try MLMultiArray(shape: [1], dataType: .float32)
        speedArray[0] = NSNumber(value: speed)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIds),
            "attention_mask": MLFeatureValue(multiArray: mask),
            "ref_s": MLFeatureValue(multiArray: refS),
            "speed": MLFeatureValue(multiArray: speedArray),
        ])

        let output = try model.prediction(from: input)

        guard let predDurArray = output.featureValue(for: "pred_dur")?.multiArrayValue else {
            throw KokoroError.inferenceFailed("Missing pred_dur output from duration model")
        }
        guard let tEnArray = output.featureValue(for: "t_en")?.multiArrayValue else {
            throw KokoroError.inferenceFailed("Missing t_en output from duration model")
        }

        // Extract pred_dur [1, N] → [Float]
        var predDur = [Float](repeating: 0, count: n)
        let durPtr = predDurArray.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<n {
            predDur[i] = durPtr[i]
        }

        // Extract t_en [1, 512, N] → [[Float]] (512 x N)
        let hiddenDim = 512
        var tEn = [[Float]](repeating: [Float](repeating: 0, count: n), count: hiddenDim)
        let tEnPtr = tEnArray.dataPointer.assumingMemoryBound(to: Float.self)
        for h in 0..<hiddenDim {
            for t in 0..<n {
                // Shape [1, 512, N]: index = 0*512*N + h*N + t
                tEn[h][t] = tEnPtr[h * Self.maxInputTokens + t]
            }
        }

        return Output(predDur: predDur, tEn: tEn, tokenCount: n)
    }
}
