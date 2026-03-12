import CoreML
import Foundation

/// Stage 2 of the two-stage pipeline: harmonic-phase vocoder targeting ANE.
///
/// Converts acoustic features to waveform audio.
///
/// Inputs:
/// - `asr` [1, 512, F] float32: acoustic sequence representation
/// - `F0_pred` [1, F*2] float32: fundamental frequency curve
/// - `N_pred` [1, F*2] float32: noise curve
/// - `ref_s` [1, 256] float32: voice style embedding
///
/// Output:
/// - `waveform` [1, S] float32: 24kHz mono PCM
final class HARDecoder: @unchecked Sendable {
    private let model: MLModel
    private let bucket: Bucket

    init(modelURL: URL, bucket: Bucket) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Allow ANE
        self.model = try MLModel(contentsOf: modelURL, configuration: config)
        self.bucket = bucket
    }

    /// Run HAR decoding to produce waveform audio.
    ///
    /// - Parameters:
    ///   - asr: Acoustic sequence representation [512][frames].
    ///   - f0Pred: Fundamental frequency curve [frames * 2].
    ///   - nPred: Noise curve [frames * 2].
    ///   - styleVector: 256-dim voice embedding.
    /// - Returns: PCM audio samples at 24kHz.
    func predict(
        asr: [[Float]],
        f0Pred: [Float],
        nPred: [Float],
        styleVector: [Float]
    ) throws -> [Float] {
        let frames = bucket.maxFrames
        let hiddenDim = 512

        // asr [1, 512, frames]
        let asrArray = try MLMultiArray(
            shape: [1, hiddenDim as NSNumber, frames as NSNumber], dataType: .float32)
        let asrPtr = asrArray.dataPointer.assumingMemoryBound(to: Float.self)
        for h in 0..<hiddenDim {
            let row = h < asr.count ? asr[h] : [Float](repeating: 0, count: frames)
            for f in 0..<frames {
                asrPtr[h * frames + f] = f < row.count ? row[f] : 0
            }
        }

        // F0_pred [1, frames*2]
        let f0Len = frames * 2
        let f0Array = try MLMultiArray(shape: [1, f0Len as NSNumber], dataType: .float32)
        let f0Ptr = f0Array.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<f0Len {
            f0Ptr[i] = i < f0Pred.count ? f0Pred[i] : 0
        }

        // N_pred [1, frames*2]
        let nArray = try MLMultiArray(shape: [1, f0Len as NSNumber], dataType: .float32)
        let nPtr = nArray.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<f0Len {
            nPtr[i] = i < nPred.count ? nPred[i] : 0
        }

        // ref_s [1, 256]
        let refS = try MLArrayHelpers.makeStyleArray(styleVector)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "asr": MLFeatureValue(multiArray: asrArray),
            "F0_pred": MLFeatureValue(multiArray: f0Array),
            "N_pred": MLFeatureValue(multiArray: nArray),
            "ref_s": MLFeatureValue(multiArray: refS),
        ])

        let output = try model.prediction(from: input)

        guard let waveform = output.featureValue(for: "waveform")?.multiArrayValue else {
            throw KokoroError.inferenceFailed("Missing waveform output from HAR decoder")
        }

        return MLArrayHelpers.extractFloats(from: waveform)
    }
}
