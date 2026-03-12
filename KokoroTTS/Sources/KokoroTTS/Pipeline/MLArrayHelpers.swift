import CoreML

/// Shared helpers for MLMultiArray operations used by the synthesis pipeline.
enum MLArrayHelpers {
    /// Extract float samples from an MLMultiArray, handling both float16 and float32 data types.
    static func extractFloats(from array: MLMultiArray, maxCount: Int? = nil) -> [Float] {
        let count = min(array.count, max(0, maxCount ?? array.count))
        guard count > 0 else { return [] }
        var samples = [Float](repeating: 0, count: count)
        if array.dataType == .float16 {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<count { samples[i] = Float(ptr[i]) }
        } else {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
            samples.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress!.update(from: ptr, count: count)
            }
        }
        return samples
    }

    /// Create a [1, 256] float32 style embedding MLMultiArray from a float vector.
    static func makeStyleArray(_ styleVector: [Float]) throws -> MLMultiArray {
        let dim = VoiceStore.styleDim
        let array = try MLMultiArray(shape: [1, dim as NSNumber], dataType: .float32)
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<min(styleVector.count, dim) {
            ptr[i] = styleVector[i]
        }
        return array
    }

    /// Create paired input_ids and attention_mask MLMultiArrays from token IDs.
    static func makeTokenInputs(
        tokenIds: [Int], maxLength: Int
    ) throws -> (inputIds: MLMultiArray, mask: MLMultiArray) {
        let n = min(tokenIds.count, maxLength)
        let inputIds = try MLMultiArray(shape: [1, maxLength as NSNumber], dataType: .int32)
        let mask = try MLMultiArray(shape: [1, maxLength as NSNumber], dataType: .int32)
        let inputPtr = inputIds.dataPointer.assumingMemoryBound(to: Int32.self)
        let maskPtr = mask.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<maxLength {
            if i < n {
                inputPtr[i] = Int32(tokenIds[i])
                maskPtr[i] = 1
            } else {
                inputPtr[i] = 0
                maskPtr[i] = 0
            }
        }
        return (inputIds, mask)
    }
}
