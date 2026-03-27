// MPSGraphDecoder — Qwen 3.5 DeltaNet inference via MPSGraph.
//
// Bypasses CoreML for linear ops to use native LUT-fused matmul.
// Build layer by layer, verify each against reference activations.

import Foundation
import Metal
import MetalPerformanceShadersGraph

/// Loads FP16 numpy arrays from disk into MTLBuffers.
final class NpyLoader {
    let device: MTLDevice

    init(device: MTLDevice) {
        self.device = device
    }

    /// Load a .npy file into a MTLBuffer. Returns (buffer, shape, elementCount).
    func load(_ path: String) -> (MTLBuffer, [Int], Int) {
        let data = try! Data(contentsOf: URL(fileURLWithPath: path))
        // Parse numpy header
        let hdrLen = Int(data[8]) | (Int(data[9]) << 8)
        let hdrEnd = 10 + hdrLen
        let hdr = String(data: data[10..<hdrEnd], encoding: .ascii)!

        // Extract shape
        let shapeMatch = hdr.range(of: #"\([\d, ]+\)"#, options: .regularExpression)!
        let shapeStr = String(hdr[shapeMatch]).dropFirst().dropLast()
        let shape = shapeStr.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        let count = shape.reduce(1, *)

        // Detect dtype
        let isFP16 = hdr.contains("float16") || hdr.contains("<f2")
        let isFP32 = hdr.contains("float32") || hdr.contains("<f4")
        let isInt32 = hdr.contains("int32") || hdr.contains("<i4")

        let raw = data[hdrEnd...]
        let bytesPerElement = isFP16 ? 2 : 4
        let bufferSize = count * bytesPerElement

        let buffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)!

        raw.withUnsafeBytes { src in
            buffer.contents().copyMemory(from: src.baseAddress!, byteCount: bufferSize)
        }

        return (buffer, shape, count)
    }

    /// Load and return as MPSGraphTensorData. Auto-detects dtype from the .npy header.
    func loadTensor(_ path: String, dataType: MPSDataType? = nil) -> (MPSGraphTensorData, [Int]) {
        let data = try! Data(contentsOf: URL(fileURLWithPath: path))
        let hdrLen = Int(data[8]) | (Int(data[9]) << 8)
        let hdr = String(data: data[10..<(10 + hdrLen)], encoding: .ascii)!

        let isFP16 = hdr.contains("float16") || hdr.contains("<f2")
        let isFP32 = hdr.contains("float32") || hdr.contains("<f4")
        let isInt32 = hdr.contains("int32") || hdr.contains("<i4")
        let detectedType: MPSDataType = isInt32 ? .int32 : (isFP32 ? .float32 : .float16)
        let dt = dataType ?? detectedType

        let (buffer, shape, _) = load(path)
        let nsShape = shape.map { $0 as NSNumber }
        let tensorData = MPSGraphTensorData(buffer, shape: nsShape, dataType: dt)
        return (tensorData, shape)
    }
}

/// Compare two buffers (FP16 or FP32) and report max absolute difference.
func compareBuffers(
    _ actual: MTLBuffer,
    _ expected: MTLBuffer,
    count: Int,
    label: String,
    tolerance: Float = 0.05,
    fp32: Bool = false
) -> Bool {
    // Handle FP32 or FP16 comparison
    guard !fp32 else {
        let a = actual.contents().bindMemory(to: Float.self, capacity: count)
        let e = expected.contents().bindMemory(to: Float.self, capacity: count)
        var maxDiff: Float = 0
        var maxIdx = 0
        var sumDiff: Float = 0
        for i in 0..<count {
            let diff = abs(a[i] - e[i])
            sumDiff += diff
            if diff > maxDiff { maxDiff = diff; maxIdx = i }
        }
        let avgDiff = sumDiff / Float(count)
        let pass = maxDiff < tolerance
        let symbol = pass ? "✓" : "✗"
        print("  [\(symbol)] \(label): maxDiff=\(String(format: "%.6f", maxDiff)) avgDiff=\(String(format: "%.8f", avgDiff)) at idx=\(maxIdx)")
        return pass
    }
    let a = actual.contents().bindMemory(to: Float16.self, capacity: count)
    let e = expected.contents().bindMemory(to: Float16.self, capacity: count)

    var maxDiff: Float = 0
    var maxIdx = 0
    var sumDiff: Float = 0

    for i in 0..<count {
        let diff = abs(Float(a[i]) - Float(e[i]))
        sumDiff += diff
        if diff > maxDiff {
            maxDiff = diff
            maxIdx = i
        }
    }

    let avgDiff = sumDiff / Float(count)
    let pass = maxDiff < tolerance

    let symbol = pass ? "✓" : "✗"
    print(
        "  [\(symbol)] \(label): maxDiff=\(String(format: "%.4f", maxDiff)) avgDiff=\(String(format: "%.6f", avgDiff)) at idx=\(maxIdx)"
    )

    return pass
}
