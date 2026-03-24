// Swift inference test: loads exported numpy inputs, runs 2-chunk pipeline,
// verifies output matches Python (top1 = 13).
//
// Build & run:
//   swiftc -O -framework CoreML -framework Foundation \
//     scripts/coreml-bench/inference.swift -o /tmp/coreml-infer
//   /tmp/coreml-infer

import CoreML
import Foundation

// MARK: - Numpy Loading

func loadNpy(_ path: String) throws -> MLMultiArray {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    // Parse numpy .npy format: 10-byte magic + header + raw data
    // Header starts at byte 8, length at bytes 8-9 (little-endian uint16)
    let headerLen = Int(data[8]) | (Int(data[9]) << 8)
    let headerEnd = 10 + headerLen
    let header = String(data: data[10..<headerEnd], encoding: .ascii)!

    // Parse shape from header: 'shape': (1, 6144, 4)
    let shapeMatch = header.range(of: #"\([\d, ]+\)"#, options: .regularExpression)!
    let shapeStr = String(header[shapeMatch]).dropFirst().dropLast()  // remove parens
    let shape = shapeStr.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }

    // Parse dtype
    let isFloat = header.contains("float32") || header.contains("<f4")
    let isInt = header.contains("int32") || header.contains("<i4")
    let isInt64 = header.contains("int64") || header.contains("<i8")

    let rawData = data[headerEnd...]
    let count = shape.reduce(1, *)

    if isFloat {
        let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        rawData.withUnsafeBytes { buf in
            let src = buf.bindMemory(to: Float.self)
            let dst = arr.dataPointer.bindMemory(to: Float.self, capacity: count)
            for i in 0..<count { dst[i] = src[i] }
        }
        return arr
    } else if isInt {
        let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
        rawData.withUnsafeBytes { buf in
            let src = buf.bindMemory(to: Int32.self)
            let dst = arr.dataPointer.bindMemory(to: Int32.self, capacity: count)
            for i in 0..<count { dst[i] = src[i] }
        }
        return arr
    } else if isInt64 {
        let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
        rawData.withUnsafeBytes { buf in
            let src = buf.bindMemory(to: Int64.self)
            let dst = arr.dataPointer.bindMemory(to: Int32.self, capacity: count)
            for i in 0..<count { dst[i] = Int32(src[i]) }
        }
        return arr
    } else {
        fatalError("Unsupported dtype in \(path): \(header)")
    }
}

func argmax(_ arr: MLMultiArray) -> Int {
    let count = arr.count
    let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: count)
    var maxIdx = 0
    var maxVal = ptr[0]
    for i in 1..<count {
        if ptr[i] > maxVal { maxVal = ptr[i]; maxIdx = i }
    }
    return maxIdx
}

// MARK: - Main

let base = "models/split2"
let inputDir = "\(base)/test_inputs"

print("Loading models...")
let t0 = CFAbsoluteTimeGetCurrent()

let configA = MLModelConfiguration()
configA.computeUnits = .cpuAndGPU
let compiledA = try! MLModel.compileModel(at: URL(fileURLWithPath: "\(base)/chunk_a.mlpackage"))
let chunkA = try! MLModel(contentsOf: compiledA, configuration: configA)

let configB = MLModelConfiguration()
configB.computeUnits = .cpuAndGPU
let compiledB = try! MLModel.compileModel(at: URL(fileURLWithPath: "\(base)/chunk_b.mlpackage"))
let chunkB = try! MLModel(contentsOf: compiledB, configuration: configB)

print("Loaded in \(String(format: "%.0f", (CFAbsoluteTimeGetCurrent() - t0) * 1000))ms")

// Load test inputs
print("Loading test inputs...")
let tokenId = try! loadNpy("\(inputDir)/token_id.npy")
let ropeCos = try! loadNpy("\(inputDir)/rope_cos.npy")
let ropeSin = try! loadNpy("\(inputDir)/rope_sin.npy")

// Build chunk A input
var dictA: [String: MLFeatureValue] = [
    "input_id": MLFeatureValue(multiArray: tokenId),
    "rope_cos": MLFeatureValue(multiArray: ropeCos),
    "rope_sin": MLFeatureValue(multiArray: ropeSin),
]
for j in 0..<9 {
    dictA["conv_state_\(j)"] = MLFeatureValue(multiArray: try! loadNpy("\(inputDir)/a_conv_\(j).npy"))
    dictA["rec_state_\(j)"] = MLFeatureValue(multiArray: try! loadNpy("\(inputDir)/a_rec_\(j).npy"))
}
let providerA = try! MLDictionaryFeatureProvider(dictionary: dictA)

// Run chunk A
// Diagnostic: verify loaded values match Python
let tidPtr = tokenId.dataPointer.bindMemory(to: Int32.self, capacity: tokenId.count)
print("  input_id: \(tidPtr[0])")
print("  rope_cos shape: \(ropeCos.shape)")

let convSum = {() -> Float in
    let conv0 = dictA["conv_state_0"]!.multiArrayValue!
    let ptr = conv0.dataPointer.bindMemory(to: Float.self, capacity: conv0.count)
    var s: Float = 0; for i in 0..<conv0.count { s += ptr[i] }; return s
}()
print("  a_conv_0 sum: \(convSum)")

// Verify all input names match model expectations
print("\nModel A expected inputs:")
for desc in chunkA.modelDescription.inputDescriptionsByName {
    let c = desc.value.multiArrayConstraint
    print("  \(desc.key): \(c?.shape ?? [])")
}
print("Provider A keys: \(dictA.keys.sorted())")

// Check for any shape mismatch
for desc in chunkA.modelDescription.inputDescriptionsByName {
    if let provided = dictA[desc.key]?.multiArrayValue,
       let expected = desc.value.multiArrayConstraint?.shape {
        let pShape = (0..<provided.shape.count).map { provided.shape[$0].intValue }
        let eShape = expected.map { $0.intValue }
        if pShape != eShape {
            print("  MISMATCH: \(desc.key) provided \(pShape) vs expected \(eShape)")
        }
    }
}

// Dump first 5 values of conv_state_0 to verify numpy loading
let conv0Check = dictA["conv_state_0"]!.multiArrayValue!
let conv0Ptr = conv0Check.dataPointer.bindMemory(to: Float.self, capacity: 5)
print("  a_conv_0 first 5: \(conv0Ptr[0]), \(conv0Ptr[1]), \(conv0Ptr[2]), \(conv0Ptr[3]), \(conv0Ptr[4])")

// Dump rec_state_0 stats
let rec0Check = dictA["rec_state_0"]!.multiArrayValue!
let rec0Ptr = rec0Check.dataPointer.bindMemory(to: Float.self, capacity: rec0Check.count)
var rec0Sum: Float = 0
var rec0Max: Float = 0
for i in 0..<rec0Check.count { rec0Sum += rec0Ptr[i]; rec0Max = max(rec0Max, abs(rec0Ptr[i])) }
print("  a_rec_0 sum: \(rec0Sum), max abs: \(rec0Max)")

print("\nRunning pipeline...")
let outA = try! chunkA.prediction(from: providerA)

// Build chunk B input
var dictB: [String: MLFeatureValue] = [
    "hidden": outA.featureValue(for: "hidden")!,
    "rope_cos": MLFeatureValue(multiArray: ropeCos),
    "rope_sin": MLFeatureValue(multiArray: ropeSin),
]
for j in 0..<9 {
    dictB["conv_state_\(j)"] = MLFeatureValue(multiArray: try! loadNpy("\(inputDir)/b_conv_\(j).npy"))
    dictB["rec_state_\(j)"] = MLFeatureValue(multiArray: try! loadNpy("\(inputDir)/b_rec_\(j).npy"))
}
let providerB = try! MLDictionaryFeatureProvider(dictionary: dictB)

// Run chunk B
let outB = try! chunkB.prediction(from: providerB)
let logits = outB.featureValue(for: "logits")!.multiArrayValue!
let top1 = argmax(logits)

let expectedTop1 = try! loadNpy("\(inputDir)/expected_top1.npy")
    .dataPointer.bindMemory(to: Int32.self, capacity: 1)[0]

print("Top1: \(top1) (expected \(expectedTop1))")
print("Match: \(top1 == Int(expectedTop1))")

// Benchmark: warm decode
print("\n--- Warm decode benchmark ---")
var times: [Double] = []

// Warmup
for _ in 0..<5 {
    let _ = try! chunkA.prediction(from: providerA)
    let oA = try! chunkA.prediction(from: providerA)
    var dB = dictB
    dB["hidden"] = oA.featureValue(for: "hidden")!
    let pB = try! MLDictionaryFeatureProvider(dictionary: dB)
    let _ = try! chunkB.prediction(from: pB)
}

for _ in 0..<20 {
    let tStart = CFAbsoluteTimeGetCurrent()

    let oA = try! chunkA.prediction(from: providerA)
    var dB = dictB
    dB["hidden"] = oA.featureValue(for: "hidden")!
    let pB = try! MLDictionaryFeatureProvider(dictionary: dB)
    let _ = try! chunkB.prediction(from: pB)

    times.append((CFAbsoluteTimeGetCurrent() - tStart) * 1000)
}

times.sort()
let median = times[10]
print("Median: \(String(format: "%.1f", median))ms (\(String(format: "%.0f", 1000.0 / median)) tok/s)")
print("Min: \(String(format: "%.1f", times[0]))ms  Max: \(String(format: "%.1f", times[19]))ms")
