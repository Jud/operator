// Realistic generation benchmark: each token depends on the previous one's output.
//
// Build & run:
//   swiftc -O -framework CoreML -framework Foundation \
//     scripts/coreml-bench/realistic_gen.swift -o /tmp/coreml-realgen
//   /tmp/coreml-realgen

import CoreML
import Foundation

func loadNpy(_ path: String) throws -> MLMultiArray {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let headerLen = Int(data[8]) | (Int(data[9]) << 8)
    let headerEnd = 10 + headerLen
    let header = String(data: data[10..<headerEnd], encoding: .ascii)!
    let shapeMatch = header.range(of: #"\([\d, ]+\)"#, options: .regularExpression)!
    let shapeStr = String(header[shapeMatch]).dropFirst().dropLast()
    let shape = shapeStr.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    let isFloat = header.contains("float32") || header.contains("<f4")
    let isInt32 = header.contains("int32") || header.contains("<i4")
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
    } else if isInt32 {
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
    }
    fatalError("Unsupported dtype: \(header)")
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

func makeTokenInput(_ tokenId: Int32) -> MLMultiArray {
    let arr = try! MLMultiArray(shape: [1, 1], dataType: .int32)
    arr.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = tokenId
    return arr
}

// MARK: - Main

let base = "models/split2"
let inputDir = "\(base)/test_inputs"

print("Loading models...")
let t0 = CFAbsoluteTimeGetCurrent()
let configGPU = MLModelConfiguration()
configGPU.computeUnits = .cpuAndGPU
let chunkA = try! MLModel(contentsOf: MLModel.compileModel(at: URL(fileURLWithPath: "\(base)/chunk_a.mlpackage")), configuration: configGPU)
let chunkB = try! MLModel(contentsOf: MLModel.compileModel(at: URL(fileURLWithPath: "\(base)/chunk_b.mlpackage")), configuration: configGPU)
print("Loaded in \(String(format: "%.0f", (CFAbsoluteTimeGetCurrent() - t0) * 1000))ms")

// Load initial DeltaNet states (from PyTorch prefill of "The capital of France is")
let ropeCos = try! loadNpy("\(inputDir)/rope_cos.npy")
let ropeSin = try! loadNpy("\(inputDir)/rope_sin.npy")

var aConv: [MLMultiArray] = []
var aRec: [MLMultiArray] = []
var bConv: [MLMultiArray] = []
var bRec: [MLMultiArray] = []
for j in 0..<9 {
    aConv.append(try! loadNpy("\(inputDir)/a_conv_\(j).npy"))
    aRec.append(try! loadNpy("\(inputDir)/a_rec_\(j).npy"))
    bConv.append(try! loadNpy("\(inputDir)/b_conv_\(j).npy"))
    bRec.append(try! loadNpy("\(inputDir)/b_rec_\(j).npy"))
}

func runFullStep(tokenId: Int32,
                 aConvStates: [MLMultiArray], aRecStates: [MLMultiArray],
                 bConvStates: [MLMultiArray], bRecStates: [MLMultiArray])
    -> (nextToken: Int, aConvOut: [MLMultiArray], aRecOut: [MLMultiArray],
        bConvOut: [MLMultiArray], bRecOut: [MLMultiArray])
{
    // Chunk A
    var dictA: [String: MLFeatureValue] = [
        "input_id": MLFeatureValue(multiArray: makeTokenInput(tokenId)),
        "rope_cos": MLFeatureValue(multiArray: ropeCos),
        "rope_sin": MLFeatureValue(multiArray: ropeSin),
    ]
    for j in 0..<9 {
        dictA["conv_state_\(j)"] = MLFeatureValue(multiArray: aConvStates[j])
        dictA["rec_state_\(j)"] = MLFeatureValue(multiArray: aRecStates[j])
    }
    let outA = try! chunkA.prediction(from: MLDictionaryFeatureProvider(dictionary: dictA))

    // Chunk B
    var dictB: [String: MLFeatureValue] = [
        "hidden": outA.featureValue(for: "hidden")!,
        "rope_cos": MLFeatureValue(multiArray: ropeCos),
        "rope_sin": MLFeatureValue(multiArray: ropeSin),
    ]
    for j in 0..<9 {
        dictB["conv_state_\(j)"] = MLFeatureValue(multiArray: bConvStates[j])
        dictB["rec_state_\(j)"] = MLFeatureValue(multiArray: bRecStates[j])
    }
    let outB = try! chunkB.prediction(from: MLDictionaryFeatureProvider(dictionary: dictB))

    // Extract outputs
    let logits = outB.featureValue(for: "logits")!.multiArrayValue!
    let nextTok = argmax(logits)

    var newAConv: [MLMultiArray] = []
    var newARec: [MLMultiArray] = []
    var newBConv: [MLMultiArray] = []
    var newBRec: [MLMultiArray] = []
    for j in 0..<9 {
        newAConv.append(outA.featureValue(for: "conv_state_\(j)_out")!.multiArrayValue!)
        newARec.append(outA.featureValue(for: "rec_state_\(j)_out")!.multiArrayValue!)
        newBConv.append(outB.featureValue(for: "conv_state_\(j)_out")!.multiArrayValue!)
        newBRec.append(outB.featureValue(for: "rec_state_\(j)_out")!.multiArrayValue!)
    }

    return (nextTok, newAConv, newARec, newBConv, newBRec)
}

// MARK: - Realistic generation benchmark

print("\n--- Realistic generation (each token depends on previous) ---")

let numTokens = 15
let startToken: Int32 = 11751  // " Paris" (first token after "The capital of France is")

// Warmup
var curAConv = aConv, curARec = aRec, curBConv = bConv, curBRec = bRec
for _ in 0..<3 {
    let (_, ac, ar, bc, br) = runFullStep(
        tokenId: startToken,
        aConvStates: curAConv, aRecStates: curARec,
        bConvStates: curBConv, bRecStates: curBRec)
    curAConv = ac; curARec = ar; curBConv = bc; curBRec = br
}

// Actual benchmark
var allTimes: [Double] = []
for run in 0..<5 {
    curAConv = aConv; curARec = aRec; curBConv = bConv; curBRec = bRec
    var currentToken: Int32 = startToken
    var generated: [Int] = []
    var stepTimes: [Double] = []

    let genStart = CFAbsoluteTimeGetCurrent()

    for _ in 0..<numTokens {
        let stepStart = CFAbsoluteTimeGetCurrent()

        let (nextTok, ac, ar, bc, br) = runFullStep(
            tokenId: currentToken,
            aConvStates: curAConv, aRecStates: curARec,
            bConvStates: curBConv, bRecStates: curBRec)

        stepTimes.append((CFAbsoluteTimeGetCurrent() - stepStart) * 1000)
        curAConv = ac; curARec = ar; curBConv = bc; curBRec = br
        generated.append(nextTok)
        currentToken = Int32(nextTok)
    }

    let totalMs = (CFAbsoluteTimeGetCurrent() - genStart) * 1000
    allTimes.append(totalMs)

    if run == 0 {
        stepTimes.sort()
        print("  Generated tokens: \(generated.prefix(10))...")
        print("  Per-step median: \(String(format: "%.1f", stepTimes[stepTimes.count/2]))ms")
        print("  Per-step range: \(String(format: "%.1f", stepTimes[0]))ms - \(String(format: "%.1f", stepTimes.last!))ms")
    }
}

allTimes.sort()
let medTotal = allTimes[2]
let tokPerSec = Double(numTokens) / (medTotal / 1000)
print("\n  \(numTokens) tokens in \(String(format: "%.0f", medTotal))ms")
print("  \(String(format: "%.1f", medTotal / Double(numTokens)))ms/tok")
print("  \(String(format: "%.0f", tokPerSec)) tok/s (realistic, with token dependencies)")
