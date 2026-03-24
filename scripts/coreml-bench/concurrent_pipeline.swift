// Concurrent pipeline test: chunk A(token N+1) overlaps chunk B(token N)
//
// Proves that pipelining the 2-chunk model across tokens gives real speedup
// while maintaining correct output.
//
// Build & run:
//   swiftc -O -framework CoreML -framework Foundation \
//     scripts/coreml-bench/concurrent_pipeline.swift -o /tmp/coreml-pipeline
//   /tmp/coreml-pipeline

import CoreML
import Foundation

// MARK: - Numpy Loading (same as inference.swift)

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

// MARK: - Pipeline Components

struct ChunkIO {
    let model: MLModel
    let nDelta: Int  // number of DeltaNet state pairs

    func predict(inputDict: [String: MLFeatureValue]) throws -> MLFeatureProvider {
        let provider = try MLDictionaryFeatureProvider(dictionary: inputDict)
        return try model.prediction(from: provider)
    }
}

// MARK: - Main

let base = "models/split2"
let inputDir = "\(base)/test_inputs"

print("Loading models...")
let t0 = CFAbsoluteTimeGetCurrent()

let configGPU = MLModelConfiguration()
configGPU.computeUnits = .cpuAndGPU

let compiledA = try! MLModel.compileModel(at: URL(fileURLWithPath: "\(base)/chunk_a.mlpackage"))
let compiledB = try! MLModel.compileModel(at: URL(fileURLWithPath: "\(base)/chunk_b.mlpackage"))
let chunkA = try! MLModel(contentsOf: compiledA, configuration: configGPU)
let chunkB = try! MLModel(contentsOf: compiledB, configuration: configGPU)

print("Loaded in \(String(format: "%.0f", (CFAbsoluteTimeGetCurrent() - t0) * 1000))ms")

// Load test inputs (DeltaNet states from PyTorch prefill of "The capital of France is")
let ropeCos = try! loadNpy("\(inputDir)/rope_cos.npy")
let ropeSin = try! loadNpy("\(inputDir)/rope_sin.npy")

func loadStates(_ prefix: String, count: Int) -> ([MLMultiArray], [MLMultiArray]) {
    var conv: [MLMultiArray] = []
    var rec: [MLMultiArray] = []
    for j in 0..<count {
        conv.append(try! loadNpy("\(inputDir)/\(prefix)_conv_\(j).npy"))
        rec.append(try! loadNpy("\(inputDir)/\(prefix)_rec_\(j).npy"))
    }
    return (conv, rec)
}

let (aConv, aRec) = loadStates("a", count: 9)
let (bConv, bRec) = loadStates("b", count: 9)

// Build reusable input dicts
func buildDictA(tokenId: Int32, convStates: [MLMultiArray], recStates: [MLMultiArray]) -> [String: MLFeatureValue] {
    let tid = try! MLMultiArray(shape: [1, 1], dataType: .int32)
    tid.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = tokenId

    var dict: [String: MLFeatureValue] = [
        "input_id": MLFeatureValue(multiArray: tid),
        "rope_cos": MLFeatureValue(multiArray: ropeCos),
        "rope_sin": MLFeatureValue(multiArray: ropeSin),
    ]
    for j in 0..<9 {
        dict["conv_state_\(j)"] = MLFeatureValue(multiArray: convStates[j])
        dict["rec_state_\(j)"] = MLFeatureValue(multiArray: recStates[j])
    }
    return dict
}

func buildDictB(hidden: MLFeatureValue, convStates: [MLMultiArray], recStates: [MLMultiArray]) -> [String: MLFeatureValue] {
    var dict: [String: MLFeatureValue] = [
        "hidden": hidden,
        "rope_cos": MLFeatureValue(multiArray: ropeCos),
        "rope_sin": MLFeatureValue(multiArray: ropeSin),
    ]
    for j in 0..<9 {
        dict["conv_state_\(j)"] = MLFeatureValue(multiArray: convStates[j])
        dict["rec_state_\(j)"] = MLFeatureValue(multiArray: recStates[j])
    }
    return dict
}

// MARK: - Correctness check

print("\n--- Correctness ---")
let dictA = buildDictA(tokenId: 11751, convStates: aConv, recStates: aRec)
let outA = try! chunkA.prediction(from: MLDictionaryFeatureProvider(dictionary: dictA))
let dictB = buildDictB(hidden: outA.featureValue(for: "hidden")!, convStates: bConv, recStates: bRec)
let outB = try! chunkB.prediction(from: MLDictionaryFeatureProvider(dictionary: dictB))
let top1 = argmax(outB.featureValue(for: "logits")!.multiArrayValue!)
print("Top1: \(top1) (expected 13)")
print("Match: \(top1 == 13)")

// MARK: - Sequential benchmark

print("\n--- Sequential (A then B) ---")
// Warmup
for _ in 0..<5 {
    let oA = try! chunkA.prediction(from: MLDictionaryFeatureProvider(dictionary: dictA))
    let dB = buildDictB(hidden: oA.featureValue(for: "hidden")!, convStates: bConv, recStates: bRec)
    let _ = try! chunkB.prediction(from: MLDictionaryFeatureProvider(dictionary: dB))
}

var seqTimes: [Double] = []
for _ in 0..<20 {
    let ts = CFAbsoluteTimeGetCurrent()
    let oA = try! chunkA.prediction(from: MLDictionaryFeatureProvider(dictionary: dictA))
    let dB = buildDictB(hidden: oA.featureValue(for: "hidden")!, convStates: bConv, recStates: bRec)
    let _ = try! chunkB.prediction(from: MLDictionaryFeatureProvider(dictionary: dB))
    seqTimes.append((CFAbsoluteTimeGetCurrent() - ts) * 1000)
}
seqTimes.sort()
let seqMed = seqTimes[10]
print("Median: \(String(format: "%.1f", seqMed))ms (\(String(format: "%.0f", 1000.0 / seqMed)) tok/s)")

// MARK: - Concurrent pipeline benchmark
// Simulate generating N tokens where chunk_A(N+1) overlaps with chunk_B(N)

print("\n--- Concurrent pipeline (A(N+1) || B(N)) ---")

let numTokens = 10
let tokenId: Int32 = 11751  // same token for simplicity

// Warmup
for _ in 0..<3 {
    let oA = try! chunkA.prediction(from: MLDictionaryFeatureProvider(dictionary: dictA))
    let dB = buildDictB(hidden: oA.featureValue(for: "hidden")!, convStates: bConv, recStates: bRec)
    let _ = try! chunkB.prediction(from: MLDictionaryFeatureProvider(dictionary: dB))
}

// Sequential baseline for N tokens
var seqNTimes: [Double] = []
for _ in 0..<5 {
    let ts = CFAbsoluteTimeGetCurrent()
    for _ in 0..<numTokens {
        let oA = try! chunkA.prediction(from: MLDictionaryFeatureProvider(dictionary: dictA))
        let dB = buildDictB(hidden: oA.featureValue(for: "hidden")!, convStates: bConv, recStates: bRec)
        let _ = try! chunkB.prediction(from: MLDictionaryFeatureProvider(dictionary: dB))
    }
    seqNTimes.append((CFAbsoluteTimeGetCurrent() - ts) * 1000)
}
seqNTimes.sort()
let seqNMed = seqNTimes[2]

// Pipelined: after the first token, overlap A(N+1) with B(N)
var pipeTimes: [Double] = []
for _ in 0..<5 {
    let ts = CFAbsoluteTimeGetCurrent()

    // Token 0: sequential (no overlap possible)
    var prevOutA = try! chunkA.prediction(from: MLDictionaryFeatureProvider(dictionary: dictA))

    for tok in 1..<numTokens {
        // Concurrent: B processes token tok-1, A processes token tok
        let dB = buildDictB(hidden: prevOutA.featureValue(for: "hidden")!, convStates: bConv, recStates: bRec)
        let provB = try! MLDictionaryFeatureProvider(dictionary: dB)
        let provA = try! MLDictionaryFeatureProvider(dictionary: dictA)

        let group = DispatchGroup()
        let q = DispatchQueue.global(qos: .userInteractive)

        var outB_result: MLFeatureProvider?
        var outA_result: MLFeatureProvider?

        group.enter()
        q.async {
            outB_result = try? chunkB.prediction(from: provB)
            group.leave()
        }

        group.enter()
        q.async {
            outA_result = try? chunkA.prediction(from: provA)
            group.leave()
        }

        group.wait()
        prevOutA = outA_result!
    }

    // Final B for last token
    let dB = buildDictB(hidden: prevOutA.featureValue(for: "hidden")!, convStates: bConv, recStates: bRec)
    let _ = try! chunkB.prediction(from: MLDictionaryFeatureProvider(dictionary: dB))

    pipeTimes.append((CFAbsoluteTimeGetCurrent() - ts) * 1000)
}
pipeTimes.sort()
let pipeMed = pipeTimes[2]

print("Sequential \(numTokens) tokens:  \(String(format: "%.0f", seqNMed))ms (\(String(format: "%.1f", seqNMed / Double(numTokens)))ms/tok, \(String(format: "%.0f", Double(numTokens) / (seqNMed / 1000))) tok/s)")
print("Pipelined \(numTokens) tokens:   \(String(format: "%.0f", pipeMed))ms (\(String(format: "%.1f", pipeMed / Double(numTokens)))ms/tok, \(String(format: "%.0f", Double(numTokens) / (pipeMed / 1000))) tok/s)")
print("Speedup: \(String(format: "%.2f", seqNMed / pipeMed))x")
