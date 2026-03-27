// CoreML speed benchmark: compare FP16 baseline vs Hadamard-2bit monolith models.
//
// Loads two monolith models (same architecture, different lm_head quantization),
// runs identical 15-token generation, and compares tok/s + correctness.
//
// Build & run:
//   swiftc -O -framework CoreML -framework Foundation \
//     scripts/experiments/quip-hadamard-lmhead/bench_coreml.swift -o /tmp/bench-hadamard
//   /tmp/bench-hadamard [baseline_dir] [experiment_dir]
//
// Defaults:
//   baseline_dir = models/monolith_kv
//   experiment_dir = models/monolith_hadamard_2bit

import CoreML
import Foundation

// MARK: - Helpers

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
    if arr.dataType == .float16 {
        let ptr = arr.dataPointer.bindMemory(to: Float16.self, capacity: count)
        var maxIdx = 0; var maxVal = ptr[0]
        for i in 1..<count { if ptr[i] > maxVal { maxVal = ptr[i]; maxIdx = i } }
        return maxIdx
    }
    let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: count)
    var maxIdx = 0; var maxVal = ptr[0]
    for i in 1..<count { if ptr[i] > maxVal { maxVal = ptr[i]; maxIdx = i } }
    return maxIdx
}

func makeTokenInput(_ tokenId: Int32) -> MLMultiArray {
    let arr = try! MLMultiArray(shape: [1, 1], dataType: .int32)
    arr.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = tokenId
    return arr
}

func makeCachePosition(_ pos: Int32) -> MLMultiArray {
    let arr = try! MLMultiArray(shape: [1], dataType: .int32)
    arr.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = pos
    return arr
}

func makeAttnMask(pos: Int, maxSeqLen: Int) -> MLMultiArray {
    let arr = try! MLMultiArray(shape: [1, 1, 1, NSNumber(value: maxSeqLen)], dataType: .float32)
    let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: maxSeqLen)
    for i in 0..<maxSeqLen { ptr[i] = (i <= pos) ? 0.0 : -1e4 }
    return arr
}

func makeRoPE(allCos: MLMultiArray, allSin: MLMultiArray, pos: Int, ropeDim: Int) -> (MLMultiArray, MLMultiArray) {
    let cos = try! MLMultiArray(shape: [1, 1, NSNumber(value: ropeDim)], dataType: .float32)
    let sin = try! MLMultiArray(shape: [1, 1, NSNumber(value: ropeDim)], dataType: .float32)
    let cosPtr = cos.dataPointer.bindMemory(to: Float.self, capacity: ropeDim)
    let sinPtr = sin.dataPointer.bindMemory(to: Float.self, capacity: ropeDim)
    let allCosPtr = allCos.dataPointer.bindMemory(to: Float.self, capacity: allCos.count)
    let allSinPtr = allSin.dataPointer.bindMemory(to: Float.self, capacity: allSin.count)
    let offset = pos * ropeDim
    for i in 0..<ropeDim { cosPtr[i] = allCosPtr[offset + i]; sinPtr[i] = allSinPtr[offset + i] }
    return (cos, sin)
}

// MARK: - State management

struct ModelState {
    var convStates: [MLMultiArray]
    var recStates: [MLMultiArray]
    var keyCache: [MLMultiArray]
    var valCache: [MLMultiArray]
}

func loadState(_ dir: String, nDelta: Int, nAttn: Int) -> ModelState {
    var conv: [MLMultiArray] = [], rec: [MLMultiArray] = []
    var key: [MLMultiArray] = [], val: [MLMultiArray] = []
    for j in 0..<nDelta {
        conv.append(try! loadNpy("\(dir)/conv_state_\(j).npy"))
        rec.append(try! loadNpy("\(dir)/rec_state_\(j).npy"))
    }
    for j in 0..<nAttn {
        key.append(try! loadNpy("\(dir)/key_cache_\(j).npy"))
        val.append(try! loadNpy("\(dir)/value_cache_\(j).npy"))
    }
    return ModelState(convStates: conv, recStates: rec, keyCache: key, valCache: val)
}

func extractState(_ output: MLFeatureProvider, nDelta: Int, nAttn: Int) -> ModelState {
    var conv: [MLMultiArray] = [], rec: [MLMultiArray] = []
    var key: [MLMultiArray] = [], val: [MLMultiArray] = []
    for j in 0..<nDelta {
        conv.append(output.featureValue(for: "conv_state_\(j)_out")!.multiArrayValue!)
        rec.append(output.featureValue(for: "rec_state_\(j)_out")!.multiArrayValue!)
    }
    for j in 0..<nAttn {
        key.append(output.featureValue(for: "key_cache_\(j)_out")!.multiArrayValue!)
        val.append(output.featureValue(for: "value_cache_\(j)_out")!.multiArrayValue!)
    }
    return ModelState(convStates: conv, recStates: rec, keyCache: key, valCache: val)
}

// MARK: - Generation

func runStep(model: MLModel, tokenId: Int32, pos: Int, state: ModelState,
             ropeCos: MLMultiArray, ropeSin: MLMultiArray,
             maxSeqLen: Int, nDelta: Int, nAttn: Int)
    -> (nextToken: Int, state: ModelState)
{
    var dict: [String: MLFeatureValue] = [
        "input_id": MLFeatureValue(multiArray: makeTokenInput(tokenId)),
        "rope_cos": MLFeatureValue(multiArray: ropeCos),
        "rope_sin": MLFeatureValue(multiArray: ropeSin),
        "cache_position": MLFeatureValue(multiArray: makeCachePosition(Int32(pos))),
        "attn_mask": MLFeatureValue(multiArray: makeAttnMask(pos: pos, maxSeqLen: maxSeqLen)),
    ]
    for j in 0..<nDelta {
        dict["conv_state_\(j)"] = MLFeatureValue(multiArray: state.convStates[j])
        dict["rec_state_\(j)"] = MLFeatureValue(multiArray: state.recStates[j])
    }
    for j in 0..<nAttn {
        dict["key_cache_\(j)"] = MLFeatureValue(multiArray: state.keyCache[j])
        dict["value_cache_\(j)"] = MLFeatureValue(multiArray: state.valCache[j])
    }

    let out = try! model.prediction(from: MLDictionaryFeatureProvider(dictionary: dict))
    let logits = out.featureValue(for: "logits")!.multiArrayValue!
    return (argmax(logits), extractState(out, nDelta: nDelta, nAttn: nAttn))
}

func benchmark(label: String, model: MLModel, firstToken: Int32, promptLen: Int,
               initState: ModelState, fullRopeCos: MLMultiArray, fullRopeSin: MLMultiArray,
               ropeDim: Int, maxSeqLen: Int, nDelta: Int, nAttn: Int,
               numTokens: Int = 15, numRuns: Int = 5) -> [Int]
{
    print("\n--- \(label) ---")

    // Warmup
    var state = initState
    for i in 0..<3 {
        let (c, s) = makeRoPE(allCos: fullRopeCos, allSin: fullRopeSin, pos: promptLen + i, ropeDim: ropeDim)
        let (_, st) = runStep(model: model, tokenId: firstToken, pos: promptLen + i, state: state,
                              ropeCos: c, ropeSin: s, maxSeqLen: maxSeqLen, nDelta: nDelta, nAttn: nAttn)
        state = st
    }

    var allTimes: [Double] = []
    var firstRunTokens: [Int] = []

    for run in 0..<numRuns {
        state = initState
        var currentToken = firstToken
        var generated: [Int] = []
        var stepTimes: [Double] = []

        let genStart = CFAbsoluteTimeGetCurrent()
        for step in 0..<numTokens {
            let stepStart = CFAbsoluteTimeGetCurrent()
            let pos = promptLen + step
            let (c, s) = makeRoPE(allCos: fullRopeCos, allSin: fullRopeSin, pos: pos, ropeDim: ropeDim)
            let (tok, st) = runStep(model: model, tokenId: currentToken, pos: pos, state: state,
                                    ropeCos: c, ropeSin: s, maxSeqLen: maxSeqLen, nDelta: nDelta, nAttn: nAttn)
            stepTimes.append((CFAbsoluteTimeGetCurrent() - stepStart) * 1000)
            state = st
            generated.append(tok)
            currentToken = Int32(tok)
        }
        let totalMs = (CFAbsoluteTimeGetCurrent() - genStart) * 1000
        allTimes.append(totalMs)

        if run == 0 {
            firstRunTokens = generated
            stepTimes.sort()
            print("  Tokens: \(generated.prefix(10))...")
            print("  Per-step median: \(String(format: "%.1f", stepTimes[stepTimes.count/2]))ms")
        }
    }

    allTimes.sort()
    let medTotal = allTimes[numRuns / 2]
    let tokPerSec = Double(numTokens) / (medTotal / 1000)
    print("  \(numTokens) tokens in \(String(format: "%.0f", medTotal))ms")
    print("  \(String(format: "%.1f", medTotal / Double(numTokens)))ms/tok")
    print("  \(String(format: "%.0f", tokPerSec)) tok/s")

    return firstRunTokens
}

// MARK: - Main

let args = CommandLine.arguments
let baselineDir = args.count > 1 ? args[1] : "models/monolith_kv"
let experimentDir = args.count > 2 ? args[2] : "models/monolith_hadamard_2bit"
let maxSeqLen = 256
let nDelta = 18  // Monolith: all 18 DeltaNet layers
let nAttn = 6    // Monolith: all 6 attention layers

print("=== Hadamard lm_head Benchmark ===")
print("Baseline:   \(baselineDir)")
print("Experiment: \(experimentDir)")

// Load models
print("\nLoading models...")
let t0 = CFAbsoluteTimeGetCurrent()
let configGPU = MLModelConfiguration()
configGPU.computeUnits = .cpuAndGPU

let baselineModel = try! MLModel(contentsOf: MLModel.compileModel(at: URL(fileURLWithPath: "\(baselineDir)/model.mlpackage")), configuration: configGPU)
let experimentModel = try! MLModel(contentsOf: MLModel.compileModel(at: URL(fileURLWithPath: "\(experimentDir)/model.mlpackage")), configuration: configGPU)
print("Loaded in \(String(format: "%.0f", (CFAbsoluteTimeGetCurrent() - t0) * 1000))ms")

// Load shared test inputs (from baseline — states are the same)
let inputDir = "\(baselineDir)/test_inputs"
print("Loading test inputs from \(inputDir)...")
let fullRopeCos = try! loadNpy("\(inputDir)/all_rope_cos.npy")
let fullRopeSin = try! loadNpy("\(inputDir)/all_rope_sin.npy")
let ropeDim = fullRopeCos.shape[2].intValue

let initState = loadState(inputDir, nDelta: nDelta, nAttn: nAttn)
let inputIdArr = try! loadNpy("\(inputDir)/input_id.npy")
let firstToken = inputIdArr.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0]
let cachePosArr = try! loadNpy("\(inputDir)/cache_position.npy")
let promptLen = Int(cachePosArr.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0])

print("  Prompt length: \(promptLen), first token: \(firstToken)")

// Also load experiment test inputs if they exist
let expInputDir = "\(experimentDir)/test_inputs"
let expState: ModelState
if FileManager.default.fileExists(atPath: "\(expInputDir)/conv_state_0.npy") {
    print("Loading experiment test inputs from \(expInputDir)...")
    expState = loadState(expInputDir, nDelta: nDelta, nAttn: nAttn)
} else {
    print("Using baseline test inputs for experiment model")
    expState = initState
}

// Run benchmarks
let baselineTokens = benchmark(
    label: "BASELINE (FP16)", model: baselineModel,
    firstToken: firstToken, promptLen: promptLen, initState: initState,
    fullRopeCos: fullRopeCos, fullRopeSin: fullRopeSin,
    ropeDim: ropeDim, maxSeqLen: maxSeqLen, nDelta: nDelta, nAttn: nAttn
)

let experimentTokens = benchmark(
    label: "EXPERIMENT (Hadamard + 2-bit)", model: experimentModel,
    firstToken: firstToken, promptLen: promptLen, initState: expState,
    fullRopeCos: fullRopeCos, fullRopeSin: fullRopeSin,
    ropeDim: ropeDim, maxSeqLen: maxSeqLen, nDelta: nDelta, nAttn: nAttn
)

// Compare
print("\n=== COMPARISON ===")
let matches = zip(baselineTokens, experimentTokens).filter { $0.0 == $0.1 }.count
print("Token match: \(matches)/\(baselineTokens.count)")
print("Baseline tokens:   \(baselineTokens)")
print("Experiment tokens: \(experimentTokens)")

// Model sizes
func dirSize(_ path: String) -> Int64 {
    let fm = FileManager.default
    guard let enumerator = fm.enumerator(atPath: path) else { return 0 }
    var total: Int64 = 0
    while let file = enumerator.nextObject() as? String {
        let full = "\(path)/\(file)"
        if let attrs = try? fm.attributesOfItem(atPath: full) {
            total += (attrs[.size] as? Int64) ?? 0
        }
    }
    return total
}

let baselineSize = dirSize("\(baselineDir)/model.mlpackage")
let experimentSize = dirSize("\(experimentDir)/model.mlpackage")
print("\nModel sizes:")
print("  Baseline:   \(String(format: "%.1f", Double(baselineSize) / 1e6)) MB")
print("  Experiment: \(String(format: "%.1f", Double(experimentSize) / 1e6)) MB")
print("  Ratio: \(String(format: "%.1fx", Double(baselineSize) / Double(experimentSize))) smaller")
