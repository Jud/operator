// A/B benchmark: FP16 vs 6-bit palettized batch prefill model.
//
// Build & run:
//   swiftc -O -framework CoreML -framework Foundation \
//     scripts/experiments/quip-hadamard-lmhead/bench_prefill_6bit.swift -o /tmp/bench-prefill-6bit
//   /tmp/bench-prefill-6bit

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
    let count = shape.reduce(1, *)
    let rawData = data[headerEnd...]
    if header.contains("float16") || header.contains("<f2") {
        let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
        rawData.withUnsafeBytes { buf in
            let src = buf.bindMemory(to: UInt16.self)
            let dst = arr.dataPointer.bindMemory(to: UInt16.self, capacity: count)
            for i in 0..<count { dst[i] = src[i] }
        }
        return arr
    } else if header.contains("float32") || header.contains("<f4") {
        let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        rawData.withUnsafeBytes { buf in
            let src = buf.bindMemory(to: Float.self)
            let dst = arr.dataPointer.bindMemory(to: Float.self, capacity: count)
            for i in 0..<count { dst[i] = src[i] }
        }
        return arr
    }
    fatalError("Unsupported dtype: \(header)")
}

// MARK: - Config

let nDelta = 18
let nAttn = 6
let maxKV = 256
let ropeDim = 64
let fixedN = 64
let promptLen = 131

let transcriptionIds: [Int32] = [367, 748, 353, 557, 1040, 2019, 310, 279, 42294, 3436]
let suffixIds: [Int32] = [510, 1405, 1400, 29, 198, 248046, 198, 248045, 74455, 198]
let allIds = transcriptionIds + suffixIds
let totalPrefill = allIds.count

let fp16Path = "models/batch_prefill_fp16/model.mlpackage"
let sixBitPath = "models/batch_prefill_6bit/model.mlpackage"
let cacheDir = "models/monolith_kv_fp16/prompt_cache"

// MARK: - Load models

print("=== FP16 vs 6-bit lm_head Prefill Benchmark ===\n")
print("Loading models...")
let t0 = CFAbsoluteTimeGetCurrent()
let cfg = MLModelConfiguration()
cfg.computeUnits = .cpuAndGPU

let fp16Model = try! MLModel(
    contentsOf: MLModel.compileModel(at: URL(fileURLWithPath: fp16Path)),
    configuration: cfg)
let sixBitModel = try! MLModel(
    contentsOf: MLModel.compileModel(at: URL(fileURLWithPath: sixBitPath)),
    configuration: cfg)
print("  Loaded in \(String(format: "%.0f", (CFAbsoluteTimeGetCurrent() - t0) * 1000))ms")

// MARK: - Load states

var convStates: [MLMultiArray] = []
var recStates: [MLMultiArray] = []
var keyCache: [MLMultiArray] = []
var valCache: [MLMultiArray] = []
for j in 0..<nDelta {
    convStates.append(try! loadNpy("\(cacheDir)/conv_state_\(j).npy"))
    recStates.append(try! loadNpy("\(cacheDir)/rec_state_\(j).npy"))
}
for j in 0..<nAttn {
    keyCache.append(try! loadNpy("\(cacheDir)/key_cache_\(j).npy"))
    valCache.append(try! loadNpy("\(cacheDir)/value_cache_\(j).npy"))
}
let ropeCos = try! loadNpy("\(cacheDir)/rope_cos.npy")
let ropeSin = try! loadNpy("\(cacheDir)/rope_sin.npy")

// MARK: - Build inputs

let paddedIds = try! MLMultiArray(shape: [1, NSNumber(value: fixedN)], dataType: .int32)
let idsPtr = paddedIds.dataPointer.bindMemory(to: Int32.self, capacity: fixedN)
for i in 0..<totalPrefill { idsPtr[i] = allIds[i] }
for i in totalPrefill..<fixedN { idsPtr[i] = 0 }

let seqLen = try! MLMultiArray(shape: [1], dataType: .int32)
seqLen.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = Int32(totalPrefill)

let startPos = try! MLMultiArray(shape: [1], dataType: .int32)
startPos.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = Int32(promptLen)

let batchCos = try! MLMultiArray(shape: [1, NSNumber(value: fixedN), NSNumber(value: ropeDim)], dataType: .float32)
let batchSin = try! MLMultiArray(shape: [1, NSNumber(value: fixedN), NSNumber(value: ropeDim)], dataType: .float32)
let allC = ropeCos.dataPointer.bindMemory(to: Float.self, capacity: ropeCos.count)
let allS = ropeSin.dataPointer.bindMemory(to: Float.self, capacity: ropeSin.count)
let cP = batchCos.dataPointer.bindMemory(to: Float.self, capacity: fixedN * ropeDim)
let sP = batchSin.dataPointer.bindMemory(to: Float.self, capacity: fixedN * ropeDim)
for i in 0..<fixedN {
    let off = (promptLen + i) * ropeDim
    for j in 0..<ropeDim { cP[i * ropeDim + j] = allC[off + j]; sP[i * ropeDim + j] = allS[off + j] }
}

let attnMask = try! MLMultiArray(shape: [1, 1, NSNumber(value: fixedN), NSNumber(value: maxKV)], dataType: .float32)
let maskPtr = attnMask.dataPointer.bindMemory(to: Float.self, capacity: fixedN * maxKV)
for i in 0..<fixedN {
    for j in 0..<maxKV {
        if i < totalPrefill {
            maskPtr[i * maxKV + j] = (j < promptLen + i + 1) ? 0.0 : -1e4
        } else {
            maskPtr[i * maxKV + j] = (j == 0) ? 0.0 : -1e4
        }
    }
}

// MARK: - Benchmark

func buildDict() -> [String: MLFeatureValue] {
    var dict: [String: MLFeatureValue] = [
        "input_ids": MLFeatureValue(multiArray: paddedIds),
        "seq_len": MLFeatureValue(multiArray: seqLen),
        "start_pos": MLFeatureValue(multiArray: startPos),
        "rope_cos": MLFeatureValue(multiArray: batchCos),
        "rope_sin": MLFeatureValue(multiArray: batchSin),
        "attn_mask": MLFeatureValue(multiArray: attnMask),
    ]
    for j in 0..<nDelta {
        dict["conv_state_\(j)"] = MLFeatureValue(multiArray: convStates[j])
        dict["rec_state_\(j)"] = MLFeatureValue(multiArray: recStates[j])
    }
    for j in 0..<nAttn {
        dict["key_cache_\(j)"] = MLFeatureValue(multiArray: keyCache[j])
        dict["value_cache_\(j)"] = MLFeatureValue(multiArray: valCache[j])
    }
    return dict
}

func runBench(label: String, model: MLModel, runs: Int = 10, warmup: Int = 3) -> (median: Double, firstToken: Int) {
    let dict = buildDict()
    let provider = try! MLDictionaryFeatureProvider(dictionary: dict)

    // Warmup
    for _ in 0..<warmup { let _ = try! model.prediction(from: provider) }

    // Benchmark
    var times: [Double] = []
    var lastOut: MLFeatureProvider?
    for _ in 0..<runs {
        let start = CFAbsoluteTimeGetCurrent()
        let out = try! model.prediction(from: provider)
        times.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
        lastOut = out
    }
    times.sort()
    let median = times[runs / 2]

    // Extract first token from logits
    var firstToken = -1
    if let out = lastOut {
        for name in out.featureNames {
            guard let arr = out.featureValue(for: name)?.multiArrayValue else { continue }
            if arr.count > 10000 && arr.shape.count == 3 {
                let vocabSize = arr.shape[2].intValue
                let offset = (totalPrefill - 1) * vocabSize
                if arr.dataType == .float16 {
                    let ptr = arr.dataPointer.bindMemory(to: Float16.self, capacity: arr.count)
                    var maxIdx = 0; var maxVal = ptr[offset]
                    for i in 1..<vocabSize { if ptr[offset + i] > maxVal { maxVal = ptr[offset + i]; maxIdx = i } }
                    firstToken = maxIdx
                } else {
                    let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: arr.count)
                    var maxIdx = 0; var maxVal = ptr[offset]
                    for i in 1..<vocabSize { if ptr[offset + i] > maxVal { maxVal = ptr[offset + i]; maxIdx = i } }
                    firstToken = maxIdx
                }
                break
            }
        }
    }

    print("  \(label): \(String(format: "%.1f", median))ms median (\(runs) runs)")
    print("    All: \(times.map { String(format: "%.1f", $0) }.joined(separator: ", "))")
    print("    First token: \(firstToken)")
    return (median, firstToken)
}

print("\n--- \(totalPrefill)-token batch prefill ---\n")

let (fp16Med, fp16Tok) = runBench(label: "FP16", model: fp16Model)
let (sixMed, sixTok) = runBench(label: "6-bit lm_head", model: sixBitModel)

print("\n=== RESULTS ===")
print("  FP16:        \(String(format: "%.1f", fp16Med))ms")
print("  6-bit head:  \(String(format: "%.1f", sixMed))ms")
let speedup = fp16Med / sixMed
if speedup > 1 {
    print("  Speedup:     \(String(format: "%.2f", speedup))x faster")
} else {
    print("  Slowdown:    \(String(format: "%.2f", 1/speedup))x slower")
}
print("  Token match: \(fp16Tok == sixTok ? "YES" : "NO") (FP16=\(fp16Tok), 6-bit=\(sixTok))")
