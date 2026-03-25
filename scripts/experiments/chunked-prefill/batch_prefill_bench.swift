// Batch prefill benchmark: measure single-call prefill latency.
//
// Loads the batch prefill model + decode model, runs prefill on transcription
// tokens, then generates with decode model. Measures end-to-end latency.
//
// Build & run:
//   swiftc -O -framework CoreML -framework Foundation \
//     scripts/experiments/chunked-prefill/batch_prefill_bench.swift -o /tmp/batch-prefill-bench
//   /tmp/batch-prefill-bench

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

// MARK: - Config

let nDelta = 18
let nAttn = 6
let maxKV = 256
let ropeDim = 64
let fixedN = 64
let promptLen = 131   // System prompt length

// Test transcription token IDs (pre-tokenized: "um so I was like going to the uh store")
let transcriptionIds: [Int32] = [
    367, 748, 353, 557, 1040, 2019, 310, 279, 42294, 3436
]

// MARK: - Main

print("Loading models...")
let t0 = CFAbsoluteTimeGetCurrent()
let cfg = MLModelConfiguration()
cfg.computeUnits = .cpuAndGPU

let prefillModel = try! MLModel(
    contentsOf: MLModel.compileModel(at: URL(fileURLWithPath: "models/batch_prefill_fp16/model.mlpackage")),
    configuration: cfg)
let decodeModel = try! MLModel(
    contentsOf: MLModel.compileModel(at: URL(fileURLWithPath: "models/monolith_kv_fp16/model.mlpackage")),
    configuration: cfg)
print("  Models loaded: \(String(format: "%.0f", (CFAbsoluteTimeGetCurrent() - t0) * 1000))ms")

// Load cached system prompt states
let cacheDir = "models/monolith_kv_fp16/prompt_cache"
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
print("  States loaded")

// MARK: - Build prefill inputs

let nTokens = transcriptionIds.count
// Also need suffix tokens: </transcription>\n<|im_end|>\n<|im_start|>assistant\n
let suffixIds: [Int32] = [510, 1405, 1400, 29, 198, 248046, 198, 248045, 74455, 198]
let allIds = transcriptionIds + suffixIds
let totalPrefill = allIds.count

// Pad to fixedN
let paddedIds = try! MLMultiArray(shape: [1, NSNumber(value: fixedN)], dataType: .int32)
let idsPtr = paddedIds.dataPointer.bindMemory(to: Int32.self, capacity: fixedN)
for i in 0..<totalPrefill { idsPtr[i] = allIds[i] }
for i in totalPrefill..<fixedN { idsPtr[i] = 0 }

let seqLen = try! MLMultiArray(shape: [1], dataType: .int32)
seqLen.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = Int32(totalPrefill)

let startPos = try! MLMultiArray(shape: [1], dataType: .int32)
startPos.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = Int32(promptLen)

// RoPE for positions promptLen..promptLen+fixedN-1
let batchCos = try! MLMultiArray(shape: [1, NSNumber(value: fixedN), NSNumber(value: ropeDim)], dataType: .float32)
let batchSin = try! MLMultiArray(shape: [1, NSNumber(value: fixedN), NSNumber(value: ropeDim)], dataType: .float32)
let allC = ropeCos.dataPointer.bindMemory(to: Float.self, capacity: ropeCos.count)
let allS = ropeSin.dataPointer.bindMemory(to: Float.self, capacity: ropeSin.count)
let cPtr = batchCos.dataPointer.bindMemory(to: Float.self, capacity: fixedN * ropeDim)
let sPtr = batchSin.dataPointer.bindMemory(to: Float.self, capacity: fixedN * ropeDim)
for i in 0..<fixedN {
    let off = (promptLen + i) * ropeDim
    for j in 0..<ropeDim {
        cPtr[i * ropeDim + j] = allC[off + j]
        sPtr[i * ropeDim + j] = allS[off + j]
    }
}

// Attention mask: [1, 1, fixedN, maxKV]
let attnMask = try! MLMultiArray(shape: [1, 1, NSNumber(value: fixedN), NSNumber(value: maxKV)], dataType: .float32)
let maskPtr = attnMask.dataPointer.bindMemory(to: Float.self, capacity: fixedN * maxKV)
for i in 0..<fixedN {
    for j in 0..<maxKV {
        if i < totalPrefill {
            maskPtr[i * maxKV + j] = (j < promptLen + i + 1) ? 0.0 : -1e4
        } else {
            maskPtr[i * maxKV + j] = (j == 0) ? 0.0 : -1e4  // padding: attend to pos 0
        }
    }
}

// MARK: - Run batch prefill

func runPrefill() -> (Double, [String: MLFeatureValue]) {
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

    let start = CFAbsoluteTimeGetCurrent()
    let out = try! prefillModel.prediction(from: MLDictionaryFeatureProvider(dictionary: dict))
    let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000

    // Extract output states as dict
    var stateDict: [String: MLFeatureValue] = [:]
    for name in out.featureNames {
        stateDict[name] = out.featureValue(for: name)
    }
    return (ms, stateDict)
}

// Warmup
print("\nWarmup...")
for _ in 0..<2 { let _ = runPrefill() }

// Benchmark prefill
print("\n--- Batch Prefill Benchmark ---")
var prefillTimes: [Double] = []
var lastOutput: [String: MLFeatureValue] = [:]
for _ in 0..<5 {
    let (ms, output) = runPrefill()
    prefillTimes.append(ms)
    lastOutput = output
}
prefillTimes.sort()
let medianPrefill = prefillTimes[2]
print("  \(totalPrefill) tokens in \(String(format: "%.0f", medianPrefill))ms (median of 5)")
print("  \(String(format: "%.1f", medianPrefill / Double(totalPrefill)))ms/tok equivalent")

// Compare: what would per-token decode take?
let perTokenEstimate = Double(totalPrefill) * 12.0  // 12ms/tok from decode model
print("  vs per-token decode: ~\(String(format: "%.0f", perTokenEstimate))ms (\(String(format: "%.1f", perTokenEstimate / medianPrefill))x slower)")

// Get logits from prefill
// The logits output has shape [1, 64, vocab] — need position totalPrefill-1
let logitsArr = lastOutput.values.first(where: { ($0.multiArrayValue?.count ?? 0) > 10000 })?.multiArrayValue
if let logits = logitsArr {
    // Find the right position's logits
    let vocabSize = logits.shape.last!.intValue
    print("  Logits shape: \(logits.shape), vocab: \(vocabSize)")

    // For [1, 64, vocab], position totalPrefill-1
    if logits.shape.count == 3 {
        let offset = (totalPrefill - 1) * vocabSize
        if logits.dataType == .float16 {
            let ptr = logits.dataPointer.bindMemory(to: Float16.self, capacity: logits.count)
            var maxIdx = 0; var maxVal = ptr[offset]
            for i in 1..<vocabSize { if ptr[offset + i] > maxVal { maxVal = ptr[offset + i]; maxIdx = i } }
            print("  First token from prefill: \(maxIdx)")
        } else {
            let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: logits.count)
            var maxIdx = 0; var maxVal = ptr[offset]
            for i in 1..<vocabSize { if ptr[offset + i] > maxVal { maxVal = ptr[offset + i]; maxIdx = i } }
            print("  First token from prefill: \(maxIdx)")
        }
    }
}

print("\n--- Summary ---")
print("  Batch prefill (\(totalPrefill) tokens): \(String(format: "%.0f", medianPrefill))ms")
print("  Speedup vs per-token: \(String(format: "%.1f", perTokenEstimate / medianPrefill))x")
