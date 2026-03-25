// Cached decode: load pre-computed system prompt states, generate tokens.
//
// No prefill model needed — states are loaded from .npy files (~13 MB, ~5ms).
// Uses the monolith_kv_fp16 decode model for generation.
//
// Build & run:
//   swiftc -O -framework CoreML -framework Foundation \
//     scripts/coreml-bench/cached_decode.swift -o /tmp/coreml-cached-decode
//   /tmp/coreml-cached-decode

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
    } else if header.contains("int32") || header.contains("<i4") {
        let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
        rawData.withUnsafeBytes { buf in
            let src = buf.bindMemory(to: Int32.self)
            let dst = arr.dataPointer.bindMemory(to: Int32.self, capacity: count)
            for i in 0..<count { dst[i] = src[i] }
        }
        return arr
    }
    fatalError("Unsupported dtype in \(path): \(header)")
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

let cacheDir = "models/monolith_kv_fp16/prompt_cache"
let modelPath = "models/monolith_kv_fp16/model.mlpackage"
let nDelta = 18
let nAttn = 6
let maxKV = 256
let ropeDim = 64
let promptLen = 151  // From meta.json
let numTokens = 30

// MARK: - Load

print("Loading model...")
let t0 = CFAbsoluteTimeGetCurrent()
let cfg = MLModelConfiguration()
cfg.computeUnits = .cpuAndGPU
let model = try! MLModel(
    contentsOf: MLModel.compileModel(at: URL(fileURLWithPath: modelPath)),
    configuration: cfg)
print("  Model: \(String(format: "%.0f", (CFAbsoluteTimeGetCurrent() - t0) * 1000))ms")

print("Loading cached states...")
let stateStart = CFAbsoluteTimeGetCurrent()
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
print("  States: \(String(format: "%.0f", (CFAbsoluteTimeGetCurrent() - stateStart) * 1000))ms")

// MARK: - Decode helpers

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

func makeAttnMask(pos: Int) -> MLMultiArray {
    let arr = try! MLMultiArray(shape: [1, 1, 1, NSNumber(value: maxKV)], dataType: .float32)
    let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: maxKV)
    for i in 0..<maxKV { ptr[i] = (i <= pos) ? 0.0 : -1e4 }
    return arr
}

func makeRoPE(pos: Int) -> (MLMultiArray, MLMultiArray) {
    let cos = try! MLMultiArray(shape: [1, 1, NSNumber(value: ropeDim)], dataType: .float32)
    let sin = try! MLMultiArray(shape: [1, 1, NSNumber(value: ropeDim)], dataType: .float32)
    let cPtr = cos.dataPointer.bindMemory(to: Float.self, capacity: ropeDim)
    let sPtr = sin.dataPointer.bindMemory(to: Float.self, capacity: ropeDim)
    let allC = ropeCos.dataPointer.bindMemory(to: Float.self, capacity: ropeCos.count)
    let allS = ropeSin.dataPointer.bindMemory(to: Float.self, capacity: ropeSin.count)
    let off = pos * ropeDim
    for i in 0..<ropeDim { cPtr[i] = allC[off + i]; sPtr[i] = allS[off + i] }
    return (cos, sin)
}

// MARK: - Generate

func decodeStep(tokenId: Int32, pos: Int) -> (Int, CFAbsoluteTime) {
    let stepStart = CFAbsoluteTimeGetCurrent()
    let (cos, sin) = makeRoPE(pos: pos)

    var dict: [String: MLFeatureValue] = [
        "input_id": MLFeatureValue(multiArray: makeTokenInput(tokenId)),
        "rope_cos": MLFeatureValue(multiArray: cos),
        "rope_sin": MLFeatureValue(multiArray: sin),
        "cache_position": MLFeatureValue(multiArray: makeCachePosition(Int32(pos))),
        "attn_mask": MLFeatureValue(multiArray: makeAttnMask(pos: pos)),
    ]
    for j in 0..<nDelta {
        dict["conv_state_\(j)"] = MLFeatureValue(multiArray: convStates[j])
        dict["rec_state_\(j)"] = MLFeatureValue(multiArray: recStates[j])
    }
    for j in 0..<nAttn {
        dict["key_cache_\(j)"] = MLFeatureValue(multiArray: keyCache[j])
        dict["value_cache_\(j)"] = MLFeatureValue(multiArray: valCache[j])
    }

    let out = try! model.prediction(from: MLDictionaryFeatureProvider(dictionary: dict))
    let logits = out.featureValue(for: "logits")!.multiArrayValue!
    let nextTok = argmax(logits)

    // Update states
    for j in 0..<nDelta {
        convStates[j] = out.featureValue(for: "conv_state_\(j)_out")!.multiArrayValue!
        recStates[j] = out.featureValue(for: "rec_state_\(j)_out")!.multiArrayValue!
    }
    for j in 0..<nAttn {
        keyCache[j] = out.featureValue(for: "key_cache_\(j)_out")!.multiArrayValue!
        valCache[j] = out.featureValue(for: "value_cache_\(j)_out")!.multiArrayValue!
    }

    return (nextTok, CFAbsoluteTimeGetCurrent() - stepStart)
}

// Warmup
print("\nWarmup...")
do {
    // Save initial states for reset
    let savedConv = convStates; let savedRec = recStates
    let savedKey = keyCache; let savedVal = valCache
    for i in 0..<3 {
        let (cos, sin) = makeRoPE(pos: promptLen + i)
        var dict: [String: MLFeatureValue] = [
            "input_id": MLFeatureValue(multiArray: makeTokenInput(248068)),
            "rope_cos": MLFeatureValue(multiArray: cos),
            "rope_sin": MLFeatureValue(multiArray: sin),
            "cache_position": MLFeatureValue(multiArray: makeCachePosition(Int32(promptLen + i))),
            "attn_mask": MLFeatureValue(multiArray: makeAttnMask(pos: promptLen + i)),
        ]
        for j in 0..<nDelta {
            dict["conv_state_\(j)"] = MLFeatureValue(multiArray: convStates[j])
            dict["rec_state_\(j)"] = MLFeatureValue(multiArray: recStates[j])
        }
        for j in 0..<nAttn {
            dict["key_cache_\(j)"] = MLFeatureValue(multiArray: keyCache[j])
            dict["value_cache_\(j)"] = MLFeatureValue(multiArray: valCache[j])
        }
        let out = try! model.prediction(from: MLDictionaryFeatureProvider(dictionary: dict))
        for j in 0..<nDelta {
            convStates[j] = out.featureValue(for: "conv_state_\(j)_out")!.multiArrayValue!
            recStates[j] = out.featureValue(for: "rec_state_\(j)_out")!.multiArrayValue!
        }
        for j in 0..<nAttn {
            keyCache[j] = out.featureValue(for: "key_cache_\(j)_out")!.multiArrayValue!
            valCache[j] = out.featureValue(for: "value_cache_\(j)_out")!.multiArrayValue!
        }
    }
    // Reset states
    convStates = savedConv; recStates = savedRec
    keyCache = savedKey; valCache = savedVal
}

// Generate
print("\n--- Generation (\(numTokens) tokens) ---")
// First token from PT: 248068 (<think>)
var currentToken: Int32 = 248068
var generated: [Int] = [Int(currentToken)]
var pos = promptLen
var totalMs: Double = 0

for _ in 0..<numTokens {
    let (tok, elapsed) = decodeStep(tokenId: currentToken, pos: pos)
    totalMs += elapsed * 1000
    generated.append(tok)
    currentToken = Int32(tok)
    pos += 1
}

let tokPerSec = Double(numTokens) / (totalMs / 1000)
print("  \(numTokens) tokens in \(String(format: "%.0f", totalMs))ms")
print("  \(String(format: "%.1f", totalMs / Double(numTokens)))ms/tok")
print("  \(String(format: "%.0f", tokPerSec)) tok/s")
print("  Token IDs: \(generated.prefix(15))...")
