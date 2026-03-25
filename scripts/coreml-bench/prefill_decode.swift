// Prefill + Decode end-to-end pipeline.
//
// Loads the MIL prefill model (while_loop) and the monolith decode model (KV cache).
// Prefills a system prompt, then generates tokens using the decode model.
//
// Build & run:
//   swiftc -O -framework CoreML -framework Foundation \
//     scripts/coreml-bench/prefill_decode.swift -o /tmp/coreml-prefill-decode
//   /tmp/coreml-prefill-decode

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

    if header.contains("float32") || header.contains("<f4") {
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

func makeAttnMask(pos: Int, maxKV: Int) -> MLMultiArray {
    let arr = try! MLMultiArray(shape: [1, 1, 1, NSNumber(value: maxKV)], dataType: .float32)
    let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: maxKV)
    for i in 0..<maxKV { ptr[i] = (i <= pos) ? 0.0 : -1e4 }
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

// MARK: - Config

let nDelta = 18   // DeltaNet layers
let nAttn = 6     // Attention layers
let maxKV = 256   // Must match both models
let ropeDim = 64

// MARK: - Main

print("Loading models...")
let t0 = CFAbsoluteTimeGetCurrent()
let gpuConfig = MLModelConfiguration()
gpuConfig.computeUnits = .cpuAndGPU

let prefillModel = try! MLModel(
    contentsOf: MLModel.compileModel(at: URL(fileURLWithPath: "models/prefill_mil/prefill_loop.mlpackage")),
    configuration: gpuConfig)
let decodeModel = try! MLModel(
    contentsOf: MLModel.compileModel(at: URL(fileURLWithPath: "models/monolith_kv_fp16/model.mlpackage")),
    configuration: gpuConfig)
print("Loaded in \(String(format: "%.0f", (CFAbsoluteTimeGetCurrent() - t0) * 1000))ms")

// Load test inputs
print("Loading test inputs...")
let promptIds = try! loadNpy("models/prefill_mil/test_inputs/prompt_ids.npy")
let ropeCos = try! loadNpy("models/prefill_mil/test_inputs/rope_cos.npy")
let ropeSin = try! loadNpy("models/prefill_mil/test_inputs/rope_sin.npy")
let promptLen = promptIds.count
print("  Prompt: \(promptLen) tokens, RoPE: \(ropeCos.shape)")

// MARK: - Step 1: Prefill

print("\n--- Prefill ---")
// Pad token IDs to maxKV
let paddedIds = try! MLMultiArray(shape: [NSNumber(value: maxKV)], dataType: .int32)
let srcPtr = promptIds.dataPointer.bindMemory(to: Int32.self, capacity: promptLen)
let dstPtr = paddedIds.dataPointer.bindMemory(to: Int32.self, capacity: maxKV)
for i in 0..<promptLen { dstPtr[i] = srcPtr[i] }
for i in promptLen..<maxKV { dstPtr[i] = 0 }

let seqLen = try! MLMultiArray(shape: [1], dataType: .int32)
seqLen.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = Int32(promptLen)

let prefillInput: [String: MLFeatureValue] = [
    "token_ids": MLFeatureValue(multiArray: paddedIds),
    "seq_len": MLFeatureValue(multiArray: seqLen),
]

let prefillStart = CFAbsoluteTimeGetCurrent()
let prefillOut = try! prefillModel.prediction(from: MLDictionaryFeatureProvider(dictionary: prefillInput))
let prefillMs = (CFAbsoluteTimeGetCurrent() - prefillStart) * 1000
print("  Prefill: \(String(format: "%.0f", prefillMs))ms for \(promptLen) tokens")

// Get first token from prefill logits
let prefillLogits = prefillOut.featureValue(for: "lm_head")!.multiArrayValue!
let firstToken = argmax(prefillLogits)
print("  First token: \(firstToken)")

// MARK: - Step 2: Map prefill states to decode inputs

// Prefill outputs: while_loop_0_{2+j*2} = conv_j, while_loop_0_{3+j*2} = rec_j (j=0..17)
//                  while_loop_0_{38+j*2} = key_j,  while_loop_0_{39+j*2} = val_j (j=0..5)
// Decode expects:  conv_state_j, rec_state_j (j=0..17), key_cache_j, value_cache_j (j=0..5)

func mapPrefillToDecodeStates(_ prefillOut: MLFeatureProvider) -> [String: MLFeatureValue] {
    var states: [String: MLFeatureValue] = [:]
    for j in 0..<nDelta {
        let convName = "while_loop_0_\(2 + j * 2)"
        let recName = "while_loop_0_\(3 + j * 2)"
        states["conv_state_\(j)"] = prefillOut.featureValue(for: convName)!
        states["rec_state_\(j)"] = prefillOut.featureValue(for: recName)!
    }
    for j in 0..<nAttn {
        let keyName = "while_loop_0_\(38 + j * 2)"
        let valName = "while_loop_0_\(39 + j * 2)"
        states["key_cache_\(j)"] = prefillOut.featureValue(for: keyName)!
        states["value_cache_\(j)"] = prefillOut.featureValue(for: valName)!
    }
    return states
}

var decodeStates = mapPrefillToDecodeStates(prefillOut)

// MARK: - Step 3: Decode loop

print("\n--- Decode ---")
let numTokens = 15
var currentToken = Int32(firstToken)
var generated: [Int] = []
var pos = promptLen  // Next position after prompt

let decodeStart = CFAbsoluteTimeGetCurrent()

for step in 0..<numTokens {
    let (stepCos, stepSin) = makeRoPE(allCos: ropeCos, allSin: ropeSin, pos: pos, ropeDim: ropeDim)

    var dict: [String: MLFeatureValue] = [
        "input_id": MLFeatureValue(multiArray: makeTokenInput(currentToken)),
        "rope_cos": MLFeatureValue(multiArray: stepCos),
        "rope_sin": MLFeatureValue(multiArray: stepSin),
        "cache_position": MLFeatureValue(multiArray: makeCachePosition(Int32(pos))),
        "attn_mask": MLFeatureValue(multiArray: makeAttnMask(pos: pos, maxKV: maxKV)),
    ]
    // Add states
    for (key, val) in decodeStates { dict[key] = val }

    let out = try! decodeModel.prediction(from: MLDictionaryFeatureProvider(dictionary: dict))

    let logits = out.featureValue(for: "logits")!.multiArrayValue!
    let nextTok = argmax(logits)
    generated.append(nextTok)

    // Update states from decode output
    var newStates: [String: MLFeatureValue] = [:]
    for j in 0..<nDelta {
        newStates["conv_state_\(j)"] = out.featureValue(for: "conv_state_\(j)_out")!
        newStates["rec_state_\(j)"] = out.featureValue(for: "rec_state_\(j)_out")!
    }
    for j in 0..<nAttn {
        newStates["key_cache_\(j)"] = out.featureValue(for: "key_cache_\(j)_out")!
        newStates["value_cache_\(j)"] = out.featureValue(for: "value_cache_\(j)_out")!
    }
    decodeStates = newStates

    currentToken = Int32(nextTok)
    pos += 1
}

let totalMs = (CFAbsoluteTimeGetCurrent() - decodeStart) * 1000
let tokPerSec = Double(numTokens) / (totalMs / 1000)

print("  Generated \(numTokens) tokens in \(String(format: "%.0f", totalMs))ms")
print("  \(String(format: "%.1f", totalMs / Double(numTokens)))ms/tok")
print("  \(String(format: "%.0f", tokPerSec)) tok/s")
print("  Token IDs: \(generated)")
print("  Prefill: \(String(format: "%.0f", prefillMs))ms")
