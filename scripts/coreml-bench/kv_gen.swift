// KV-cached generation benchmark: correct multi-token generation with padded KV cache.
//
// Validates that the CoreML 2-chunk model with KV cache produces correct output,
// then benchmarks realistic generation speed.
//
// Build & run:
//   swiftc -O -framework CoreML -framework Foundation \
//     scripts/coreml-bench/kv_gen.swift -o /tmp/coreml-kvgen
//   /tmp/coreml-kvgen

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
    if arr.dataType == .float16 {
        let ptr = arr.dataPointer.bindMemory(to: Float16.self, capacity: count)
        var maxIdx = 0; var maxVal = ptr[0]
        for i in 1..<count { if ptr[i] > maxVal { maxVal = ptr[i]; maxIdx = i } }
        return maxIdx
    }
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

func makeCachePosition(_ pos: Int32) -> MLMultiArray {
    let arr = try! MLMultiArray(shape: [1], dataType: .int32)
    arr.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = pos
    return arr
}

func makeAttnMask(pos: Int, maxSeqLen: Int) -> MLMultiArray {
    let arr = try! MLMultiArray(shape: [1, 1, 1, NSNumber(value: maxSeqLen)], dataType: .float32)
    let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: maxSeqLen)
    for i in 0..<maxSeqLen {
        ptr[i] = (i <= pos) ? 0.0 : -1e4
    }
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
    for i in 0..<ropeDim {
        cosPtr[i] = allCosPtr[offset + i]
        sinPtr[i] = allSinPtr[offset + i]
    }
    return (cos, sin)
}

// MARK: - State management

struct ChunkState {
    var convStates: [MLMultiArray]   // DeltaNet conv states
    var recStates: [MLMultiArray]    // DeltaNet recurrent states
    var keyCache: [MLMultiArray]     // Attention key cache
    var valCache: [MLMultiArray]     // Attention value cache
}

func loadInitialState(_ dir: String, prefix: String, nDelta: Int, nAttn: Int) -> ChunkState {
    var conv: [MLMultiArray] = []
    var rec: [MLMultiArray] = []
    var key: [MLMultiArray] = []
    var val: [MLMultiArray] = []
    for j in 0..<nDelta {
        conv.append(try! loadNpy("\(dir)/\(prefix)_conv_\(j).npy"))
        rec.append(try! loadNpy("\(dir)/\(prefix)_rec_\(j).npy"))
    }
    for j in 0..<nAttn {
        key.append(try! loadNpy("\(dir)/\(prefix)_key_\(j).npy"))
        val.append(try! loadNpy("\(dir)/\(prefix)_val_\(j).npy"))
    }
    return ChunkState(convStates: conv, recStates: rec, keyCache: key, valCache: val)
}

func extractState(_ output: MLFeatureProvider, nDelta: Int, nAttn: Int) -> ChunkState {
    var conv: [MLMultiArray] = []
    var rec: [MLMultiArray] = []
    var key: [MLMultiArray] = []
    var val: [MLMultiArray] = []
    for j in 0..<nDelta {
        conv.append(output.featureValue(for: "conv_state_\(j)_out")!.multiArrayValue!)
        rec.append(output.featureValue(for: "rec_state_\(j)_out")!.multiArrayValue!)
    }
    for j in 0..<nAttn {
        key.append(output.featureValue(for: "key_cache_\(j)_out")!.multiArrayValue!)
        val.append(output.featureValue(for: "value_cache_\(j)_out")!.multiArrayValue!)
    }
    return ChunkState(convStates: conv, recStates: rec, keyCache: key, valCache: val)
}

// MARK: - Main

let base = "models/split2_kv"
let inputDir = "\(base)/test_inputs"
let maxSeqLen = 256
let nDelta = 9   // DeltaNet layers per chunk
let nAttn = 3    // Attention layers per chunk

print("Loading models...")
let t0 = CFAbsoluteTimeGetCurrent()
let configGPU = MLModelConfiguration()
configGPU.computeUnits = .cpuAndGPU
let chunkA = try! MLModel(contentsOf: MLModel.compileModel(at: URL(fileURLWithPath: "\(base)/chunk_a.mlpackage")), configuration: configGPU)
let chunkB = try! MLModel(contentsOf: MLModel.compileModel(at: URL(fileURLWithPath: "\(base)/chunk_b.mlpackage")), configuration: configGPU)
print("Loaded in \(String(format: "%.0f", (CFAbsoluteTimeGetCurrent() - t0) * 1000))ms")

// Load full RoPE table and initial states
print("Loading test inputs...")
let fullRopeCos = try! loadNpy("\(inputDir)/all_rope_cos.npy")
let fullRopeSin = try! loadNpy("\(inputDir)/all_rope_sin.npy")
let ropeDim = fullRopeCos.shape[2].intValue
print("  RoPE table: \(fullRopeCos.shape) (dim=\(ropeDim))")

let initAState = loadInitialState(inputDir, prefix: "a", nDelta: nDelta, nAttn: nAttn)
let initBState = loadInitialState(inputDir, prefix: "b", nDelta: nDelta, nAttn: nAttn)
let refToken = try! loadNpy("\(inputDir)/ref_token.npy")
let refTokenId = refToken.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0]

// First token from prefill
let inputIdArr = try! loadNpy("\(inputDir)/input_id.npy")
let firstToken = inputIdArr.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0]

// Cache position from test inputs
let cachePosArr = try! loadNpy("\(inputDir)/cache_position.npy")
let promptLen = Int(cachePosArr.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0])

print("  Prompt length: \(promptLen)")
print("  First decode token: \(firstToken)")
print("  Expected next token: \(refTokenId)")

// MARK: - Single-step correctness test

func runStep(tokenId: Int32, pos: Int, aState: ChunkState, bState: ChunkState,
             ropeCos: MLMultiArray, ropeSin: MLMultiArray)
    -> (nextToken: Int, aStateOut: ChunkState, bStateOut: ChunkState)
{
    let cachePos = makeCachePosition(Int32(pos))
    let mask = makeAttnMask(pos: pos, maxSeqLen: maxSeqLen)

    // Chunk A
    var dictA: [String: MLFeatureValue] = [
        "input_id": MLFeatureValue(multiArray: makeTokenInput(tokenId)),
        "rope_cos": MLFeatureValue(multiArray: ropeCos),
        "rope_sin": MLFeatureValue(multiArray: ropeSin),
        "cache_position": MLFeatureValue(multiArray: cachePos),
        "attn_mask": MLFeatureValue(multiArray: mask),
    ]
    for j in 0..<nDelta {
        dictA["conv_state_\(j)"] = MLFeatureValue(multiArray: aState.convStates[j])
        dictA["rec_state_\(j)"] = MLFeatureValue(multiArray: aState.recStates[j])
    }
    for j in 0..<nAttn {
        dictA["key_cache_\(j)"] = MLFeatureValue(multiArray: aState.keyCache[j])
        dictA["value_cache_\(j)"] = MLFeatureValue(multiArray: aState.valCache[j])
    }
    let outA = try! chunkA.prediction(from: MLDictionaryFeatureProvider(dictionary: dictA))

    // Chunk B
    var dictB: [String: MLFeatureValue] = [
        "hidden": outA.featureValue(for: "hidden")!,
        "rope_cos": MLFeatureValue(multiArray: ropeCos),
        "rope_sin": MLFeatureValue(multiArray: ropeSin),
        "cache_position": MLFeatureValue(multiArray: cachePos),
        "attn_mask": MLFeatureValue(multiArray: mask),
    ]
    for j in 0..<nDelta {
        dictB["conv_state_\(j)"] = MLFeatureValue(multiArray: bState.convStates[j])
        dictB["rec_state_\(j)"] = MLFeatureValue(multiArray: bState.recStates[j])
    }
    for j in 0..<nAttn {
        dictB["key_cache_\(j)"] = MLFeatureValue(multiArray: bState.keyCache[j])
        dictB["value_cache_\(j)"] = MLFeatureValue(multiArray: bState.valCache[j])
    }
    let outB = try! chunkB.prediction(from: MLDictionaryFeatureProvider(dictionary: dictB))

    let logits = outB.featureValue(for: "logits")!.multiArrayValue!
    let nextTok = argmax(logits)
    let newA = extractState(outA, nDelta: nDelta, nAttn: nAttn)
    let newB = extractState(outB, nDelta: nDelta, nAttn: nAttn)

    return (nextTok, newA, newB)
}

print("\n--- Correctness test (single step) ---")
let (testCos, testSin) = makeRoPE(allCos: fullRopeCos, allSin: fullRopeSin, pos: promptLen, ropeDim: ropeDim)
let (nextTok, _, _) = runStep(
    tokenId: firstToken, pos: promptLen,
    aState: initAState, bState: initBState,
    ropeCos: testCos, ropeSin: testSin
)
print("  CoreML output token: \(nextTok)")
print("  PyTorch ref token:   \(refTokenId)")
if nextTok == Int(refTokenId) {
    print("  ✓ MATCH — CoreML KV cache produces correct output!")
} else {
    print("  ✗ MISMATCH — check model conversion")
}

// MARK: - Multi-token generation benchmark

// For multi-token generation, we need RoPE for all positions.
// Pre-compute from the test cos/sin (single position) — we can't easily get all positions
// without the full RoPE table. So we'll export it during conversion.
// For now, use the same cos/sin (position invariant for benchmark timing purposes).
// NOTE: This means generated text is WRONG for benchmark but timing is accurate.

print("\n--- Multi-token generation benchmark ---")
print("  (Using full RoPE table — correct positions)")

let numTokens = 15

// Warmup
var curAState = initAState
var curBState = initBState
for i in 0..<3 {
    let (wCos, wSin) = makeRoPE(allCos: fullRopeCos, allSin: fullRopeSin, pos: promptLen + i, ropeDim: ropeDim)
    let (_, a, b) = runStep(
        tokenId: firstToken, pos: promptLen + i,
        aState: curAState, bState: curBState,
        ropeCos: wCos, ropeSin: wSin
    )
    curAState = a
    curBState = b
}

// Benchmark
var allTimes: [Double] = []
for run in 0..<5 {
    curAState = initAState
    curBState = initBState
    var currentToken = firstToken
    var generated: [Int] = []
    var stepTimes: [Double] = []

    let genStart = CFAbsoluteTimeGetCurrent()

    for step in 0..<numTokens {
        let stepStart = CFAbsoluteTimeGetCurrent()
        let pos = promptLen + step

        let (stepCos, stepSin) = makeRoPE(allCos: fullRopeCos, allSin: fullRopeSin, pos: pos, ropeDim: ropeDim)
        let (tok, a, b) = runStep(
            tokenId: currentToken, pos: pos,
            aState: curAState, bState: curBState,
            ropeCos: stepCos, ropeSin: stepSin
        )

        stepTimes.append((CFAbsoluteTimeGetCurrent() - stepStart) * 1000)
        curAState = a
        curBState = b
        generated.append(tok)
        currentToken = Int32(tok)
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
print("  \(String(format: "%.0f", tokPerSec)) tok/s (with KV cache)")
