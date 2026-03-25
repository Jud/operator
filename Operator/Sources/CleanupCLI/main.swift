// CleanupCLI — Speech-to-text cleanup using Qwen3.5 on CoreML.
//
// Reads raw transcriptions from stdin (one per line), outputs cleaned text.
// Uses pre-computed system prompt states + monolith KV decode model.
//
// Usage:
//   swift run CleanupCLI [--model-dir DIR]
//   echo "um so I was like going to the uh store" | swift run CleanupCLI

import CoreML
import Foundation
import QwenTokenizer

// MARK: - Config

let defaultModelDir = "models/monolith_kv_fp16"
let defaultPrefillDir = "models/batch_prefill_fp16"
let nDelta = 18
let nAttn = 6
let maxKV = 256
let ropeDim = 64
let fixedN = 64
let maxGenTokens = 300
let eosTokenId = 248044  // <|endoftext|>
let imEndTokenId = 248046  // <|im_end|>
let thinkTokenId = 248068  // <think>
let thinkEndTokenId = 248069  // </think>

// MARK: - .npy loader

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

// MARK: - Decode helpers

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

func makeAttnMask(pos: Int) -> MLMultiArray {
    let arr = try! MLMultiArray(shape: [1, 1, 1, NSNumber(value: maxKV)], dataType: .float32)
    let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: maxKV)
    for i in 0..<maxKV { ptr[i] = (i <= pos) ? 0.0 : -1e4 }
    return arr
}

// MARK: - Model state

class CleanupEngine {
    let decodeModel: MLModel
    let prefillModel: MLModel
    let tokenizer: Tokenizer
    let ropeCos: MLMultiArray
    let ropeSin: MLMultiArray

    // Cached system prompt states (immutable, reused per request)
    let cachedConv: [MLMultiArray]
    let cachedRec: [MLMultiArray]
    let cachedKey: [MLMultiArray]
    let cachedVal: [MLMultiArray]
    let promptLen: Int

    // Suffix tokens: </transcription>\n<|im_end|>\n<|im_start|>assistant\n
    let suffixIds: [Int]

    init(modelDir: String, prefillDir: String) throws {
        let t0 = CFAbsoluteTimeGetCurrent()

        // Load tokenizer
        let tokDir = URL(fileURLWithPath: "\(modelDir)/tokenizer")
        tokenizer = try AutoTokenizer.from(modelFolder: tokDir)

        // Load models
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndGPU
        let decodeURL = URL(fileURLWithPath: "\(modelDir)/model.mlpackage")
        decodeModel = try MLModel(contentsOf: MLModel.compileModel(at: decodeURL), configuration: cfg)
        let prefillURL = URL(fileURLWithPath: "\(prefillDir)/model.mlpackage")
        prefillModel = try MLModel(contentsOf: MLModel.compileModel(at: prefillURL), configuration: cfg)

        // Load cached states
        let cacheDir = "\(modelDir)/prompt_cache"
        var conv: [MLMultiArray] = [], rec: [MLMultiArray] = []
        var key: [MLMultiArray] = [], val: [MLMultiArray] = []
        for j in 0..<nDelta {
            conv.append(try loadNpy("\(cacheDir)/conv_state_\(j).npy"))
            rec.append(try loadNpy("\(cacheDir)/rec_state_\(j).npy"))
        }
        for j in 0..<nAttn {
            key.append(try loadNpy("\(cacheDir)/key_cache_\(j).npy"))
            val.append(try loadNpy("\(cacheDir)/value_cache_\(j).npy"))
        }
        cachedConv = conv; cachedRec = rec; cachedKey = key; cachedVal = val

        // Load RoPE
        ropeCos = try loadNpy("\(cacheDir)/rope_cos.npy")
        ropeSin = try loadNpy("\(cacheDir)/rope_sin.npy")

        // Load metadata
        let metaData = try Data(contentsOf: URL(fileURLWithPath: "\(cacheDir)/meta.json"))
        let meta = try JSONSerialization.jsonObject(with: metaData) as! [String: Any]
        promptLen = meta["prompt_len"] as! Int

        // Suffix token IDs (hardcoded — Swift tokenizer doesn't handle special tokens)
        // </transcription>\n<|im_end|>\n<|im_start|>assistant\n
        suffixIds = [510, 1405, 1400, 29, 198, 248046, 198, 248045, 74455, 198]

        let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        fputs("Loaded in \(String(format: "%.0f", ms))ms (prompt: \(promptLen) tokens, suffix: \(suffixIds.count) tokens)\n", stderr)
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

    /// Run one decode step: feed a token, get next token + update states.
    func step(tokenId: Int32, pos: Int,
              conv: inout [MLMultiArray], rec: inout [MLMultiArray],
              key: inout [MLMultiArray], val: inout [MLMultiArray]) -> Int {
        let (cos, sin) = makeRoPE(pos: pos)
        var dict: [String: MLFeatureValue] = [
            "input_id": MLFeatureValue(multiArray: makeTokenInput(tokenId)),
            "rope_cos": MLFeatureValue(multiArray: cos),
            "rope_sin": MLFeatureValue(multiArray: sin),
            "cache_position": MLFeatureValue(multiArray: makeCachePosition(Int32(pos))),
            "attn_mask": MLFeatureValue(multiArray: makeAttnMask(pos: pos)),
        ]
        for j in 0..<nDelta {
            dict["conv_state_\(j)"] = MLFeatureValue(multiArray: conv[j])
            dict["rec_state_\(j)"] = MLFeatureValue(multiArray: rec[j])
        }
        for j in 0..<nAttn {
            dict["key_cache_\(j)"] = MLFeatureValue(multiArray: key[j])
            dict["value_cache_\(j)"] = MLFeatureValue(multiArray: val[j])
        }

        let out = try! decodeModel.prediction(from: MLDictionaryFeatureProvider(dictionary: dict))

        // Update states
        for j in 0..<nDelta {
            conv[j] = out.featureValue(for: "conv_state_\(j)_out")!.multiArrayValue!
            rec[j] = out.featureValue(for: "rec_state_\(j)_out")!.multiArrayValue!
        }
        for j in 0..<nAttn {
            key[j] = out.featureValue(for: "key_cache_\(j)_out")!.multiArrayValue!
            val[j] = out.featureValue(for: "value_cache_\(j)_out")!.multiArrayValue!
        }

        return argmax(out.featureValue(for: "logits")!.multiArrayValue!)
    }

    /// Batch prefill: process all tokens in one predict() call.
    func batchPrefill(_ allIds: [Int]) -> (Int, [MLMultiArray], [MLMultiArray], [MLMultiArray], [MLMultiArray], Int) {
        let totalPrefill = allIds.count

        // Pad token IDs to fixedN
        let paddedIds = try! MLMultiArray(shape: [1, NSNumber(value: fixedN)], dataType: .int32)
        let idsPtr = paddedIds.dataPointer.bindMemory(to: Int32.self, capacity: fixedN)
        for i in 0..<min(totalPrefill, fixedN) { idsPtr[i] = Int32(allIds[i]) }
        for i in totalPrefill..<fixedN { idsPtr[i] = 0 }

        let seqLen = try! MLMultiArray(shape: [1], dataType: .int32)
        seqLen.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = Int32(totalPrefill)

        let startPos = try! MLMultiArray(shape: [1], dataType: .int32)
        startPos.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = Int32(promptLen)

        // Build batch RoPE: [1, fixedN, ropeDim]
        let batchCos = try! MLMultiArray(shape: [1, NSNumber(value: fixedN), NSNumber(value: ropeDim)], dataType: .float32)
        let batchSin = try! MLMultiArray(shape: [1, NSNumber(value: fixedN), NSNumber(value: ropeDim)], dataType: .float32)
        let allC = ropeCos.dataPointer.bindMemory(to: Float.self, capacity: ropeCos.count)
        let allS = ropeSin.dataPointer.bindMemory(to: Float.self, capacity: ropeSin.count)
        let cPtr = batchCos.dataPointer.bindMemory(to: Float.self, capacity: fixedN * ropeDim)
        let sPtr = batchSin.dataPointer.bindMemory(to: Float.self, capacity: fixedN * ropeDim)
        for i in 0..<fixedN {
            let off = (promptLen + i) * ropeDim
            for j in 0..<ropeDim { cPtr[i * ropeDim + j] = allC[off + j]; sPtr[i * ropeDim + j] = allS[off + j] }
        }

        // Build attention mask: [1, 1, fixedN, maxKV]
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

        var dict: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: paddedIds),
            "seq_len": MLFeatureValue(multiArray: seqLen),
            "start_pos": MLFeatureValue(multiArray: startPos),
            "rope_cos": MLFeatureValue(multiArray: batchCos),
            "rope_sin": MLFeatureValue(multiArray: batchSin),
            "attn_mask": MLFeatureValue(multiArray: attnMask),
        ]
        for j in 0..<nDelta {
            dict["conv_state_\(j)"] = MLFeatureValue(multiArray: cachedConv[j])
            dict["rec_state_\(j)"] = MLFeatureValue(multiArray: cachedRec[j])
        }
        for j in 0..<nAttn {
            dict["key_cache_\(j)"] = MLFeatureValue(multiArray: cachedKey[j])
            dict["value_cache_\(j)"] = MLFeatureValue(multiArray: cachedVal[j])
        }

        let out = try! prefillModel.prediction(from: MLDictionaryFeatureProvider(dictionary: dict))

        // Extract logits — shape [1, 64, 248320], find by matching vocab dim
        let logitsName = out.featureNames.first(where: {
            let arr = out.featureValue(for: $0)!.multiArrayValue!
            return arr.shape.count == 3 && arr.shape[2].intValue > 200000
        })!
        let logitsArr = out.featureValue(for: logitsName)!.multiArrayValue!
        let vocabSize = logitsArr.shape.last!.intValue
        fputs("  Logits: name=\(logitsName) shape=\(logitsArr.shape) dtype=\(logitsArr.dataType.rawValue) vocab=\(vocabSize) total=\(totalPrefill)\n", stderr)

        // MLMultiArray strides: for [1, 64, V], strides = [64*V, V, 1]
        let stride1 = logitsArr.strides[1].intValue  // should be vocabSize
        let offset = (totalPrefill - 1) * stride1
        fputs("  stride1=\(stride1) offset=\(offset)\n", stderr)

        var maxIdx = 0; var maxVal: Float = -.infinity
        if logitsArr.dataType == .float16 {
            let ptr = logitsArr.dataPointer.bindMemory(to: Float16.self, capacity: logitsArr.count)
            for i in 0..<vocabSize { let v = Float(ptr[offset + i]); if v > maxVal { maxVal = v; maxIdx = i } }
        } else {
            let ptr = logitsArr.dataPointer.bindMemory(to: Float.self, capacity: logitsArr.count)
            for i in 0..<vocabSize { if ptr[offset + i] > maxVal { maxVal = ptr[offset + i]; maxIdx = i } }
        }
        fputs("  Prefill argmax: \(maxIdx) (val=\(maxVal))\n", stderr)

        // Extract output states by iterating ALL outputs and classifying by shape.
        // Output order: logits, then alternating (conv, rec) × 18, then alternating (key, val) × 6.
        // We classify by shape and use clip_ prefix for rec states (stable across rebuilds).
        var outConv: [MLMultiArray] = []
        var outRec: [MLMultiArray] = []
        var outKey: [MLMultiArray] = []
        var outVal: [MLMultiArray] = []

        // Collect conv and rec entries with their numeric suffixes for stable ordering
        var convEntries: [(Int, MLMultiArray)] = []
        var recEntries: [(Int, MLMultiArray)] = []
        var keyEntries: [(Int, MLMultiArray)] = []
        var valEntries: [(Int, MLMultiArray)] = []

        let convK = 4  // conv kernel size
        for name in out.featureNames {
            guard let arr = out.featureValue(for: name)?.multiArrayValue else { continue }
            let shape = arr.shape.map { $0.intValue }
            if shape == [1, 6144, convK + fixedN] {
                // Conv padded output: [1, 6144, 68] — need to slice [conv_k+seq_len-conv_k : conv_k+seq_len]
                // Extract the conv_k window at the real token end position
                let realEnd = convK + totalPrefill
                let sliced = try! MLMultiArray(shape: [1, 6144, NSNumber(value: convK)], dataType: arr.dataType)
                for ch in 0..<6144 {
                    for p in 0..<convK {
                        sliced[[0, ch, p] as [NSNumber]] = arr[[0, ch, realEnd - convK + p] as [NSNumber]]
                    }
                }
                let digits = name.unicodeScalars.filter { CharacterSet.decimalDigits.contains($0) }
                let num = Int(String(digits)) ?? 0
                convEntries.append((num, sliced))
            } else if shape == [1, 16, 128, 128] {
                // Rec state: clip_N — extract N for sort
                let digits = name.unicodeScalars.filter { CharacterSet.decimalDigits.contains($0) }
                let num = Int(String(digits)) ?? 0
                recEntries.append((num, arr))
            } else if name.hasPrefix("new_key_cache") {
                // Extract numeric suffix for ordering (no suffix = highest = last layer)
                let num = Int(name.replacingOccurrences(of: "new_key_cache", with: "").replacingOccurrences(of: "_", with: "")) ?? 999
                keyEntries.append((num, arr))
            } else if name.hasPrefix("new_value_cache") {
                let num = Int(name.replacingOccurrences(of: "new_value_cache", with: "").replacingOccurrences(of: "_", with: "")) ?? 999
                valEntries.append((num, arr))
            }
        }
        outConv = convEntries.sorted { $0.0 < $1.0 }.map { $0.1 }
        outRec = recEntries.sorted { $0.0 < $1.0 }.map { $0.1 }
        outKey = keyEntries.sorted { $0.0 < $1.0 }.map { $0.1 }
        outVal = valEntries.sorted { $0.0 < $1.0 }.map { $0.1 }

        // Debug: check conv output strides
        for name in out.featureNames {
            if let arr = out.featureValue(for: name)?.multiArrayValue, arr.shape.count == 3 && arr.shape[2].intValue == convK + fixedN {
                fputs("  Conv \(name): shape=\(arr.shape) strides=\(arr.strides)\n", stderr)
                break
            }
        }
        fputs("  States: \(outConv.count) conv, \(outRec.count) rec, \(outKey.count) key, \(outVal.count) val\n", stderr)
        let pos = promptLen + totalPrefill
        return (maxIdx, outConv, outRec, outKey, outVal, pos)
    }

    /// Clean up a raw transcription.
    func cleanup(_ text: String) -> String {
        let t0 = CFAbsoluteTimeGetCurrent()

        // Tokenize: transcription text + suffix
        let textIds = tokenizer.encode(text: text)
        let allPrefillIds = textIds + suffixIds

        // Hybrid prefill: batch the text tokens, per-token the suffix
        // This synchronizes states with the decode model for compatible generation
        let prefillStart = CFAbsoluteTimeGetCurrent()

        // Step 1: Batch prefill text tokens only
        var (_, conv, rec, key, val, pos) = batchPrefill(textIds)

        // Step 2: Per-token decode the suffix tokens (syncs states with decode model)
        var lastToken = 0
        for id in suffixIds {
            lastToken = step(tokenId: Int32(id), pos: pos, conv: &conv, rec: &rec, key: &key, val: &val)
            pos += 1
        }
        let prefillMs = (CFAbsoluteTimeGetCurrent() - prefillStart) * 1000

        // If first token is <think>, force-feed </think>\n to skip thinking
        fputs("  First gen token: \(lastToken) (think=\(thinkTokenId))\n", stderr)
        if lastToken == thinkTokenId {
            _ = step(tokenId: Int32(thinkEndTokenId), pos: pos, conv: &conv, rec: &rec, key: &key, val: &val)
            pos += 1
            lastToken = step(tokenId: Int32(198), pos: pos, conv: &conv, rec: &rec, key: &key, val: &val) // 198 = \n
            pos += 1
        }

        // Generate (respect KV cache limit)
        let maxGen = min(maxGenTokens, maxKV - pos - 1)
        var generated: [Int] = [lastToken]
        var currentToken = Int32(lastToken)
        for _ in 0..<maxGen {
            let tok = step(tokenId: currentToken, pos: pos, conv: &conv, rec: &rec, key: &key, val: &val)
            if tok == eosTokenId || tok == imEndTokenId { break }
            generated.append(tok)
            currentToken = Int32(tok)
            pos += 1
        }

        let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000

        // Decode tokens to text
        var output = tokenizer.decode(tokens: generated)

        // Strip thinking tags if present
        if let thinkEnd = output.range(of: "</think>") {
            output = String(output[thinkEnd.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
        }

        let genToks = generated.count
        let genMs = ms - prefillMs
        fputs("  [batch \(textIds.count)tok + suffix \(suffixIds.count)tok = \(String(format: "%.0f", prefillMs))ms | gen \(genToks)tok = \(String(format: "%.0f", genMs))ms | total \(String(format: "%.0f", ms))ms]\n", stderr)

        return output
    }
}

// MARK: - Main

let modelDir: String
let prefillDir: String
if CommandLine.arguments.count > 2 && CommandLine.arguments[1] == "--model-dir" {
    modelDir = CommandLine.arguments[2]
    prefillDir = CommandLine.arguments.count > 4 && CommandLine.arguments[3] == "--prefill-dir"
        ? CommandLine.arguments[4] : defaultPrefillDir
} else {
    modelDir = defaultModelDir
    prefillDir = defaultPrefillDir
}

let engine = try CleanupEngine(modelDir: modelDir, prefillDir: prefillDir)

fputs("Ready. Enter transcriptions (one per line, Ctrl-D to quit):\n", stderr)

while let line = readLine() {
    let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.isEmpty { continue }
    let cleaned = engine.cleanup(trimmed)
    print(cleaned)
}
