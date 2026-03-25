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
let nDelta = 18
let nAttn = 6
let maxKV = 256
let ropeDim = 64
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
    let model: MLModel
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

    init(modelDir: String) throws {
        let t0 = CFAbsoluteTimeGetCurrent()

        // Load tokenizer
        let tokDir = URL(fileURLWithPath: "\(modelDir)/tokenizer")
        tokenizer = try AutoTokenizer.from(modelFolder: tokDir)

        // Load model
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndGPU
        let modelURL = URL(fileURLWithPath: "\(modelDir)/model.mlpackage")
        model = try MLModel(contentsOf: MLModel.compileModel(at: modelURL), configuration: cfg)

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

        // Pre-tokenize suffix: close transcription, start assistant turn
        suffixIds = tokenizer.encode(text: "</transcription>\n<|im_end|>\n<|im_start|>assistant\n")

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

        let out = try! model.prediction(from: MLDictionaryFeatureProvider(dictionary: dict))

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

    /// Clean up a raw transcription.
    func cleanup(_ text: String) -> String {
        let t0 = CFAbsoluteTimeGetCurrent()

        // Start from cached system prompt states
        var conv = cachedConv
        var rec = cachedRec
        var key = cachedKey
        var val = cachedVal
        var pos = promptLen

        // Tokenize the user's transcription (just the raw text, no tags)
        let textIds = tokenizer.encode(text: text)

        // Feed transcription tokens (prefill — ignore logits)
        for id in textIds {
            _ = step(tokenId: Int32(id), pos: pos, conv: &conv, rec: &rec, key: &key, val: &val)
            pos += 1
        }

        // Feed suffix tokens (</transcription>\n<|im_end|>\n<|im_start|>assistant\n)
        var lastToken = 0
        for id in suffixIds {
            lastToken = step(tokenId: Int32(id), pos: pos, conv: &conv, rec: &rec, key: &key, val: &val)
            pos += 1
        }

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

        let prefillToks = textIds.count + suffixIds.count
        let genToks = generated.count
        fputs("  [\(prefillToks) prefill + \(genToks) gen in \(String(format: "%.0f", ms))ms]\n", stderr)

        return output
    }
}

// MARK: - Main

let modelDir: String
if CommandLine.arguments.count > 2 && CommandLine.arguments[1] == "--model-dir" {
    modelDir = CommandLine.arguments[2]
} else {
    modelDir = defaultModelDir
}

let engine = try CleanupEngine(modelDir: modelDir)

fputs("Ready. Enter transcriptions (one per line, Ctrl-D to quit):\n", stderr)

while let line = readLine() {
    let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.isEmpty { continue }
    let cleaned = engine.cleanup(trimmed)
    print(cleaned)
}
