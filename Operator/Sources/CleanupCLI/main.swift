// CleanupCLI v2 — Speech cleanup via CoreML batch prefill + decode.
//
// Architecture:
//   Rust tokenizer (via UniFFI) ← tokenize/detokenize
//   Swift (this file) ← CoreML inference + state management
//
// Usage:
//   swift run -c release CleanupCLI
//   echo "um so I was like going to the uh store" | swift run -c release CleanupCLI

import CoreML
import Foundation
import QwenTokenizerRust

// MARK: - Config

let nDelta = 18
let nAttn = 6
let maxKV = 256
let ropeDim = 64
let fixedN = 64
let maxGenTokens = 200
let eosTokenId: Int32 = 248_044
let imEndTokenId: Int32 = 248_046
let thinkTokenId: Int32 = 248_068
let thinkEndTokenId: Int32 = 248_069

// MARK: - Rust tokenizer (via UniFFI)

class TokenizerHelper {
    let inner: QwenTokenizer

    init(tokenizerJsonPath: String) {
        do {
            inner = try loadTokenizer(tokenizerJsonPath: tokenizerJsonPath)
        } catch {
            fatalError("Failed to load tokenizer from \(tokenizerJsonPath): \(error)")
        }
    }

    func encode(_ text: String) -> [Int32] {
        inner.encode(text: text)
    }

    func decode(_ ids: [Int32]) -> String {
        (try? inner.decode(ids: ids)) ?? ""
    }

    func suffixIds() -> [Int32] {
        inner.suffixIds()
    }
}

// MARK: - Npy loader

func loadNpy(_ path: String) throws -> MLMultiArray {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let hdrLen = Int(data[8]) | (Int(data[9]) << 8)
    let hdrEnd = 10 + hdrLen
    let hdr = String(data: data[10 ..< hdrEnd], encoding: .ascii)!
    let shapeStr = String(hdr[hdr.range(of: #"\([\d, ]+\)"#, options: .regularExpression)!]).dropFirst().dropLast()
    let shape = shapeStr.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    let count = shape.reduce(1, *)
    let raw = data[hdrEnd...]
    let isFP16 = hdr.contains("float16") || hdr.contains("<f2")
    let dt: MLMultiArrayDataType = isFP16 ? .float16 : .float32
    let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: dt)
    if isFP16 {
        raw.withUnsafeBytes { b in
            let s = b.bindMemory(to: UInt16.self)
            let d = arr.dataPointer.bindMemory(to: UInt16.self, capacity: count)
            for i in 0 ..< count { d[i] = s[i] }
        }
    } else {
        raw.withUnsafeBytes { b in
            let s = b.bindMemory(to: Float.self)
            let d = arr.dataPointer.bindMemory(to: Float.self, capacity: count)
            for i in 0 ..< count { d[i] = s[i] }
        }
    }
    return arr
}

// MARK: - MLMultiArray helpers

func argmax(_ a: MLMultiArray, offset: Int = 0, count n: Int? = nil) -> Int {
    let cnt = n ?? a.count
    if a.dataType == .float16 {
        let p = a.dataPointer.bindMemory(to: Float16.self, capacity: a.count)
        var mi = 0; var mv = p[offset]
        for i in 1 ..< cnt { if p[offset + i] > mv { mv = p[offset + i]; mi = i } }
        return mi
    }
    let p = a.dataPointer.bindMemory(to: Float.self, capacity: a.count)
    var mi = 0; var mv = p[offset]
    for i in 1 ..< cnt { if p[offset + i] > mv { mv = p[offset + i]; mi = i } }
    return mi
}

func int32Array(_ v: Int32) -> MLMultiArray {
    let a = try! MLMultiArray(shape: [1], dataType: .int32)
    a.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = v; return a
}

func f32Array(_ shape: [Int], fill: Float = 0) -> MLMultiArray {
    let c = shape.reduce(1, *)
    let a = try! MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
    let p = a.dataPointer.bindMemory(to: Float.self, capacity: c)
    for i in 0 ..< c { p[i] = fill }; return a
}

// MARK: - Engine

class CleanupEngine {
    let prefill: MLModel
    let decode: MLModel
    let tok: TokenizerHelper
    let cos: MLMultiArray
    let sin: MLMultiArray
    let suffix: [Int32]
    let initConv: [MLMultiArray]
    let initRec: [MLMultiArray]
    let initKey: [MLMultiArray]
    let initVal: [MLMultiArray]
    let sysLen: Int

    /// Compile .mlpackage to .mlmodelc next to it (reuses if already compiled and up-to-date).
    static func compiledURL(for mlpackagePath: String) throws -> URL {
        let src = URL(fileURLWithPath: mlpackagePath)
        let compiled = src.deletingPathExtension().appendingPathExtension("mlmodelc")
        let fm = FileManager.default

        // Reuse if compiled model exists and is newer than source
        if fm.fileExists(atPath: compiled.path) {
            let srcDate = (try? fm.attributesOfItem(atPath: src.path)[.modificationDate] as? Date) ?? .distantPast
            let cmpDate = (try? fm.attributesOfItem(atPath: compiled.path)[.modificationDate] as? Date) ?? .distantPast
            if cmpDate >= srcDate {
                return compiled
            }
            try? fm.removeItem(at: compiled)
        }

        // Compile to temp, copy to permanent location
        let tempCompiled = try MLModel.compileModel(at: src)
        do {
            if fm.fileExists(atPath: compiled.path) { try fm.removeItem(at: compiled) }
            try fm.copyItem(at: tempCompiled, to: compiled)
            try? fm.removeItem(at: tempCompiled)
            fputs("  Compiled \(src.lastPathComponent) → \(compiled.lastPathComponent)\n", stderr)
        } catch {
            fputs("  Warning: could not cache compiled model: \(error.localizedDescription)\n", stderr)
            fputs("    src: \(tempCompiled.path)\n    dst: \(compiled.path)\n", stderr)
            return tempCompiled  // fall back to temp
        }
        return compiled
    }

    init(decodeDir: String, prefillDir: String, tokenizerJsonPath: String) throws {
        let t0 = CFAbsoluteTimeGetCurrent()
        tok = TokenizerHelper(tokenizerJsonPath: tokenizerJsonPath)
        suffix = tok.suffixIds()

        let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndGPU
        decode = try MLModel(contentsOf: Self.compiledURL(for: "\(decodeDir)/model.mlpackage"), configuration: cfg)
        prefill = try MLModel(contentsOf: Self.compiledURL(for: "\(prefillDir)/model.mlpackage"), configuration: cfg)

        let cd = "\(decodeDir)/prompt_cache"
        var c = [MLMultiArray](), r = [MLMultiArray](), k = [MLMultiArray](), v = [MLMultiArray]()
        for j in 0 ..< nDelta { c.append(try loadNpy("\(cd)/conv_state_\(j).npy")); r.append(try loadNpy("\(cd)/rec_state_\(j).npy")) }
        for j in 0 ..< nAttn { k.append(try loadNpy("\(cd)/key_cache_\(j).npy")); v.append(try loadNpy("\(cd)/value_cache_\(j).npy")) }
        initConv = c; initRec = r; initKey = k; initVal = v
        cos = try loadNpy("\(cd)/rope_cos.npy"); sin = try loadNpy("\(cd)/rope_sin.npy")
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: "\(cd)/meta.json"))) as! [String: Any]
        sysLen = meta["prompt_len"] as! Int
        fputs("Loaded in \(String(format: "%.0f", (CFAbsoluteTimeGetCurrent() - t0) * 1000))ms\n", stderr)
    }

    // MARK: Prefill

    func batchPrefill(_ textIds: [Int32]) -> (Int32, [MLMultiArray], [MLMultiArray], [MLMultiArray], [MLMultiArray], Int) {
        let all = textIds + suffix; let n = all.count
        let buildStart = CFAbsoluteTimeGetCurrent()

        let ids = try! MLMultiArray(shape: [1, NSNumber(value: fixedN)], dataType: .int32)
        let ip = ids.dataPointer.bindMemory(to: Int32.self, capacity: fixedN)
        for i in 0 ..< fixedN { ip[i] = i < n ? all[i] : 0 }

        let bc = f32Array([1, fixedN, ropeDim])
        let bs = f32Array([1, fixedN, ropeDim])
        let ac = cos.dataPointer.bindMemory(to: Float.self, capacity: cos.count)
        let as2 = sin.dataPointer.bindMemory(to: Float.self, capacity: sin.count)
        let cp = bc.dataPointer.bindMemory(to: Float.self, capacity: fixedN * ropeDim)
        let sp = bs.dataPointer.bindMemory(to: Float.self, capacity: fixedN * ropeDim)
        for i in 0 ..< fixedN where sysLen + i < maxKV {
            let o = (sysLen + i) * ropeDim
            for j in 0 ..< ropeDim { cp[i * ropeDim + j] = ac[o + j]; sp[i * ropeDim + j] = as2[o + j] }
        }

        let mask = f32Array([1, 1, fixedN, maxKV], fill: -1e4)
        let mp = mask.dataPointer.bindMemory(to: Float.self, capacity: fixedN * maxKV)
        for i in 0 ..< fixedN {
            if i < n { for j in 0 ..< (sysLen + i + 1) { mp[i * maxKV + j] = 0 } }
            else { mp[i * maxKV] = 0 }
        }

        var d: [String: MLFeatureValue] = [
            "input_ids": .init(multiArray: ids), "seq_len": .init(multiArray: int32Array(Int32(n))),
            "start_pos": .init(multiArray: int32Array(Int32(sysLen))),
            "rope_cos": .init(multiArray: bc), "rope_sin": .init(multiArray: bs), "attn_mask": .init(multiArray: mask),
        ]
        for j in 0 ..< nDelta { d["conv_state_\(j)"] = .init(multiArray: initConv[j]); d["rec_state_\(j)"] = .init(multiArray: initRec[j]) }
        for j in 0 ..< nAttn { d["key_cache_\(j)"] = .init(multiArray: initKey[j]); d["value_cache_\(j)"] = .init(multiArray: initVal[j]) }

        let buildMs = (CFAbsoluteTimeGetCurrent() - buildStart) * 1000
        let predictStart = CFAbsoluteTimeGetCurrent()
        let out = try! prefill.prediction(from: MLDictionaryFeatureProvider(dictionary: d))
        let predictMs = (CFAbsoluteTimeGetCurrent() - predictStart) * 1000
        let extractStart = CFAbsoluteTimeGetCurrent()

        let logits = out.featureValue(for: "logits")!.multiArrayValue!
        let vocab = logits.shape.last!.intValue
        let firstTok = Int32(argmax(logits, offset: (n - 1) * vocab, count: vocab))

        // Extract conv states at the REAL token boundary, not the padded tail.
        // Conv output is [1, 6144, conv_k + fixedN]. The last 4 real QKV projections
        // start at offset `n` (= conv_k + n_real_tokens - conv_k = n_real_tokens).
        let convStart = n  // = actual token count (conv_k prepended + n real - conv_k)
        var co = [MLMultiArray](), ro = [MLMultiArray]()
        for j in 0 ..< nDelta {
            let full = out.featureValue(for: "conv_state_\(j)_out")!.multiArrayValue!
            let w = full.shape[2].intValue
            let sl = try! MLMultiArray(shape: [1, 6144, 4], dataType: full.dataType)
            if full.dataType == .float16 {
                let s = full.dataPointer.bindMemory(to: UInt16.self, capacity: full.count)
                let dd = sl.dataPointer.bindMemory(to: UInt16.self, capacity: sl.count)
                for ch in 0 ..< 6144 { for kk in 0 ..< 4 { dd[ch * 4 + kk] = s[ch * w + convStart + kk] } }
            } else {
                let s = full.dataPointer.bindMemory(to: Float.self, capacity: full.count)
                let dd = sl.dataPointer.bindMemory(to: Float.self, capacity: sl.count)
                for ch in 0 ..< 6144 { for kk in 0 ..< 4 { dd[ch * 4 + kk] = s[ch * w + convStart + kk] } }
            }
            co.append(sl); ro.append(out.featureValue(for: "rec_state_\(j)_out")!.multiArrayValue!)
        }
        var ko = [MLMultiArray](), vo = [MLMultiArray]()
        for j in 0 ..< nAttn {
            ko.append(out.featureValue(for: "key_cache_\(j)_out")!.multiArrayValue!)
            vo.append(out.featureValue(for: "value_cache_\(j)_out")!.multiArrayValue!)
        }
        let extractMs = (CFAbsoluteTimeGetCurrent() - extractStart) * 1000
        fputs("    prefill breakdown: build=\(String(format: "%.0f", buildMs))ms predict=\(String(format: "%.0f", predictMs))ms extract=\(String(format: "%.0f", extractMs))ms\n", stderr)
        return (firstTok, co, ro, ko, vo, sysLen + n)
    }

    // MARK: Decode step

    func step(_ token: Int32, _ pos: Int, _ cv: inout [MLMultiArray], _ rc: inout [MLMultiArray], _ kc: inout [MLMultiArray], _ vc: inout [MLMultiArray]) -> Int32 {
        let ta = try! MLMultiArray(shape: [1, 1], dataType: .int32)
        ta.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = token
        let cosA = f32Array([1, 1, ropeDim]); let sinA = f32Array([1, 1, ropeDim])
        let ac = cos.dataPointer.bindMemory(to: Float.self, capacity: cos.count)
        let as2 = sin.dataPointer.bindMemory(to: Float.self, capacity: sin.count)
        let cp2 = cosA.dataPointer.bindMemory(to: Float.self, capacity: ropeDim)
        let sp2 = sinA.dataPointer.bindMemory(to: Float.self, capacity: ropeDim)
        let o = pos * ropeDim; for i in 0 ..< ropeDim { cp2[i] = ac[o + i]; sp2[i] = as2[o + i] }
        let mask = f32Array([1, 1, 1, maxKV], fill: -1e4)
        let mp = mask.dataPointer.bindMemory(to: Float.self, capacity: maxKV)
        for i in 0 ... pos { mp[i] = 0 }

        var d: [String: MLFeatureValue] = [
            "input_id": .init(multiArray: ta), "rope_cos": .init(multiArray: cosA), "rope_sin": .init(multiArray: sinA),
            "cache_position": .init(multiArray: int32Array(Int32(pos))), "attn_mask": .init(multiArray: mask),
        ]
        for j in 0 ..< nDelta { d["conv_state_\(j)"] = .init(multiArray: cv[j]); d["rec_state_\(j)"] = .init(multiArray: rc[j]) }
        for j in 0 ..< nAttn { d["key_cache_\(j)"] = .init(multiArray: kc[j]); d["value_cache_\(j)"] = .init(multiArray: vc[j]) }

        let out = try! decode.prediction(from: MLDictionaryFeatureProvider(dictionary: d))
        for j in 0 ..< nDelta { cv[j] = out.featureValue(for: "conv_state_\(j)_out")!.multiArrayValue!; rc[j] = out.featureValue(for: "rec_state_\(j)_out")!.multiArrayValue! }
        for j in 0 ..< nAttn { kc[j] = out.featureValue(for: "key_cache_\(j)_out")!.multiArrayValue!; vc[j] = out.featureValue(for: "value_cache_\(j)_out")!.multiArrayValue! }
        return Int32(argmax(out.featureValue(for: "logits")!.multiArrayValue!))
    }

    // MARK: Full pipeline

    func cleanup(_ text: String) -> String {
        let wall = CFAbsoluteTimeGetCurrent()

        // 1. Tokenize
        var t = CFAbsoluteTimeGetCurrent()
        let textIds = tok.encode(text)
        let tokenizeMs = (CFAbsoluteTimeGetCurrent() - t) * 1000

        // 2. Batch prefill (input build + predict + state extract)
        t = CFAbsoluteTimeGetCurrent()
        var (firstTok, cv, rc, kc, vc, pos) = batchPrefill(textIds)
        let prefillMs = (CFAbsoluteTimeGetCurrent() - t) * 1000

        // 3. Skip thinking
        var thinkMs = 0.0
        if firstTok == thinkTokenId {
            t = CFAbsoluteTimeGetCurrent()
            _ = step(thinkEndTokenId, pos, &cv, &rc, &kc, &vc); pos += 1
            firstTok = step(198, pos, &cv, &rc, &kc, &vc); pos += 1
            thinkMs = (CFAbsoluteTimeGetCurrent() - t) * 1000
        }

        // 4. Generate (per-step timing)
        t = CFAbsoluteTimeGetCurrent()
        var gen: [Int32] = [firstTok]; var cur = firstTok
        var stepTimes = [Double]()
        for _ in 0 ..< min(maxGenTokens, maxKV - pos - 1) {
            let s = CFAbsoluteTimeGetCurrent()
            let tok = step(cur, pos, &cv, &rc, &kc, &vc)
            stepTimes.append((CFAbsoluteTimeGetCurrent() - s) * 1000)
            if tok == eosTokenId || tok == imEndTokenId { break }
            gen.append(tok); cur = tok; pos += 1
        }
        let genMs = (CFAbsoluteTimeGetCurrent() - t) * 1000

        // 5. Detokenize
        t = CFAbsoluteTimeGetCurrent()
        let output = tok.decode(gen)
        let detokMs = (CFAbsoluteTimeGetCurrent() - t) * 1000

        let totalMs = (CFAbsoluteTimeGetCurrent() - wall) * 1000

        // Detailed timing breakdown
        stepTimes.sort()
        let medStep = stepTimes.isEmpty ? 0 : stepTimes[stepTimes.count / 2]
        let tokPerSec = stepTimes.isEmpty ? 0 : 1000.0 / medStep

        fputs("  tokenize: \(String(format: "%.0f", tokenizeMs))ms\n", stderr)
        fputs("  prefill:  \(String(format: "%.0f", prefillMs))ms (\(textIds.count)+\(suffix.count)=\(textIds.count + suffix.count) tokens)\n", stderr)
        if thinkMs > 0 { fputs("  no-think: \(String(format: "%.0f", thinkMs))ms (2 decode steps)\n", stderr) }
        fputs("  generate: \(String(format: "%.0f", genMs))ms (\(gen.count) tokens, \(String(format: "%.1f", medStep))ms/tok median, \(String(format: "%.0f", tokPerSec)) tok/s)\n", stderr)
        fputs("  detokize: \(String(format: "%.0f", detokMs))ms\n", stderr)
        fputs("  TOTAL:    \(String(format: "%.0f", totalMs))ms\n", stderr)

        return output
    }
}

// MARK: - Main

let modelDir: String
let prefillDir: String
if let idx = CommandLine.arguments.firstIndex(of: "--model-dir"), idx + 1 < CommandLine.arguments.count {
    modelDir = CommandLine.arguments[idx + 1]
} else {
    modelDir = "models/monolith_kv_fp16"
}
if let idx = CommandLine.arguments.firstIndex(of: "--prefill-dir"), idx + 1 < CommandLine.arguments.count {
    prefillDir = CommandLine.arguments[idx + 1]
} else {
    prefillDir = "models/batch_prefill_fp16"
}

// Resolve tokenizer.json — check model dir first, then fallback paths
let tokenizerJson: String = {
    let fm = FileManager.default
    for candidate in [
        "\(modelDir)/tokenizer/tokenizer.json",
        "models/monolith_kv_fp16/tokenizer/tokenizer.json",
        "../models/monolith_kv_fp16/tokenizer/tokenizer.json",
    ] {
        if fm.fileExists(atPath: candidate) { return candidate }
    }
    fatalError("tokenizer.json not found. Expected at \(modelDir)/tokenizer/tokenizer.json")
}()

let engine = try CleanupEngine(
    decodeDir: modelDir,
    prefillDir: prefillDir,
    tokenizerJsonPath: tokenizerJson
)
fputs("Ready. Enter transcriptions (Ctrl-D to quit):\n", stderr)
while let line = Swift.readLine() {
    let t = line.trimmingCharacters(in: .whitespacesAndNewlines)
    if t.isEmpty { continue }
    fputs("  Processing: \(t.prefix(40))...\n", stderr)
    let result = engine.cleanup(t)
    fputs("  Result: \(result.prefix(40))...\n", stderr)
    print(result)
}
