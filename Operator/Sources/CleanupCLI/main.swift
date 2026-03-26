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

private let nDelta = 18
private let nAttn = 6
private let maxKV = 256
private let ropeDim = 64
private let fixedN = 64
private let maxGenTokens = 200
private let eosTokenId: Int32 = 248_044
private let imEndTokenId: Int32 = 248_046
private let thinkTokenId: Int32 = 248_068
private let thinkEndTokenId: Int32 = 248_069

// MARK: - Rust tokenizer (via UniFFI)

private class TokenizerHelper {
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

private func loadNpy(_ path: String) throws -> MLMultiArray {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let hdrLen = Int(data[8]) | (Int(data[9]) << 8)
    let hdrEnd = 10 + hdrLen
    guard let hdr = String(data: data[10..<hdrEnd], encoding: .ascii) else {
        throw EngineError.missingOutput("npy header at \(path)")
    }
    guard let shapeRange = hdr.range(of: #"\([\d, ]+\)"#, options: .regularExpression) else {
        throw EngineError.missingOutput("npy shape at \(path)")
    }
    let shapeStr = String(hdr[shapeRange]).dropFirst().dropLast()
    let shape = shapeStr.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    let count = shape.reduce(1, *)
    let raw = data[hdrEnd...]
    let isFP16 = hdr.contains("float16") || hdr.contains("<f2")
    let dt: MLMultiArrayDataType = isFP16 ? .float16 : .float32
    let arr = try MLMultiArray(shape: shape.map { $0 as NSNumber }, dataType: dt)
    if isFP16 {
        raw.withUnsafeBytes { b in
            let s = b.bindMemory(to: UInt16.self)
            let d = arr.dataPointer.bindMemory(to: UInt16.self, capacity: count)
            for i in 0..<count { d[i] = s[i] }
        }
    } else {
        raw.withUnsafeBytes { b in
            let s = b.bindMemory(to: Float.self)
            let d = arr.dataPointer.bindMemory(to: Float.self, capacity: count)
            for i in 0..<count { d[i] = s[i] }
        }
    }
    return arr
}

// MARK: - Model output helpers

private enum EngineError: Error {
    case missingOutput(String)
}

private func modelArray(_ output: MLFeatureProvider, _ name: String) throws -> MLMultiArray {
    guard let val = output.featureValue(for: name)?.multiArrayValue else {
        throw EngineError.missingOutput(name)
    }
    return val
}

// MARK: - MLMultiArray helpers

private func argmax(_ a: MLMultiArray, offset: Int = 0, count n: Int? = nil) -> Int {
    let cnt = n ?? a.count
    if a.dataType == .float16 {
        let p = a.dataPointer.bindMemory(to: Float16.self, capacity: a.count)
        var mi = 0
        var mv = p[offset]
        for i in 1..<cnt where p[offset + i] > mv {
            mv = p[offset + i]
            mi = i
        }
        return mi
    }
    let p = a.dataPointer.bindMemory(to: Float.self, capacity: a.count)
    var mi = 0
    var mv = p[offset]
    for i in 1..<cnt where p[offset + i] > mv {
        mv = p[offset + i]
        mi = i
    }
    return mi
}

private func int32Array(_ v: Int32) throws -> MLMultiArray {
    let a = try MLMultiArray(shape: [1], dataType: .int32)
    a.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = v
    return a
}

private func f32Array(_ shape: [Int], fill: Float = 0) throws -> MLMultiArray {
    let c = shape.reduce(1, *)
    let a = try MLMultiArray(shape: shape.map { $0 as NSNumber }, dataType: .float32)
    let p = a.dataPointer.bindMemory(to: Float.self, capacity: c)
    for i in 0..<c { p[i] = fill }
    return a
}

// MARK: - Prefill result

private struct PrefillResult {
    var firstToken: Int32
    var convStates: [MLMultiArray]
    var recStates: [MLMultiArray]
    var keyCache: [MLMultiArray]
    var valCache: [MLMultiArray]
    var position: Int
}

// MARK: - Engine

private class CleanupEngine {
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

        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndGPU
        decode = try MLModel(contentsOf: Self.compiledURL(for: "\(decodeDir)/model.mlpackage"), configuration: cfg)
        prefill = try MLModel(contentsOf: Self.compiledURL(for: "\(prefillDir)/model.mlpackage"), configuration: cfg)

        let cd = "\(decodeDir)/prompt_cache"
        var c: [MLMultiArray] = []
        var r: [MLMultiArray] = []
        var k: [MLMultiArray] = []
        var v: [MLMultiArray] = []
        for j in 0..<nDelta {
            c.append(try loadNpy("\(cd)/conv_state_\(j).npy"))
            r.append(try loadNpy("\(cd)/rec_state_\(j).npy"))
        }
        for j in 0..<nAttn {
            k.append(try loadNpy("\(cd)/key_cache_\(j).npy"))
            v.append(try loadNpy("\(cd)/value_cache_\(j).npy"))
        }
        initConv = c
        initRec = r
        initKey = k
        initVal = v
        cos = try loadNpy("\(cd)/rope_cos.npy")
        sin = try loadNpy("\(cd)/rope_sin.npy")
        guard
            let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: "\(cd)/meta.json")))
                as? [String: Any],
            let promptLen = meta["prompt_len"] as? Int
        else {
            throw NSError(domain: "CleanupEngine", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid meta.json"])
        }
        sysLen = promptLen
        fputs("Loaded in \(String(format: "%.0f", (CFAbsoluteTimeGetCurrent() - t0) * 1_000))ms\n", stderr)
    }

    // MARK: Prefill

    private func extractConvSlice(
        _ full: MLMultiArray,
        startOffset: Int
    ) throws -> MLMultiArray {
        let w = full.shape[2].intValue
        let sl = try MLMultiArray(shape: [1, 6_144, 4], dataType: full.dataType)
        if full.dataType == .float16 {
            let src = full.dataPointer.bindMemory(to: UInt16.self, capacity: full.count)
            let dst = sl.dataPointer.bindMemory(to: UInt16.self, capacity: sl.count)
            for ch in 0..<6_144 {
                for kk in 0..<4 { dst[ch * 4 + kk] = src[ch * w + startOffset + kk] }
            }
        } else {
            let src = full.dataPointer.bindMemory(to: Float.self, capacity: full.count)
            let dst = sl.dataPointer.bindMemory(to: Float.self, capacity: sl.count)
            for ch in 0..<6_144 {
                for kk in 0..<4 { dst[ch * 4 + kk] = src[ch * w + startOffset + kk] }
            }
        }
        return sl
    }

    private func buildPrefillInputs(
        _ tokenIds: [Int32]
    ) throws -> [String: MLFeatureValue] {
        let n = tokenIds.count

        let ids = try MLMultiArray(shape: [1, fixedN as NSNumber], dataType: .int32)
        let ip = ids.dataPointer.bindMemory(to: Int32.self, capacity: fixedN)
        for i in 0..<fixedN { ip[i] = i < n ? tokenIds[i] : 0 }

        let bc = try f32Array([1, fixedN, ropeDim])
        let bs = try f32Array([1, fixedN, ropeDim])
        let ac = cos.dataPointer.bindMemory(to: Float.self, capacity: cos.count)
        let as2 = sin.dataPointer.bindMemory(to: Float.self, capacity: sin.count)
        let cp = bc.dataPointer.bindMemory(to: Float.self, capacity: fixedN * ropeDim)
        let sp = bs.dataPointer.bindMemory(to: Float.self, capacity: fixedN * ropeDim)
        for i in 0..<fixedN where sysLen + i < maxKV {
            let o = (sysLen + i) * ropeDim
            for j in 0..<ropeDim {
                cp[i * ropeDim + j] = ac[o + j]
                sp[i * ropeDim + j] = as2[o + j]
            }
        }

        let mask = try f32Array([1, 1, fixedN, maxKV], fill: -1e4)
        let mp = mask.dataPointer.bindMemory(to: Float.self, capacity: fixedN * maxKV)
        for i in 0..<fixedN {
            if i < n {
                for j in 0..<(sysLen + i + 1) { mp[i * maxKV + j] = 0 }
            } else {
                mp[i * maxKV] = 0
            }
        }

        var d: [String: MLFeatureValue] = [
            "input_ids": .init(multiArray: ids),
            "seq_len": .init(multiArray: try int32Array(Int32(n))),
            "start_pos": .init(multiArray: try int32Array(Int32(sysLen))),
            "rope_cos": .init(multiArray: bc),
            "rope_sin": .init(multiArray: bs),
            "attn_mask": .init(multiArray: mask)
        ]
        for j in 0..<nDelta {
            d["conv_state_\(j)"] = .init(multiArray: initConv[j])
            d["rec_state_\(j)"] = .init(multiArray: initRec[j])
        }
        for j in 0..<nAttn {
            d["key_cache_\(j)"] = .init(multiArray: initKey[j])
            d["value_cache_\(j)"] = .init(multiArray: initVal[j])
        }
        return d
    }

    func batchPrefill(_ textIds: [Int32]) throws -> PrefillResult {
        let all = textIds + suffix
        let n = all.count
        let buildStart = CFAbsoluteTimeGetCurrent()
        let d = try buildPrefillInputs(all)

        let buildMs = (CFAbsoluteTimeGetCurrent() - buildStart) * 1_000
        let predictStart = CFAbsoluteTimeGetCurrent()
        let out = try prefill.prediction(from: MLDictionaryFeatureProvider(dictionary: d))
        let predictMs = (CFAbsoluteTimeGetCurrent() - predictStart) * 1_000
        let extractStart = CFAbsoluteTimeGetCurrent()

        let logits = try modelArray(out, "logits")
        guard let vocabNS = logits.shape.last else {
            throw EngineError.missingOutput("logits.shape")
        }

        let vocab = vocabNS.intValue
        let firstTok = Int32(argmax(logits, offset: (n - 1) * vocab, count: vocab))

        // Extract conv states at the REAL token boundary, not the padded tail.
        var co: [MLMultiArray] = []
        var ro: [MLMultiArray] = []
        for j in 0..<nDelta {
            let full = try modelArray(out, "conv_state_\(j)_out")
            co.append(try extractConvSlice(full, startOffset: n))
            ro.append(try modelArray(out, "rec_state_\(j)_out"))
        }
        var ko: [MLMultiArray] = []
        var vo: [MLMultiArray] = []
        for j in 0..<nAttn {
            ko.append(try modelArray(out, "key_cache_\(j)_out"))
            vo.append(try modelArray(out, "value_cache_\(j)_out"))
        }

        let extractMs = (CFAbsoluteTimeGetCurrent() - extractStart) * 1_000
        fputs(
            "    prefill: build=\(String(format: "%.0f", buildMs))ms"
                + " predict=\(String(format: "%.0f", predictMs))ms"
                + " extract=\(String(format: "%.0f", extractMs))ms\n",
            stderr
        )
        return PrefillResult(
            firstToken: firstTok,
            convStates: co,
            recStates: ro,
            keyCache: ko,
            valCache: vo,
            position: sysLen + n
        )
    }

    // MARK: Decode step

    func step(
        _ token: Int32,
        _ pos: Int,
        _ cv: inout [MLMultiArray],
        _ rc: inout [MLMultiArray],
        _ kc: inout [MLMultiArray],
        _ vc: inout [MLMultiArray]
    ) throws -> Int32 {
        let ta = try MLMultiArray(shape: [1, 1], dataType: .int32)
        ta.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = token
        let cosA = try f32Array([1, 1, ropeDim])
        let sinA = try f32Array([1, 1, ropeDim])
        let ac = cos.dataPointer.bindMemory(to: Float.self, capacity: cos.count)
        let as2 = sin.dataPointer.bindMemory(to: Float.self, capacity: sin.count)
        let cp2 = cosA.dataPointer.bindMemory(to: Float.self, capacity: ropeDim)
        let sp2 = sinA.dataPointer.bindMemory(to: Float.self, capacity: ropeDim)
        let o = pos * ropeDim
        for i in 0..<ropeDim {
            cp2[i] = ac[o + i]
            sp2[i] = as2[o + i]
        }
        let mask = try f32Array([1, 1, 1, maxKV], fill: -1e4)
        let mp = mask.dataPointer.bindMemory(to: Float.self, capacity: maxKV)
        for i in 0...pos { mp[i] = 0 }

        var d: [String: MLFeatureValue] = [
            "input_id": .init(multiArray: ta), "rope_cos": .init(multiArray: cosA), "rope_sin": .init(multiArray: sinA),
            "cache_position": .init(multiArray: try int32Array(Int32(pos))), "attn_mask": .init(multiArray: mask)
        ]
        for j in 0..<nDelta {
            d["conv_state_\(j)"] = .init(multiArray: cv[j])
            d["rec_state_\(j)"] = .init(multiArray: rc[j])
        }
        for j in 0..<nAttn {
            d["key_cache_\(j)"] = .init(multiArray: kc[j])
            d["value_cache_\(j)"] = .init(multiArray: vc[j])
        }

        let out = try decode.prediction(from: MLDictionaryFeatureProvider(dictionary: d))
        for j in 0..<nDelta {
            cv[j] = try modelArray(out, "conv_state_\(j)_out")
            rc[j] = try modelArray(out, "rec_state_\(j)_out")
        }
        for j in 0..<nAttn {
            kc[j] = try modelArray(out, "key_cache_\(j)_out")
            vc[j] = try modelArray(out, "value_cache_\(j)_out")
        }
        return Int32(argmax(try modelArray(out, "logits")))
    }

    // MARK: Full pipeline

    func cleanup(_ text: String) throws -> String {
        let wall = CFAbsoluteTimeGetCurrent()

        // 1. Tokenize
        var t = CFAbsoluteTimeGetCurrent()
        let textIds = tok.encode(text)
        let tokenizeMs = (CFAbsoluteTimeGetCurrent() - t) * 1_000

        // 2. Batch prefill (input build + predict + state extract)
        t = CFAbsoluteTimeGetCurrent()
        var pf = try batchPrefill(textIds)
        let prefillMs = (CFAbsoluteTimeGetCurrent() - t) * 1_000

        // 3. Skip thinking
        var thinkMs = 0.0
        if pf.firstToken == thinkTokenId {
            t = CFAbsoluteTimeGetCurrent()
            _ = try step(thinkEndTokenId, pf.position, &pf.convStates, &pf.recStates, &pf.keyCache, &pf.valCache)
            pf.position += 1
            pf.firstToken = try step(198, pf.position, &pf.convStates, &pf.recStates, &pf.keyCache, &pf.valCache)
            pf.position += 1
            thinkMs = (CFAbsoluteTimeGetCurrent() - t) * 1_000
        }

        // 4. Generate (per-step timing)
        t = CFAbsoluteTimeGetCurrent()
        var gen: [Int32] = [pf.firstToken]
        var cur = pf.firstToken
        var stepTimes: [Double] = []
        for _ in 0..<min(maxGenTokens, maxKV - pf.position - 1) {
            let s = CFAbsoluteTimeGetCurrent()
            let tok = try step(cur, pf.position, &pf.convStates, &pf.recStates, &pf.keyCache, &pf.valCache)
            stepTimes.append((CFAbsoluteTimeGetCurrent() - s) * 1_000)
            if tok == eosTokenId || tok == imEndTokenId { break }
            gen.append(tok)
            cur = tok
            pf.position += 1
        }
        let genMs = (CFAbsoluteTimeGetCurrent() - t) * 1_000

        // 5. Detokenize
        t = CFAbsoluteTimeGetCurrent()
        let output = tok.decode(gen)
        let detokMs = (CFAbsoluteTimeGetCurrent() - t) * 1_000

        let totalMs = (CFAbsoluteTimeGetCurrent() - wall) * 1_000

        // Detailed timing breakdown
        stepTimes.sort()
        let medStep = stepTimes.isEmpty ? 0 : stepTimes[stepTimes.count / 2]
        let tokPerSec = stepTimes.isEmpty ? 0 : 1_000.0 / medStep

        let nTok = textIds.count
        let nSfx = suffix.count

        fputs("  tokenize: \(String(format: "%.0f", tokenizeMs))ms\n", stderr)
        fputs("  prefill:  \(String(format: "%.0f", prefillMs))ms (\(nTok)+\(nSfx)=\(nTok + nSfx) tokens)\n", stderr)
        if thinkMs > 0 {
            fputs("  no-think: \(String(format: "%.0f", thinkMs))ms (2 decode steps)\n", stderr)
        }
        let genFmt = String(format: "%.0f", genMs)
        let medFmt = String(format: "%.1f", medStep)
        let tpsFmt = String(format: "%.0f", tokPerSec)
        fputs("  generate: \(genFmt)ms (\(gen.count) tok, \(medFmt)ms/tok, \(tpsFmt) tok/s)\n", stderr)
        fputs("  detokize: \(String(format: "%.0f", detokMs))ms\n", stderr)
        fputs("  TOTAL:    \(String(format: "%.0f", totalMs))ms\n", stderr)

        return output
    }
}

// MARK: - Main

private func parseArg(_ flag: String, default defaultValue: String) -> String {
    if let idx = CommandLine.arguments.firstIndex(of: flag), idx + 1 < CommandLine.arguments.count {
        return CommandLine.arguments[idx + 1]
    }
    return defaultValue
}

private func resolveTokenizerJson(modelDir: String) -> String {
    let fm = FileManager.default
    for candidate in [
        "\(modelDir)/tokenizer/tokenizer.json",
        "models/monolith_kv_fp16/tokenizer/tokenizer.json",
        "../models/monolith_kv_fp16/tokenizer/tokenizer.json"
    ] where fm.fileExists(atPath: candidate) {
        return candidate
    }
    fatalError("tokenizer.json not found. Expected at \(modelDir)/tokenizer/tokenizer.json")
}

private let modelDir = parseArg("--model-dir", default: "models/monolith_kv_fp16")
private let prefillDir = parseArg("--prefill-dir", default: "models/batch_prefill_fp16")
private let tokenizerJson = resolveTokenizerJson(modelDir: modelDir)

private let engine = try CleanupEngine(
    decodeDir: modelDir,
    prefillDir: prefillDir,
    tokenizerJsonPath: tokenizerJson
)

fputs("Ready. Enter transcriptions (Ctrl-D to quit):\n", stderr)
while let line = Swift.readLine() {
    let text = line.trimmingCharacters(in: .whitespacesAndNewlines)
    if text.isEmpty { continue }
    fputs("  Processing: \(text.prefix(40))...\n", stderr)
    let result = try engine.cleanup(text)
    fputs("  Result: \(result.prefix(40))...\n", stderr)
    print(result)
}
