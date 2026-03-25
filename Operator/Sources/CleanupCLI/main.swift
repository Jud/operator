// CleanupCLI v2 — Speech cleanup via CoreML batch prefill + decode.
//
// Architecture:
//   Python helper (tokenizer_helper.py) ← tokenize/detokenize
//   Swift (this file) ← CoreML inference + state management
//
// Usage:
//   swift run -c release CleanupCLI
//   echo "um so I was like going to the uh store" | swift run -c release CleanupCLI

import CoreML
import Foundation

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

// MARK: - Python tokenizer helper

class TokenizerHelper {
    let process: Process
    let stdinHandle: FileHandle
    let reader: FileHandle

    init(pythonPath: String, scriptPath: String) {
        process = Process()
        process.executableURL = URL(fileURLWithPath: pythonPath)
        process.arguments = [scriptPath]
        process.environment = ProcessInfo.processInfo.environment

        let inPipe = Pipe()
        let outPipe = Pipe()
        process.standardInput = inPipe
        process.standardOutput = outPipe
        process.standardError = FileHandle.nullDevice

        stdinHandle = inPipe.fileHandleForWriting
        reader = outPipe.fileHandleForReading

        try! process.run()

        // Wait for "ready"
        let ready = readOneLine()
        precondition(ready == "ready", "Tokenizer helper did not start: got '\(ready ?? "nil")'")
    }

    private func readOneLine() -> String? {
        var buf = Data()
        while true {
            let byte = reader.readData(ofLength: 1)
            if byte.isEmpty { return nil }
            if byte[0] == 0x0A { break }
            buf.append(byte)
        }
        return String(data: buf, encoding: .utf8)
    }

    private func send(_ cmd: String) -> String {
        stdinHandle.write(Data((cmd + "\n").utf8))
        return readOneLine() ?? ""
    }

    func encode(_ text: String) -> [Int32] {
        send("encode \(text)").split(separator: " ").compactMap { Int32($0) }
    }

    func decode(_ ids: [Int32]) -> String {
        send("decode \(ids.map { String($0) }.joined(separator: " "))")
    }

    func suffixIds() -> [Int32] {
        send("suffix").split(separator: " ").compactMap { Int32($0) }
    }

    deinit { stdinHandle.write(Data("quit\n".utf8)); process.terminate() }
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

    init(decodeDir: String, prefillDir: String, python: String, script: String) throws {
        let t0 = CFAbsoluteTimeGetCurrent()
        tok = TokenizerHelper(pythonPath: python, scriptPath: script)
        suffix = tok.suffixIds()

        let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndGPU
        decode = try MLModel(contentsOf: MLModel.compileModel(at: URL(fileURLWithPath: "\(decodeDir)/model.mlpackage")), configuration: cfg)
        prefill = try MLModel(contentsOf: MLModel.compileModel(at: URL(fileURLWithPath: "\(prefillDir)/model.mlpackage")), configuration: cfg)

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

        let out = try! prefill.prediction(from: MLDictionaryFeatureProvider(dictionary: d))

        let logits = out.featureValue(for: "logits")!.multiArrayValue!
        let vocab = logits.shape.last!.intValue
        let firstTok = Int32(argmax(logits, offset: (n - 1) * vocab, count: vocab))

        var co = [MLMultiArray](), ro = [MLMultiArray]()
        for j in 0 ..< nDelta {
            let full = out.featureValue(for: "conv_state_\(j)_out")!.multiArrayValue!
            let w = full.shape[2].intValue
            let sl = try! MLMultiArray(shape: [1, 6144, 4], dataType: full.dataType)
            if full.dataType == .float16 {
                let s = full.dataPointer.bindMemory(to: UInt16.self, capacity: full.count)
                let dd = sl.dataPointer.bindMemory(to: UInt16.self, capacity: sl.count)
                for ch in 0 ..< 6144 { for kk in 0 ..< 4 { dd[ch * 4 + kk] = s[ch * w + w - 4 + kk] } }
            } else {
                let s = full.dataPointer.bindMemory(to: Float.self, capacity: full.count)
                let dd = sl.dataPointer.bindMemory(to: Float.self, capacity: sl.count)
                for ch in 0 ..< 6144 { for kk in 0 ..< 4 { dd[ch * 4 + kk] = s[ch * w + w - 4 + kk] } }
            }
            co.append(sl); ro.append(out.featureValue(for: "rec_state_\(j)_out")!.multiArrayValue!)
        }
        var ko = [MLMultiArray](), vo = [MLMultiArray]()
        for j in 0 ..< nAttn {
            ko.append(out.featureValue(for: "key_cache_\(j)_out")!.multiArrayValue!)
            vo.append(out.featureValue(for: "value_cache_\(j)_out")!.multiArrayValue!)
        }
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
        let t0 = CFAbsoluteTimeGetCurrent()
        let textIds = tok.encode(text)
        let pStart = CFAbsoluteTimeGetCurrent()
        var (firstTok, cv, rc, kc, vc, pos) = batchPrefill(textIds)
        let pMs = (CFAbsoluteTimeGetCurrent() - pStart) * 1000

        if firstTok == thinkTokenId {
            _ = step(thinkEndTokenId, pos, &cv, &rc, &kc, &vc); pos += 1
            firstTok = step(198, pos, &cv, &rc, &kc, &vc); pos += 1
        }

        let gStart = CFAbsoluteTimeGetCurrent()
        var gen: [Int32] = [firstTok]; var cur = firstTok
        for _ in 0 ..< min(maxGenTokens, maxKV - pos - 1) {
            let t = step(cur, pos, &cv, &rc, &kc, &vc)
            if t == eosTokenId || t == imEndTokenId { break }
            gen.append(t); cur = t; pos += 1
        }
        let gMs = (CFAbsoluteTimeGetCurrent() - gStart) * 1000
        let output = tok.decode(gen)
        fputs("  [\(textIds.count)+\(suffix.count) prefill \(String(format: "%.0f", pMs))ms | \(gen.count) gen \(String(format: "%.0f", gMs))ms | \(String(format: "%.0f", (CFAbsoluteTimeGetCurrent() - t0) * 1000))ms total]\n", stderr)
        return output
    }
}

// MARK: - Main

let engine = try CleanupEngine(
    decodeDir: "models/monolith_kv_fp16",
    prefillDir: "models/batch_prefill_fp16",
    python: "/tmp/coreml-venv/bin/python",
    script: "scripts/coreml-convert/tokenizer_helper.py"
)
fputs("Ready. Enter transcriptions (Ctrl-D to quit):\n", stderr)
while let line = Swift.readLine() {
    let t = line.trimmingCharacters(in: .whitespacesAndNewlines)
    if t.isEmpty { continue }
    print(engine.cleanup(t))
}
