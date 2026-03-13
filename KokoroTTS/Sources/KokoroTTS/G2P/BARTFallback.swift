import Accelerate
import Foundation

/// BART-tiny neural G2P fallback for out-of-vocabulary words.
///
/// 752K-param BART (d_model=128, 1 encoder layer, 1 decoder layer, vocab 63).
/// Weights from `PeterReid/graphemes_to_phonemes_en_us` (3 MB safetensors).
/// Pure Accelerate implementation — zero external dependencies.
final class BARTFallback {
    private let d = 128          // d_model
    private let ffnDim = 1024
    private let vocabSize = 63
    private let maxPos = 64
    private let posOff = 2       // BART position offset
    private let bosId = 1
    private let eosId = 2
    private let scale: Float     // 1/sqrt(d_model)

    private let graphemeToId: [Character: Int]
    private let idToPhoneme: [Int: Character]

    // All weights stored as flat [Float] arrays.
    // Convention: "W" = weight matrix [outDim, inDim], "B" = bias [outDim].
    private let sharedW: [Float]       // [63, 128]
    private let logitBias: [Float]     // [63]

    // Encoder
    private let ePosW: [Float]         // [66, 128]
    private let eLnEW: [Float], eLnEB: [Float]
    private let eQW: [Float], eQB: [Float], eKW: [Float], eKB: [Float]
    private let eVW: [Float], eVB: [Float], eOW: [Float], eOB: [Float]
    private let eSaLnW: [Float], eSaLnB: [Float]
    private let eF1W: [Float], eF1B: [Float], eF2W: [Float], eF2B: [Float]
    private let eFLnW: [Float], eFLnB: [Float]

    // Decoder
    private let dPosW: [Float]         // [66, 128]
    private let dLnEW: [Float], dLnEB: [Float]
    private let dSQW: [Float], dSQB: [Float], dSKW: [Float], dSKB: [Float]
    private let dSVW: [Float], dSVB: [Float], dSOW: [Float], dSOB: [Float]
    private let dSaLnW: [Float], dSaLnB: [Float]
    private let dCQW: [Float], dCQB: [Float], dCKW: [Float], dCKB: [Float]
    private let dCVW: [Float], dCVB: [Float], dCOW: [Float], dCOB: [Float]
    private let dCaLnW: [Float], dCaLnB: [Float]
    private let dF1W: [Float], dF1B: [Float], dF2W: [Float], dF2B: [Float]
    private let dFLnW: [Float], dFLnB: [Float]

    init?(weightsURL: URL, configURL: URL) {
        guard let cfgData = try? Data(contentsOf: configURL),
              let cfg = try? JSONSerialization.jsonObject(with: cfgData) as? [String: Any],
              let gChars = cfg["grapheme_chars"] as? String,
              let pChars = cfg["phoneme_chars"] as? String
        else { return nil }

        var g2id = [Character: Int]()
        for (i, c) in gChars.enumerated() where i >= 4 { g2id[c] = i }
        self.graphemeToId = g2id

        var id2p = [Int: Character]()
        for (i, c) in pChars.enumerated() where i >= 4 { id2p[i] = c }
        self.idToPhoneme = id2p
        self.scale = 1.0 / sqrtf(128.0)

        guard let st = try? SafetensorsLoader(url: weightsURL) else { return nil }
        do {
            sharedW = try st.floats(for: "model.shared.weight")
            logitBias = try st.floats(for: "final_logits_bias")

            ePosW = try st.floats(for: "model.encoder.embed_positions.weight")
            eLnEW = try st.floats(for: "model.encoder.layernorm_embedding.weight")
            eLnEB = try st.floats(for: "model.encoder.layernorm_embedding.bias")
            eQW = try st.floats(for: "model.encoder.layers.0.self_attn.q_proj.weight")
            eQB = try st.floats(for: "model.encoder.layers.0.self_attn.q_proj.bias")
            eKW = try st.floats(for: "model.encoder.layers.0.self_attn.k_proj.weight")
            eKB = try st.floats(for: "model.encoder.layers.0.self_attn.k_proj.bias")
            eVW = try st.floats(for: "model.encoder.layers.0.self_attn.v_proj.weight")
            eVB = try st.floats(for: "model.encoder.layers.0.self_attn.v_proj.bias")
            eOW = try st.floats(for: "model.encoder.layers.0.self_attn.out_proj.weight")
            eOB = try st.floats(for: "model.encoder.layers.0.self_attn.out_proj.bias")
            eSaLnW = try st.floats(for: "model.encoder.layers.0.self_attn_layer_norm.weight")
            eSaLnB = try st.floats(for: "model.encoder.layers.0.self_attn_layer_norm.bias")
            eF1W = try st.floats(for: "model.encoder.layers.0.fc1.weight")
            eF1B = try st.floats(for: "model.encoder.layers.0.fc1.bias")
            eF2W = try st.floats(for: "model.encoder.layers.0.fc2.weight")
            eF2B = try st.floats(for: "model.encoder.layers.0.fc2.bias")
            eFLnW = try st.floats(for: "model.encoder.layers.0.final_layer_norm.weight")
            eFLnB = try st.floats(for: "model.encoder.layers.0.final_layer_norm.bias")

            dPosW = try st.floats(for: "model.decoder.embed_positions.weight")
            dLnEW = try st.floats(for: "model.decoder.layernorm_embedding.weight")
            dLnEB = try st.floats(for: "model.decoder.layernorm_embedding.bias")
            dSQW = try st.floats(for: "model.decoder.layers.0.self_attn.q_proj.weight")
            dSQB = try st.floats(for: "model.decoder.layers.0.self_attn.q_proj.bias")
            dSKW = try st.floats(for: "model.decoder.layers.0.self_attn.k_proj.weight")
            dSKB = try st.floats(for: "model.decoder.layers.0.self_attn.k_proj.bias")
            dSVW = try st.floats(for: "model.decoder.layers.0.self_attn.v_proj.weight")
            dSVB = try st.floats(for: "model.decoder.layers.0.self_attn.v_proj.bias")
            dSOW = try st.floats(for: "model.decoder.layers.0.self_attn.out_proj.weight")
            dSOB = try st.floats(for: "model.decoder.layers.0.self_attn.out_proj.bias")
            dSaLnW = try st.floats(for: "model.decoder.layers.0.self_attn_layer_norm.weight")
            dSaLnB = try st.floats(for: "model.decoder.layers.0.self_attn_layer_norm.bias")
            dCQW = try st.floats(for: "model.decoder.layers.0.encoder_attn.q_proj.weight")
            dCQB = try st.floats(for: "model.decoder.layers.0.encoder_attn.q_proj.bias")
            dCKW = try st.floats(for: "model.decoder.layers.0.encoder_attn.k_proj.weight")
            dCKB = try st.floats(for: "model.decoder.layers.0.encoder_attn.k_proj.bias")
            dCVW = try st.floats(for: "model.decoder.layers.0.encoder_attn.v_proj.weight")
            dCVB = try st.floats(for: "model.decoder.layers.0.encoder_attn.v_proj.bias")
            dCOW = try st.floats(for: "model.decoder.layers.0.encoder_attn.out_proj.weight")
            dCOB = try st.floats(for: "model.decoder.layers.0.encoder_attn.out_proj.bias")
            dCaLnW = try st.floats(for: "model.decoder.layers.0.encoder_attn_layer_norm.weight")
            dCaLnB = try st.floats(for: "model.decoder.layers.0.encoder_attn_layer_norm.bias")
            dF1W = try st.floats(for: "model.decoder.layers.0.fc1.weight")
            dF1B = try st.floats(for: "model.decoder.layers.0.fc1.bias")
            dF2W = try st.floats(for: "model.decoder.layers.0.fc2.weight")
            dF2B = try st.floats(for: "model.decoder.layers.0.fc2.bias")
            dFLnW = try st.floats(for: "model.decoder.layers.0.final_layer_norm.weight")
            dFLnB = try st.floats(for: "model.decoder.layers.0.final_layer_norm.bias")
        } catch {
            return nil
        }
    }

    /// Load from the module's bundled resources.
    static func fromBundle() -> BARTFallback? {
        guard let wURL = Bundle.module.url(forResource: "bart_g2p", withExtension: "safetensors"),
              let cURL = Bundle.module.url(forResource: "bart_g2p_config", withExtension: "json")
        else { return nil }
        return BARTFallback(weightsURL: wURL, configURL: cURL)
    }

    /// Predict IPA phonemes for a lowercased word.
    func predict(_ word: String) -> String? {
        let gIds = word.compactMap { graphemeToId[$0] }
        guard !gIds.isEmpty, gIds.count + 2 <= maxPos else { return nil }

        let encIn = [bosId] + gIds + [eosId]
        let encOut = encode(encIn)
        let pIds = decode(encOut: encOut, encLen: encIn.count)
        guard !pIds.isEmpty else { return nil }

        let chars = pIds.compactMap { idToPhoneme[$0] }
        return chars.isEmpty ? nil : String(chars)
    }

    // MARK: - Encoder

    private func encode(_ ids: [Int]) -> [Float] {
        let n = ids.count

        // Token + positional embeddings
        var x = [Float](repeating: 0, count: n * d)
        for i in 0..<n {
            let tOff = ids[i] * d
            let pOff = (i + posOff) * d
            for j in 0..<d { x[i * d + j] = sharedW[tOff + j] + ePosW[pOff + j] }
        }
        layerNorm(&x, rows: n, w: eLnEW, b: eLnEB)

        // Self-attention
        var a = fullAttn(x, ctx: x, n: n, cLen: n, mask: nil,
                         qW: eQW, qB: eQB, kW: eKW, kB: eKB,
                         vW: eVW, vB: eVB, oW: eOW, oB: eOB)
        addInPlace(&a, x)
        layerNorm(&a, rows: n, w: eSaLnW, b: eSaLnB)

        // FFN
        var f = ffn(a, rows: n, w1: eF1W, b1: eF1B, w2: eF2W, b2: eF2B)
        addInPlace(&f, a)
        layerNorm(&f, rows: n, w: eFLnW, b: eFLnB)

        return f
    }

    // MARK: - Decoder (greedy)

    private func decode(encOut: [Float], encLen: Int) -> [Int] {
        // Pre-compute cross-attention K, V from encoder output.
        let cK = linear(encOut, rows: encLen, wt: dCKW, b: dCKB, outD: d)
        let cV = linear(encOut, rows: encLen, wt: dCVW, b: dCVB, outD: d)

        var tokens = [bosId]
        let maxSteps = min(50, maxPos - 2)

        for _ in 0..<maxSteps {
            let n = tokens.count

            // Token + positional embeddings
            var x = [Float](repeating: 0, count: n * d)
            for i in 0..<n {
                let tOff = tokens[i] * d
                let pOff = (i + posOff) * d
                for j in 0..<d { x[i * d + j] = sharedW[tOff + j] + dPosW[pOff + j] }
            }
            layerNorm(&x, rows: n, w: dLnEW, b: dLnEB)

            // Causal self-attention
            let mask = causalMask(n)
            var sa = fullAttn(x, ctx: x, n: n, cLen: n, mask: mask,
                              qW: dSQW, qB: dSQB, kW: dSKW, kB: dSKB,
                              vW: dSVW, vB: dSVB, oW: dSOW, oB: dSOB)
            addInPlace(&sa, x)
            layerNorm(&sa, rows: n, w: dSaLnW, b: dSaLnB)

            // Cross-attention (Q from decoder state, K/V from encoder)
            let cQ = linear(sa, rows: n, wt: dCQW, b: dCQB, outD: d)
            var ca = sdpa(q: cQ, k: cK, v: cV, qLen: n, kvLen: encLen, mask: nil)
            ca = linear(ca, rows: n, wt: dCOW, b: dCOB, outD: d)
            addInPlace(&ca, sa)
            layerNorm(&ca, rows: n, w: dCaLnW, b: dCaLnB)

            // FFN
            var f = ffn(ca, rows: n, w1: dF1W, b1: dF1B, w2: dF2W, b2: dF2B)
            addInPlace(&f, ca)
            layerNorm(&f, rows: n, w: dFLnW, b: dFLnB)

            // LM head on last position: logits = h @ shared^T + bias
            let lastOff = (n - 1) * d
            var logits = [Float](repeating: 0, count: vocabSize)
            for v in 0..<vocabSize {
                var dot: Float = 0
                for j in 0..<d { dot += f[lastOff + j] * sharedW[v * d + j] }
                logits[v] = dot + logitBias[v]
            }

            // Greedy argmax
            var best = 0
            var bestVal = logits[0]
            for i in 1..<vocabSize {
                if logits[i] > bestVal { bestVal = logits[i]; best = i }
            }
            if best == eosId { break }
            tokens.append(best)
        }

        // Drop seed BOS and any generated special tokens (BOS/EOS/PAD).
        return tokens.dropFirst().filter { $0 >= 4 }
    }

    // MARK: - Linear Algebra

    /// output = input @ W^T + bias.  W is [outD, inD] row-major.
    private func linear(_ input: [Float], rows: Int, wt: [Float], b: [Float],
                        outD: Int, inD: Int? = nil) -> [Float] {
        let k = inD ?? d
        var out = [Float](repeating: 0, count: rows * outD)
        input.withUnsafeBufferPointer { inBuf in
            wt.withUnsafeBufferPointer { wBuf in
                out.withUnsafeMutableBufferPointer { oBuf in
                    cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans,
                        Int32(rows), Int32(outD), Int32(k),
                        1.0, inBuf.baseAddress!, Int32(k),
                        wBuf.baseAddress!, Int32(k),
                        0.0, oBuf.baseAddress!, Int32(outD))
                }
            }
        }
        for r in 0..<rows {
            for j in 0..<outD { out[r * outD + j] += b[j] }
        }
        return out
    }

    /// Full attention: project Q/K/V, compute SDPA, project output.
    private func fullAttn(
        _ x: [Float], ctx: [Float], n: Int, cLen: Int, mask: [Float]?,
        qW: [Float], qB: [Float], kW: [Float], kB: [Float],
        vW: [Float], vB: [Float], oW: [Float], oB: [Float]
    ) -> [Float] {
        let q = linear(x, rows: n, wt: qW, b: qB, outD: d)
        let k = linear(ctx, rows: cLen, wt: kW, b: kB, outD: d)
        let v = linear(ctx, rows: cLen, wt: vW, b: vB, outD: d)
        let a = sdpa(q: q, k: k, v: v, qLen: n, kvLen: cLen, mask: mask)
        return linear(a, rows: n, wt: oW, b: oB, outD: d)
    }

    /// Scaled dot-product attention.
    private func sdpa(q: [Float], k: [Float], v: [Float],
                      qLen: Int, kvLen: Int, mask: [Float]?) -> [Float] {
        // scores = Q @ K^T * scale  → [qLen, kvLen]
        var scores = [Float](repeating: 0, count: qLen * kvLen)
        q.withUnsafeBufferPointer { qBuf in
            k.withUnsafeBufferPointer { kBuf in
                scores.withUnsafeMutableBufferPointer { sBuf in
                    cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans,
                        Int32(qLen), Int32(kvLen), Int32(d),
                        scale, qBuf.baseAddress!, Int32(d),
                        kBuf.baseAddress!, Int32(d),
                        0.0, sBuf.baseAddress!, Int32(kvLen))
                }
            }
        }

        if let mask {
            for i in 0..<scores.count { scores[i] += mask[i] }
        }

        // Softmax per row
        for r in 0..<qLen {
            let off = r * kvLen
            var maxV: Float = scores[off]
            for j in 1..<kvLen { if scores[off + j] > maxV { maxV = scores[off + j] } }
            var sum: Float = 0
            for j in 0..<kvLen {
                scores[off + j] = expf(scores[off + j] - maxV)
                sum += scores[off + j]
            }
            if sum > 0 { for j in 0..<kvLen { scores[off + j] /= sum } }
        }

        // output = weights @ V  → [qLen, d]
        var out = [Float](repeating: 0, count: qLen * d)
        scores.withUnsafeBufferPointer { sBuf in
            v.withUnsafeBufferPointer { vBuf in
                out.withUnsafeMutableBufferPointer { oBuf in
                    cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        Int32(qLen), Int32(d), Int32(kvLen),
                        1.0, sBuf.baseAddress!, Int32(kvLen),
                        vBuf.baseAddress!, Int32(d),
                        0.0, oBuf.baseAddress!, Int32(d))
                }
            }
        }
        return out
    }

    /// FFN: gelu(x @ W1^T + b1) @ W2^T + b2
    private func ffn(_ input: [Float], rows: Int,
                     w1: [Float], b1: [Float], w2: [Float], b2: [Float]) -> [Float] {
        var h = linear(input, rows: rows, wt: w1, b: b1, outD: ffnDim)
        let invSqrt2: Float = 1.0 / sqrtf(2.0)
        for i in 0..<h.count { h[i] = h[i] * 0.5 * (1.0 + erff(h[i] * invSqrt2)) }
        return linear(h, rows: rows, wt: w2, b: b2, outD: d, inD: ffnDim)
    }

    /// Layer normalization over last dimension (d_model).
    private func layerNorm(_ x: inout [Float], rows: Int, w: [Float], b: [Float]) {
        let eps: Float = 1e-5
        for r in 0..<rows {
            let off = r * d
            var mean: Float = 0
            for j in 0..<d { mean += x[off + j] }
            mean /= Float(d)
            var variance: Float = 0
            for j in 0..<d {
                let diff = x[off + j] - mean
                variance += diff * diff
            }
            variance /= Float(d)
            let inv = 1.0 / sqrtf(variance + eps)
            for j in 0..<d { x[off + j] = (x[off + j] - mean) * inv * w[j] + b[j] }
        }
    }

    /// Element-wise a += b.
    private func addInPlace(_ a: inout [Float], _ b: [Float]) {
        for i in 0..<a.count { a[i] += b[i] }
    }

    /// Lower-triangular causal mask: 0 for attend, -1e9 for block.
    private func causalMask(_ n: Int) -> [Float] {
        var m = [Float](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in (i + 1)..<n { m[i * n + j] = -1e9 }
        }
        return m
    }
}
