import CoreML
import Foundation

/// CoreML BART encoder-decoder for grapheme-to-phoneme conversion of OOV words.
///
/// Uses separate encoder and decoder models with autoregressive greedy decoding.
final class G2PModel: @unchecked Sendable {
    private let encoder: MLModel
    private let decoder: MLModel
    private let graphemeToId: [String: Int]
    private let idToPhoneme: [Int: String]
    private let bosId: Int
    private let eosId: Int
    private let padId: Int

    private static let maxInputLen = 64
    private static let maxOutputLen = 64

    init(encoderURL: URL, decoderURL: URL, vocabURL: URL) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly  // G2P is small, CPU is faster than ANE dispatch
        self.encoder = try MLModel(contentsOf: encoderURL, configuration: config)
        self.decoder = try MLModel(contentsOf: decoderURL, configuration: config)

        // Load G2P vocabulary
        let data = try Data(contentsOf: vocabURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw KokoroError.modelLoadFailed("Invalid g2p_vocab.json format")
        }

        self.graphemeToId = (json["grapheme_to_id"] as? [String: Int]) ?? [:]
        let rawIdToPhoneme = (json["id_to_phoneme"] as? [String: String]) ?? [:]
        self.idToPhoneme = Dictionary(uniqueKeysWithValues: rawIdToPhoneme.compactMap { k, v in
            Int(k).map { ($0, v) }
        })
        self.bosId = (json["bos_token_id"] as? Int) ?? 1
        self.eosId = (json["eos_token_id"] as? Int) ?? 2
        self.padId = (json["pad_token_id"] as? Int) ?? 0
    }

    /// Load G2P model from a directory containing encoder, decoder, and vocab files.
    static func load(from directory: URL) throws -> G2PModel? {
        let encoderURL = directory.appendingPathComponent("G2PEncoder.mlmodelc")
        let decoderURL = directory.appendingPathComponent("G2PDecoder.mlmodelc")
        let vocabURL = directory.appendingPathComponent("g2p_vocab.json")
        let fm = FileManager.default

        guard fm.fileExists(atPath: encoderURL.path),
              fm.fileExists(atPath: decoderURL.path),
              fm.fileExists(atPath: vocabURL.path) else {
            return nil
        }

        return try G2PModel(encoderURL: encoderURL, decoderURL: decoderURL, vocabURL: vocabURL)
    }

    /// Convert a word to its phoneme representation using neural G2P.
    func phonemize(_ word: String) -> String? {
        guard !graphemeToId.isEmpty else { return nil }

        // Encode graphemes
        var inputIds: [Int32] = [Int32(bosId)]
        for char in word {
            let s = String(char)
            if let id = graphemeToId[s] {
                inputIds.append(Int32(id))
            } else if let id = graphemeToId[s.lowercased()] {
                inputIds.append(Int32(id))
            } else {
                inputIds.append(Int32(graphemeToId["<unk>"] ?? 3))
            }
        }
        inputIds.append(Int32(eosId))

        let seqLen = inputIds.count
        guard seqLen <= Self.maxInputLen else { return nil }

        do {
            // Run encoder
            let encInput = try MLMultiArray(shape: [1, seqLen as NSNumber], dataType: .int32)
            let encPtr = encInput.dataPointer.assumingMemoryBound(to: Int32.self)
            for i in 0..<seqLen { encPtr[i] = inputIds[i] }

            let encFeatures = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(multiArray: encInput),
            ])
            let encOutput = try encoder.prediction(from: encFeatures)
            guard let hiddenStates = encOutput.featureValue(
                for: "encoder_hidden_states")?.multiArrayValue else {
                return nil
            }

            // Autoregressive decoding
            var decoderIds: [Int32] = [Int32(bosId)]

            for step in 0..<Self.maxOutputLen {
                let decLen = decoderIds.count

                let decInput = try MLMultiArray(shape: [1, decLen as NSNumber], dataType: .int32)
                let decPtr = decInput.dataPointer.assumingMemoryBound(to: Int32.self)
                for i in 0..<decLen { decPtr[i] = decoderIds[i] }

                let posIds = try MLMultiArray(shape: [1, decLen as NSNumber], dataType: .int32)
                let posPtr = posIds.dataPointer.assumingMemoryBound(to: Int32.self)
                for i in 0..<decLen { posPtr[i] = Int32(i) }

                let mask = try MLMultiArray(
                    shape: [1, decLen as NSNumber, decLen as NSNumber], dataType: .float32)
                let maskPtr = mask.dataPointer.assumingMemoryBound(to: Float.self)
                for i in 0..<decLen {
                    for j in 0..<decLen {
                        maskPtr[i * decLen + j] = (j <= i) ? 0.0 : -Float.greatestFiniteMagnitude
                    }
                }

                let decFeatures = try MLDictionaryFeatureProvider(dictionary: [
                    "decoder_input_ids": MLFeatureValue(multiArray: decInput),
                    "encoder_hidden_states": MLFeatureValue(multiArray: hiddenStates),
                    "position_ids": MLFeatureValue(multiArray: posIds),
                    "causal_mask": MLFeatureValue(multiArray: mask),
                ])

                let decOutput = try decoder.prediction(from: decFeatures)
                guard let logits = decOutput.featureValue(for: "logits")?.multiArrayValue else {
                    break
                }

                // Greedy: argmax of last position
                let vocabSize = logits.shape.last!.intValue
                let lastOffset = step * vocabSize
                var maxId = 0
                var maxVal: Float = -.infinity

                if logits.dataType == .float16 {
                    let lPtr = logits.dataPointer.assumingMemoryBound(to: Float16.self)
                    for v in 0..<vocabSize {
                        let val = Float(lPtr[lastOffset + v])
                        if val > maxVal { maxVal = val; maxId = v }
                    }
                } else {
                    let lPtr = logits.dataPointer.assumingMemoryBound(to: Float.self)
                    for v in 0..<vocabSize {
                        let val = lPtr[lastOffset + v]
                        if val > maxVal { maxVal = val; maxId = v }
                    }
                }

                if maxId == eosId { break }
                decoderIds.append(Int32(maxId))
            }

            // Convert IDs to phonemes
            var result = ""
            for id in decoderIds.dropFirst() {
                let intId = Int(id)
                if intId != padId && intId != bosId && intId != eosId,
                   let phoneme = idToPhoneme[intId] {
                    result += phoneme
                }
            }
            return result.isEmpty ? nil : result
        } catch {
            return nil
        }
    }
}
