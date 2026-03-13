import Foundation

/// Minimal safetensors parser for loading float32 tensors.
///
/// Format: 8-byte LE header size + JSON header + raw tensor data.
/// Supports only F32 dtype (sufficient for the BART G2P model).
struct SafetensorsLoader {
    struct TensorInfo {
        let shape: [Int]
        let dataOffset: Int
        let dataLength: Int
    }

    private let tensors: [String: TensorInfo]
    private let data: Data
    private let dataStart: Int

    init(url: URL) throws {
        let data = try Data(contentsOf: url)
        guard data.count >= 8 else {
            throw KokoroError.modelLoadFailed("Safetensors file too small")
        }

        let headerSize: UInt64 = data.withUnsafeBytes { $0.load(as: UInt64.self) }
        let dataStart = 8 + Int(headerSize)
        guard dataStart <= data.count else {
            throw KokoroError.modelLoadFailed("Safetensors header exceeds file size")
        }

        let headerData = data[8..<dataStart]
        guard let json = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw KokoroError.modelLoadFailed("Invalid safetensors header JSON")
        }

        var tensors = [String: TensorInfo]()
        for (name, value) in json {
            if name == "__metadata__" { continue }
            guard let info = value as? [String: Any],
                  let shape = info["shape"] as? [Int],
                  let dtype = info["dtype"] as? String, dtype == "F32",
                  let offsets = info["data_offsets"] as? [Int], offsets.count == 2
            else { continue }
            tensors[name] = TensorInfo(
                shape: shape,
                dataOffset: offsets[0],
                dataLength: offsets[1] - offsets[0])
        }

        self.tensors = tensors
        self.data = data
        self.dataStart = dataStart
    }

    func floats(for name: String) throws -> [Float] {
        guard let info = tensors[name] else {
            throw KokoroError.modelLoadFailed("Missing tensor: \(name)")
        }
        let count = info.dataLength / MemoryLayout<Float>.size
        let start = dataStart + info.dataOffset
        return data.withUnsafeBytes { ptr in
            let base = ptr.baseAddress!.advanced(by: start)
                .assumingMemoryBound(to: Float.self)
            return Array(UnsafeBufferPointer(start: base, count: count))
        }
    }
}
