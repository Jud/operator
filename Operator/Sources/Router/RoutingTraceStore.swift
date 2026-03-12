import Foundation

/// Persists routing traces as JSONL (one JSON object per line) for offline analysis.
///
/// Traces are appended to a rotating file in ~/Library/Application Support/Operator/routing-traces/.
/// When the file exceeds `maxTraces`, older entries are pruned on the next append.
/// Thread-safe via actor isolation.
public actor RoutingTraceStore {
    private static let logger = Log.logger(for: "RoutingTraceStore")

    /// Maximum number of traces retained in the file.
    private let maxTraces: Int

    /// Directory containing trace files.
    private let directory: URL

    /// The active trace file path.
    private let filePath: URL

    /// JSON encoder configured for trace serialization.
    private let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }()

    /// JSON decoder configured for trace deserialization.
    private let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()

    /// Creates a trace store with the default Application Support location.
    ///
    /// - Parameter maxTraces: Maximum traces to retain. Defaults to 500.
    public init(maxTraces: Int = 500) {
        let appSupport =
            FileManager.default.urls(
                for: .applicationSupportDirectory,
                in: .userDomainMask
            ).first
            ?? FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support")
        let dir = appSupport.appendingPathComponent("Operator/routing-traces", isDirectory: true)
        self.directory = dir
        self.filePath = dir.appendingPathComponent("traces.jsonl")
        self.maxTraces = maxTraces
    }

    /// Creates a trace store at a custom directory (for testing).
    public init(directory: URL, maxTraces: Int = 500) {
        self.directory = directory
        self.filePath = directory.appendingPathComponent("traces.jsonl")
        self.maxTraces = maxTraces
    }

    /// Append a trace to the store.
    public func append(_ trace: RoutingTrace) {
        do {
            try ensureDirectory()
            var data = try encoder.encode(trace)
            data.append(0x0A)  // newline byte

            if FileManager.default.fileExists(atPath: filePath.path) {
                let handle = try FileHandle(forWritingTo: filePath)
                defer { try? handle.close() }
                handle.seekToEndOfFile()
                handle.write(data)
            } else {
                try data.write(to: filePath, options: .atomic)
            }

            Self.logger.debug("Appended routing trace \(trace.id)")
        } catch {
            Self.logger.error("Failed to append routing trace: \(error)")
        }
    }

    /// Update the annotation on an existing trace by ID.
    public func annotate(traceId: UUID, annotation: RoutingTrace.Annotation) {
        do {
            var traces = try loadAll()
            guard let index = traces.firstIndex(where: { $0.id == traceId }) else {
                Self.logger.warning("Trace \(traceId) not found for annotation")
                return
            }
            traces[index].annotation = annotation
            try writeAll(traces)
            Self.logger.info("Annotated trace \(traceId): correct=\(annotation.correct)")
        } catch {
            Self.logger.error("Failed to annotate trace: \(error)")
        }
    }

    /// Read all non-empty lines from the trace file, or an empty array if the file
    /// does not exist.
    private func readLines() throws -> [String] {
        guard FileManager.default.fileExists(atPath: filePath.path) else {
            return []
        }
        let content = try String(contentsOf: filePath, encoding: .utf8)
        return content.components(separatedBy: "\n").filter { !$0.isEmpty }
    }

    /// Load all traces from the store.
    public func loadAll() throws -> [RoutingTrace] {
        try readLines().compactMap { line in
            guard let data = line.data(using: .utf8) else {
                return nil
            }
            return try? decoder.decode(RoutingTrace.self, from: data)
        }
    }

    /// Get the most recent trace (for HUD display).
    ///
    /// Reads the file backwards to find the last non-empty line,
    /// avoiding a full parse of all traces.
    public func lastTrace() throws -> RoutingTrace? {
        guard FileManager.default.fileExists(atPath: filePath.path) else {
            return nil
        }
        let content = try String(contentsOf: filePath, encoding: .utf8)
        // Find the last non-empty line without splitting the entire string.
        guard let lastNewline = content.dropLast().lastIndex(of: "\n") else {
            // Single line (no preceding newline); use the whole content trimmed.
            let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, let data = trimmed.data(using: .utf8) else {
                return nil
            }
            return try decoder.decode(RoutingTrace.self, from: data)
        }
        let lastLine = String(content[content.index(after: lastNewline)...])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !lastLine.isEmpty, let data = lastLine.data(using: .utf8) else {
            return nil
        }
        return try decoder.decode(RoutingTrace.self, from: data)
    }

    /// Prune old traces if the file exceeds maxTraces.
    public func pruneIfNeeded() {
        do {
            var traces = try loadAll()
            guard traces.count > maxTraces else {
                return
            }
            let excess = traces.count - maxTraces
            traces.removeFirst(excess)
            try writeAll(traces)
            Self.logger.info("Pruned \(excess) old traces, \(traces.count) remaining")
        } catch {
            Self.logger.error("Failed to prune traces: \(error)")
        }
    }

    // MARK: - Private

    private func ensureDirectory() throws {
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
    }

    private func writeAll(_ traces: [RoutingTrace]) throws {
        try ensureDirectory()
        var output = Data()
        for trace in traces {
            var line = try encoder.encode(trace)
            line.append(0x0A)
            output.append(line)
        }
        try output.write(to: filePath, options: .atomic)
    }
}
