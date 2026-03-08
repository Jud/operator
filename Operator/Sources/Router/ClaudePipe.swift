import Foundation
import os

/// Errors that can occur when running claude -p subprocess.
public enum ClaudePipeError: Error, CustomStringConvertible {
    /// The subprocess exceeded its timeout and was terminated.
    case timeout(seconds: TimeInterval)

    /// The subprocess exited with a non-zero status code.
    case nonZeroExit(status: Int32, stderr: String)

    /// The subprocess output did not contain valid JSON.
    case invalidResponse(output: String)

    /// The `claude` CLI binary was not found on the system PATH.
    case cliNotFound

    /// A human-readable description of the error.
    public var description: String {
        switch self {
        case .timeout(let seconds):
            return "claude -p timed out after \(Int(seconds)) seconds"

        case .nonZeroExit(let status, let stderr):
            return "claude -p exited with status \(status): \(stderr)"

        case .invalidResponse(let output):
            return "claude -p returned invalid response: \(output.prefix(200))"

        case .cliNotFound:
            return "claude CLI not found"
        }
    }
}

/// A `RoutingEngine` backed by the real `ClaudePipe` subprocess.
public struct ClaudePipeRoutingEngine: RoutingEngine {
    /// Creates a new routing engine backed by the `claude -p` subprocess.
    public init() {}

    /// Run a prompt through the Claude CLI and return parsed JSON.
    public func run(prompt: String, timeout: TimeInterval) async throws -> [String: Any] {
        try await ClaudePipe.run(prompt: prompt, timeout: timeout)
    }
}

/// Subprocess wrapper for `claude -p` that handles timeout and JSON extraction.
///
/// Used for message routing classification and interruption handling.
/// The wrapper invokes `claude -p <prompt>` via /usr/bin/env, applies a configurable
/// timeout via DispatchSource timer, and extracts the first JSON object from the output
/// (since claude -p may emit preamble text before the JSON response).
public enum ClaudePipe {
    // MARK: - Subtypes

    /// Mutable state used during the single-pass JSON extraction scan.
    private struct JSONScanState {
        var braceDepth = 0
        var startIndex: String.Index?
        var inString = false
        var escaped = false

        /// Process a single character during the scan.
        ///
        /// Returns the extracted JSON string if a complete valid object is found.
        mutating func processCharacter(
            _ char: Character,
            at idx: String.Index,
            in text: String
        ) -> String? {
            if handleEscapeSequence(char) {
                return nil
            }
            if handleQuote(char) {
                return nil
            }
            if inString {
                return nil
            }
            return handleBrace(char, at: idx, in: text)
        }

        /// Handle escape tracking.
        ///
        /// Returns true if the character was consumed.
        private mutating func handleEscapeSequence(_ char: Character) -> Bool {
            if escaped {
                escaped = false
                return true
            }
            if char == "\\" && inString {
                escaped = true
                return true
            }
            return false
        }

        /// Handle quote toggling.
        ///
        /// Returns true if the character was a quote.
        private mutating func handleQuote(_ char: Character) -> Bool {
            guard char == "\"" else {
                return false
            }
            if braceDepth > 0 {
                inString.toggle()
            }
            return true
        }

        /// Handle open/close braces.
        ///
        /// Returns the extracted JSON if a valid object completes.
        private mutating func handleBrace(
            _ char: Character,
            at idx: String.Index,
            in text: String
        ) -> String? {
            if char == "{" {
                if braceDepth == 0 {
                    startIndex = idx
                    inString = false
                    escaped = false
                }
                braceDepth += 1
            } else if char == "}" {
                guard braceDepth > 0 else {
                    return nil
                }
                braceDepth -= 1
                if let result = tryExtractObject(endingAt: idx, in: text) {
                    return result
                }
            }
            return nil
        }

        /// If brace depth reaches zero, extract and validate the candidate JSON object.
        private mutating func tryExtractObject(
            endingAt idx: String.Index,
            in text: String
        ) -> String? {
            guard braceDepth == 0, let start = startIndex else {
                return nil
            }

            let endIdx = text.index(after: idx)
            let candidate = String(text[start..<endIdx])

            if ClaudePipe.isValidJSONObject(candidate) {
                return candidate
            }
            // Brace-matched but invalid JSON; keep searching
            startIndex = nil
            return nil
        }
    }

    // MARK: - Properties

    private static let logger = Log.logger(for: "ClaudePipe")

    /// Default timeout for claude -p invocations (10 seconds per spec).
    public static let defaultTimeout: TimeInterval = 10

    // MARK: - Public Methods

    /// Run a prompt through `claude -p` and extract the JSON response.
    ///
    /// - Parameters:
    ///   - prompt: The full prompt text to send to claude -p.
    ///   - timeout: Maximum time to wait for a response. Defaults to 10 seconds.
    /// - Returns: A dictionary parsed from the JSON object found in the output.
    /// - Throws: `ClaudePipeError` on timeout, non-zero exit, or invalid response.
    public static func run(prompt: String, timeout: TimeInterval = defaultTimeout) async throws -> [String: Any] {
        let jsonString = try await runRaw(prompt: prompt, timeout: timeout)
        guard let data = jsonString.data(using: .utf8),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            throw ClaudePipeError.invalidResponse(output: jsonString)
        }
        return json
    }

    /// Run a prompt through `claude -p` and return the raw extracted JSON string.
    ///
    /// - Parameters:
    ///   - prompt: The full prompt text to send to claude -p.
    ///   - timeout: Maximum time to wait for a response.
    /// - Returns: The first JSON object found in the output as a string.
    /// - Throws: `ClaudePipeError` on timeout, non-zero exit, or invalid response.
    public static func runRaw(prompt: String, timeout: TimeInterval = defaultTimeout) async throws -> String {
        logger.info("Running claude -p with \(Int(timeout))s timeout")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["claude", "-p", prompt]

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        let timedOut = OSAllocatedUnfairLock(initialState: false)

        let timer = DispatchSource.makeTimerSource(queue: .global())
        timer.schedule(deadline: .now() + timeout)
        timer.setEventHandler {
            timedOut.withLock { $0 = true }
            process.terminate()
        }
        timer.resume()

        defer {
            timer.cancel()
            if timedOut.withLock({ $0 }) {
                logger.warning("claude -p was terminated due to timeout")
            }
        }

        try launchProcess(process)

        return try collectOutput(
            process: process,
            stdoutPipe: stdoutPipe,
            stderrPipe: stderrPipe,
            timedOut: timedOut,
            timeout: timeout
        )
    }

    /// Extract the first JSON object from a string that may contain surrounding text.
    ///
    /// Uses brace-counting to find the candidate substring, then validates it with
    /// `JSONSerialization`. The brace counter properly tracks JSON string boundaries
    /// and escape sequences, and resets state at each top-level `{` so that stray
    /// quotes in preamble text don't corrupt parsing.
    ///
    /// Performance: O(n) single pass over the input using String.Index iteration
    /// (no `offsetBy` calls that would make it O(n²) on Swift strings).
    public static func extractJSON(from text: String) -> String? {
        var state = JSONScanState()

        for idx in text.indices {
            let char = text[idx]

            if let result = state.processCharacter(char, at: idx, in: text) {
                return result
            }
        }

        return nil
    }

    // MARK: - Internal Methods

    /// Validates whether a candidate substring is a JSON dictionary.
    static func isValidJSONObject(_ candidate: String) -> Bool {
        guard let data = candidate.data(using: .utf8),
            let obj = try? JSONSerialization.jsonObject(with: data),
            obj is [String: Any]
        else {
            return false
        }
        return true
    }

    // MARK: - Private Methods

    /// Launch a Process, converting launch errors to ClaudePipeError.
    private static func launchProcess(_ process: Process) throws {
        do {
            try process.run()
        } catch {
            let desc = error.localizedDescription.lowercased()
            if desc.contains("no such file") || desc.contains("not found") {
                throw ClaudePipeError.cliNotFound
            }
            throw ClaudePipeError.nonZeroExit(
                status: -1,
                stderr: "Failed to launch: \(error.localizedDescription)"
            )
        }
    }

    /// Collect output from a completed process and extract JSON.
    private static func collectOutput(
        process: Process,
        stdoutPipe: Pipe,
        stderrPipe: Pipe,
        timedOut: OSAllocatedUnfairLock<Bool>,
        timeout: TimeInterval
    ) throws -> String {
        let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
        let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()

        process.waitUntilExit()

        if timedOut.withLock({ $0 }) {
            throw ClaudePipeError.timeout(seconds: timeout)
        }

        guard process.terminationStatus == 0 else {
            let stderrText = String(data: stderrData, encoding: .utf8) ?? ""
            if process.terminationStatus == 127
                || stderrText.lowercased().contains("not found")
                || stderrText.lowercased().contains("no such file")
            {
                throw ClaudePipeError.cliNotFound
            }
            throw ClaudePipeError.nonZeroExit(status: process.terminationStatus, stderr: stderrText)
        }

        let output = String(data: stdoutData, encoding: .utf8) ?? ""
        logger.debug("claude -p output length: \(output.count) chars")

        guard let jsonString = extractJSON(from: output) else {
            throw ClaudePipeError.invalidResponse(output: output)
        }

        return jsonString
    }
}
