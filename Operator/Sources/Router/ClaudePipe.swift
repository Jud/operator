import Foundation
import OSLog

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

    public var description: String {
        switch self {
        case let .timeout(seconds):
            return "claude -p timed out after \(Int(seconds)) seconds"

        case let .nonZeroExit(status, stderr):
            return "claude -p exited with status \(status): \(stderr)"

        case let .invalidResponse(output):
            return "claude -p returned invalid response: \(output.prefix(200))"

        case .cliNotFound:
            return "claude CLI not found"
        }
    }
}

/// Subprocess wrapper for `claude -p` that handles timeout and JSON extraction.
///
/// Used for message routing classification and interruption handling.
/// The wrapper invokes `claude -p <prompt>` via /usr/bin/env, applies a configurable
/// timeout via DispatchSource timer, and extracts the first JSON object from the output
/// (since claude -p may emit preamble text before the JSON response).
public enum ClaudePipe {
    private static let logger = Logger(subsystem: "com.operator.app", category: "ClaudePipe")

    /// Default timeout for claude -p invocations (10 seconds per spec).
    public static let defaultTimeout: TimeInterval = 10

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

        let timedOut = UnsafeMutablePointer<Bool>.allocate(capacity: 1)
        timedOut.initialize(to: false)

        let timer = DispatchSource.makeTimerSource(queue: .global())
        timer.schedule(deadline: .now() + timeout)
        timer.setEventHandler {
            timedOut.pointee = true
            process.terminate()
        }
        timer.resume()

        defer {
            timer.cancel()
            let wasTimedOut = timedOut.pointee
            timedOut.deinitialize(count: 1)
            timedOut.deallocate()
            if wasTimedOut {
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
        timedOut: UnsafeMutablePointer<Bool>,
        timeout: TimeInterval
    ) throws -> String {
        let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
        let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()

        process.waitUntilExit()

        if timedOut.pointee {
            throw ClaudePipeError.timeout(seconds: timeout)
        }

        guard process.terminationStatus == 0 else {
            let stderrText = String(data: stderrData, encoding: .utf8) ?? ""
            if process.terminationStatus == 127
                || stderrText.lowercased().contains("not found")
                || stderrText.lowercased().contains("no such file") {
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

    /// Extract the first JSON object from a string that may contain surrounding text.
    ///
    /// Uses brace-counting to handle nested objects correctly, unlike the simple regex
    /// `\{[^}]+\}` which fails on nested braces. Falls back to the simple regex if
    /// brace counting finds nothing (defensive).
    public static func extractJSON(from text: String) -> String? {
        var braceDepth = 0
        var startIndex: String.Index?
        var inString = false
        var escaped = false

        for (offset, char) in text.enumerated() {
            let idx = text.index(text.startIndex, offsetBy: offset)

            if escaped {
                escaped = false
                continue
            }

            if char == "\\" && inString {
                escaped = true
                continue
            }

            if char == "\"" {
                inString.toggle()
                continue
            }

            if inString { continue }

            if char == "{" {
                if braceDepth == 0 {
                    startIndex = idx
                }
                braceDepth += 1
            } else if char == "}" {
                braceDepth -= 1
                if braceDepth == 0, let start = startIndex {
                    let endIdx = text.index(after: idx)
                    return String(text[start..<endIdx])
                }
            }
        }

        // Fallback: simple regex for flat JSON objects
        if let range = text.range(of: #"\{[^}]+\}"#, options: .regularExpression) {
            return String(text[range])
        }

        return nil
    }
}
