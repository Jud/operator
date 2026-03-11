import Foundation

/// HTTP client for communicating with the Operator daemon.
///
/// All requests are best-effort: errors are caught internally and logged
/// to stderr. Methods return `nil` on failure rather than throwing.
internal struct DaemonClient: Sendable {
    /// Base URL of the daemon (e.g., "http://localhost:7420").
    let baseURL: String
    /// Bearer token for daemon authentication.
    let token: String

    /// POST a JSON-encoded body to the daemon and return the raw response data.
    ///
    /// Returns `nil` if the request fails for any reason (network error,
    /// non-2xx status, encoding failure). Failures are logged to stderr.
    func post<T: Encodable & Sendable>(path: String, body: T) async -> Data? {
        guard let url = URL(string: "\(baseURL)\(path)") else {
            log("Invalid URL: \(baseURL)\(path)")
            return nil
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")

        do {
            request.httpBody = try JSONEncoder().encode(body)
        } catch {
            log("Failed to encode request body for \(path): \(error)")
            return nil
        }

        do {
            let (data, response) = try await URLSession.shared.data(for: request)

            if let httpResponse = response as? HTTPURLResponse,
                !(200...299).contains(httpResponse.statusCode)
            {
                log("Daemon error \(httpResponse.statusCode) for \(path)")
                return nil
            }

            return data
        } catch {
            log("Daemon unreachable for \(path): \(error)")
            return nil
        }
    }

    /// POST and decode the response as a specific Decodable type.
    ///
    /// Returns `nil` if the request fails or the response cannot be decoded.
    func post<T: Encodable & Sendable, R: Decodable>(path: String, body: T) async -> R? {
        guard let data: Data = await post(path: path, body: body) else {
            return nil
        }

        do {
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            return try decoder.decode(R.self, from: data)
        } catch {
            log("Failed to decode response for \(path): \(error)")
            return nil
        }
    }

    /// Write a diagnostic message to stderr.
    private func log(_ message: String) {
        let line = "[OperatorMCP] \(message)\n"
        if let data = line.data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
    }
}
