import Foundation

/// HTTP client for communicating with the Operator daemon.
///
/// All requests are best-effort: errors are caught internally and logged
/// to stderr. Methods return `nil` on failure rather than throwing.
public struct DaemonClient: Sendable {
    /// Base URL of the daemon (e.g., "http://localhost:7420").
    public let baseURL: String
    /// Bearer token for daemon authentication (from environment).
    public let token: String

    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    /// Creates a new daemon client.
    public init(baseURL: String, token: String) {
        self.baseURL = baseURL
        self.token = token
    }

    /// Read the current bearer token, preferring the token file over the cached value.
    private func currentToken() -> String {
        let tokenPath = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".operator/token").path
        if let data = FileManager.default.contents(atPath: tokenPath),
            let fileToken = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines),
            !fileToken.isEmpty
        {
            return fileToken
        }
        return token
    }

    /// POST a JSON-encoded body to the daemon and return the raw response data.
    ///
    /// Returns `nil` if the request fails for any reason (network error,
    /// non-2xx status, encoding failure). Failures are logged to stderr.
    func postRaw<T: Encodable & Sendable>(path: String, body: T) async -> Data? {
        guard let url = URL(string: "\(baseURL)\(path)") else {
            MCPLog.write("Invalid URL: \(baseURL)\(path)")
            return nil
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(currentToken())", forHTTPHeaderField: "Authorization")

        do {
            request.httpBody = try encoder.encode(body)
        } catch {
            MCPLog.write("Failed to encode request body for \(path): \(error)")
            return nil
        }

        do {
            let (data, response) = try await URLSession.shared.data(for: request)

            if let httpResponse = response as? HTTPURLResponse,
                !(200...299).contains(httpResponse.statusCode)
            {
                MCPLog.write("Daemon error \(httpResponse.statusCode) for \(path)")
                return nil
            }

            return data
        } catch {
            MCPLog.write("Daemon unreachable for \(path): \(error)")
            return nil
        }
    }

    /// POST and decode the response as a specific Decodable type.
    ///
    /// Returns `nil` if the request fails or the response cannot be decoded.
    func post<T: Encodable & Sendable, R: Decodable>(path: String, body: T) async -> R? {
        guard let data = await postRaw(path: path, body: body) else {
            return nil
        }

        do {
            return try decoder.decode(R.self, from: data)
        } catch {
            MCPLog.write("Failed to decode response for \(path): \(error)")
            return nil
        }
    }
}
