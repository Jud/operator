import Foundation
import Hummingbird

/// JSON body for POST /speak.
///
/// Matches the MCP server's speak() tool output: message text, optional
/// session name for voice lookup, and optional priority level.
public struct SpeakRequest: Decodable, Sendable {
    let message: String
    let session: String?
    let priority: String?
}

/// JSON response for successful operations.
public struct OkResponse: ResponseEncodable, Sendable {
    let ok: Bool
}

/// JSON response for POST /speak confirming message was queued.
public struct QueuedResponse: ResponseEncodable, Sendable {
    let queued: Bool
}

/// JSON response for GET /state.
///
/// Returns daemon state, audio queue depth, and all registered sessions.
public struct StateResponse: ResponseEncodable, Sendable {
    let state: String
    let queueLength: Int
    let sessions: [SessionSnapshot]
}

/// JSON body for POST /hook/session-start.
///
/// Accepts the Claude Code hook payload for session start events.
public struct HookSessionStartRequest: Decodable, Sendable {
    enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
        case tty
        case cwd
    }

    let sessionId: String
    let tty: String
    let cwd: String
}

/// JSON body for POST /hook/stop.
///
/// Accepts the Claude Code hook payload for stop events, including
/// the last assistant message for context tracking.
public struct HookStopRequest: Decodable, Sendable {
    enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
        case tty
        case lastAssistantMessage = "last_assistant_message"
    }

    let sessionId: String
    let tty: String
    let lastAssistantMessage: String?
}

/// JSON body for POST /hook/session-end.
///
/// Accepts the Claude Code hook payload for session end events.
public struct HookSessionEndRequest: Decodable, Sendable {
    enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
        case tty
    }

    let sessionId: String
    let tty: String
}

/// Middleware that validates bearer token authentication on all requests.
///
/// Reads the expected token from ~/.operator/token (created at first launch
/// with 0600 permissions). Rejects requests with missing or invalid tokens
/// with HTTP 401 Unauthorized.
///
/// Reference: technical-spec.md Component 10, "Bearer token middleware"
public struct BearerAuthMiddleware: RouterMiddleware {
    /// The request context type used by this middleware.
    public typealias Context = BasicRequestContext

    private static let logger = Log.logger(for: "BearerAuth")

    private let expectedToken: String

    /// Creates a new bearer auth middleware by reading the token from disk.
    public init() throws {
        let tokenPath = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".operator")
            .appendingPathComponent("token")
            .path

        guard let tokenData = FileManager.default.contents(atPath: tokenPath),
            let token = String(data: tokenData, encoding: .utf8)
        else {
            Self.logger.error("Failed to read bearer token from \(tokenPath)")
            throw OperatorHTTPServerError.tokenFileUnreadable
        }

        self.expectedToken = token.trimmingCharacters(in: .whitespacesAndNewlines)
        Self.logger.info("Bearer auth middleware initialized")
    }

    /// Validates the bearer token on each incoming request.
    public func handle(
        _ request: Request,
        context: BasicRequestContext,
        next: (Request, BasicRequestContext) async throws -> Response
    ) async throws -> Response {
        guard let authHeader = request.headers[.authorization] else {
            Self.logger.warning("Request missing Authorization header")
            throw HTTPError(.unauthorized, message: "Missing Authorization header")
        }

        let prefix = "Bearer "
        guard authHeader.hasPrefix(prefix) else {
            Self.logger.warning("Authorization header not Bearer scheme")
            throw HTTPError(.unauthorized, message: "Invalid Authorization scheme")
        }

        let token = String(authHeader.dropFirst(prefix.count))
            .trimmingCharacters(in: .whitespacesAndNewlines)

        guard token == expectedToken else {
            Self.logger.warning("Invalid bearer token")
            throw HTTPError(.unauthorized, message: "Invalid token")
        }

        return try await next(request, context)
    }
}

/// Errors specific to the HTTP server setup.
public enum OperatorHTTPServerError: Error, CustomStringConvertible {
    case tokenFileUnreadable

    case portInUse

    /// A human-readable description of the error.
    public var description: String {
        switch self {
        case .tokenFileUnreadable:
            return "Could not read bearer token from ~/.operator/token"

        case .portInUse:
            return "Port 7420 is already in use. Another Operator instance may be running."
        }
    }
}

/// Hummingbird HTTP server on localhost:7420 with bearer token authentication.
///
/// Exposes endpoints for the MCP server and Claude Code hooks to communicate
/// with the daemon:
/// - POST /speak: Queue a speech message
/// - GET /state: Query daemon state
/// - POST /hook/session-start: Handle Claude Code session start hook
/// - POST /hook/stop: Handle Claude Code stop hook
/// - POST /hook/session-end: Handle Claude Code session end hook
///
/// All mutations are dispatched to the SessionRegistry and AudioQueue actors
/// for thread-safe state management.
///
/// Reference: technical-spec.md Component 10; design.md Section 3.1.13
public final class OperatorHTTPServer: Sendable {
    private static let logger = Log.logger(for: "HTTPServer")

    private let sessionRegistry: SessionRegistry
    private let audioQueue: AudioQueue
    private let port: Int

    /// Closure that returns the current daemon state string.
    ///
    /// Queries the StateMachine on @MainActor. Called from the HTTP request handler
    /// for GET /state.
    private let stateProvider: @Sendable () async -> String

    /// - Parameters:
    ///   - sessionRegistry: Actor managing session state.
    ///   - audioQueue: Actor managing speech playback queue.
    ///   - port: TCP port to bind (default 7420).
    ///   - stateProvider: Async closure returning the current StateMachine state string.
    public init(
        sessionRegistry: SessionRegistry,
        audioQueue: AudioQueue,
        port: Int = 7_420,
        stateProvider: @escaping @Sendable () async -> String = { "running" }
    ) {
        self.port = port
        self.sessionRegistry = sessionRegistry
        self.audioQueue = audioQueue
        self.stateProvider = stateProvider
        Self.logger.info("OperatorHTTPServer initialized on port \(port)")
    }

    /// Start the HTTP server.
    ///
    /// This method runs until the server is shut down.
    /// Binds to 127.0.0.1 only (localhost), not exposed to the network.
    /// Bearer token middleware is applied to all routes.
    public func start() async throws {
        let router = Router()

        let authMiddleware = try BearerAuthMiddleware()
        router.add(middleware: authMiddleware)

        let registry = self.sessionRegistry
        let queue = self.audioQueue

        configureSpeakRoute(router: router, registry: registry, queue: queue)
        configureStateRoute(router: router, registry: registry, queue: queue)
        configureHookRoutes(router: router, registry: registry)

        let app = Application(
            router: router,
            configuration: .init(
                address: .hostname("127.0.0.1", port: self.port)
            )
        )

        Self.logger.info("Starting HTTP server on 127.0.0.1:\(self.port)")
        try await app.runService()
    }
}

// MARK: - Route Configuration

extension OperatorHTTPServer {
    /// Configure POST /speak route.
    private func configureSpeakRoute(
        router: Router<BasicRequestContext>,
        registry: SessionRegistry,
        queue: AudioQueue
    ) {
        router.post("/speak") { request, context -> QueuedResponse in
            let body = try await context.requestDecoder.decode(
                SpeakRequest.self,
                from: request,
                context: context
            )

            guard !body.message.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw HTTPError(.badRequest, message: "Message cannot be empty")
            }

            let sessionName = body.session ?? "Unknown"
            let priority: AudioQueue.QueuedMessage.Priority =
                body.priority == "urgent" ? .urgent : .normal
            async let voice = registry.voiceFor(session: body.session)
            async let pitch = registry.pitchFor(session: body.session)

            await queue.enqueue(
                AudioQueue.QueuedMessage(
                    sessionName: sessionName,
                    text: body.message,
                    priority: priority,
                    voice: voice,
                    pitchMultiplier: pitch
                )
            )

            Self.logger.info(
                "Queued speech from '\(sessionName)' via HTTP (priority: \(String(describing: priority)))"
            )
            return QueuedResponse(queued: true)
        }
    }

    /// Configure GET /state route.
    private func configureStateRoute(
        router: Router<BasicRequestContext>,
        registry: SessionRegistry,
        queue: AudioQueue
    ) {
        let getState = self.stateProvider
        router.get("/state") { _, _ -> StateResponse in
            let currentState = await getState()
            let registryState = await registry.getState()
            let queueLength = await queue.pendingCount

            return StateResponse(
                state: currentState,
                queueLength: queueLength,
                sessions: registryState.sessions
            )
        }
    }

    /// Configure hook routes for Claude Code integration.
    ///
    /// These endpoints receive raw hook JSON from Claude Code and route
    /// the payloads to the session registry for lifecycle management.
    private func configureHookRoutes(router: Router<BasicRequestContext>, registry: SessionRegistry) {
        router.post("/hook/session-start") { request, context -> OkResponse in
            let body = try await context.requestDecoder.decode(
                HookSessionStartRequest.self,
                from: request,
                context: context
            )
            await registry.handleSessionStart(sessionId: body.sessionId, tty: body.tty, cwd: body.cwd)
            Self.logger.info("Hook session-start for session \(body.sessionId)")
            return OkResponse(ok: true)
        }

        router.post("/hook/stop") { request, context -> OkResponse in
            let body = try await context.requestDecoder.decode(
                HookStopRequest.self,
                from: request,
                context: context
            )
            await registry.handleStop(
                sessionId: body.sessionId,
                tty: body.tty,
                lastAssistantMessage: body.lastAssistantMessage
            )
            Self.logger.info("Hook stop for session \(body.sessionId)")
            return OkResponse(ok: true)
        }

        router.post("/hook/session-end") { request, context -> OkResponse in
            let body = try await context.requestDecoder.decode(
                HookSessionEndRequest.self,
                from: request,
                context: context
            )
            await registry.handleSessionEnd(sessionId: body.sessionId, tty: body.tty)
            Self.logger.info("Hook session-end for session \(body.sessionId)")
            return OkResponse(ok: true)
        }
    }
}
