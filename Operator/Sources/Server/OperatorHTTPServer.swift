import Foundation
import Hummingbird
import OSLog

/// JSON body for POST /register.
///
/// Matches the MCP server's register() tool output: session name, TTY path,
/// and optional working directory and context summary.
public struct RegisterRequest: Decodable, Sendable {
    let name: String
    let tty: String
    let cwd: String?
    let context: String?
}

/// JSON body for POST /speak.
///
/// Matches the MCP server's speak() tool output: message text, optional
/// session name for voice lookup, and optional priority level.
public struct SpeakRequest: Decodable, Sendable {
    let message: String
    let session: String?
    let priority: String?
}

/// JSON body for POST /update.
///
/// Matches the MCP server's update_context() tool output: TTY to identify
/// the session, and optional summary and recent messages for routing context.
public struct UpdateRequest: Decodable, Sendable {
    enum CodingKeys: String, CodingKey {
        case tty
        case summary
        case recentMessages = "recent_messages"
    }

    let tty: String
    let summary: String?
    let recentMessages: [SessionMessage]?
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

/// JSON response for POST /register with session details.
public struct RegisterResponse: ResponseEncodable, Sendable {
    let ok: Bool
    let name: String
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
    public typealias Context = BasicRequestContext

    private static let logger = Logger(subsystem: "com.operator.app", category: "BearerAuth")

    private let expectedToken: String

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
/// Exposes four endpoints for the MCP server to communicate with the daemon:
/// - POST /register: Register a Claude Code session
/// - POST /speak: Queue a speech message
/// - POST /update: Update session context
/// - GET /state: Query daemon state
///
/// All mutations are dispatched to the SessionRegistry and AudioQueue actors
/// for thread-safe state management.
///
/// Reference: technical-spec.md Component 10; design.md Section 3.1.13
public final class OperatorHTTPServer: Sendable {
    private static let logger = Logger(subsystem: "com.operator.app", category: "HTTPServer")

    private let sessionRegistry: SessionRegistry
    private let audioQueue: AudioQueue
    private let port: Int

    /// Closure that returns the current daemon state string by querying the
    /// StateMachine on @MainActor. Called from the HTTP request handler
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

    /// Start the HTTP server. This method runs until the server is shut down.
    ///
    /// Binds to 127.0.0.1 only (localhost), not exposed to the network.
    /// Bearer token middleware is applied to all routes.
    public func start() async throws {
        let router = Router()

        let authMiddleware = try BearerAuthMiddleware()
        router.add(middleware: authMiddleware)

        let registry = self.sessionRegistry
        let queue = self.audioQueue

        configureRegisterRoute(router: router, registry: registry, queue: queue)
        configureSpeakRoute(router: router, registry: registry, queue: queue)
        configureUpdateRoute(router: router, registry: registry)
        configureStateRoute(router: router, registry: registry, queue: queue)

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
    /// Configure POST /register route.
    private func configureRegisterRoute(
        router: Router<BasicRequestContext>,
        registry: SessionRegistry,
        queue: AudioQueue
    ) {
        router.post("/register") { request, context -> RegisterResponse in
            let body = try await context.requestDecoder.decode(
                RegisterRequest.self, from: request, context: context
            )

            let name = body.name.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty else {
                throw HTTPError(.badRequest, message: "Session name cannot be empty")
            }

            let ok = await registry.register(
                name: name, tty: body.tty, cwd: body.cwd ?? "", context: body.context
            )
            guard ok else {
                throw HTTPError(.badRequest, message: "Name 'operator' is reserved")
            }
            let voice = await registry.voiceFor(session: name)
            let pitch = await registry.pitchFor(session: name)

            await queue.enqueue(
                AudioQueue.QueuedMessage(
                    sessionName: "Operator",
                    text: "\(name) connected.",
                    priority: .urgent,
                    voice: voice,
                    pitchMultiplier: pitch
                )
            )
            Self.logger.info("Registered session '\(name)' via HTTP")
            return RegisterResponse(ok: true, name: name, tty: body.tty)
        }
    }

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
            let voice = await registry.voiceFor(session: body.session)
            let pitch = await registry.pitchFor(session: body.session)

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

    /// Configure POST /update route.
    private func configureUpdateRoute(router: Router<BasicRequestContext>, registry: SessionRegistry) {
        router.post("/update") { request, context -> OkResponse in
            let body = try await context.requestDecoder.decode(
                UpdateRequest.self,
                from: request,
                context: context
            )

            await registry.updateContext(
                tty: body.tty,
                summary: body.summary ?? "",
                recentMessages: body.recentMessages ?? []
            )

            Self.logger.info("Updated context for TTY \(body.tty) via HTTP")
            return OkResponse(ok: true)
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
}
