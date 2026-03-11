import Foundation
import OperatorMCPCore

@main
internal enum OperatorMCPApp {
    static func main() async {
        let config = loadConfig()
        let tty = detectTTY()
        let terminalType = detectTerminalType()
        let sessionName = deriveSessionName()
        let cwd = FileManager.default.currentDirectoryPath

        let client = DaemonClient(baseURL: config.baseURL, token: config.token)

        let ghosttyResolver = GhosttyResolver(client: client, tty: tty)

        let server = MCPServer(client: client, sessionName: sessionName)

        let heartbeat = Heartbeat(
            client: client,
            ghosttyResolver: ghosttyResolver,
            sessionName: sessionName,
            tty: tty,
            cwd: cwd,
            terminalType: terminalType
        )

        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                await heartbeat.heartbeatLoop()
            }
            group.addTask {
                await server.stdinLoop()
            }
        }
    }
}
