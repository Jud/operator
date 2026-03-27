import Foundation
import OperatorShared

/// Operator CLI — voice output for Claude Code sessions.
///
/// Usage:
///   operator speak "message"                     # Speak with normal priority
///   operator speak --priority urgent "message"    # Urgent — jumps the queue
///   operator speak --priority low "message"       # Low — droppable under saturation
///   operator speak --session myproject "message"  # Override session name

private let daemonURL = "http://127.0.0.1:7420"
private let validPriorities = ["normal", "urgent", "low"]

@main
internal enum OperatorCLI {
    static func main() async {
        let args = CommandLine.arguments

        guard args.count >= 2 else {
            printUsage()
            exit(1)
        }

        switch args[1] {
        case "speak":
            await handleSpeak(args: Array(args.dropFirst(2)))

        case "--help", "-h":
            printUsage()

        default:
            StderrLog.write("Unknown command: \(args[1])", tag: "OperatorCLI")
            printUsage()
            exit(1)
        }
    }

    private static func handleSpeak(args: [String]) async {
        let parsed = parseArgs(args)

        guard let text = parsed.message, !text.isEmpty else {
            StderrLog.write("No message provided", tag: "OperatorCLI")
            exit(1)
        }

        guard let token = DaemonClient.readToken() else {
            StderrLog.write(
                "Cannot read token from ~/.operator/token",
                tag: "OperatorCLI"
            )
            exit(1)
        }

        let client = DaemonClient(baseURL: daemonURL, token: token)
        let request = SpeakRequest(
            message: text,
            session: parsed.session,
            priority: parsed.priority
        )

        guard await client.postRaw(path: "/speak", body: request) != nil else {
            StderrLog.write("Operator daemon unreachable", tag: "OperatorCLI")
            exit(1)
        }

        print("{\"queued\":true}")
    }
}

// MARK: - Argument Parsing

private struct ParsedArgs {
    var priority = "normal"
    var session = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .lastPathComponent
    var message: String?
}

private func parseArgs(_ args: [String]) -> ParsedArgs {
    var result = ParsedArgs()
    var idx = 0

    while idx < args.count {
        switch args[idx] {
        case "--priority":
            idx += 1
            guard idx < args.count else { break }
            let val = args[idx]
            guard validPriorities.contains(val) else {
                StderrLog.write(
                    "Invalid priority: \(val). Use normal, urgent, or low.",
                    tag: "OperatorCLI"
                )
                exit(1)
            }
            result.priority = val

        case "--session":
            idx += 1
            guard idx < args.count else { break }
            result.session = args[idx]

        case "--help", "-h":
            printUsage()
            exit(0)

        default:
            if result.message == nil {
                result.message = args[idx]
            } else {
                result.message = (result.message ?? "") + " " + args[idx]
            }
        }
        idx += 1
    }

    return result
}

private func printUsage() {
    let usage = """
        Usage: operator <command> [options]

        Commands:
          speak <message>    Send a voice message through Operator

        Options (speak):
          --priority <level>  Priority: normal (default), urgent, low
          --session <name>    Session name (default: current directory name)
          --help, -h          Show this help message
        """
    print(usage)
}
