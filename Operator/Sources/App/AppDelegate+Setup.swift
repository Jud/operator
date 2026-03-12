import Foundation

// swiftlint:disable type_contents_order
// MARK: - Auth, CLI Symlinks, and Plugin Registration

extension AppDelegate {
    static let operatorDir: URL = {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".operator")
    }()

    func setupAuthToken() {
        let dir = Self.operatorDir
        let tokenFile = dir.appendingPathComponent("token")

        guard !FileManager.default.fileExists(atPath: tokenFile.path) else {
            Self.logger.info("Auth token already exists at \(tokenFile.path)")
            return
        }

        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let token = UUID().uuidString
            try token.write(to: tokenFile, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o600],
                ofItemAtPath: tokenFile.path
            )
            Self.logger.info("Generated auth token at \(tokenFile.path)")
        } catch {
            Self.logger.error("Failed to set up auth token: \(error)")
        }
    }

    func installCLISymlinks() {
        let fm = FileManager.default
        let binDir = Self.operatorDir.appendingPathComponent("bin")

        do {
            try fm.createDirectory(at: binDir, withIntermediateDirectories: true)
        } catch {
            Self.logger.error("Failed to create ~/.operator/bin: \(error)")
            return
        }

        let appMacOS = Bundle.main.bundleURL
            .appendingPathComponent("Contents/MacOS")
        let tool = "operator-mcp"
        let source = appMacOS.appendingPathComponent(tool)
        let link = binDir.appendingPathComponent(tool)

        try? fm.removeItem(at: link)

        do {
            try fm.createSymbolicLink(at: link, withDestinationURL: source)
            Self.logger.info("Symlinked \(link.path) -> \(source.path)")
        } catch {
            Self.logger.error("Failed to symlink \(tool): \(error)")
        }
    }

    func registerPlugin() {
        guard let claudePath = Self.findClaude() else {
            Self.logger.warning("Claude CLI not found — skipping plugin registration")
            return
        }
        Task.detached {
            await Self.runPluginRegistration(claudePath: claudePath)
        }
    }

    private static func findClaude() -> String? {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        for candidate in [
            "/usr/local/bin/claude",
            "/opt/homebrew/bin/claude",
            "\(home)/.claude/bin/claude"
        ] where FileManager.default.isExecutableFile(atPath: candidate) {
            return candidate
        }
        return nil
    }

    private static func runPluginRegistration(claudePath: String) async {
        for (args, label) in [
            (["plugin", "marketplace", "add", "Jud/operator-plugin"], "marketplace"),
            (["plugin", "install", "operator@operator"], "plugin")
        ] {
            let result = runProcess(claudePath, args)
            if result.status != 0 {
                logger.warning("Failed to register \(label): \(result.error)")
            } else {
                logger.info("\(label) registered")
            }
        }
    }

    private static func runProcess(
        _ path: String,
        _ arguments: [String]
    ) -> (status: Int32, output: String, error: String) {  // swiftlint:disable:this large_tuple
        let process = Process()
        process.executableURL = URL(fileURLWithPath: path)
        process.arguments = arguments
        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe
        do {
            try process.run()
        } catch {
            return (-1, "", error.localizedDescription)
        }
        let out = outPipe.fileHandleForReading.readDataToEndOfFile()
        let err = errPipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (
            process.terminationStatus,
            String(data: out, encoding: .utf8) ?? "",
            String(data: err, encoding: .utf8) ?? ""
        )
    }
}
// swiftlint:enable type_contents_order
