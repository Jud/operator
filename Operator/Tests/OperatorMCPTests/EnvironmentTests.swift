// swiftlint:disable:this file_name
import Foundation
import Testing

@testable import OperatorMCPCore

@Suite("Environment - TTY Normalization")
internal struct TTYNormalizationTests {
    @Test("prepends /dev/ when prefix is missing")
    func prependsDevPrefix() {
        let result = normalizeTTY("ttys004")
        #expect(result == "/dev/ttys004")
    }

    @Test("preserves /dev/ prefix when already present")
    func preservesDevPrefix() {
        let result = normalizeTTY("/dev/ttys004")
        #expect(result == "/dev/ttys004")
    }

    @Test("returns unknown for nil input")
    func nilReturnsUnknown() {
        let result = normalizeTTY(nil)
        #expect(result == "unknown")
    }

    @Test("returns unknown for empty string")
    func emptyReturnsUnknown() {
        let result = normalizeTTY("")
        #expect(result == "unknown")
    }

    @Test("returns unknown for whitespace-only string")
    func whitespaceReturnsUnknown() {
        let result = normalizeTTY("   \n  ")
        #expect(result == "unknown")
    }

    @Test("trims surrounding whitespace before normalizing")
    func trimsWhitespace() {
        let result = normalizeTTY("  ttys007\n")
        #expect(result == "/dev/ttys007")
    }
}

@Suite("Environment - Session Name")
internal struct SessionNameTests {
    @Test("derives session name from current directory basename")
    func currentDirectoryBasename() {
        let name = deriveSessionName()
        let expected = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath
        ).lastPathComponent
        #expect(name == expected)
        #expect(!name.isEmpty)
    }
}

@Suite("Environment - Config Loading")
internal struct ConfigLoadingTests {
    @Test("defaults port to 7420 when OPERATOR_PORT is unset")
    func defaultPort() {
        let config = loadConfig()
        if ProcessInfo.processInfo.environment["OPERATOR_PORT"] == nil {
            #expect(config.port == 7_420)
        }
    }

    @Test("baseURL is constructed from port")
    func baseURLFormat() {
        let config = loadConfig()
        #expect(config.baseURL == "http://localhost:\(config.port)")
    }

    @Test("token defaults to empty string when OPERATOR_TOKEN is unset")
    func defaultToken() {
        let config = loadConfig()
        if ProcessInfo.processInfo.environment["OPERATOR_TOKEN"] == nil {
            #expect(config.token.isEmpty)
        }
    }
}

@Suite("Environment - Terminal Type")
internal struct TerminalTypeTests {
    @Test("returns iterm2 when TERM_PROGRAM is not ghostty")
    func defaultTerminalType() {
        let termType = detectTerminalType()
        let termProgram = ProcessInfo.processInfo.environment["TERM_PROGRAM"]
        if termProgram == "ghostty" {
            #expect(termType == "ghostty")
        } else {
            #expect(termType == "iterm2")
        }
    }
}
