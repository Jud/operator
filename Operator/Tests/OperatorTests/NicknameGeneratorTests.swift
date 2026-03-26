import Testing

@testable import OperatorCore

internal enum NicknameGeneratorTests {
    @Suite("NicknameGenerator - Nickname Derivation")
    internal struct NicknameTests {
        @Test("single short word stays as-is, capitalized")
        func singleShortWord() {
            #expect(NicknameGenerator.nickname(from: "operator") == "Operator")
        }

        @Test("already capitalized single word preserved")
        func alreadyCapitalized() {
            #expect(NicknameGenerator.nickname(from: "Sudo") == "Sudo")
        }

        @Test("hyphenated name picks first non-generic word")
        func hyphenatedName() {
            #expect(NicknameGenerator.nickname(from: "kokoro-tts-swift") == "Kokoro")
        }

        @Test("underscored name picks first non-generic word")
        func underscoredName() {
            #expect(NicknameGenerator.nickname(from: "kokoro_tts_swift") == "Kokoro")
        }

        @Test("generic first word skipped for distinctive second word")
        func genericFirstWord() {
            #expect(NicknameGenerator.nickname(from: "my-dashboard") == "Dashboard")
        }

        @Test("all generic words joins first two")
        func allGenericWords() {
            #expect(NicknameGenerator.nickname(from: "my-app") == "MyApp")
        }

        @Test("spaced name picks first non-generic word")
        func spacedName() {
            #expect(NicknameGenerator.nickname(from: "cool project name") == "Cool")
        }

        @Test("empty string returns empty")
        func emptyString() {
            #expect(NicknameGenerator.nickname(from: "").isEmpty)
        }
    }

    @Suite("NicknameGenerator - NATO Deduplication")
    internal struct DeduplicationTests {
        @Test("unique nickname returned unchanged")
        func uniqueNickname() {
            let result = NicknameGenerator.deduplicate("Kokoro", existing: ["Operator"])
            #expect(result == "Kokoro")
        }

        @Test("duplicate gets Alpha suffix")
        func duplicateGetsAlpha() {
            let result = NicknameGenerator.deduplicate("Operator", existing: ["Operator"])
            #expect(result == "Operator Alpha")
        }

        @Test("second duplicate gets Bravo")
        func secondDuplicateGetsBravo() {
            let result = NicknameGenerator.deduplicate(
                "Operator",
                existing: ["Operator", "Operator Alpha"]
            )
            #expect(result == "Operator Bravo")
        }

        @Test("case-insensitive conflict detection")
        func caseInsensitiveConflict() {
            let result = NicknameGenerator.deduplicate("operator", existing: ["Operator"])
            #expect(result == "operator Alpha")
        }

        @Test("NATO variant also counts as conflict")
        func natoVariantConflict() {
            let result = NicknameGenerator.deduplicate(
                "Operator",
                existing: ["Operator Alpha"]
            )
            #expect(result == "Operator Bravo")
        }

        @Test("empty existing list returns unchanged")
        func emptyExisting() {
            let result = NicknameGenerator.deduplicate("Operator", existing: [])
            #expect(result == "Operator")
        }
    }
}
