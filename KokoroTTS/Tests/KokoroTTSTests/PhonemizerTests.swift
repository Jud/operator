import Testing
@testable import KokoroTTS

// MisakiSwift tests require Metal GPU (MLX), which isn't available in
// the SPM test runner. The EnglishG2P integration is verified through
// the TTS benchmark (`./scripts/run-benchmarks.sh tts`) which runs
// the full pipeline end-to-end with real models.
//
// Tokenizer tests (IPA → token IDs) are in TokenizerTests.swift and
// don't require MLX.
