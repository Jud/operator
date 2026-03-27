// VerifyMPSGraph — Layer-by-layer verification of MPSGraph decoder.
//
// Usage:
//   swift run VerifyMPSGraph [ref_dir]
//   swift run VerifyMPSGraph models/reference_activations

import Foundation

let refDir: String
if CommandLine.arguments.count > 1 {
    refDir = CommandLine.arguments[1]
} else {
    refDir = "models/reference_activations"
}

guard FileManager.default.fileExists(atPath: "\(refDir)/meta.json") else {
    fputs("Reference dir not found: \(refDir)\n", stderr)
    fputs("Run: /tmp/coreml-venv/bin/python scripts/pipeline/dump_reference.py\n", stderr)
    exit(1)
}

// FP32 first — zero tolerance for bugs
verifyAllFP32(refDir: refDir)
