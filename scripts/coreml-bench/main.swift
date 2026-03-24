// Minimal Swift test: can CoreML run two models concurrently?
//
// Build & run:
//   swiftc -O -framework CoreML -framework Foundation scripts/coreml-bench/main.swift -o /tmp/coreml-bench
//   /tmp/coreml-bench models/disaggregated/deltanet_block_0.mlpackage models/disaggregated/deltanet_block_1.mlpackage

import CoreML
import Foundation

func loadModel(_ path: String, computeUnits: MLComputeUnits = .all) throws -> MLModel {
    let url = URL(fileURLWithPath: path)
    let config = MLModelConfiguration()
    config.computeUnits = computeUnits
    return try MLModel(contentsOf: MLModel.compileModel(at: url), configuration: config)
}

func buildInput(model: MLModel) throws -> MLDictionaryFeatureProvider {
    var dict: [String: MLFeatureValue] = [:]
    for desc in model.modelDescription.inputDescriptionsByName {
        let name = desc.key
        let constraint = desc.value.multiArrayConstraint!
        let _ = constraint.shape.map { $0.intValue }
        let arr = try MLMultiArray(shape: constraint.shape, dataType: .float32)
        // Fill with small random values
        for i in 0..<arr.count {
            arr[i] = NSNumber(value: Float.random(in: -0.01...0.01))
        }
        dict[name] = MLFeatureValue(multiArray: arr)
    }
    return try MLDictionaryFeatureProvider(dictionary: dict)
}

func benchmark(_ label: String, iterations: Int = 20, _ block: () throws -> Void) rethrows -> Double {
    // Warmup
    for _ in 0..<5 { try block() }

    var times: [Double] = []
    for _ in 0..<iterations {
        let t0 = CFAbsoluteTimeGetCurrent()
        try block()
        times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
    }
    times.sort()
    let median = times[iterations / 2]
    print("  \(label): \(String(format: "%.1f", median))ms")
    return median
}

guard CommandLine.arguments.count >= 3 else {
    print("Usage: coreml-bench <model_a.mlpackage> <model_b.mlpackage>")
    exit(1)
}

let pathA = CommandLine.arguments[1]
let pathB = CommandLine.arguments[2]

print("Loading models...")
let modelA_gpu = try! loadModel(pathA, computeUnits: .cpuAndGPU)
let modelB_gpu = try! loadModel(pathB, computeUnits: .cpuAndGPU)
let modelA_ane = try! loadModel(pathA, computeUnits: .cpuAndNeuralEngine)
let modelB_ane = try! loadModel(pathB, computeUnits: .cpuAndNeuralEngine)

let inputA = try! buildInput(model: modelA_gpu)
let inputB = try! buildInput(model: modelB_gpu)

print("Loaded.\n")

// Single model
let single = try benchmark("Single block (GPU)") {
    let _ = try modelA_gpu.prediction(from: inputA)
}

// Sequential
let seq = try benchmark("Sequential (2 × GPU)") {
    let _ = try modelA_gpu.prediction(from: inputA)
    let _ = try modelB_gpu.prediction(from: inputB)
}

// Concurrent: GCD
let concGCD = benchmark("Concurrent GCD (2 × GPU)") {
    let group = DispatchGroup()
    let q = DispatchQueue.global(qos: .userInteractive)

    group.enter()
    q.async {
        let _ = try? modelA_gpu.prediction(from: inputA)
        group.leave()
    }
    group.enter()
    q.async {
        let _ = try? modelB_gpu.prediction(from: inputB)
        group.leave()
    }
    group.wait()
}

// Concurrent: GPU + ANE
let concMixed = benchmark("Concurrent GCD (GPU + ANE)") {
    let group = DispatchGroup()
    let q = DispatchQueue.global(qos: .userInteractive)

    group.enter()
    q.async {
        let _ = try? modelA_gpu.prediction(from: inputA)
        group.leave()
    }
    group.enter()
    q.async {
        let _ = try? modelB_ane.prediction(from: inputB)
        group.leave()
    }
    group.wait()
}

// Concurrent: async/await
if #available(macOS 13.0, *) {
    // Use DispatchSemaphore to run the async test synchronously
    let sem = DispatchSemaphore(value: 0)
    var asyncTime: Double = 0

    Task {
        // Warmup
        for _ in 0..<5 {
            async let r1 = modelA_gpu.prediction(from: inputA)
            async let r2 = modelB_gpu.prediction(from: inputB)
            let _ = try await (r1, r2)
        }

        var times: [Double] = []
        for _ in 0..<20 {
            let t0 = CFAbsoluteTimeGetCurrent()
            async let r1 = modelA_gpu.prediction(from: inputA)
            async let r2 = modelB_gpu.prediction(from: inputB)
            let _ = try await (r1, r2)
            times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }
        times.sort()
        asyncTime = times[10]
        print("  Concurrent async let (2 × GPU): \(String(format: "%.1f", asyncTime))ms")
        sem.signal()
    }
    sem.wait()
}

print("\n--- Summary ---")
print(String(format: "Sequential:     %.1fms", seq))
print(String(format: "Concurrent GCD: %.1fms (%.2fx)", concGCD, seq / concGCD))
print(String(format: "GPU + ANE:      %.1fms (%.2fx)", concMixed, seq / concMixed))

if concGCD < seq * 0.7 || concMixed < seq * 0.7 {
    print("\n✓ Concurrent execution provides speedup!")
} else {
    print("\n✗ CoreML serializes execution even in Swift.")
}
