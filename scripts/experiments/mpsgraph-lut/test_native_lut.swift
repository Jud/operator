#!/usr/bin/env swift
// Test MPSGraph's NATIVE LUT dequantize → matmul fusion.
// Uses dequantizeTensor:LUTTensor:axis:name: which is the real API.

import Foundation
import MetalPerformanceShadersGraph

let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!

let M = 1
let K = 2048
let N = 6144
let nGroups = N / 16  // 384 groups

print("=== MPSGraph native LUT dequantize + matmul ===")
print("  [\(M), \(K)] × [\(N), \(K)]  (\(nGroups) groups × 16 centroids)")
print()

// --- Graph 1: Native LUT dequant + matmul ---
let graph1 = MPSGraph()

let x1 = graph1.placeholder(shape: [M as NSNumber, K as NSNumber], dataType: .float16, name: "x")

// Quantized weights: [N, K] uint8 (4-bit values 0-15 stored in uint8)
let qw1 = graph1.placeholder(shape: [N as NSNumber, K as NSNumber], dataType: .uInt8, name: "qw")

// LUT: [256] — uint8 indices need 256-entry LUT (2^8)
// We map 384 groups × 16 centroids → 256 global centroids (near-zero mapping error)
let lut1 = graph1.placeholder(shape: [256 as NSNumber], dataType: .float16, name: "lut")

// Native LUT dequantize with 256-entry global LUT
let dequant1 = graph1.dequantize(qw1, LUTTensor: lut1, name: "dequant")
let wT1 = graph1.transposeTensor(dequant1, dimension: 0, withDimension: 1, name: nil)
let out1 = graph1.matrixMultiplication(primary: x1, secondary: wT1, name: "matmul")

// --- Graph 2: FP16 matmul baseline ---
let graph2 = MPSGraph()
let x2 = graph2.placeholder(shape: [M as NSNumber, K as NSNumber], dataType: .float16, name: "x")
let w2 = graph2.placeholder(shape: [K as NSNumber, N as NSNumber], dataType: .float16, name: "w")
let out2 = graph2.matrixMultiplication(primary: x2, secondary: w2, name: "matmul")

// --- Data ---
func makeRandom<T: BinaryFloatingPoint>(_ count: Int, range: ClosedRange<Float> = -1...1) -> [T] {
    (0..<count).map { _ in T(Float.random(in: range)) }
}

let xVals: [Float16] = makeRandom(M * K)
let lutVals: [Float16] = makeRandom(256, range: -2...2)
let qVals: [UInt8] = (0..<N*K).map { _ in UInt8.random(in: 0...255) }
let wVals: [Float16] = makeRandom(K * N)

func tensor(_ vals: UnsafeRawBufferPointer, _ shape: [NSNumber], _ dt: MPSDataType) -> MPSGraphTensorData {
    let buf = device.makeBuffer(bytes: vals.baseAddress!, length: vals.count, options: .storageModeShared)!
    return MPSGraphTensorData(buf, shape: shape, dataType: dt)
}

let xT = xVals.withUnsafeBytes { tensor($0, [M as NSNumber, K as NSNumber], .float16) }
let lutT = lutVals.withUnsafeBytes { tensor($0, [256 as NSNumber], .float16) }
let qT = qVals.withUnsafeBytes { tensor($0, [N as NSNumber, K as NSNumber], .uInt8) }
let wT = wVals.withUnsafeBytes { tensor($0, [K as NSNumber, N as NSNumber], .float16) }

// --- Benchmark ---
func bench(_ g: MPSGraph, _ f: [MPSGraphTensor: MPSGraphTensorData], _ t: [MPSGraphTensor], _ label: String) {
    for _ in 0..<10 {
        let _ = g.run(with: commandQueue, feeds: f, targetTensors: t, targetOperations: nil)
    }
    var times: [Double] = []
    for _ in 0..<30 {
        let t0 = CFAbsoluteTimeGetCurrent()
        let result = g.run(with: commandQueue, feeds: f, targetTensors: t, targetOperations: nil)
        let arr = result[t[0]]!.mpsndarray()
        var dummy = [Float16](repeating: 0, count: M * N)
        dummy.withUnsafeMutableBytes { ptr in arr.readBytes(ptr.baseAddress!, strideBytes: nil) }
        times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
    }
    times.sort()
    print("  \(label): p50=\(String(format: "%.2f", times[15]))ms  min=\(String(format: "%.2f", times.first!))ms")
}

let feeds1: [MPSGraphTensor: MPSGraphTensorData] = [x1: xT, qw1: qT, lut1: lutT]
let feeds2: [MPSGraphTensor: MPSGraphTensorData] = [x2: xT, w2: wT]

bench(graph1, feeds1, [out1], "Native LUT dequant + matmul")
bench(graph2, feeds2, [out2], "FP16 matmul baseline        ")
