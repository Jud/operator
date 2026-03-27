#!/usr/bin/env swift
// Test MPSGraph's native dequantize → matmul fusion with lookup table quantization.
//
// From WWDC24: "If your graph contains a dequantize operation on the weights
// preceding a matrix multiplication, MPSGraph will replace the two operations
// with a single quantized matrix multiplication operation."
//
// This test verifies that claim for LUT-based quantization.

import Foundation
import MetalPerformanceShadersGraph

let device = MTLCreateSystemDefaultDevice()!
let graph = MPSGraph()

let M = 1
let K = 2048    // realistic input dim
let N = 6144    // realistic output dim (matches Qwen3.5 2B qkv projection)
let nGroups = N / 16  // 384 groups, gs=16

print("=== MPSGraph native dequantize + matmul test ===")
print("  [\(M), \(K)] × [\(N), \(K)]  (\(nGroups) groups × 16 centroids)")
print()

// Input: [M, K] float16
let xData = graph.placeholder(shape: [M as NSNumber, K as NSNumber], dataType: .float16, name: "x")

// Quantized weight: [N, K] uint8 with values 0-15 (4-bit packed as uint8)
let qWeight = graph.placeholder(shape: [N as NSNumber, K as NSNumber], dataType: .uInt8, name: "qWeight")

// LUT: [nGroups, 16] float16
let lut = graph.placeholder(
    shape: [nGroups as NSNumber, 16 as NSNumber],
    dataType: .float16,
    name: "lut"
)

// Approach 1: Use MPSGraph.dequantize if available
// MPSGraph has dequantizeTensor(tensor:scaleTensor:zeroPointTensor:) for affine
// For LUT, we need to use gather-based dequantization

// Compute group IDs: row i → group i/16
let groupIdValues = (0..<N).map { Int32($0 / 16) }
let groupIdData = groupIdValues.withUnsafeBytes { Data($0) }
let groupIds = graph.constant(groupIdData, shape: [N as NSNumber, 1], dataType: .int32)
let groupIdsBroadcast = graph.broadcast(groupIds, shape: [N as NSNumber, K as NSNumber], name: nil)

// LUT index: group * 16 + quantized_value
let sixteen = graph.constant(16.0, dataType: .int32)
let qInt32 = graph.cast(qWeight, to: .int32, name: nil)
let flatIdx = graph.addition(
    graph.multiplication(groupIdsBroadcast, sixteen, name: nil),
    qInt32,
    name: "flatIdx"
)

// Flatten LUT and gather
let lutFlat = graph.reshape(lut, shape: [(nGroups * 16) as NSNumber], name: nil)
let flatIdxFlat = graph.reshape(flatIdx, shape: [(N * K) as NSNumber], name: nil)

// Gather dequantized values
let dequantFlat = graph.gatherND(
    withUpdatesTensor: lutFlat,
    indicesTensor: graph.reshape(flatIdxFlat, shape: [(N * K) as NSNumber, 1], name: nil),
    batchDimensions: 0,
    name: "dequant"
)
let dequantWeight = graph.reshape(dequantFlat, shape: [N as NSNumber, K as NSNumber], name: nil)

// Matmul: [M, K] × [K, N] → [M, N]
let weightT = graph.transposeTensor(dequantWeight, dimension: 0, withDimension: 1, name: nil)
let output = graph.matrixMultiplication(primary: xData, secondary: weightT, name: "matmul")

// Also build a plain FP16 matmul for comparison
let graph2 = MPSGraph()
let xData2 = graph2.placeholder(shape: [M as NSNumber, K as NSNumber], dataType: .float16, name: "x")
let wData2 = graph2.placeholder(shape: [K as NSNumber, N as NSNumber], dataType: .float16, name: "w")
let output2 = graph2.matrixMultiplication(primary: xData2, secondary: wData2, name: "matmul")

// --- Prepare data ---
let commandQueue = device.makeCommandQueue()!

var xValues = [Float16](repeating: 0, count: M * K)
for i in 0..<xValues.count { xValues[i] = Float16.random(in: -1...1) }

var lutValues = [Float16](repeating: 0, count: nGroups * 16)
for i in 0..<lutValues.count { lutValues[i] = Float16.random(in: -2...2) }

var qValues = [UInt8](repeating: 0, count: N * K)
for i in 0..<qValues.count { qValues[i] = UInt8.random(in: 0..<16) }

var wValues = [Float16](repeating: 0, count: K * N)
for i in 0..<wValues.count { wValues[i] = Float16.random(in: -1...1) }

func makeTensor(_ values: UnsafeRawBufferPointer, _ shape: [NSNumber], _ dt: MPSDataType) -> MPSGraphTensorData {
    let buf = device.makeBuffer(bytes: values.baseAddress!, length: values.count, options: .storageModeShared)!
    return MPSGraphTensorData(buf, shape: shape, dataType: dt)
}

let xTensor = xValues.withUnsafeBytes { makeTensor($0, [M as NSNumber, K as NSNumber], .float16) }
let lutTensor = lutValues.withUnsafeBytes { makeTensor($0, [nGroups as NSNumber, 16 as NSNumber], .float16) }
let qTensor = qValues.withUnsafeBytes { makeTensor($0, [N as NSNumber, K as NSNumber], .uInt8) }
let wTensor = wValues.withUnsafeBytes { makeTensor($0, [K as NSNumber, N as NSNumber], .float16) }

let feeds1: [MPSGraphTensor: MPSGraphTensorData] = [xData: xTensor, qWeight: qTensor, lut: lutTensor]
let feeds2: [MPSGraphTensor: MPSGraphTensorData] = [xData2: xTensor, wData2: wTensor]

// --- Benchmark ---
func bench(_ g: MPSGraph, _ f: [MPSGraphTensor: MPSGraphTensorData], _ t: [MPSGraphTensor], _ label: String) {
    // Warmup
    for _ in 0..<10 {
        let _ = g.run(with: commandQueue, feeds: f, targetTensors: t, targetOperations: nil)
    }

    var times: [Double] = []
    for _ in 0..<30 {
        let t0 = CFAbsoluteTimeGetCurrent()
        let result = g.run(with: commandQueue, feeds: f, targetTensors: t, targetOperations: nil)
        let arr = result[t[0]]!.mpsndarray()
        var dummy = [Float16](repeating: 0, count: M * N)
        dummy.withUnsafeMutableBytes { ptr in
            arr.readBytes(ptr.baseAddress!, strideBytes: nil)
        }
        times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
    }
    times.sort()
    print("  \(label): p50=\(String(format: "%.2f", times[15]))ms  min=\(String(format: "%.2f", times.first!))ms")
}

bench(graph, feeds1, [output], "LUT dequant + matmul (grouped)")
bench(graph2, feeds2, [output2], "FP16 matmul (baseline)       ")
