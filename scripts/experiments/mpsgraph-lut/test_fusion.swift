#!/usr/bin/env swift
// Minimal test: does MPSGraph fuse grouped LUT dequantize + matmul?
//
// Creates a [256, 256] weight with 16 groups × 16 centroids (PAL4 gs=16),
// builds an MPSGraph that dequantizes and matmuls, and times it.
//
// Run: swift scripts/experiments/mpsgraph-lut/test_fusion.swift

import Foundation
import MetalPerformanceShadersGraph

let device = MTLCreateSystemDefaultDevice()!
let graph = MPSGraph()

// --- Config ---
let M = 1      // batch (single token decode)
let K = 256    // input dim
let N = 256    // output dim
let nGroups = 16  // groups along output dim (gs=16 means N/16 = 16 groups)
let nCentroids = 16  // 4-bit = 16 entries per group

// --- Build the graph ---

// Input activation: [M, K] float16
let xData = graph.placeholder(shape: [M as NSNumber, K as NSNumber], dataType: .float16, name: "x")

// LUT: [nGroups, nCentroids] float16 — the palettized centroids
let lutShape: [NSNumber] = [nGroups as NSNumber, nCentroids as NSNumber]
let lutPlaceholder = graph.placeholder(shape: lutShape, dataType: .float16, name: "lut")

// Indices: [N, K] uint8 — 4-bit packed as uint8 (values 0-15)
let idxPlaceholder = graph.placeholder(shape: [N as NSNumber, K as NSNumber], dataType: .uInt8, name: "indices")

// Group assignment: each output row belongs to a group
// Row i → group i / (N/nGroups)
let groupSize = N / nGroups

// Dequantize: for each (i, j), weight[i,j] = lut[i / groupSize, indices[i,j]]
// MPSGraph approach: use gather to look up LUT values
// First, compute group IDs for each row
let rowIndices = graph.constant(
    Data(bytes: (0..<N).map { Int32($0 / groupSize) }, count: N * 4),
    shape: [N as NSNumber, 1],
    dataType: .int32
)

// Broadcast group IDs to [N, K]
let groupIds = graph.broadcast(rowIndices, shape: [N as NSNumber, K as NSNumber], name: "groupIds")

// Cast indices to int32 for gather
let idxInt32 = graph.cast(idxPlaceholder, to: .int32, name: "idxInt32")

// Compute flat LUT index: group * nCentroids + local_index
let nCentroidsConst = graph.constant(Double(nCentroids), dataType: .int32)
let lutOffset = graph.multiplication(groupIds, nCentroidsConst, name: "lutOffset")
let flatIdx = graph.addition(lutOffset, idxInt32, name: "flatIdx")

// Flatten LUT and gather
let lutFlat = graph.reshape(lutPlaceholder, shape: [(nGroups * nCentroids) as NSNumber], name: "lutFlat")
let weight = graph.gatherND(
    withUpdatesTensor: lutFlat,
    indicesTensor: graph.reshape(flatIdx, shape: [(N * K) as NSNumber, 1], name: nil),
    batchDimensions: 0,
    name: "gatherWeight"
)
let weightReshaped = graph.reshape(weight, shape: [N as NSNumber, K as NSNumber], name: "weightMatrix")

// Matmul: output = x @ weight.T → [M, N]
let weightT = graph.transposeTensor(weightReshaped, dimension: 0, withDimension: 1, name: "weightT")
let output = graph.matrixMultiplication(primary: xData, secondary: weightT, name: "output")

// --- Prepare test data ---
let commandQueue = device.makeCommandQueue()!

func makeBuffer<T>(_ values: [T]) -> MTLBuffer {
    values.withUnsafeBytes { ptr in
        device.makeBuffer(bytes: ptr.baseAddress!, length: ptr.count, options: .storageModeShared)!
    }
}

// Random input
var xValues = [Float16](repeating: 0, count: M * K)
for i in 0..<xValues.count { xValues[i] = Float16(Float.random(in: -1...1)) }

// Random LUT centroids
var lutValues = [Float16](repeating: 0, count: nGroups * nCentroids)
for i in 0..<lutValues.count { lutValues[i] = Float16(Float.random(in: -2...2)) }

// Random indices (0-15)
var idxValues = [UInt8](repeating: 0, count: N * K)
for i in 0..<idxValues.count { idxValues[i] = UInt8.random(in: 0..<UInt8(nCentroids)) }

let xBuf = makeBuffer(xValues)
let lutBuf = makeBuffer(lutValues)
let idxBuf = makeBuffer(idxValues)

let xTensor = MPSGraphTensorData(xBuf, shape: [M as NSNumber, K as NSNumber], dataType: .float16)
let lutTensor = MPSGraphTensorData(lutBuf, shape: lutShape, dataType: .float16)
let idxTensor = MPSGraphTensorData(idxBuf, shape: [N as NSNumber, K as NSNumber], dataType: .uInt8)

let feeds: [MPSGraphTensor: MPSGraphTensorData] = [
    xData: xTensor,
    lutPlaceholder: lutTensor,
    idxPlaceholder: idxTensor,
]

// --- Warmup ---
print("Warming up...")
for _ in 0..<5 {
    let _ = graph.run(with: commandQueue, feeds: feeds, targetTensors: [output], targetOperations: nil)
}

// --- Benchmark ---
let N_ITERS = 50
var times: [Double] = []
print("Benchmarking \(N_ITERS) iterations...")

for _ in 0..<N_ITERS {
    let t0 = CFAbsoluteTimeGetCurrent()
    let result = graph.run(with: commandQueue, feeds: feeds, targetTensors: [output], targetOperations: nil)
    // Force GPU sync
    let arr = result[output]!.mpsndarray()
    var dummy = [Float16](repeating: 0, count: M * N)
    dummy.withUnsafeMutableBytes { ptr in
        arr.readBytes(ptr.baseAddress!, strideBytes: nil)
    }
    let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
    times.append(elapsed)
}

times.sort()
let p50 = times[N_ITERS / 2]
print("")
print("=== MPSGraph grouped LUT matmul ===")
print("  Matrix: [\(M), \(K)] × [\(N), \(K)]  (\(nGroups) groups × \(nCentroids) centroids)")
print("  p50: \(String(format: "%.2f", p50)) ms")
print("  min: \(String(format: "%.2f", times.first!)) ms")
print("  max: \(String(format: "%.2f", times.last!)) ms")
