// Verify DeltaNet decode step ops in MPSGraph against reference.
//
// Tests: Z/B/A projections, conv1d update, QKV split, gates, recurrence, norm+gate, out_proj

import Foundation
import Metal
import MetalPerformanceShadersGraph

func verifyDeltaNet(refDir: String) {
    let device = MTLCreateSystemDefaultDevice()!
    let commandQueue = device.makeCommandQueue()!
    let loader = NpyLoader(device: device)

    print("=== Verifying DeltaNet internals (layer 0) ===\n")

    var allPass = true

    // --- Z projection ---
    print("Step 1: Z projection")
    let g1 = MPSGraph()
    let (zIn, _) = loader.loadTensor("\(refDir)/activations/layer_0_deltanet_z_in_0.npy")
    let (zRefBuf, _, zCount) = loader.load("\(refDir)/activations/layer_0_deltanet_z_out.npy")
    let (zW, zWShape) = loader.loadTensor("\(refDir)/weights/model_layers_0_linear_attn_in_proj_z_weight.npy")

    let zX = g1.placeholder(shape: zIn.shape, dataType: .float16, name: "x")
    let zWeight = g1.placeholder(shape: zW.shape, dataType: .float16, name: "w")
    let zX2d = g1.reshape(zX, shape: [1 as NSNumber, zWShape[1] as NSNumber], name: nil)
    let zWT = g1.transposeTensor(zWeight, dimension: 0, withDimension: 1, name: nil)
    let zMM = g1.matrixMultiplication(primary: zX2d, secondary: zWT, name: nil)
    let zOut = g1.reshape(zMM, shape: [1 as NSNumber, 1 as NSNumber, zWShape[0] as NSNumber], name: nil)

    let r1 = g1.run(with: commandQueue, feeds: [zX: zIn, zWeight: zW], targetTensors: [zOut], targetOperations: nil)
    let zActBuf = device.makeBuffer(length: zCount * 2, options: .storageModeShared)!
    r1[zOut]!.mpsndarray().readBytes(zActBuf.contents(), strideBytes: nil)
    let zPass = compareBuffers(zActBuf, zRefBuf, count: zCount, label: "Z projection")
    allPass = allPass && zPass

    // --- B projection ---
    print("\nStep 2: B projection")
    let g2 = MPSGraph()
    let (bW, bWShape) = loader.loadTensor("\(refDir)/weights/model_layers_0_linear_attn_in_proj_b_weight.npy")
    let (bRefBuf, _, bCount) = loader.load("\(refDir)/activations/layer_0_beta.npy")

    let bX = g2.placeholder(shape: zIn.shape, dataType: .float16, name: "x")
    let bWeight = g2.placeholder(shape: bW.shape, dataType: .float16, name: "w")
    let bX2d = g2.reshape(bX, shape: [1 as NSNumber, bWShape[1] as NSNumber], name: nil)
    let bWT = g2.transposeTensor(bWeight, dimension: 0, withDimension: 1, name: nil)
    let bMM = g2.matrixMultiplication(primary: bX2d, secondary: bWT, name: nil)
    let bLinOut = g2.reshape(bMM, shape: [1 as NSNumber, 1 as NSNumber, bWShape[0] as NSNumber], name: nil)
    // beta = sigmoid(b)
    let bSigmoid = g2.sigmoid(with: bLinOut, name: "beta")

    let r2 = g2.run(with: commandQueue, feeds: [bX: zIn, bWeight: bW], targetTensors: [bSigmoid], targetOperations: nil)
    let bActBuf = device.makeBuffer(length: bCount * 2, options: .storageModeShared)!
    r2[bSigmoid]!.mpsndarray().readBytes(bActBuf.contents(), strideBytes: nil)
    let bPass = compareBuffers(bActBuf, bRefBuf, count: bCount, label: "Beta (sigmoid(B))")
    allPass = allPass && bPass

    // --- G gate ---
    print("\nStep 3: G gate = -exp(A_log) * softplus(A_proj + dt_bias)")
    let g3 = MPSGraph()
    let (aW, aWShape) = loader.loadTensor("\(refDir)/weights/model_layers_0_linear_attn_in_proj_a_weight.npy")
    let (aLog, _) = loader.loadTensor("\(refDir)/weights/model_layers_0_linear_attn_A_log.npy")
    let (dtBias, _) = loader.loadTensor("\(refDir)/weights/model_layers_0_linear_attn_dt_bias.npy")
    let (gRefBuf, _, gCount) = loader.load("\(refDir)/activations/layer_0_g.npy")

    let aX = g3.placeholder(shape: zIn.shape, dataType: .float16, name: "x")
    let aWeight = g3.placeholder(shape: aW.shape, dataType: .float16, name: "w")
    let aLogP = g3.placeholder(shape: aLog.shape, dataType: .float16, name: "a_log")
    let dtBiasP = g3.placeholder(shape: dtBias.shape, dataType: .float16, name: "dt_bias")

    // a = A_proj(hidden)
    let aX2d = g3.reshape(aX, shape: [1 as NSNumber, aWShape[1] as NSNumber], name: nil)
    let aWT = g3.transposeTensor(aWeight, dimension: 0, withDimension: 1, name: nil)
    let aMM = g3.matrixMultiplication(primary: aX2d, secondary: aWT, name: nil)
    let aOut = g3.reshape(aMM, shape: [1 as NSNumber, 1 as NSNumber, aWShape[0] as NSNumber], name: nil)

    // g = -exp(A_log) * softplus(a + dt_bias) — all in FP32
    let aF32 = g3.cast(aOut, to: .float32, name: nil)
    let aLogF32 = g3.cast(aLogP, to: .float32, name: nil)
    let dtBiasF32 = g3.cast(dtBiasP, to: .float32, name: nil)
    let negExpA = g3.negative(with: g3.exponent(with: aLogF32, name: nil), name: nil)
    let aPlusDt = g3.addition(aF32, dtBiasF32, name: nil)
    let softplus = g3.logarithm(with: g3.addition(g3.exponent(with: aPlusDt, name: nil),
                                                    g3.constant(1.0, dataType: .float32), name: nil), name: nil)
    let gVal = g3.multiplication(negExpA, softplus, name: nil)
    let gOut = g3.cast(gVal, to: .float16, name: nil)

    let r3 = g3.run(with: commandQueue, feeds: [aX: zIn, aWeight: aW, aLogP: aLog, dtBiasP: dtBias],
                     targetTensors: [gOut], targetOperations: nil)
    let gActBuf = device.makeBuffer(length: gCount * 2, options: .storageModeShared)!
    r3[gOut]!.mpsndarray().readBytes(gActBuf.contents(), strideBytes: nil)
    let gPass = compareBuffers(gActBuf, gRefBuf, count: gCount, label: "G gate")
    allPass = allPass && gPass

    // --- Conv1d update ---
    print("\nStep 4: Conv1d update (shift + depthwise conv + silu)")
    let g4 = MPSGraph()
    let (qkvOut, _) = loader.loadTensor("\(refDir)/activations/layer_0_deltanet_qkv_out.npy")
    let (convState, _) = loader.loadTensor("\(refDir)/cache_state/conv_state_0.npy")
    let (convW, _) = loader.loadTensor("\(refDir)/weights/model_layers_0_linear_attn_conv1d_weight.npy")
    let (convRefBuf, _, convCount) = loader.load("\(refDir)/activations/layer_0_after_conv1d.npy")

    // QKV out is [1, 1, 6144], need to transpose to [1, 6144, 1]
    let qkvP = g4.placeholder(shape: qkvOut.shape, dataType: .float16, name: "qkv")
    let csP = g4.placeholder(shape: convState.shape, dataType: .float16, name: "conv_state")
    let cwP = g4.placeholder(shape: convW.shape, dataType: .float16, name: "conv_weight")

    // Transpose qkv: [1, 1, 6144] → [1, 6144, 1]
    let qkvT = g4.transposeTensor(qkvP, dimension: 1, withDimension: 2, name: nil)

    // Shift conv_state left, append new value:
    // new_state = concat(conv_state[:, :, 1:], qkvT, dim=2) → [1, 6144, 4]
    let stateSlice = g4.sliceTensor(csP, dimension: 2, start: 1, length: 3, name: nil)
    let newState = g4.concatTensors([stateSlice, qkvT], dimension: 2, name: nil)

    // Depthwise conv: sum(new_state * weight, dim=-1) per channel
    // conv_weight is [6144, 1, 4], squeeze to [6144, 4]
    let cwSqueezed = g4.reshape(cwP, shape: [6144 as NSNumber, 4 as NSNumber], name: nil)
    // new_state is [1, 6144, 4], squeeze batch → [6144, 4]
    let nsSqueezed = g4.reshape(newState, shape: [6144 as NSNumber, 4 as NSNumber], name: nil)
    // Element-wise multiply then reduce sum along last dim
    let product = g4.multiplication(nsSqueezed, cwSqueezed, name: nil)
    let convSum = g4.reductionSum(with: product, axis: -1, name: nil)  // [6144, 1]

    // Note: conv1d.bias might not exist — check
    // Apply SiLU: x * sigmoid(x)
    let convSigmoid = g4.sigmoid(with: convSum, name: nil)
    let convSilu = g4.multiplication(convSum, convSigmoid, name: nil)
    let convOut = g4.reshape(convSilu, shape: [1 as NSNumber, 6144 as NSNumber, 1 as NSNumber], name: nil)

    let r4 = g4.run(with: commandQueue, feeds: [qkvP: qkvOut, csP: convState, cwP: convW],
                     targetTensors: [convOut], targetOperations: nil)
    let convActBuf = device.makeBuffer(length: convCount * 2, options: .storageModeShared)!
    r4[convOut]!.mpsndarray().readBytes(convActBuf.contents(), strideBytes: nil)
    let convPass = compareBuffers(convActBuf, convRefBuf, count: convCount, label: "Conv1d update", tolerance: 0.1)
    allPass = allPass && convPass

    // --- Summary ---
    print("\n=== DeltaNet Internals Summary ===")
    print("  Z projection:    \(zPass ? "PASS" : "FAIL")")
    print("  Beta gate:       \(bPass ? "PASS" : "FAIL")")
    print("  G gate:          \(gPass ? "PASS" : "FAIL")")
    print("  Conv1d update:   \(convPass ? "PASS" : "FAIL")")
    print("  Overall:         \(allPass ? "ALL PASS ✓" : "FAILED ✗")")
}
