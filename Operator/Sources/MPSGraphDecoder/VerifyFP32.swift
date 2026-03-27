// FP32 verification of all ops against PyTorch reference.
// ALL computation in FP32. Zero tolerance for bugs.
// FP16 conversion is a separate step AFTER FP32 verification passes.

import Foundation
import Metal
import MetalPerformanceShadersGraph

func verifyAllFP32(refDir: String) {
    let device = MTLCreateSystemDefaultDevice()!
    let commandQueue = device.makeCommandQueue()!
    let loader = NpyLoader(device: device)

    print("=== FP32 Verification (zero tolerance for bugs) ===\n")
    let tol: Float = 1e-4  // FP32 should match PyTorch within floating point epsilon

    var allPass = true

    func verify(_ graph: MPSGraph, _ feeds: [MPSGraphTensor: MPSGraphTensorData],
                _ target: MPSGraphTensor, _ refPath: String, _ label: String) -> Bool {
        let result = graph.run(with: commandQueue, feeds: feeds, targetTensors: [target], targetOperations: nil)
        let (refBuf, _, count) = loader.load(refPath)
        let actBuf = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        result[target]!.mpsndarray().readBytes(actBuf.contents(), strideBytes: nil)
        return compareBuffers(actBuf, refBuf, count: count, label: label, tolerance: tol, fp32: true)
    }

    // --- Embedding ---
    print("1. Embedding")
    let g1 = MPSGraph()
    let (tokenId, _) = loader.loadTensor("\(refDir)/decode_inputs/token_id.npy", dataType: .int32)
    let (embedW, _) = loader.loadTensor("\(refDir)/weights/model_embed_tokens_weight.npy")
    let pTok = g1.placeholder(shape: [1], dataType: .int32, name: "tok")
    let pEW = g1.placeholder(shape: embedW.shape, dataType: .float32, name: "ew")
    let embedOut = g1.reshape(
        g1.gatherND(withUpdatesTensor: pEW,
                     indicesTensor: g1.reshape(pTok, shape: [1 as NSNumber, 1], name: nil),
                     batchDimensions: 0, name: nil),
        shape: [1 as NSNumber, 1 as NSNumber, 2048 as NSNumber], name: nil)
    let p1 = verify(g1, [pTok: tokenId, pEW: embedW], embedOut,
                     "\(refDir)/activations/embed_out.npy", "Embedding")
    allPass = allPass && p1

    // --- RMSNorm ---
    print("\n2. RMSNorm (1 + weight)")
    let g2 = MPSGraph()
    let (normIn, _) = loader.loadTensor("\(refDir)/activations/layer_0_input_norm_in_0.npy")
    let (normW, _) = loader.loadTensor("\(refDir)/weights/model_layers_0_input_layernorm_weight.npy")
    let pNI = g2.placeholder(shape: normIn.shape, dataType: .float32, name: "x")
    let pNW = g2.placeholder(shape: normW.shape, dataType: .float32, name: "w")
    let wPlusOne = g2.addition(pNW, g2.constant(1.0, dataType: .float32), name: nil)
    let sq2 = g2.multiplication(pNI, pNI, name: nil)
    let var2 = g2.mean(of: sq2, axes: [-1], name: nil)
    let rsqrt2 = g2.reverseSquareRoot(with: g2.addition(var2, g2.constant(1e-6, dataType: .float32), name: nil), name: nil)
    let normOut = g2.multiplication(g2.multiplication(pNI, rsqrt2, name: nil), wPlusOne, name: nil)
    let p2 = verify(g2, [pNI: normIn, pNW: normW], normOut,
                     "\(refDir)/activations/layer_0_input_norm_out.npy", "RMSNorm")
    allPass = allPass && p2

    // --- QKV projection ---
    print("\n3. QKV projection")
    let g3 = MPSGraph()
    let (qkvIn, _) = loader.loadTensor("\(refDir)/activations/layer_0_deltanet_qkv_in_0.npy")
    let (qkvW, qkvWShape) = loader.loadTensor("\(refDir)/weights/model_layers_0_linear_attn_in_proj_qkv_weight.npy")
    let pQI = g3.placeholder(shape: qkvIn.shape, dataType: .float32, name: "x")
    let pQW = g3.placeholder(shape: qkvW.shape, dataType: .float32, name: "w")
    let x2d = g3.reshape(pQI, shape: [1 as NSNumber, qkvWShape[1] as NSNumber], name: nil)
    let wT3 = g3.transposeTensor(pQW, dimension: 0, withDimension: 1, name: nil)
    let mm3 = g3.matrixMultiplication(primary: x2d, secondary: wT3, name: nil)
    let qkvOut = g3.reshape(mm3, shape: [1 as NSNumber, 1 as NSNumber, qkvWShape[0] as NSNumber], name: nil)
    let p3 = verify(g3, [pQI: qkvIn, pQW: qkvW], qkvOut,
                     "\(refDir)/activations/layer_0_deltanet_qkv_out.npy", "QKV projection")
    allPass = allPass && p3

    // --- Z projection ---
    print("\n4. Z projection")
    let g4 = MPSGraph()
    let (zW, zWShape) = loader.loadTensor("\(refDir)/weights/model_layers_0_linear_attn_in_proj_z_weight.npy")
    let pZI = g4.placeholder(shape: qkvIn.shape, dataType: .float32, name: "x")
    let pZW = g4.placeholder(shape: zW.shape, dataType: .float32, name: "w")
    let z2d = g4.reshape(pZI, shape: [1 as NSNumber, zWShape[1] as NSNumber], name: nil)
    let zWT = g4.transposeTensor(pZW, dimension: 0, withDimension: 1, name: nil)
    let zmm = g4.matrixMultiplication(primary: z2d, secondary: zWT, name: nil)
    let zOut = g4.reshape(zmm, shape: [1 as NSNumber, 1 as NSNumber, zWShape[0] as NSNumber], name: nil)
    let p4 = verify(g4, [pZI: qkvIn, pZW: zW], zOut,
                     "\(refDir)/activations/layer_0_deltanet_z_out.npy", "Z projection")
    allPass = allPass && p4

    // --- B projection ---
    print("\n5. B projection")
    let g5 = MPSGraph()
    let (bW, bWShape) = loader.loadTensor("\(refDir)/weights/model_layers_0_linear_attn_in_proj_b_weight.npy")
    let pBI = g5.placeholder(shape: qkvIn.shape, dataType: .float32, name: "x")
    let pBW = g5.placeholder(shape: bW.shape, dataType: .float32, name: "w")
    let b2d = g5.reshape(pBI, shape: [1 as NSNumber, bWShape[1] as NSNumber], name: nil)
    let bWT = g5.transposeTensor(pBW, dimension: 0, withDimension: 1, name: nil)
    let bmm = g5.matrixMultiplication(primary: b2d, secondary: bWT, name: nil)
    let bOut = g5.reshape(bmm, shape: [1 as NSNumber, 1 as NSNumber, bWShape[0] as NSNumber], name: nil)
    let p5 = verify(g5, [pBI: qkvIn, pBW: bW], bOut,
                     "\(refDir)/activations/layer_0_deltanet_b_out.npy", "B projection")
    allPass = allPass && p5

    // --- A projection ---
    print("\n6. A projection")
    let g6 = MPSGraph()
    let (aW, aWShape) = loader.loadTensor("\(refDir)/weights/model_layers_0_linear_attn_in_proj_a_weight.npy")
    let pAI = g6.placeholder(shape: qkvIn.shape, dataType: .float32, name: "x")
    let pAW = g6.placeholder(shape: aW.shape, dataType: .float32, name: "w")
    let a2d = g6.reshape(pAI, shape: [1 as NSNumber, aWShape[1] as NSNumber], name: nil)
    let aWT = g6.transposeTensor(pAW, dimension: 0, withDimension: 1, name: nil)
    let amm = g6.matrixMultiplication(primary: a2d, secondary: aWT, name: nil)
    let aOut = g6.reshape(amm, shape: [1 as NSNumber, 1 as NSNumber, aWShape[0] as NSNumber], name: nil)
    let p6 = verify(g6, [pAI: qkvIn, pAW: aW], aOut,
                     "\(refDir)/activations/layer_0_deltanet_a_out.npy", "A projection")
    allPass = allPass && p6

    // --- Summary ---
    print("\n=== FP32 Summary ===")
    print("  All pass: \(allPass ? "YES ✓" : "NO ✗")")
    if allPass {
        print("  All ops match PyTorch within FP32 tolerance (\(tol))")
        print("  Safe to proceed to FP16 conversion")
    }
}
