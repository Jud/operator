// Verify MPSGraph layer 0 against PyTorch reference activations.
//
// Layer 0 is a DeltaNet layer:
//   1. Embedding lookup
//   2. RMSNorm (input_layernorm)
//   3. QKV projection (in_proj_qkv): [2048] → [6144]
//
// Run: swift run VerifyMPSGraph

import Foundation
import Metal
import MetalPerformanceShadersGraph

func verifyLayer0(refDir: String) {
    let device = MTLCreateSystemDefaultDevice()!
    let commandQueue = device.makeCommandQueue()!
    let loader = NpyLoader(device: device)

    print("=== Verifying Layer 0 (DeltaNet) ===\n")

    // --- Step 1: Verify embedding ---
    print("Step 1: Embedding lookup")

    let graph1 = MPSGraph()

    // Load reference — use token_id.npy (int32), not embed_in_0.npy (FP16 overflow)
    let (embedIn, _) = loader.loadTensor("\(refDir)/decode_inputs/token_id.npy", dataType: .int32)
    let (embedRefOut, embedShape) = loader.loadTensor("\(refDir)/activations/embed_out.npy")
    let (embedWeight, embedWShape) = loader.loadTensor("\(refDir)/weights/model_embed_tokens_weight.npy")

    // Build embedding graph: gather(weight, indices)
    // token_id is [1], reshape to [1, 1] for consistency
    let inputIds = graph1.placeholder(
        shape: [1 as NSNumber],
        dataType: .int32,
        name: "input_ids"
    )
    let embedW = graph1.placeholder(
        shape: embedWeight.shape,
        dataType: .float16,
        name: "embed_weight"
    )

    // Flatten input for gather, then reshape back
    let flatIds = graph1.reshape(inputIds, shape: [-1 as NSNumber], name: nil)
    let flatIds32 = graph1.cast(flatIds, to: .int32, name: nil)
    let gathered = graph1.gatherND(
        withUpdatesTensor: embedW,
        indicesTensor: graph1.reshape(flatIds32, shape: [-1 as NSNumber, 1], name: nil),
        batchDimensions: 0,
        name: "embed_gather"
    )
    let embedOut = graph1.reshape(
        gathered,
        shape: embedRefOut.shape,
        name: "embed_out"
    )

    let result1 = graph1.run(
        with: commandQueue,
        feeds: [inputIds: embedIn, embedW: embedWeight],
        targetTensors: [embedOut],
        targetOperations: nil
    )

    // Compare
    let embedActual = result1[embedOut]!
    let embedActualBuf = device.makeBuffer(length: embedShape.reduce(1, *) * 2, options: .storageModeShared)!
    embedActual.mpsndarray().readBytes(embedActualBuf.contents(), strideBytes: nil)

    let (embedExpectedBuf, _, embedCount) = loader.load("\(refDir)/activations/embed_out.npy")
    let embedPass = compareBuffers(embedActualBuf, embedExpectedBuf, count: embedCount, label: "Embedding")

    // --- Step 2: Verify RMSNorm ---
    print("\nStep 2: RMSNorm (layer 0 input_layernorm)")

    let graph2 = MPSGraph()

    // Load reference
    let (normIn, normInShape) = loader.loadTensor("\(refDir)/activations/layer_0_input_norm_in_0.npy")
    let (normRefBuf, _, normCount) = loader.load("\(refDir)/activations/layer_0_input_norm_out.npy")
    let (normWeight, _) = loader.loadTensor("\(refDir)/weights/model_layers_0_input_layernorm_weight.npy")

    // Build RMSNorm: x * weight / sqrt(mean(x^2) + eps)
    let normX = graph2.placeholder(shape: normIn.shape, dataType: .float16, name: "x")
    let normW = graph2.placeholder(shape: normWeight.shape, dataType: .float16, name: "weight")

    // RMSNorm (Qwen3_5 variant): output = x * rsqrt(mean(x^2) + eps) * (1 + weight)
    // All in FP32, cast result to FP16 at the end
    let xf32 = graph2.cast(normX, to: .float32, name: nil)
    let wf32 = graph2.cast(normW, to: .float32, name: nil)
    let one = graph2.constant(1.0, dataType: .float32)
    let wPlusOne = graph2.addition(wf32, one, name: "w_plus_one")
    let squared = graph2.multiplication(xf32, xf32, name: "squared")
    let variance = graph2.mean(of: squared, axes: [-1], name: "variance")
    let eps = graph2.constant(1e-6, dataType: .float32)
    let varPlusEps = graph2.addition(variance, eps, name: "var_eps")
    let rsqrt = graph2.reverseSquareRoot(with: varPlusEps, name: "rsqrt")
    let normalized = graph2.multiplication(xf32, rsqrt, name: "normalized")
    let scaled = graph2.multiplication(normalized, wPlusOne, name: "scaled")
    let normOut = graph2.cast(scaled, to: .float16, name: "rms_norm_out")

    let result2 = graph2.run(
        with: commandQueue,
        feeds: [normX: normIn, normW: normWeight],
        targetTensors: [normOut],
        targetOperations: nil
    )

    let normActualBuf = device.makeBuffer(length: normCount * 2, options: .storageModeShared)!
    result2[normOut]!.mpsndarray().readBytes(normActualBuf.contents(), strideBytes: nil)
    let normPass = compareBuffers(normActualBuf, normRefBuf, count: normCount, label: "RMSNorm")

    // --- Step 3: Verify QKV projection (linear) ---
    print("\nStep 3: QKV projection (in_proj_qkv)")

    let graph3 = MPSGraph()

    let (qkvIn, _) = loader.loadTensor("\(refDir)/activations/layer_0_deltanet_qkv_in_0.npy")
    let (qkvRefBuf, _, qkvCount) = loader.load("\(refDir)/activations/layer_0_deltanet_qkv_out.npy")
    let (qkvWeight, qkvWShape) = loader.loadTensor(
        "\(refDir)/weights/model_layers_0_linear_attn_in_proj_qkv_weight.npy"
    )

    // Build linear: y = x @ W.T (no bias on this layer)
    let linX = graph3.placeholder(shape: qkvIn.shape, dataType: .float16, name: "x")
    let linW = graph3.placeholder(shape: qkvWeight.shape, dataType: .float16, name: "weight")

    // W is [6144, 2048], x is [1, 1, 2048]
    let x2d = graph3.reshape(linX, shape: [1 as NSNumber, qkvWShape[1] as NSNumber], name: nil)
    let wT = graph3.transposeTensor(linW, dimension: 0, withDimension: 1, name: nil)
    let matmul = graph3.matrixMultiplication(primary: x2d, secondary: wT, name: "qkv_matmul")
    // Reshape to [1, 1, 6144]
    let outShape: [NSNumber] = [1, 1, qkvWShape[0] as NSNumber]
    let qkvOut = graph3.reshape(matmul, shape: outShape, name: "qkv_out")

    let result3 = graph3.run(
        with: commandQueue,
        feeds: [linX: qkvIn, linW: qkvWeight],
        targetTensors: [qkvOut],
        targetOperations: nil
    )

    let qkvActualBuf = device.makeBuffer(length: qkvCount * 2, options: .storageModeShared)!
    result3[qkvOut]!.mpsndarray().readBytes(qkvActualBuf.contents(), strideBytes: nil)
    let qkvPass = compareBuffers(qkvActualBuf, qkvRefBuf, count: qkvCount, label: "QKV projection")

    // --- Summary ---
    print("\n=== Layer 0 Summary ===")
    let allPass = embedPass && normPass && qkvPass
    print("  Embedding: \(embedPass ? "PASS" : "FAIL")")
    print("  RMSNorm:   \(normPass ? "PASS" : "FAIL")")
    print("  QKV proj:  \(qkvPass ? "PASS" : "FAIL")")
    print("  Overall:   \(allPass ? "ALL PASS ✓" : "FAILED ✗")")
}
