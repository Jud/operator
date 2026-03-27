// Verify the full DeltaNet decode step as a black box:
//   Inputs:  qkv_out, z_out, b_out, a_out, conv_state, rec_state, weights
//   Output:  out_proj_input (before out_proj linear)
//
// Covers: conv1d update, QKV split, L2 norm, gates, recurrence, group norm + z gate

import Foundation
import Metal
import MetalPerformanceShadersGraph

func verifyDeltaNetBlackBox(refDir: String) {
    let device = MTLCreateSystemDefaultDevice()!
    let commandQueue = device.makeCommandQueue()!
    let loader = NpyLoader(device: device)

    print("=== DeltaNet Black Box (layer 0) ===\n")

    let graph = MPSGraph()

    // --- Load all inputs ---
    let (qkvOut, _) = loader.loadTensor("\(refDir)/activations/layer_0_deltanet_qkv_out.npy")
    let (zOut, _) = loader.loadTensor("\(refDir)/activations/layer_0_deltanet_z_out.npy")
    let (bOut, _) = loader.loadTensor("\(refDir)/activations/layer_0_deltanet_b_out.npy")
    let (aOut, _) = loader.loadTensor("\(refDir)/activations/layer_0_deltanet_a_out.npy")
    let (convState, _) = loader.loadTensor("\(refDir)/cache_state/conv_state_0.npy")
    let (recState, _) = loader.loadTensor("\(refDir)/cache_state/rec_state_0.npy")

    // Weights
    let (convW, _) = loader.loadTensor("\(refDir)/weights/model_layers_0_linear_attn_conv1d_weight.npy")
    let (aLogW, _) = loader.loadTensor("\(refDir)/weights/model_layers_0_linear_attn_A_log.npy")
    let (dtBiasW, _) = loader.loadTensor("\(refDir)/weights/model_layers_0_linear_attn_dt_bias.npy")
    let (normW, _) = loader.loadTensor("\(refDir)/weights/model_layers_0_linear_attn_norm_weight.npy")

    // Reference output
    let (outRefBuf, _, outCount) = loader.load("\(refDir)/activations/layer_0_deltanet_out_in_0.npy")

    // --- Placeholders ---
    let pQKV = graph.placeholder(shape: [1, 1, 6144], dataType: .float16, name: "qkv")
    let pZ = graph.placeholder(shape: [1, 1, 2048], dataType: .float16, name: "z")
    let pB = graph.placeholder(shape: [1, 1, 16], dataType: .float16, name: "b")
    let pA = graph.placeholder(shape: [1, 1, 16], dataType: .float16, name: "a")
    let pConvState = graph.placeholder(shape: [1, 6144, 4], dataType: .float16, name: "conv_state")
    let pRecState = graph.placeholder(shape: [1, 16, 128, 128], dataType: .float16, name: "rec_state")
    let pConvW = graph.placeholder(shape: [6144, 1, 4], dataType: .float16, name: "conv_w")
    let pALog = graph.placeholder(shape: [16], dataType: .float16, name: "a_log")
    let pDtBias = graph.placeholder(shape: [16], dataType: .float16, name: "dt_bias")
    let pNormW = graph.placeholder(shape: [128], dataType: .float16, name: "norm_w")

    // --- Step 1: Conv1d update ---
    // QKV: [1, 1, 6144] → transpose → [1, 6144, 1]
    let qkvT = graph.transposeTensor(pQKV, dimension: 1, withDimension: 2, name: nil)
    // Shift conv_state left, append qkvT
    let stateSlice = graph.sliceTensor(pConvState, dimension: 2, start: 1, length: 3, name: nil)
    let newConvState = graph.concatTensors([stateSlice, qkvT], dimension: 2, name: nil)
    // Depthwise conv: element-wise multiply [1, 6144, 4] × [6144, 1, 4] squeezed
    let cwSq = graph.reshape(pConvW, shape: [1 as NSNumber, 6144 as NSNumber, 4 as NSNumber], name: nil)
    let convProd = graph.multiplication(newConvState, cwSq, name: nil)
    let convSum = graph.reductionSum(with: convProd, axis: 2, name: nil)  // [1, 6144, 1]
    // SiLU
    let convSig = graph.sigmoid(with: convSum, name: nil)
    let convSilu = graph.multiplication(convSum, convSig, name: nil)  // [1, 6144, 1]
    // Transpose back: [1, 6144, 1] → [1, 1, 6144]
    let convOut = graph.transposeTensor(convSilu, dimension: 1, withDimension: 2, name: nil)

    // --- Step 2: Split Q/K/V and reshape ---
    // [1, 1, 6144] → q[1,1,2048], k[1,1,2048], v[1,1,2048]
    let q_raw = graph.sliceTensor(convOut, dimension: 2, start: 0, length: 2048, name: nil)
    let k_raw = graph.sliceTensor(convOut, dimension: 2, start: 2048, length: 2048, name: nil)
    let v_raw = graph.sliceTensor(convOut, dimension: 2, start: 4096, length: 2048, name: nil)

    // Reshape to [1, 1, 16, 128]
    let headShape: [NSNumber] = [1, 1, 16, 128]
    let q4d = graph.reshape(q_raw, shape: headShape, name: nil)
    let k4d = graph.reshape(k_raw, shape: headShape, name: nil)
    let v4d = graph.reshape(v_raw, shape: headShape, name: nil)

    // --- Step 3: L2 normalize Q and K ---
    // normalize(x, dim=-1): x / max(||x||₂, eps)
    func l2Normalize(_ x: MPSGraphTensor) -> MPSGraphTensor {
        let f32 = graph.cast(x, to: .float32, name: nil)
        let sq = graph.multiplication(f32, f32, name: nil)
        let sumSq = graph.reductionSum(with: sq, axis: -1, name: nil)  // sum along head_dim
        let norm = graph.squareRoot(
            with: graph.addition(sumSq, graph.constant(1e-12, dataType: .float32), name: nil),
            name: nil
        )
        let normalized = graph.division(f32, norm, name: nil)
        return graph.cast(normalized, to: .float16, name: nil)
    }
    let qNorm = l2Normalize(q4d)
    let kNorm = l2Normalize(k4d)

    // --- Step 4: Gates ---
    // beta = sigmoid(b) → [1, 1, 16]
    let beta = graph.sigmoid(with: pB, name: nil)
    // g = -exp(A_log) * softplus(a + dt_bias) → [1, 1, 16], computed in FP32
    let aF32 = graph.cast(pA, to: .float32, name: nil)
    let aLogF32 = graph.cast(pALog, to: .float32, name: nil)
    let dtBiasF32 = graph.cast(pDtBias, to: .float32, name: nil)
    let negExpALog = graph.negative(with: graph.exponent(with: aLogF32, name: nil), name: nil)
    let softplusArg = graph.addition(aF32, dtBiasF32, name: nil)
    let softplus = graph.logarithm(
        with: graph.addition(
            graph.exponent(with: softplusArg, name: nil),
            graph.constant(1.0, dataType: .float32),
            name: nil
        ),
        name: nil
    )
    let gGate = graph.multiplication(negExpALog, softplus, name: nil)  // [1, 1, 16] FP32

    // --- Step 5: Recurrence (single step) ---
    // All in FP32
    // Transpose to [batch, heads, seq, head_dim]: [1, 16, 1, 128]
    let qF32 = graph.cast(
        graph.transposeTensor(qNorm, dimension: 1, withDimension: 2, name: nil),
        to: .float32,
        name: nil
    )
    let kF32 = graph.cast(
        graph.transposeTensor(kNorm, dimension: 1, withDimension: 2, name: nil),
        to: .float32,
        name: nil
    )
    let vF32 = graph.cast(
        graph.transposeTensor(v4d, dimension: 1, withDimension: 2, name: nil),
        to: .float32,
        name: nil
    )
    let betaF32 = graph.cast(
        graph.transposeTensor(beta, dimension: 1, withDimension: 2, name: nil),
        to: .float32,
        name: nil
    )
    // g is already [1, 1, 16] FP32, transpose to [1, 16, 1]
    let gT = graph.transposeTensor(gGate, dimension: 1, withDimension: 2, name: nil)

    // Scale query
    let scale = graph.constant(1.0 / 128.0.squareRoot(), dataType: .float32)
    let qScaled = graph.multiplication(qF32, scale, name: nil)  // [1, 16, 1, 128]

    // state = rec_state (FP32) [1, 16, 128, 128]
    let stateF32 = graph.cast(pRecState, to: .float32, name: nil)

    // g_t = exp(g) [1, 16, 1] → [1, 16, 1, 1] for broadcast
    let gExp = graph.exponent(with: gT, name: nil)  // [1, 16, 1]
    let gExp4d = graph.reshape(gExp, shape: [1 as NSNumber, 16 as NSNumber, 1 as NSNumber, 1 as NSNumber], name: nil)

    // state = state * g_t
    let stateDecayed = graph.multiplication(stateF32, gExp4d, name: nil)

    // kv_mem = sum(state * k, dim=-2) = (state @ k) squeezed
    // state: [1, 16, 128, 128], k: [1, 16, 1, 128] → need k as [1, 16, 128, 1]?
    // Actually: kv_mem = sum(state * k.unsqueeze(-1), dim=-2)
    // k: [1, 16, 1, 128] → unsqueeze(-1) → [1, 16, 1, 128, 1]... no
    // Looking at the code: kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
    // k_t: [1, 16, 128] (squeezed from [1, 16, 1, 128])
    // k_t.unsqueeze(-1): [1, 16, 128, 1]
    // state * k_t.unsqueeze(-1): [1, 16, 128, 128] * [1, 16, 128, 1] → [1, 16, 128, 128]
    // .sum(dim=-2): [1, 16, 128]

    // Squeeze seq dim: [1, 16, 1, 128] → [16, 128]
    let kSq = graph.reshape(kF32, shape: [16 as NSNumber, 128 as NSNumber], name: nil)
    let vSq = graph.reshape(vF32, shape: [16 as NSNumber, 128 as NSNumber], name: nil)
    let qSq = graph.reshape(qScaled, shape: [16 as NSNumber, 128 as NSNumber], name: nil)
    let betaSq = graph.reshape(betaF32, shape: [16 as NSNumber, 1 as NSNumber], name: nil)

    // Reshape state: [1, 16, 128, 128] → [16, 128, 128]
    let state3d = graph.reshape(stateDecayed, shape: [16 as NSNumber, 128 as NSNumber, 128 as NSNumber], name: nil)

    // kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)
    // Equivalent to: kv_mem[h] = state[h].T @ k[h] (matmul)
    // k: [16, 128] → [16, 128, 1]
    let kCol = graph.reshape(kSq, shape: [16 as NSNumber, 128 as NSNumber, 1 as NSNumber], name: nil)
    // state.T @ k: [16, 128, 128] @ [16, 128, 1] → [16, 128, 1]
    let kvMemCol = graph.matrixMultiplication(
        primary: graph.transposeTensor(state3d, dimension: 1, withDimension: 2, name: nil),
        secondary: kCol,
        name: nil
    )
    let kvMem = graph.reshape(kvMemCol, shape: [16 as NSNumber, 128 as NSNumber], name: nil)

    // delta = (v - kv_mem) * beta
    let vMinusKV = graph.subtraction(vSq, kvMem, name: nil)
    let delta = graph.multiplication(vMinusKV, betaSq, name: nil)  // [16, 128]

    // state = state + outer(k, delta)
    // outer(k, delta) = k.unsqueeze(-1) @ delta.unsqueeze(-2)
    // k: [16, 128, 1], delta: [16, 1, 128]
    let deltaRow = graph.reshape(delta, shape: [16 as NSNumber, 1 as NSNumber, 128 as NSNumber], name: nil)
    let outerKD = graph.matrixMultiplication(primary: kCol, secondary: deltaRow, name: nil)  // [16, 128, 128]
    let stateUpdated = graph.addition(state3d, outerKD, name: nil)

    // output = (state * q.unsqueeze(-1)).sum(dim=-2)
    // Equivalent to: output[h] = state[h].T @ q[h]
    let qCol = graph.reshape(qSq, shape: [16 as NSNumber, 128 as NSNumber, 1 as NSNumber], name: nil)
    let outCol = graph.matrixMultiplication(
        primary: graph.transposeTensor(stateUpdated, dimension: 1, withDimension: 2, name: nil),
        secondary: qCol,
        name: nil
    )  // [16, 128, 1]
    let recOut = graph.reshape(outCol, shape: [16 as NSNumber, 128 as NSNumber], name: nil)

    // --- Step 6: Group norm + z gating ---
    // recOut already [16, 128] and FP32
    // RMSNorm per head (Qwen variant with 1+weight)
    let recF32 = recOut  // already FP32
    let normWF32 = graph.cast(pNormW, to: .float32, name: nil)
    let normWPlusOne = graph.addition(normWF32, graph.constant(1.0, dataType: .float32), name: nil)
    let sq = graph.multiplication(recF32, recF32, name: nil)
    let variance = graph.mean(of: sq, axes: [-1], name: nil)
    let eps = graph.constant(1e-6, dataType: .float32)
    let rsqrt = graph.reverseSquareRoot(with: graph.addition(variance, eps, name: nil), name: nil)
    let normed = graph.multiplication(graph.multiplication(recF32, rsqrt, name: nil), normWPlusOne, name: nil)

    // z gating: output * silu(z)
    // z: [1, 1, 2048] → reshape to [16, 128]
    let zFlat = graph.reshape(pZ, shape: [16 as NSNumber, 128 as NSNumber], name: nil)
    let zF32 = graph.cast(zFlat, to: .float32, name: nil)
    let zSig = graph.sigmoid(with: zF32, name: nil)
    let zSilu = graph.multiplication(zF32, zSig, name: nil)
    let gated = graph.multiplication(normed, zSilu, name: nil)

    // Cast back to FP16, reshape to [1, 1, 2048]
    let finalOut = graph.reshape(
        graph.cast(gated, to: .float16, name: nil),
        shape: [1 as NSNumber, 1 as NSNumber, 2048 as NSNumber],
        name: "deltanet_out"
    )

    // --- Run ---
    let feeds: [MPSGraphTensor: MPSGraphTensorData] = [
        pQKV: qkvOut, pZ: zOut, pB: bOut, pA: aOut,
        pConvState: convState, pRecState: recState,
        pConvW: convW, pALog: aLogW, pDtBias: dtBiasW, pNormW: normW
    ]

    let result = graph.run(with: commandQueue, feeds: feeds, targetTensors: [finalOut], targetOperations: nil)
    let actBuf = device.makeBuffer(length: outCount * 2, options: .storageModeShared)!
    result[finalOut]!.mpsndarray().readBytes(actBuf.contents(), strideBytes: nil)

    let pass = compareBuffers(
        actBuf,
        outRefBuf,
        count: outCount,
        label: "DeltaNet black box → out_proj input",
        tolerance: 0.5
    )

    print("\n  Result: \(pass ? "PASS ✓" : "FAIL ✗")")
    if !pass {
        print("  Note: tolerance set to 0.5 for FP16 accumulation across complex ops")
    }
}
