#!/usr/bin/env python3
"""
Inject a `speed` input into Kokoro unified CoreML models.

Modifies the model graph to scale predicted durations by 1/speed before
alignment, enabling native speech rate control (0.5x-2.0x).

Usage:
    python3 scripts/inject-speed.py <input.mlpackage> <output.mlpackage>
    xcrun coremlc compile <output.mlpackage> <output_dir>

Requires: pip install coremltools (Python 3.12)
"""
import sys
import os
import shutil


FP16 = 11  # CoreML protobuf dataType for float16


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.mlpackage> <output.mlpackage>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    import coremltools as ct
    from coremltools.models.utils import load_spec, save_spec
    from coremltools.proto import MIL_pb2

    print(f"Loading {input_path}...")
    spec = load_spec(input_path)

    # --- 1. Add 'speed' to model description inputs (FLOAT32 for user API) ---
    speed_input = spec.description.input.add()
    speed_input.name = "speed"
    speed_input.type.multiArrayType.shape.append(1)
    speed_input.type.multiArrayType.dataType = 65568  # FLOAT32

    # --- 2. Add 'speed' as function input (FP16 to match model internals) ---
    func = spec.mlProgram.functions["main"]
    speed_func_input = func.inputs.add()
    speed_func_input.name = "speed"
    speed_func_input.type.tensorType.dataType = FP16
    speed_func_input.type.tensorType.rank = 1
    speed_func_input.type.tensorType.dimensions.add().constant.size = 1

    block_name = list(func.block_specializations.keys())[0]
    block = func.block_specializations[block_name]
    print(f"Using block specialization: {block_name}")

    # --- 3. Find pred_dur op and its consumers ---
    pred_dur_idx = None
    pred_dur_dim = None
    for i, op in enumerate(block.operations):
        for out in op.outputs:
            if out.name == "pred_dur":
                pred_dur_idx = i
                pred_dur_dim = out.type.tensorType.dimensions[1].constant.size
                break
        if pred_dur_idx is not None:
            break

    if pred_dur_idx is None:
        print("ERROR: pred_dur operation not found")
        sys.exit(1)

    print(f"Found pred_dur at op #{pred_dur_idx} (dim={pred_dur_dim})")

    consumer_refs = []
    for i, op in enumerate(block.operations):
        if i <= pred_dur_idx:
            continue
        for inp_name, inp_val in op.inputs.items():
            for j, arg in enumerate(inp_val.arguments):
                if arg.name == "pred_dur":
                    consumer_refs.append((i, inp_name, j))
                    print(f"  Consumer: op #{i} ({op.type}) input '{inp_name}'")

    # --- 4. Build new operations ---
    def make_name_arg(name):
        arg = MIL_pb2.Argument()
        arg.arguments.add().name = name
        return arg

    def make_scalar_const(name, value):
        op = MIL_pb2.Operation()
        op.type = "const"
        val_attr = op.attributes["val"]
        val_attr.type.tensorType.dataType = FP16
        val_attr.type.tensorType.rank = 0
        val_attr.immediateValue.tensor.floats.values.append(value)
        out = op.outputs.add()
        out.name = name
        out.type.tensorType.dataType = FP16
        out.type.tensorType.rank = 0
        return op

    def make_output(name, rank, dims):
        out = MIL_pb2.NamedValueType()
        out.name = name
        out.type.tensorType.dataType = FP16
        out.type.tensorType.rank = rank
        for d in dims:
            out.type.tensorType.dimensions.add().constant.size = d
        return out

    new_ops = []

    # scalar consts for clip bounds
    new_ops.append(make_scalar_const("speed_clip_alpha", 1.0))
    new_ops.append(make_scalar_const("speed_clip_beta", 65504.0))  # FP16 max

    # real_div(pred_dur, speed) -> pred_dur_scaled
    op_div = MIL_pb2.Operation()
    op_div.type = "real_div"
    op_div.inputs["x"].CopyFrom(make_name_arg("pred_dur"))
    op_div.inputs["y"].CopyFrom(make_name_arg("speed"))
    op_div.outputs.append(make_output("pred_dur_scaled", 2, [1, pred_dur_dim]))
    new_ops.append(op_div)

    # round(pred_dur_scaled) -> pred_dur_rounded
    op_round = MIL_pb2.Operation()
    op_round.type = "round"
    op_round.inputs["x"].CopyFrom(make_name_arg("pred_dur_scaled"))
    op_round.outputs.append(make_output("pred_dur_rounded", 2, [1, pred_dur_dim]))
    new_ops.append(op_round)

    # clip(pred_dur_rounded, alpha=1.0, beta=max) -> pred_dur_clamped
    op_clip = MIL_pb2.Operation()
    op_clip.type = "clip"
    op_clip.inputs["x"].CopyFrom(make_name_arg("pred_dur_rounded"))
    op_clip.inputs["alpha"].CopyFrom(make_name_arg("speed_clip_alpha"))
    op_clip.inputs["beta"].CopyFrom(make_name_arg("speed_clip_beta"))
    op_clip.outputs.append(make_output("pred_dur_clamped", 2, [1, pred_dur_dim]))
    new_ops.append(op_clip)

    # --- 5. Insert after pred_dur ---
    insert_idx = pred_dur_idx + 1
    for i, new_op in enumerate(new_ops):
        block.operations.insert(insert_idx + i, new_op)
    num_inserted = len(new_ops)
    print(f"Inserted {num_inserted} ops at #{insert_idx}")

    # --- 6. Update consumers ---
    for orig_idx, inp_name, arg_idx in consumer_refs:
        shifted_idx = orig_idx + num_inserted
        op = block.operations[shifted_idx]
        op.inputs[inp_name].arguments[arg_idx].name = "pred_dur_clamped"
        print(f"  Updated op #{shifted_idx} ({op.type}) '{inp_name}' -> pred_dur_clamped")

    # --- 7. Update block output ---
    for i, out_name in enumerate(block.outputs):
        if out_name == "pred_dur":
            block.outputs[i] = "pred_dur_clamped"

    for out in spec.description.output:
        if out.name == "pred_dur":
            out.name = "pred_dur_clamped"

    # --- 8. Save ---
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    weights_dir = os.path.join(input_path, "Data", "com.apple.CoreML", "weights")
    save_spec(spec, output_path, weights_dir=weights_dir)
    print(f"\nSaved to {output_path}")

    # Verify
    verify_model = ct.models.MLModel(output_path)
    verify_spec = verify_model.get_spec()
    print(f"\nVerification:")
    print(f"  Inputs: {[i.name for i in verify_spec.description.input]}")
    print(f"  Outputs: {[o.name for o in verify_spec.description.output]}")


if __name__ == "__main__":
    main()
