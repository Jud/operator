#!/usr/bin/env python3
"""
Inject a `speed` input into compiled Kokoro CoreML models (.mlmodelc).

Modifies both model.mil (text) and coremldata.bin (binary header) to add
speed-based duration scaling: real_div(pred_dur, speed) → round → clip.

Usage:
    python3 scripts/inject-speed-compiled.py <model.mlmodelc>

Creates a backup at <model.mlmodelc>.bak before modifying in-place.
"""
import os
import re
import shutil
import sys


def inject_mil(mil_path: str, token_dim: int) -> None:
    """Modify model.mil to add speed input and duration scaling ops."""
    with open(mil_path, "r") as f:
        text = f.read()

    # 1. Add speed to function signature
    text = re.sub(
        r"(func main<\w+>\([^)]*tensor<fp32, \[1, 256\]> ref_s)\)",
        r"\1, tensor<fp32, [1]> speed)",
        text,
    )

    # 2. Insert speed ops after pred_dur computation
    speed_ops = f"""
            tensor<fp32, []> speed_clip_alpha = const()[val = tensor<fp32, []>(0x1p+0)];
            tensor<fp32, []> speed_clip_beta = const()[val = tensor<fp32, []>(0x1.ffcp+15)];
            tensor<fp32, [1, {token_dim}]> pred_dur_scaled = real_div(x = pred_dur, y = speed);
            tensor<fp32, [1, {token_dim}]> pred_dur_rounded = round(x = pred_dur_scaled);
            tensor<fp32, [1, {token_dim}]> pred_dur_clamped = clip(alpha = speed_clip_alpha, beta = speed_clip_beta, x = pred_dur_rounded);"""

    # Find the line defining pred_dur (LHS of assignment) and insert speed ops after
    lines = text.split("\n")
    new_lines = []
    inserted = False
    for line in lines:
        new_lines.append(line)
        if not inserted:
            # Match: tensor<fp32, [1, N]> pred_dur = ...
            if re.search(r">\s+pred_dur\s*=\s", line):
                for op_line in speed_ops.strip().split("\n"):
                    new_lines.append(op_line)
                inserted = True

    if not inserted:
        raise RuntimeError("Could not find pred_dur definition in model.mil")

    text = "\n".join(new_lines)

    # 3. Replace pred_dur consumers with pred_dur_clamped
    # reduce_sum and cumsum that consume pred_dur
    text = re.sub(r"x = pred_dur\)", "x = pred_dur_clamped)", text)

    # 4. Update return statement
    text = re.sub(r"-> \(audio, audio_length_samples, pred_dur\);",
                  "-> (audio, audio_length_samples, pred_dur_clamped);", text)

    with open(mil_path, "w") as f:
        f.write(text)


def inject_coremldata(bin_path: str, token_dim: int) -> None:
    """Modify coremldata.bin to add speed input and rename pred_dur output."""
    with open(bin_path, "rb") as f:
        data = bytearray(f.read())

    # Speed input descriptor (protobuf-encoded):
    speed_input = (
        b"\x0a\x12"        # field 1, length 18
        b"\x0a\x05speed"   # name = "speed"
        b"\x1a\x09"        # field 3, length 9
        b"\x2a\x07"        # multiArrayType, length 7
        b"\x0a\x01\x01"    # shape = [1]
        b"\x10\xa0\x80\x04" # dataType = FLOAT32
    )

    # Find where outputs start (field tag 0x52 = field 10, wire type 2)
    output_start = data.index(b"\x52")

    # Insert speed input before first output
    data[output_start:output_start] = speed_input

    # Replace "pred_dur" with "pred_dur_clamped" in output descriptor
    old_output = b"pred_dur"
    new_output = b"pred_dur_clamped"

    # Find pred_dur in output section (after the insertion point)
    search_from = output_start + len(speed_input)
    pos = data.index(old_output, search_from)

    if data[pos:pos + len(new_output)] != new_output:
        size_diff = len(new_output) - len(old_output)

        # Find the 52 tag before this pred_dur
        tag_pos = pos
        while tag_pos > 0 and data[tag_pos] != 0x52:
            tag_pos -= 1

        # Update outer length (52 LL) and inner length (0a LL2)
        data[tag_pos + 1] += size_diff
        inner_tag_pos = tag_pos + 2
        if data[inner_tag_pos] == 0x0a:
            data[inner_tag_pos + 1] += size_diff

        # Replace the name
        data[pos:pos + len(old_output)] = new_output

    # Update the data section length at fixed offset 0x4B (little-endian u16).
    # Verified across kokoro_21_5s, kokoro_24_10s, kokoro_24_10s.speed:
    #   5s orig:  0x0164 (356)
    #   10s orig: 0x0167 (359)
    #   10s speed: 0x0183 (387) — diff = 28 = 20 + 8
    increase = len(speed_input) + (len(new_output) - len(old_output))
    old_len = data[0x4B] | (data[0x4C] << 8)
    new_len = old_len + increase
    data[0x4B] = new_len & 0xFF
    data[0x4C] = (new_len >> 8) & 0xFF

    with open(bin_path, "wb") as f:
        f.write(data)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model.mlmodelc>")
        sys.exit(1)

    model_path = sys.argv[1]

    mil_path = os.path.join(model_path, "model.mil")
    bin_path = os.path.join(model_path, "coremldata.bin")

    if not os.path.exists(mil_path):
        print(f"ERROR: {mil_path} not found")
        sys.exit(1)

    # Read model.mil to determine token dimension
    with open(mil_path, "r") as f:
        header = f.readline() + f.readline() + f.readline() + f.readline()

    # Extract token dim from input_ids shape: tensor<int32, [1, N]>
    m = re.search(r"tensor<int32, \[1, (\d+)\]> input_ids", header)
    if not m:
        print("ERROR: Could not determine token dimension from model.mil")
        sys.exit(1)
    token_dim = int(m.group(1))

    # Check if speed already injected
    if "speed" in header:
        print(f"Speed input already present in {model_path}")
        sys.exit(0)

    print(f"Model: {model_path}")
    print(f"Token dimension: {token_dim}")

    # Create backup
    backup_path = model_path + ".pre-speed"
    if not os.path.exists(backup_path):
        shutil.copytree(model_path, backup_path)
        print(f"Backup: {backup_path}")

    # Inject into model.mil
    inject_mil(mil_path, token_dim)
    print("Injected speed ops into model.mil")

    # Inject into coremldata.bin
    inject_coremldata(bin_path, token_dim)
    print("Updated coremldata.bin header")

    # Verify
    with open(mil_path, "r") as f:
        new_header = ""
        for _ in range(5):
            new_header += f.readline()
    if "speed" in new_header and "pred_dur_clamped" in new_header:
        print("Verification: speed input found in modified model.mil")
    else:
        print("WARNING: Verification failed — check model.mil manually")


if __name__ == "__main__":
    main()
