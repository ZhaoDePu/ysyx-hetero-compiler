"""Build and check a real tied-LM-head projection slice against frozen MMIO."""

import argparse
import json
import mmap
from pathlib import Path
import struct
import subprocess

import numpy as np

from hetero_compiler.c11_w4_projection import COLUMNS, GROUP, lower_projection
from tools.c11_qwen_export import MODEL_SHA256, VOCAB, HIDDEN, WEIGHT_SHIFT, Tensors, fp32, sha256


def open_panels(args):
    receipt = json.loads(args.weights.with_name("w4a8.json").read_text())
    record = receipt["tensor_extents"][receipt["tied_lm_head"]]
    rows, width = record["shape"]
    if (rows, width) != (VOCAB, HIDDEN) or args.first + args.channels > rows:
        raise ValueError("LM head slice exceeds the official embedding shape")
    groups = width // GROUP
    target = slice(args.first, args.first + args.channels)
    if args.variant == "B0":
        with args.original.open("rb") as original, mmap.mmap(original.fileno(), 0, access=mmap.ACCESS_READ) as image:
            buf = fp32(Tensors(image).bf16(receipt["tied_lm_head"], (rows, width))[target])
        values = buf.reshape(args.channels, groups, GROUP).transpose(1, 0, 2)
        scale = np.maximum(1, np.rint(np.abs(values).max(axis=-1).astype(np.float64)
                                      * ((1 << WEIGHT_SHIFT) / 127))).astype("<u4")
        weight = np.rint(values.astype(np.float64) * (1 << WEIGHT_SHIFT) / scale[..., None])
        quantized = weight.clip(-127, 127).astype(np.int8)
        payload = quantized.tobytes()
    else:
        packed = np.memmap(args.weights, dtype=np.uint8, mode="r",
                           offset=record["packed_offset"], shape=(groups, rows, GROUP // 2))
        payload = packed[:, target].tobytes()
        packed = np.frombuffer(payload, np.uint8).reshape(groups, args.channels, GROUP // 2)
        int4 = np.empty((groups, args.channels, GROUP), np.int32)
        int4[..., 0::2] = (packed & 15).astype(np.int32)
        int4[..., 1::2] = (packed >> 4).astype(np.int32)
        quantized = ((int4 ^ 8) - 8).astype(np.int8)
        scale = np.memmap(args.weights, dtype="<u4", mode="r",
                          offset=record["scale_offset"], shape=(groups, rows))[:, target]
    return groups, payload, quantized, scale


def recover(partials, scales, input_scale):
    mul = (partials.astype(np.int64) * scales.astype(np.int64)).sum(axis=0) * input_scale
    fixed = np.sign(mul) * ((np.abs(mul) + (1 << (WEIGHT_SHIFT - 1))) >> WEIGHT_SHIFT)
    return np.clip(fixed, -(1 << 26), 1 << 26).astype("<i4")


def replay(args):
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.lm_input.read_bytes()
    if len(raw) != 4 + HIDDEN:
        raise ValueError("Malformed C99 LM input capture")
    input_scale, = struct.unpack_from("<I", raw)
    groups, payload, quantized, scale = open_panels(args)
    a8 = np.frombuffer(raw[4:], dtype=np.int8).reshape(groups, GROUP)
    expected = np.sum(quantized.astype(np.int32) * a8[:, None, :], axis=-1, dtype=np.int32)
    metrics = lower_projection(args.out, payload, rows=1, groups=groups,
                               channels=args.channels, variant=args.variant)
    (args.out / "activation.bin").write_bytes(a8.tobytes())
    (args.out / "partial.golden.bin").write_bytes(expected.astype("<i4").tobytes())
    (args.out / "compile.json").write_text(json.dumps(metrics, indent=2) + "\n")
    partial_out = args.out / "partial.rtl.bin"
    with (args.out / "rtl.log").open("w") as log:
        subprocess.run([str(args.runner), str(args.out / "program.yhb"), "--backend", "verilator",
                        "--input", f"activation={args.out / 'activation.bin'}",
                        "--output", f"partial={partial_out}", "--dump-stats"],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    partial = np.fromfile(partial_out, dtype="<i4").reshape(expected.shape)
    errors = np.count_nonzero(partial != expected)
    if errors:
        which = np.argwhere(partial != expected)[0]
        raise ValueError(f"{errors} RTL partial sums disagree, first {which.tolist()}")
    output = recover(partial, scale, input_scale)
    if args.variant != "B0":
        reference = np.memmap(args.prefill_q16, mode="r", dtype="<i4", offset=(
            args.prefill_q16.stat().st_size - 4 * VOCAB), shape=(VOCAB,))
        if not np.array_equal(reference[args.first:args.first + args.channels], output):
            raise ValueError("Scaled RTL partial sums did not reproduce C99 logits")
    output.tofile(args.out / "logits.rtl-plus-host-scale.bin")
    metrics.update({"first_channel": args.first, "rtl_partial_bytes_exact": int(expected.nbytes),
                    "host_scaled_logit_bytes": int(output.nbytes),
                    "c99_w4_logits_exact": args.variant != "B0",
                    "rtl_log": "rtl.log"})
    (args.out / "replay.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"C11_LM_SLICE_PASS {args.variant} raw_bytes={expected.nbytes} logits_bytes={output.nbytes}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("weights", "lm-input", "prefill-q16", "runner", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--channels", type=int, default=2048)
    parser.add_argument("--variant", choices=("B0", "B1", "B2"), required=True)
    args = parser.parse_args()
    if args.first < 0 or args.channels < COLUMNS or args.channels % COLUMNS:
        parser.error("channel count must be a positive multiple of 16 and start must be nonnegative")
    if args.variant == "B0" and (not args.original or sha256(args.original) != MODEL_SHA256):
        parser.error("B0 requires the pinned official BF16 weights")
    replay(args)


if __name__ == "__main__":
    main()
