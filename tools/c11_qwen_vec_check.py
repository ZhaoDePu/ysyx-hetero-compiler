"""Replay native RMSNorm and SwiGLU using official Layer-0 operands."""

import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from hetero_compiler.npu import _z15_ewise
from hetero_compiler.profile import load_profile
from hetero_compiler.yhb import emit_yhb
from tools.c11_qwen_export import HIDDEN, INTER
from tools.c11_qwen_gqa_check import quantize


A, B, INTERMEDIATE, OUT = 0, 0x2000, 0x4000, 0x6000


def receipts(weights, captured):
    meta = json.loads(weights.with_name("w4a8.json").read_text())
    gamma = []
    for name in ("input_layernorm", "post_attention_layernorm"):
        weight = meta["tensor_extents"][f"model.layers.0.{name}.weight"]
        table = np.memmap(weights, mode="r", dtype="<i4", offset=weight["offset"], shape=(HIDDEN,))
        gamma.append(quantize(table))
    data = np.fromfile(captured, dtype="<i4").reshape(10, 4 * HIDDEN + 3 * INTER)
    for pos, row in enumerate(data):
        x = row[:2 * HIDDEN].reshape(2, HIDDEN)
        norms = row[2 * HIDDEN:4 * HIDDEN].reshape(2, HIDDEN)
        for n in range(2):
            src, xs = quantize(x[n])
            multiplier, gs = gamma[n]
            _, dst_scale = quantize(norms[n])
            shift = 16
            gain = round(math.sqrt(HIDDEN) * gs / dst_scale * (1 << (shift - 7)))
            while gain > 65535 and shift:
                shift -= 1
                gain = round(math.sqrt(HIDDEN) * gs / dst_scale * (1 << (shift - 7)))
            epsilon = round(4295 * HIDDEN / (xs * xs))
            yield (f"p{pos}/rms{n}", "rms", src, multiplier, gain, epsilon, shift,
                   None, dst_scale / 65536, norms[n])
        gate, up, swiglu = row[4 * HIDDEN:].reshape(3, INTER)
        gate8, gs = quantize(gate)
        up8, us = quantize(up)
        candidates = np.arange(256, dtype=np.uint8).view(np.int8).astype(np.float64) * gs / 65536
        activated = candidates / (1 + np.exp(-candidates))
        scale = max(1, math.ceil(float(np.abs(activated).max()) * 65536 / 127))
        lut = np.rint(activated * 65536 / scale).clip(-127, 127).astype(np.int8)
        shift = 7
        y_scale = scale * us * (1 << shift) / (1 << 32)
        yield (f"p{pos}/swiglu", "swiglu", gate8, up8, 0, 0, shift, lut, y_scale, swiglu)


def native_bundle(out, items):
    count_input = sum(2 * len(item[2]) for item in items)
    count_output = count_input // 2
    profile = load_profile("fpga_zu15eg_balanced")
    commands, stages, native = [], [], []

    def transfer(direction, ddr, scratch, length):
        commands.extend(({"type": "dma", "direction": direction, "ddr_offset": ddr,
                          "scratch_offset": scratch, "bytes": length}, {"type": "dma-fence"}))

    def stage(label, spec, operands):
        config = {"activation": None, "activation_mode": None, **spec}
        stages.append({"id": label, "zone": 0, "device": "VPU"})
        native.append({"stage": label, "device": "VPU",
                       "writes": _z15_ewise(config, operands)})
        commands.extend(({"type": "launch", "stage": label}, {"type": "fence", "wait_for": label}))

    offset = 0
    for name, mode, left, _, gain, epsilon, shift, lut, _, _ in items:
        width = len(left)
        transfer("load", offset * 2, A, width)
        transfer("load", offset * 2 + width, B, width)
        if mode == "rms":
            stage(name, {"op": 7, "num_bytes": width, "h": 1, "ca": width,
                         "scale": gain, "epsilon": epsilon, "shift": shift},
                  {"a": (A,), "b": (B,), "c": (OUT,)})
        else:
            stage(name + "/silu", {"op": 11, "num_bytes": width, "table": lut.tolist()},
                  {"a": (A,), "c": (INTERMEDIATE,)})
            stage(name + "/mul", {"op": 6, "num_bytes": width, "scale": 1, "shift": shift},
                  {"a": (INTERMEDIATE,), "b": (B,), "c": (OUT,)})
        transfer("store", count_input + offset, OUT, width)
        offset += width

    manifest = {
        "profile": profile, "plan": {"stages": stages, "allocations": {}},
        "graph": {"inputs": ["inputs"], "outputs": ["outputs"],
                  "tensors": {"inputs": {"shape": [count_input], "dtype": "int8"},
                              "outputs": {"shape": [count_output], "dtype": "int8"}}},
        "input_bindings": {"inputs": {"ddr_offset": 0, "size": count_input, "needs_data": True}},
        "output_bindings": {"outputs": {"ddr_offset": count_input, "size": count_output}},
        "backends": {"gpu": {"stages": []}, "npu": {"commands": native}},
        "initial_ddr_extents": [],
        "execution": commands,
    }
    emit_yhb(out, manifest)
    return out / "program.yhb"


def verify(args):
    args.out.mkdir(parents=True, exist_ok=True)
    items = list(receipts(args.weights, args.capture))
    binary = native_bundle(args.out, items)
    fixture, inp, golden, table = [args.out / f"{name}.bin" for name in ("fixture", "input", "gold", "lut")]
    reference = bytearray()
    with fixture.open("wb") as archive:
        for name, mode, left, right, gain, eps, shift, lut, _, _ in items:
            payload = left.tobytes() + right.tobytes()
            archive.write(payload)
            inp.write_bytes(payload)
            if lut is not None:
                table.write_bytes(lut.tobytes())
            subprocess.run([str(args.reference), mode, str(len(left)), str(gain), str(eps), str(shift),
                            str(inp), str(table), str(golden)], check=True)
            reference.extend(golden.read_bytes())
    observed = args.out / "output.rtl.bin"
    with (args.out / "rtl.log").open("w") as log:
        subprocess.run([str(args.runner), str(binary), "--backend", "verilator", "--dump-stats",
                        "--input", f"inputs={fixture}", "--output", f"outputs={observed}"],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    if observed.read_bytes() != reference:
        raise ValueError("VPU byte outputs differ from the C99 native reference")
    offset, errors = 0, []
    actual = np.frombuffer(reference, dtype=np.int8)
    for name, mode, left, _, _, _, _, _, scale, q16 in items:
        size = len(left)
        error = np.abs(actual[offset:offset + size].astype(np.float64) * scale - q16 / 65536)
        errors.append({"sample": name, "operation": mode, "output_bytes": size,
                       "mean_abs_vs_q16": float(np.mean(error)), "max_abs_vs_q16": float(np.max(error))})
        offset += size
    (args.out / "errors.json").write_text(json.dumps(errors, indent=2) + "\n")
    print(f"C11_LAYER0_VPU_RMS_SWIGLU_PASS cases={len(items)} bytes={len(reference)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("weights", "capture", "reference", "runner", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    verify(parser.parse_args())


if __name__ == "__main__":
    main()
