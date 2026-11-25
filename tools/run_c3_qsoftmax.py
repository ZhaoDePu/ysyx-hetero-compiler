"""Verify byte-exact C3 native RV32 Q/DQ Softmax on the frozen Balanced MMIO bridge."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

from examples.generate_quantized_onnx import FLOAT, INT8, UINT8, Model


def build_softmax(shape, scores, *, mask=None, axis=-1, scale=1/256, score_scale=1/16,
                  score_type=INT8, point=0):
    model = Model("standard_onnx_qdq_softmax")
    model.input("scores", shape, score_type, scores)
    model.tensor("score_scale", [], FLOAT, [score_scale])
    model.tensor("score_zero", [], score_type, [point])
    model.tensor("prob_scale", [], FLOAT, [scale])
    model.tensor("prob_zero", [], UINT8, [0])
    logits = model.op("dequant", "DequantizeLinear", ["scores", "score_scale", "score_zero"], "float_scores")
    if mask is not None:
        mask_shape, mask_values = mask
        model.tensor("causal_mask", mask_shape, FLOAT, mask_values)
        logits = model.op("mask", "Add", [logits, "causal_mask"], "float_masked")
    model.op("probability", "Softmax", [logits], "float_probability", axis=axis)
    model.op("quant", "QuantizeLinear", ["float_probability", "prob_scale", "prob_zero"], "probability")
    model.output("probability", shape, UINT8)
    return model.encode()


SCRATCH = Path("/dev/shm/ysyx_compiler_c4a/softmax")
PRIORITY = ["nice", "-n", "10", "ionice", "-c2", "-n7"]


def cases():
    floor = float("-inf")
    mask = [0 if col <= row else floor for row in range(4) for col in range(4)]
    yields = {
        "causal": build_softmax([4, 4], [
            ((23*i) % 223)-103 for i in range(16)
        ], mask=([4, 4], mask)),
        "strided_uint8": build_softmax(
            [2, 3, 4], [(39*i)%256 for i in range(24)], axis=1, score_type=UINT8,
            point=108, score_scale=1/8, scale=1/128),
        "rne_ties": build_softmax([3, 2], [0, 0, 5, 5, -64, -64], scale=1),
        "high_precision": build_softmax(
            [4, 5], [(-45 + 103*i) % 249 - 123 for i in range(20)], score_scale=1/32768),
    }
    yields["finite_mask"] = build_softmax(
        [2, 2], [-15, 127, 1, 2], mask=([2, 2], [0, -1.0e9, 0, 0]))
    for name, model in yields.items():
        yield name, model


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="qsoftmax-rtl-", dir=SCRATCH))
    stats = {}
    for name, blob in cases():
        model = out / (name + ".onnx")
        bundle = out / name
        replay = out / (name + ".rtl")
        model.write_bytes(blob)
        subprocess.run(PRIORITY + [
            sys.executable, "-B", "-m", "hetero_compiler", "--graph", str(model), "--profile",
            "fpga_zu15eg_balanced", "--out", str(bundle)
        ], check=True)
        proc = subprocess.run(PRIORITY + [
            sys.executable, "-B", "-m", "tools.verify_bundle", "--bundle", str(bundle),
            "--case", str(replay), "--execute-plan"
        ], capture_output=True, text=True)
        (out / (name + ".log")).write_text(proc.stdout + proc.stderr)
        print(proc.stdout + proc.stderr, end="", flush=True)
        proc.check_returncode()
        observed, = [line.partition(" ")[2] for line in proc.stdout.splitlines()
                     if line.startswith("Z15_METRICS ")]
        stats[name] = json.loads(observed)
    (out / "result.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"QSOFTMAX_RTL_PASS evidence={out}", flush=True)


if __name__ == "__main__":
    main()
