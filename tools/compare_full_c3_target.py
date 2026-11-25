"""Compare both frozen tensor geometries on one standard quantized ONNX model."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from examples.generate_quantized_onnx import Model, INT8, INT32
from hetero_compiler.compiler import compile_bundle, tensor_bytes
from hetero_compiler.ir import normalize
from hetero_compiler.onnx import parse_model
from hetero_compiler.profile import load_profile
from hetero_compiler.reference import evaluate
from hetero_compiler.yhb import inspect_yhb

ROOT = Path("/dev/shm/ysyx_compiler_c4vpu/targets")
PRIORITY = ["nice", "-n", "10", "ionice", "-c2", "-n7"]
SIZES = (72, 72, 64)
CLOCKS = {16: (164.555, 124.239), 24: (153.022, 126.968)}


def model():
    m, n, k = SIZES
    mb = Model("Phase12C_tensor_geometry_72x72x64")
    a = mb.input("x", [m, k], INT8, [((i * 11) % 37) - 18 for i in range(m * k)])
    b = mb.tensor("w", [k, n], INT8, [((i * 7 + i // n) % 19) - 9 for i in range(k * n)])
    mb.output(mb.op("dot", "MatMulInteger", [a, b], "out"), [m, n], INT32)
    return mb.encode()


def compare(path):
    onnx = path / "same-source-matmul.onnx"
    onnx.write_bytes(model())
    raw = parse_model(onnx.read_bytes())
    graph = normalize(raw)
    golden = tensor_bytes(evaluate(graph)["out"], "int32")
    stats = []
    for name, dim in (("fpga_zu15eg_balanced", 16), ("fpga_zu15eg_full", 24)):
        dest = path / name
        profile = load_profile(name)
        bundle = compile_bundle(raw, profile, dest)
        stages = bundle["plan"]["stages"]
        if len(stages) != 1 or stages[0]["device"] != "NPU":
            raise RuntimeError("fixture no longer lowers to one native NPU descriptor")
        command, = bundle["backends"]["npu"]["commands"]
        registers = dict(command["writes"])
        if len(registers) != 11 or registers.get(0x10) != 8:
            raise RuntimeError("fixture must use the frozen GEMM register subset")
        regs = dest / "registers.txt"
        regs.write_text(" ".join(str(registers[i]) for i in range(0x60, 0x88, 4)) + "\n",
                        encoding="ascii")
        initial = dest / "initial.bin"
        expected_image = bytearray(initial.read_bytes())
        offset = bundle["plan"]["allocations"]["out"]["offset"]
        expected_image[offset:offset + len(golden)] = golden
        expected = dest / "expected.bin"
        expected.write_bytes(expected_image)
        run = subprocess.run(PRIORITY + [str(ROOT / f"tensor-{dim}/tensor-sim"), str(regs),
                                         str(initial), str(expected)], capture_output=True, text=True,
                             check=False, timeout=80)
        log = dest / "direct-tensor.log"
        log.write_text(run.stdout + run.stderr, encoding="utf-8")
        if run.returncode or not run.stdout.startswith("C3_TENSOR_PASS"):
            raise RuntimeError(f"{name} frozen tensor check failed: {run.stderr}")
        observed = {key: int(value) for key, value in
                    (item.split("=") for item in run.stdout.split()[1:])}
        timing = CLOCKS[dim]
        yhb = (dest / "program.yhb").read_bytes()
        stats.append({"profile": name, "array": [dim, dim], "scope": "direct tensor RTL",
                      "status": "byte_exact", "npu_fmax_mhz": timing[0], "gpu_fmax_mhz": timing[1],
                      "yhb_target": inspect_yhb(yhb)["target"], "yhb_bytes": len(yhb),
                      **observed, "wall_us_at_npu_fmax": observed["cycles"] / timing[0]})

    result = {"workload": list(SIZES), "graph_sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
              "target_rows": stats, "whole_board_comparison": "unavailable on frozen Full board MMIO",
              "direct_cycle_speedup": stats[0]["cycles"] / stats[1]["cycles"],
              "direct_fmax_speedup": stats[0]["wall_us_at_npu_fmax"] / stats[1]["wall_us_at_npu_fmax"]}
    (path / "c3-full-compare.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    path = args.out.resolve()
    if ROOT not in path.parents or path.exists():
        cli.error("output must be a fresh path below the C3 targets scratch directory")
    path.mkdir()
    compare(path)


if __name__ == "__main__":
    main()
