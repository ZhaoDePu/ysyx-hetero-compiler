"""Run a high-address native ASIC-IP YHB through the functional model."""

import argparse
import json
import struct
from pathlib import Path

from hetero_compiler.asic_ip import replay
from hetero_compiler.compiler import compile_bundle
from hetero_compiler.ir import normalize
from hetero_compiler.profile import load_profile
from hetero_compiler.reference import evaluate
from hetero_compiler.yhb import inspect_yhb


SCRATCH = Path("/dev/shm/ysyx_compiler_c3/targets")


def fixture():
    shape = [336, 64]
    count = shape[0] * shape[1]
    a = [((13 * i + 5) % 127) - 64 for i in range(count)]
    b = [((7 * i * i + 9) % 31) - 15 for i in range(count)]
    rows = [
        ("Add", ["a", "b"], "sum", {}),
        ("Mul", ["sum", "b"], "product", {}),
        ("SiLU", ["product"], "silu", {}),
        ("GELU", ["silu"], "gelu", {}),
        ("RMSNorm", ["gelu"], "normalized", {}),
        ("RowSoftmax", ["normalized"], "probs", {"causal_mask": True}),
    ]
    raw = {"name": "native_64k_shared_vpu",
           "tensors": {n: {"shape": shape, "dtype": "int8", "kind": "input", "data": v}
                       for n, v in (("a", a), ("b", b))},
           "nodes": [{"id": f"s{i}", "op": op, "inputs": args, "output": dst, "attrs": attrs}
                     for i, (op, args, dst, attrs) in enumerate(rows)], "outputs": ["probs"]}
    return raw, {"a": list(reversed(a)), "b": b[1:] + b[:1]}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    if SCRATCH not in args.out.resolve().parents or args.out.exists():
        cli.error("output must be a fresh directory in C3 targets scratch")
    raw, inputs = fixture()
    bundle = compile_bundle(raw, load_profile("asic_nangate45_ip"), args.out)
    path = args.out / "program.yhb"
    code = path.read_bytes()
    measured = replay(code, {name: struct.pack(f"<{len(values)}b", *values)
                              for name, values in inputs.items()})
    expected = bytes(evaluate(normalize(raw), inputs)["probs"])
    if measured != {"probs": expected}:
        raise RuntimeError("native ASIC IP replay differs from the graph golden")
    report = {"status": "byte_exact", "measurement": "software descriptor/arena replay",
              "target_id": inspect_yhb(code)["target"], "yhb_bytes": len(code),
              "arena_bytes": bundle["profile"]["sram"]["capacity"],
              "scratch_live_peak": bundle["plan"]["peak_live_bytes"],
              "scratch_high_water": bundle["plan"]["high_water_bytes"],
              "shared_vpu_stages": [s["op"] for s in bundle["plan"]["stages"]],
              "compared_output_bytes": len(expected)}
    (args.out / "c3-asic-replay.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
