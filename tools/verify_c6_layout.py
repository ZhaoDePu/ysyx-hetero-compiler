"""Check physical byte reindex output in frozen Z15 RTL."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from hetero_compiler.c6_layout import plan_layout
from hetero_compiler.c6_memory import Lifetime, reserve_ddr
from hetero_compiler.c6_schedule import emit_program, emit_rtl_control


def build_fixture(directory):
    shape = (1, 32, 10, 10)
    physical = np.random.default_rng(62119).integers(-128, 128, size=(1, 10, 10, 32), dtype=np.int8)
    packed = physical.transpose(0, 3, 1, 2).reshape(1, 2, 16, 100)
    swapped = packed.swapaxes(2, 3)
    indices = (2, 0, 3, 1)
    generic = swapped.transpose(indices)
    plans = [
        plan_layout("nhwc_to_qkv", "Reshape", shape, packed.shape, input_layout="NHWC"),
        plan_layout("swap_heads", "Transpose", packed.shape, swapped.shape, perm=(0, 1, 3, 2)),
        plan_layout("generic", "Transpose", swapped.shape, generic.shape, perm=indices),
    ]
    initial = {"input": physical.tobytes()}
    expected = {"packed": packed.tobytes(), "swapped": swapped.tobytes(), "generic": generic.tobytes()}
    for i, plan in enumerate(plans):
        if plan.constant_data:
            initial[f"index_{i}"] = plan.constant_data
    slots, _ = reserve_ddr(Lifetime(n, len(blob), 0, 1) for n, blob in (initial | expected).items())
    names = ["input", *expected]
    layers = []
    for i, plan in enumerate(plans):
        constant = slots.get(f"index_{i}")
        layers.append(plan.tiles(slots[names[i]].offset, slots[names[i + 1]].offset,
                                 constant.offset if constant else None))
    graph = {n: {"shape": list(a.shape), "dtype": "int8"} for n, a in
             zip(names, (physical, packed, swapped, generic), strict=True)}
    manifest = emit_program(directory, layers, graph, ["input"], list(expected), slots, initial,
                            external_inputs=False)
    return emit_rtl_control(directory, manifest, initial, expected), {
        "scratch_peak_by_half": manifest["plan"]["scratch_peak_by_half"],
        "dma_bytes": manifest["plan"]["dma_bytes"],
        "stages": len(manifest["plan"]["stages"]),
        "checked_bytes": sum(map(len, expected.values())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner", type=Path)
    args = parser.parse_args()
    control, metrics = build_fixture(args.output)
    print(json.dumps(metrics), flush=True)
    if args.runner:
        subprocess.run([args.runner, control], check=True)


if __name__ == "__main__":
    main()
