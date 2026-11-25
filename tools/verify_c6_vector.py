"""Replay a tiled vector/layout chain on the frozen Z15 Balanced MMIO shell."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from hetero_compiler.c6_memory import Lifetime, reserve_ddr
from hetero_compiler.c6_schedule import emit_program, emit_rtl_control
from hetero_compiler.c6_vector import vector_tiles


def build_fixture(directory):
    rng = np.random.default_rng(32126)
    initial = {name: rng.integers(-128, 128, size=(1, 7, 8, channels), dtype=np.int8)
               for name, channels in (("a", 16), ("b", 32), ("c", 16))}
    joined = np.concatenate(tuple(initial.values()), axis=3)
    resized = joined.repeat(2, axis=1).repeat(2, axis=2)
    padded = np.pad(resized, ((0, 0), (2, 2), (2, 2), (0, 0)), constant_values=-128)
    pool = np.lib.stride_tricks.sliding_window_view(padded, (5, 5), axis=(1, 2)).max(axis=(-1, -2))
    left, middle, right = np.split(pool, [16, 48], axis=3)
    summed = np.clip(left.astype(np.int16) + right, -128, 127).astype(np.int8)
    multiplied = np.clip((left.astype(np.int32) * right * 13 + 64) >> 7, -128, 127).astype(np.int8)
    codes = np.arange(256, dtype=np.uint8).view(np.int8).astype(np.float64)
    lut = np.clip(np.rint(codes * 1.75 + 3), -128, 127).astype(np.int8)
    result = lut[summed.view(np.uint8)]
    expected = dict(joined=joined, resized=resized, pool=pool, left=left, middle=middle, right=right,
                    summed=summed, multiplied=multiplied, result=result)
    tensors = initial | expected
    slots, _ = reserve_ddr(Lifetime(name, int(arr.size), 0, 1) for name, arr in tensors.items())

    def layer(name, op, inputs, outputs, **attrs):
        shapes = lambda names: [tensors[n].shape for n in names]
        offsets = lambda names: [slots[n].offset for n in names]
        return vector_tiles(name, op, shapes(inputs), shapes(outputs), offsets(inputs), offsets(outputs),
                            **attrs)

    layers = (
        layer("join", "Concat", ["a", "b", "c"], ["joined"], axis=3, tile_rows=3),
        layer("resize", "Resize", ["joined"], ["resized"], tile_rows=2),
        layer("pool", "MaxPool", ["resized"], ["pool"], tile_rows=4),
        layer("split", "Split", ["pool"], ["left", "middle", "right"], axis=3, tile_rows=4),
        layer("add", "EWAdd", ["left", "right"], ["summed"], tile_rows=3),
        layer("mul", "EWMul", ["left", "right"], ["multiplied"], mul=13, shift=7, tile_rows=4),
        layer("requant", "RequantLUT", ["summed"], ["result"], table=lut, tile_rows=5),
    )
    initial_bytes = {name: arr.tobytes() for name, arr in initial.items()}
    expected_bytes = {name: arr.tobytes() for name, arr in expected.items()}
    graph = {name: {"shape": list(arr.shape), "dtype": "int8"} for name, arr in tensors.items()}
    manifest = emit_program(directory, layers, graph, list(initial), list(expected), slots,
                            initial_bytes, external_inputs=False)
    return emit_rtl_control(directory, manifest, initial_bytes, expected_bytes), {
        "scratch_peak_by_half": manifest["plan"]["scratch_peak_by_half"],
        "dma_bytes": manifest["plan"]["dma_bytes"],
        "layers": len(layers), "tiles": sum(map(len, layers)),
        "checked_bytes": sum(map(len, expected_bytes.values())),
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
