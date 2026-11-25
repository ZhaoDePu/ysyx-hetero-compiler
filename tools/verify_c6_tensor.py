#!/usr/bin/env python3
"""Build C6 Conv tiling bundles and independent NumPy RTL replay controls."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from hetero_compiler.c6_memory import Lifetime, reserve_ddr
from hetero_compiler.c6_reduce_conv import combine_tiles, raw_conv_tiles, reduction_widths
from hetero_compiler.c6_schedule import emit_program, emit_rtl_control
from hetero_compiler.c6_tensor import Conv, conv_tiles, pack_weights


def conv_reference(data, weights, bias, mul, shift, pad, stride, lut=None):
    """Independent direct INT32 cross correlation with native floor requant."""
    k = weights.shape[0]
    padded = np.pad(data.astype(np.int32), ((pad, pad), (pad, pad), (0, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k), axis=(0, 1))
    windows = windows[::stride, ::stride].transpose(0, 1, 3, 4, 2)
    acc = windows.reshape(-1, np.prod(weights.shape[:-1])) @ weights.astype(np.int32).reshape(
        -1, weights.shape[-1]) + bias
    scaled = np.clip((acc.astype(np.int64) * mul) >> shift, -128, 127).astype(np.int8)
    if lut is not None:
        scaled = lut[scaled.view(np.uint8)]
    return scaled.reshape(*windows.shape[:2], weights.shape[-1])


def build_fixture(directory, spec, seed=62026):
    rng = np.random.default_rng(seed)
    nx, ny = spec.height, spec.width
    ci, co = spec.input_channels, spec.output_channels
    data = rng.integers(-60, 61, size=(nx, ny, ci), dtype=np.int8)
    filt = rng.integers(-12, 13, size=(spec.kernel, spec.kernel, ci, co), dtype=np.int8)
    bias = rng.integers(-275, 275, size=co, dtype=np.int32)
    mul = rng.integers(21, 70, size=co, dtype=np.int64)
    shift = rng.integers(9, 13, size=co, dtype=np.int64)
    codes = np.arange(256, dtype=np.uint8).view(np.int8).astype(np.float64)
    lut = np.rint(codes / (1 + np.exp(-codes / 21))).astype(np.int8)
    input_image = np.zeros((1, nx, ny, spec.padded_channels), dtype=np.int8)
    input_image[0, :, :, :ci] = data
    output = conv_reference(data, filt, bias, mul, shift, spec.pad, spec.stride, lut)
    initial = {"images": input_image.tobytes(), "bias": bias.astype("<i4").tobytes()}
    partial_size = {}
    if spec.depth <= 511:
        initial["weights"] = pack_weights(spec, filt)
    else:
        widths = reduction_widths(spec)
        first = 0
        for idx, channels in enumerate(widths):
            part = Conv(spec.name, nx, ny, channels, co, spec.kernel, spec.stride, spec.pad)
            initial[f"weights{idx}"] = pack_weights(part, filt[:, :, first:first + channels, :])
            partial_size[f"partial{idx}"] = output.size * 4
            first += channels
        initial["params"] = (mul | shift << 16).astype("<u4").tobytes()
        initial["lut"] = lut.tobytes()
    expected = {"result": output.tobytes()}
    slots, _ = reserve_ddr([Lifetime(name, len(data), 0, 1)
                            for name, data in (initial | expected).items()] +
                           [Lifetime(name, size, 0, 1) for name, size in partial_size.items()])
    if partial_size:
        partials = [slots[name].offset for name in partial_size]
        weights = [slots[f"weights{idx}"].offset for idx in range(len(partials))]
        layers = (raw_conv_tiles(spec, input_ddr=slots["images"].offset, weights_ddr=weights,
                                 bias_ddr=slots["bias"].offset, partials_ddr=partials),
                  combine_tiles(spec, output_ddr=slots["result"].offset, partials_ddr=partials,
                                params_ddr=slots["params"].offset, lut_ddr=slots["lut"].offset))
    else:
        layers = (conv_tiles(spec, input_ddr=slots["images"].offset, output_ddr=slots["result"].offset,
                             weights_ddr=slots["weights"].offset, bias_ddr=slots["bias"].offset,
                             mul=mul, shift=shift, lut=lut),)
    tensors = {"images": {"shape": list(input_image.shape), "dtype": "int8"},
               "result": {"shape": [1, *output.shape], "dtype": "int8"}}
    manifest = emit_program(directory, layers, tensors, ["images"], ["result"], slots, initial,
                            external_inputs=False)
    control = emit_rtl_control(directory, manifest, initial, expected)
    return control, {"geometry": [nx, ny, ci, co, spec.kernel, spec.stride, spec.pad],
                     "scratch_peak": manifest["plan"]["scratch_peak_by_half"],
                     "dma_bytes": manifest["plan"]["dma_bytes"], "tiles": sum(map(len, layers)),
                     "output_bytes": output.size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--case", choices=("smoke", "stem128", "stem320", "partial", "deep"), required=True)
    args = parser.parse_args()
    cases = {
        "smoke": Conv("smoke", 16, 32, 3, 16, 3, 2, 1),
        "stem128": Conv("stem128", 128, 128, 3, 16, 3, 2, 1),
        "stem320": Conv("stem320", 320, 320, 3, 16, 3, 2, 1),
        "partial": Conv("partial", 48, 64, 32, 48, 3, 1, 1),
        "deep": Conv("deep", 16, 16, 64, 16, 3, 1, 1),
    }
    control, metrics = build_fixture(args.output, cases[args.case])
    print(json.dumps(metrics), flush=True)
    if args.runner:
        subprocess.run([args.runner, control], check=True)


if __name__ == "__main__":
    main()
