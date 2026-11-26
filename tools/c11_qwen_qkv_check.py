"""Verify real Qwen Layer-0 QKV projection and signed bias on frozen RTL."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from hetero_compiler.c11_w4_projection import lower_projection
from hetero_compiler.c11_w4_requant import lower_reduction
from tools.c11_qwen_export import GROUP, HIDDEN, INTER, KV_HEADS, HEAD_DIM


def mapped(path, record, key, dtype, shape):
    return np.memmap(path, dtype=dtype, mode="r", offset=record[key], shape=shape)


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(args.weights.with_name("w4a8.json").read_text())["tensor_extents"]
    groups = HIDDEN // GROUP
    matrix, scales, biases = [], [], []
    for name in ("q", "k", "v"):
        prefix = f"model.layers.0.self_attn.{name}_proj."
        info = meta[prefix + "weight"]
        width = info["shape"][0]
        matrix.append(mapped(args.weights, info, "packed_offset", "u1", (groups, width, GROUP // 2)))
        scales.append(mapped(args.weights, info, "scale_offset", "<u4", (groups, width)))
        biases.append(mapped(args.weights, meta[prefix + "bias"], "offset", "<i4", (width,)))
    payload = np.concatenate(matrix, axis=1).tobytes()
    multipliers = np.concatenate(scales, axis=1)
    bias = np.concatenate(biases)
    channels = len(bias)
    v_width = KV_HEADS * HEAD_DIM
    vectors = np.fromfile(args.vectors, dtype="<i4").reshape(10, 4 * HIDDEN + 3 * INTER)
    source = vectors[args.pos, 2 * HIDDEN:3 * HIDDEN].astype(np.int64)
    peak = max(1, int(np.abs(source).max()))
    scale = max(1, (peak + 63) // 127)
    input_a8 = (np.sign(source) * ((np.abs(source) + scale // 2) // scale)).clip(-127, 127).astype(np.int8)

    traces = np.fromfile(args.qkv, dtype="<i4").reshape(10, 2 * (HIDDEN + v_width) + v_width)
    before = traces[args.pos, :(HIDDEN + v_width)].reshape(KV_HEADS, HIDDEN // KV_HEADS + HEAD_DIM)
    expected = np.concatenate((before[:, :-HEAD_DIM].reshape(-1), before[:, -HEAD_DIM:].reshape(-1),
                               traces[args.pos, -v_width:]))
    path = args.out / "activation.bin"
    path.write_bytes(input_a8.tobytes())
    dot = args.out / "dot"
    lower_projection(dot, payload, rows=1, channels=channels, groups=groups, variant="B2")
    partial = args.out / "partial.rtl.bin"
    with (args.out / "dot-rtl.log").open("w") as log:
        subprocess.run([str(args.runner), str(dot / "program.yhb"), "--backend", "verilator",
                        "--input", f"activation={path}", "--output", f"partial={partial}",
                        "--dump-stats"], stdout=log, stderr=subprocess.STDOUT, check=True)
    recover = args.out / "requant"
    program = lower_reduction(recover, multipliers.tobytes(), scale,
                              channels=channels, bias=bias.tobytes())
    result = args.out / "qkv.rtl.bin"
    with (args.out / "requant-rtl.log").open("w") as log:
        subprocess.run([str(args.runner), str(program), "--backend", "verilator",
                        "--input", f"partial={partial}", "--output", f"logits={result}",
                        "--dump-stats"], stdout=log, stderr=subprocess.STDOUT, check=True)
    actual = np.fromfile(result, dtype="<i4")
    if not np.array_equal(actual, expected):
        index = int(np.flatnonzero(actual != expected)[0])
        raise ValueError(f"QKV mismatch at {index}: {actual[index]} vs {expected[index]}")
    print(f"C11_REAL_LAYER0_QKV_PASS pos={args.pos} channels={channels} bytes={expected.nbytes}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("weights", "vectors", "qkv", "runner", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--pos", type=int, choices=range(10), default=8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
