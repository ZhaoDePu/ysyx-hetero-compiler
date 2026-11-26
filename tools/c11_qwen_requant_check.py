"""Replay Q24 weight scale reduction on the FPGA GPU from verified dot products."""

import argparse
import json
from pathlib import Path
import struct
import subprocess

import numpy as np

from hetero_compiler.c11_w4_requant import CHANNELS, GROUPS, lower_reduction
from tools.c11_qwen_export import VOCAB


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("weights", "lm-input", "partial", "prefill-q16", "runner", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    info = json.loads(args.weights.with_name("w4a8.json").read_text())
    extent = info["tensor_extents"][info["tied_lm_head"]]
    scale, = struct.unpack_from("<I", args.lm_input.read_bytes())
    values = np.memmap(args.weights, dtype="<u4", mode="r", offset=extent["scale_offset"],
                       shape=(GROUPS, VOCAB))[:, :CHANNELS]
    raw = np.fromfile(args.partial, dtype="<i4").reshape(GROUPS, CHANNELS)
    largest = max((np.abs(raw.astype(np.int64)) * values.astype(np.int64)).sum(axis=0))
    if int(largest) * scale + (1 << 23) >= 1 << 55:
        raise ValueError("Q24 scale reduction would overflow the signed int32 result before saturation")
    program = lower_reduction(args.out, values.tobytes(), scale)
    result = args.out / "logits.rtl.bin"
    with (args.out / "rtl.log").open("w") as log:
        subprocess.run([str(args.runner), str(program), "--backend", "verilator", "--dump-stats",
                        "--input", f"partial={args.partial}", "--output", f"logits={result}"],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    expected = np.memmap(args.prefill_q16, mode="r", dtype="<i4", offset=(
        args.prefill_q16.stat().st_size - VOCAB * 4), shape=(VOCAB,))
    actual = np.fromfile(result, dtype="<i4")
    if not np.array_equal(expected[:CHANNELS], actual):
        mismatch = np.flatnonzero(expected[:CHANNELS] != actual)[0]
        raise ValueError(f"RTL scaled logit {mismatch}: {actual[mismatch]} vs {expected[mismatch]}")
    print(f"C11_Q24_RTL_LOGITS_PASS channels={CHANNELS} exact_bytes={actual.nbytes}")


if __name__ == "__main__":
    main()
