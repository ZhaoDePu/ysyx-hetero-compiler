"""Replay actual Layer-0 query and key tensors through the frozen G3 RoPE."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from hetero_compiler.c11_qwen_rope import POSITIONS, TRIG_BYTES, lower_rope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("weights", "capture", "runner", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    extent = json.loads(args.weights.with_name("w4a8.json").read_text())["tensor_extents"]["rope_q16"]
    with args.weights.open("rb") as image:
        image.seek(extent["offset"])
        trig = image.read(TRIG_BYTES * POSITIONS)
    encoded = lower_rope(args.out, trig)
    capture = np.fromfile(args.capture, dtype="<i4").reshape(POSITIONS, 2 * 2 * 8 * 64 + 2 * 64)
    qk = capture[:, :2 * 2 * 8 * 64].reshape(POSITIONS, 2, 2, 8, 64)
    src = args.out / "qk.bin"
    dst = args.out / "rotated.bin"
    raw = args.out / "rtl.log"
    qk[:, 0].tofile(src)
    with raw.open("w") as log:
        subprocess.run([str(args.runner), str(encoded), "--backend", "verilator", "--dump-stats",
                        "--input", f"qk={src}", "--output", f"rotated={dst}"],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    rotated = np.fromfile(dst, dtype="<i4").reshape(qk[:, 1].shape)
    if not np.array_equal(rotated, qk[:, 1]):
        which = tuple(np.argwhere(rotated != qk[:, 1])[0])
        raise AssertionError(f"GPU RoPE mismatch at {which}: {rotated[which]} vs {qk[:, 1][which]}")
    print(f"C11_REAL_LAYER0_ROPE_PASS positions={POSITIONS} qheads=14 kvheads=2 bytes={rotated.nbytes}")


if __name__ == "__main__":
    main()
