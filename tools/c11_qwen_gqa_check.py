"""Verify real Layer-0 GQA context and byte-domain resident KV on frozen RTL."""

import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from hetero_compiler.c11_qwen_gqa import (
    GROUPS, POSITIONS, QUERIES, STATES, WIDTH, compile_gqa,
)


def quantize(vector):
    bound = max(1, int(np.abs(vector.astype(np.int64)).max()))
    scale = 1 << (((bound + 126) // 127 - 1).bit_length())
    shifted = (np.abs(vector.astype(np.int64)) + scale // 2) // scale
    return np.clip(np.sign(vector) * shifted, -127, 127).astype(np.int8), scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("capture", "runner", "reference", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    count = GROUPS * (QUERIES + 1) * WIDTH
    raw = np.fromfile(args.capture, dtype="<i4").reshape(POSITIONS, 2 * count + GROUPS * WIDTH)
    qk = raw[:, count:2 * count].reshape(POSITIONS, GROUPS, QUERIES + 1, WIDTH)
    values = raw[:, 2 * count:].reshape(POSITIONS, GROUPS, WIDTH)
    q, qs = quantize(qk[:, :, :QUERIES])
    k, ks = quantize(qk[:, :, QUERIES])
    v, vs = quantize(values)
    shift = min(31, max(0, round(math.log2(8 * math.log(2) * (1 << 32) / (qs * ks)))))
    data = np.concatenate((q, k[:, :, None], v[:, :, None]), axis=2)
    source = args.out / "qkv.bin"
    data.tofile(source)
    program = compile_gqa(args.out, shift)
    cpu = [args.out / f"{name}.c99.bin" for name in ("context", "kv")]
    rtl = [args.out / f"{name}.rtl.bin" for name in ("context", "kv")]
    subprocess.run([str(args.reference), str(source), *map(str, cpu), str(shift)], check=True)
    with (args.out / "rtl.log").open("w") as log:
        subprocess.run([str(args.runner), str(program), "--backend", "verilator", "--dump-stats",
                        "--input", f"qkv={source}", "--output", f"context={rtl[0]}",
                        "--output", f"kv={rtl[1]}"], stdout=log, stderr=subprocess.STDOUT, check=True)
    for c_file, r_file in zip(cpu, rtl, strict=True):
        if c_file.read_bytes() != r_file.read_bytes():
            raise ValueError(f"Native GQA and C99 differ for {r_file.name}")
    receipt = {"positions": POSITIONS, "gqa_groups": GROUPS, "queries_per_group": QUERIES,
               "context_bytes_exact": rtl[0].stat().st_size, "kv_bytes_exact": rtl[1].stat().st_size,
               "checkpoint_positions": STATES, "shift": shift,
               "per_operand_q16_scales": {"q": qs, "k": ks, "v": vs}}
    (args.out / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("C11_NATIVE_GQA_KV_PASS", json.dumps(receipt))


if __name__ == "__main__":
    main()
