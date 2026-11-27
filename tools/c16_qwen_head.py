"""Produce a vocabulary-permuted tied W4 head for the C16 precision experiment."""

import argparse
import contextlib
import hashlib
import json
import mmap
from pathlib import Path
import shutil

import numpy as np

from tools.c11_qwen_export import HIDDEN, MODEL_SHA256, VOCAB, Tensors, fp32, sha256
from tools.c12_smoothquant import group_quantize, pack4
from tools.c16_qwen_repack import MAGIC


def repack(source, original, layout, calibration, output):
    desc = json.loads(layout.read_text())
    group, spec = desc["group"], desc["tensors"]["embedding/lm_head"]
    if source.resolve() == output.resolve() or spec["bits"] != 8 or spec["shape"] != [VOCAB, HIDDEN]:
        raise ValueError("requires an existing C16 image with the W8 tied head and a separate output")
    groups = HIDDEN // group
    packed = spec["offset"]
    scales = packed + VOCAB * HIDDEN // 2
    following = scales + groups * VOCAB * 4
    packed_k = group // 2
    with (original.open("rb") as f, contextlib.closing(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as raw,
          source.open("rb") as ref, output.open("w+b") as dst, np.load(calibration) as calibration_values):
        table = Tensors(raw).bf16("model.embed_tokens.weight", (VOCAB, HIDDEN))
        amplitude = np.empty(VOCAB, dtype=np.float32)
        for first in range(0, VOCAB, 1024):
            amplitude[first:first + 1024] = np.abs(fp32(table[first:first + 1024])).max(axis=1)
        order = np.argsort(amplitude, kind="stable").astype("<u4")
        header = bytearray(ref.read(64))
        if header[:8] != MAGIC or not header[48] & 4:
            raise ValueError("requires a C16 W8 head")
        header[48] &= 0xfb
        dst.write(header)
        dst.seek(following)
        ref.seek(spec["extent_end"])
        shutil.copyfileobj(ref, dst, length=1 << 20)
        dst.write(order.tobytes())
        importance = calibration_values["m/head"]
        for first in range(0, VOCAB, 1024):
            rows = fp32(table[order[first:first + 1024]])
            quant, q24 = group_quantize(rows, group, 4, importance)
            codes = pack4(quant)
            for g in range(groups):
                dst.seek(packed + (g * VOCAB + first) * packed_k)
                dst.write(codes[:, g].tobytes())
                dst.seek(scales + (g * VOCAB + first) * 4)
                dst.write(q24[:, g].tobytes())
        del table
    return {"bytes": output.stat().st_size, "head_bits": 4, "group": group,
            "vocab_permutation_sha256": hashlib.sha256(order.tobytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "original", "layout", "calibration", "output", "ledger"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    if sha256(a.original) != MODEL_SHA256:
        raise ValueError("official model digest mismatch")
    result = repack(a.source, a.original, a.layout, a.calibration, a.output)
    a.ledger.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
