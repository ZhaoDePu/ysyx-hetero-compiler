"""Build a Valence-C C16 mixed single-/dual-plane experimental Qwen image."""

import argparse
import contextlib
import hashlib
import json
import mmap
from pathlib import Path
import shutil

import numpy as np

from tools.c11_qwen_export import HIDDEN, INTER, LAYERS, MODEL_SHA256, Tensors, fp32, sha256
from tools.c12_smoothquant import SCALE, pack4, smooth_scale
from tools.c14_block16 import BLOCK_TAG, cluster

MAGIC = b"YSQC16H\0"
SINGLE_TAG = 0xB1600016
PROJECTIONS = ("mlp.gate_proj", "mlp.up_proj")


def packed_size(spec):
    rows, cols = spec["shape"]
    return rows * cols * spec["bits"] // 8


def projection(tensors, layer, name):
    return fp32(tensors.bf16(f"model.layers.{layer}.{name}.weight", (INTER, HIDDEN)))


def shared_order(weights, group):
    """Keep each Down projection and dynamic A8 scale group intact."""
    features = []
    for value in weights:
        peaks = np.maximum(np.abs(value).reshape(INTER, -1, group).max(axis=2), 1e-6)
        logs = np.log(peaks)
        features.append(logs - logs.mean(axis=1, keepdims=True))
    features = np.concatenate(features, axis=1)
    return np.concatenate([cluster(features, np.arange(start, start + group))
                           for start in range(0, INTER, group)])


def shared_quantize(weight, importance, permutation, group):
    weights = weight[permutation].reshape(INTER // 16, 16, HIDDEN // group, group).transpose(2, 0, 1, 3)
    diag = np.maximum(importance.reshape(-1, 1, 1, group), 1e-9)
    peak = np.max(np.abs(weights), axis=3)
    center = np.exp(np.log(np.maximum(peak, 1e-6)).mean(axis=0))
    gain = np.clip(center / np.median(center, axis=1, keepdims=True), 1 / 16, 16)
    best_scale = best_values = None
    for _ in range(2):
        gain_q16 = np.rint(gain * 65536).clip(1, 16 * 65536).astype("<u4")
        gain = gain_q16.astype(np.float32) / 65536
        normalized = weights / gain[None, :, :, None]
        max_positive = normalized.max(axis=(2, 3)) / 7
        max_negative = -normalized.min(axis=(2, 3)) / 8
        ceiling = np.maximum(max_positive, max_negative)
        loss = np.full(ceiling.shape, np.inf, dtype=np.float32)
        best_scale = np.ones(ceiling.shape, dtype="<u4")
        best_values = np.zeros(weights.shape, dtype=np.int8)
        for clip in (.45, .55, .65, .73, .80, .86, .91, .95, 1.0):
            scale = np.maximum(1, np.rint(ceiling * (clip * SCALE))).astype("<u4")
            step = scale.astype(np.float32)[..., None, None] / SCALE
            values = np.rint(normalized / step).clip(-8, 7).astype(np.int8)
            recon = values * step * gain[None, :, :, None]
            error = np.sum((recon - weights)**2 * diag, axis=(2, 3))
            improved = error < loss
            loss[improved] = error[improved]
            best_scale[improved] = scale[improved]
            best_values[improved] = values[improved]
        unscaled = best_values * (best_scale.astype(np.float32)[..., None, None] / SCALE)
        numerator = (weights * unscaled * diag).sum(axis=(0, 3))
        denominator = (unscaled**2 * diag).sum(axis=(0, 3))
        gain = np.where(denominator > 0, numerator / np.maximum(denominator, 1e-20), gain)
    mse = float(loss.sum() / np.maximum(np.sum(weights**2 * diag), 1e-20))
    packed = pack4(best_values.transpose(1, 0, 2, 3).copy())
    # The runtime keeps both gate branches in physical order. Down columns
    # already use this same order, including within each preserved scale group.
    metadata = best_scale.tobytes() + np.arange(INTER, dtype="<u4").tobytes() + gain_q16.tobytes()
    return packed.tobytes(), metadata, mse


def write_tensor(output, spec, packed, metadata, group):
    rows, cols = spec["shape"]
    space = rows * cols // group * 4
    if len(packed) != packed_size(spec) or len(metadata) + 4 > space:
        raise ValueError("unexpected panel or metadata extent")
    output.seek(spec["offset"])
    output.write(packed)
    output.seek(spec["scale_offset"])
    output.write(metadata)
    output.write(bytes(space - len(metadata) - 4))
    output.write(SINGLE_TAG.to_bytes(4, "little"))


def repack(source, layout, dual, weights, calibration, p1, p2, keep):
    desc = json.loads(layout.read_text())
    specs, group = desc["tensors"], desc["group"]
    if group not in (64, 128) or not desc["smooth_w4_only"] or "down" not in desc["W8"]:
        raise ValueError("requires SmoothQuant W8 Down groups")
    if len({item.resolve() for item in (source, dual, p1, p2)}) != 4:
        raise ValueError("use two distinct output images")
    width = INTER * HIDDEN // 2
    fine_base = source.stat().st_size
    if dual.stat().st_size != fine_base + (LAYERS - 2) * 2 * width:
        raise ValueError("expected the C14 S3-H2 image")
    reports = []
    shutil.copyfile(source, p1)
    with (weights.open("rb") as f, contextlib.closing(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as raw,
          np.load(calibration) as stats, p1.open("r+b") as out):
        tensors = Tensors(raw)
        out.write(MAGIC)
        for layer in range(1, LAYERS - 1):
            names = [f"{layer}.{suffix}" for suffix in PROJECTIONS]
            values = [projection(tensors, layer, name) for name in PROJECTIONS]
            scale = smooth_scale(stats["p/" + names[0]], np.maximum.reduce([
                np.abs(v).max(axis=0) for v in values
            ]), desc["alpha"])
            values = [value * scale for value in values]
            permutation = shared_order(values, group)
            if not np.array_equal(permutation // group, np.arange(INTER) // group):
                raise ValueError("permutation crossed a Down scale group")
            row = {"layer": layer, "permutation_sha256": hashlib.sha256(permutation.astype("<u4").tobytes()).hexdigest()}
            for name, value in zip(names, values, strict=True):
                packed, meta, loss = shared_quantize(value, stats["m/" + name] / (scale * scale), permutation, group)
                write_tensor(out, specs[name], packed, meta, group)
                row[name.split(".")[-1] + "_relative_mse"] = loss
            spec = specs[f"{layer}.mlp.down_proj"]
            data = np.memmap(source, mode="r", dtype="i1", offset=spec["offset"],
                             shape=(INTER // group, HIDDEN, group))
            reorder = (permutation % group).reshape(-1, group)
            permuted = np.take_along_axis(data, reorder[:, None, :], axis=2)
            out.seek(spec["offset"])
            out.write(permuted.tobytes())
            reports.append(row)
            print(json.dumps(row), flush=True)
    protected = {row["layer"] for row in sorted(reports, key=lambda row:
                  row["gate_proj_relative_mse"] + row["up_proj_relative_mse"], reverse=True)[:keep]}
    shutil.copyfile(p1, p2)
    with (dual.open("rb") as f, contextlib.closing(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as h2,
          p2.open("r+b") as out):
        for layer in sorted(protected):
            for name in (*PROJECTIONS, "mlp.down_proj"):
                spec = specs[f"{layer}.{name}"]
                start, end = spec["offset"], spec["extent_end"]
                out.seek(start)
                out.write(h2[start:end])
                if name in PROJECTIONS and int.from_bytes(h2[end-4:end], "little") != BLOCK_TAG:
                    raise ValueError("expected the C14 dual-plane tag")
        out.seek(0, 2)
        for layer in sorted(protected):
            start = fine_base + (layer - 1) * 2 * width
            out.write(h2[start:start + 2 * width])
    return {"group": group, "p1_bytes": p1.stat().st_size, "p2_bytes": p2.stat().st_size,
            "p2_double_plane_layers": sorted(protected), "layers": reports}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "layout", "dual", "weights", "calibration", "p1", "p2", "ledger"):
        cli.add_argument(arg, type=Path)
    cli.add_argument("--keep", type=int, choices=range(LAYERS - 1), default=4)
    args = cli.parse_args()
    if sha256(args.weights) != MODEL_SHA256:
        raise ValueError("official model digest mismatch")
    result = repack(args.source, args.layout, args.dual, args.weights, args.calibration, args.p1, args.p2, args.keep)
    args.ledger.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
