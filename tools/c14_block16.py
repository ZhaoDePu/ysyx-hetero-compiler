"""Repack the pinned C12 KV8 image for block-16 P17 native scaling."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


Q16 = 65536
BLOCK = 16
PANEL_MAGIC = b"YSQC14P\0"
RESIDUAL_MAGIC = b"YSQC14D\0"
HYBRID_MAGIC = b"YSQC14H\0"
BLOCK_TAG = 0xB1600014
MAGIC = b"YSQC14\0\0"


def cluster(scales, channels):
    if channels.size <= BLOCK:
        return channels
    dims = np.var(scales[channels], axis=0).argmax()
    order = channels[np.argsort(scales[channels, dims], kind="stable")]
    middle = ((channels.size // BLOCK) // 2) * BLOCK
    return np.concatenate((cluster(scales, order[:middle]), cluster(scales, order[middle:])))


def block16(packed, scales, strategy):
    """Operate on the C12 SmoothQuant weight values implied by its nibbles."""
    g, n, packed_k = packed.shape
    if scales.shape != (g, n) or n % BLOCK:
        raise ValueError("expected group-major weights with 16-aligned channels")
    permutation = np.arange(n)
    balance = np.ones(n, dtype=np.float64)
    if strategy in {"s2", "s3"}:
        log = np.log(scales.astype(np.float64))
        channel_level = log.mean(axis=0)
        balance = np.exp(channel_level - np.median(channel_level))
        permutation = cluster((log - channel_level).T, permutation)
    balance_code = np.clip(np.rint(balance * Q16), 1, 16 * Q16).astype("<u4")
    scale = (scales.astype(np.float64) * (Q16 / balance_code)[None, :])[:, permutation]

    reordered = packed[:, permutation, :]
    weights = np.empty((g, n, 2 * packed_k), dtype=np.int8)
    weights[..., ::2] = ((reordered & 15) ^ 8).astype(np.int8) - 8
    weights[..., 1::2] = ((reordered >> 4) ^ 8).astype(np.int8) - 8
    target = (weights.astype(np.float32) * scale.astype(np.float32)[..., None]).reshape(g, n // BLOCK, -1)
    positive = target.max(axis=-1) / 7
    negative = -target.min(axis=-1) / 8
    ceiling = np.maximum(np.maximum(positive, negative), 1)
    best_scale = np.ones_like(ceiling)
    best_error = np.full(ceiling.shape, np.inf)
    # Optimize each group's common scale against its actual reconstructed codes.
    # The 1.0 endpoint includes the highest non-clipping scale.
    for width in ([1.0] if strategy == "s3" else np.arange(0.4, 1.0001, 0.05)):
        scale_try = np.maximum(1, np.rint(ceiling * width))
        q = np.clip(np.rint(target / scale_try[..., None]), -8, 7)
        mse = np.mean((q * scale_try[..., None] - target) ** 2, axis=-1)
        choose = mse < best_error
        best_scale[choose] = scale_try[choose]
        best_error[choose] = mse[choose]
    scaled = target / best_scale[..., None]
    codes = np.clip(np.rint(scaled), -8, 7)
    residual = np.clip(np.rint((scaled - codes) * 16), -8, 7) if strategy == "s3" else None

    def pack(values):
        q = values.astype(np.int8).reshape(weights.shape)
        return ((q[..., ::2] & 15) | ((q[..., 1::2] & 15) << 4)).astype(np.uint8)

    if residual is not None:
        best_error = np.mean(((codes + residual / 16) * best_scale[..., None] - target) ** 2, axis=-1)
    return (pack(codes), best_scale.astype("<u4"), permutation.astype("<u4"),
            balance_code[permutation], float(best_error.mean()), float(np.mean(target**2)),
            pack(residual) if residual is not None else None)


def repack(source, layout, destination, strategy, panel_major=False, protect=()):
    description = json.loads(layout.read_text())
    group = description["group"]
    if group not in {64, 128}:
        raise ValueError("P17 requires G64 or G128")
    if source.resolve() == destination.resolve():
        raise ValueError("use a separate destination for the experimental image")
    with source.open("rb") as f:
        if f.read(len(MAGIC)) != b"YSQC12\0\0":
            raise ValueError("expected the pinned C12 mixed image")
    shutil.copyfile(source, destination)
    if strategy == "s3":
        panel_major = True
    if protect and strategy != "s3":
        raise ValueError("exact layer protection is available only for S3")
    logs = []
    for name, spec in description["tensors"].items():
        if spec["bits"] != 4:
            continue
        if int(name.split(".", 1)[0]) in protect:
            logs.append({"tensor": name, "exact_fallback": True})
            continue
        n, k = spec["shape"]
        packed = np.memmap(source, mode="r", dtype="u1", offset=spec["offset"], shape=(k//group, n, group//2))
        scale = np.memmap(source, mode="r", dtype="<u4", offset=spec["scale_offset"], shape=(k//group, n))
        out, shared, indices, gain, mse, power, fine = block16(packed, scale, strategy)
        def panel_order(data):
            return data.reshape(k // group, n // BLOCK, BLOCK, group // 2).transpose(1, 0, 2, 3)

        if panel_major:
            out = panel_order(out)
        metadata = shared.tobytes() + indices.tobytes() + gain.tobytes()
        if len(metadata) + 4 > scale.nbytes:
            raise ValueError(f"{name}: missing scale space for metadata")
        with destination.open("r+b") as f:
            f.seek(spec["offset"])
            f.write(out.tobytes())
            f.seek(spec["scale_offset"])
            f.write(metadata)
            f.write(bytes(scale.nbytes - len(metadata) - 4))
            f.write(BLOCK_TAG.to_bytes(4, "little"))
        if fine is not None:
            with destination.open("ab") as f:
                f.write(panel_order(fine).tobytes())
        logs.append({"tensor": name, "blocks": int(shared.size), "mse_q24_squared": mse,
                     "rmse_over_rms": (mse / power)**0.5 if power else 0.0})
    with destination.open("r+b") as f:
        f.write(HYBRID_MAGIC if protect else RESIDUAL_MAGIC if strategy == "s3"
                else PANEL_MAGIC if panel_major else MAGIC)
    return logs


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "layout", "destination"):
        cli.add_argument(arg, type=Path)
    cli.add_argument("strategy", choices=("s1", "s2", "s3"))
    cli.add_argument("--panel-major", action="store_true", help="place all groups of one physical panel together")
    cli.add_argument("--protect", type=int, choices=range(24), action="append", default=[])
    args = cli.parse_args()
    print(json.dumps({"strategy": args.strategy, "tensors": repack(args.source, args.layout, args.destination,
                                                                   args.strategy, args.panel_major, args.protect)}, indent=2))


if __name__ == "__main__":
    main()
