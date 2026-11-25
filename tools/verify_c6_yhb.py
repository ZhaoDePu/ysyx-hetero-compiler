"""Replay the unseeded YOLO26n YHB through the strict C99 Verilator HAL."""

import argparse
import json
import re
from pathlib import Path
import subprocess
import time


RAW = ["boxes_stride8", "scores_stride8", "boxes_stride16", "scores_stride16",
       "boxes_stride32", "scores_stride32"]


def replay(executor, model, vectors, outdir, scenes=(), timeout=2400):
    outdir.mkdir(parents=True, exist_ok=True)
    cases = sorted(vectors.glob("*.input_nhwc_s8.bin"))
    if scenes:
        cases = [vectors / (scene + ".input_nhwc_s8.bin") for scene in scenes]
    if not cases:
        raise ValueError("no quantized image vectors")
    for rgb in cases:
        scene = rgb.name.split(".", 1)[0]
        case = outdir / scene
        case.mkdir(exist_ok=True)
        src = rgb.read_bytes()
        padded = bytearray((len(src) // 3) * 4)
        for channel in range(3):
            padded[channel::4] = src[channel::3]
        input_path = case / "images.bin"
        input_path.write_bytes(padded)
        cmd = [str(executor), str(model), "--backend", "verilator", "--dump-stats",
               "--host-timing", "--input", f"images={input_path}"]
        for name in RAW + ["c6_head_scales", "c6_score_lut"]:
            cmd.extend(["--output", f"{name}={case / (name + '.bin')}"])
        start = time.monotonic()
        log = case / "replay.log"
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(cmd, check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
        seconds = time.monotonic() - start
        verified_bytes = 0
        for index, name in enumerate(RAW):
            expected = (vectors / f"{scene}.pl_out_{index}.bin").read_bytes()
            got = (case / (name + ".bin")).read_bytes()
            if expected != got:
                difference = next((i for i, (a, b) in enumerate(zip(expected, got)) if a != b), "length")
                raise ValueError(f"{scene}: {name} differs at {difference}")
            verified_bytes += len(got)
        stats_line, = (line for line in log.read_text().splitlines() if line.startswith("STATS "))
        stats = {key: int(value, 0) for key, value in re.findall(r"(\w+)=(0x[\da-f]+|\d+)", stats_line)}
        if stats.get("abi_error") != 0 or stats.get("invalid_dma_descriptors") != 0:
            raise ValueError(f"{scene}: the physical replay reported an invalid descriptor or ABI error")
        result = {"scene": scene, "raw_head_bytes": verified_bytes, "rtl_seconds": seconds, **stats}
        (case / "qualified.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("executor", "model", "vectors", "outdir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=2400)
    args = parser.parse_args()
    replay(args.executor, args.model, args.vectors, args.outdir, args.scene, args.timeout)


if __name__ == "__main__":
    main()
