"""Validate the C99 RGB/PPM front end and the C6 YHB loader boundaries."""

import argparse
import json
from pathlib import Path
import struct
import subprocess

from hetero_compiler.yhb import inspect_yhb
from runtime.verify_z15_abi import corrupt_words


def verify_scene(executable, bundle, vectors, output, rgb, mode):
    kind = "rgb" if rgb.suffix == ".bin" else "ppm"
    suffix = ".rgb_nhwc_u8.bin" if kind == "rgb" else ".ppm"
    scene = rgb.name[:-len(suffix)]
    result = subprocess.run([str(executable), str(bundle), "--" + kind, str(rgb),
                             "--sim", "--conf", "0"], text=True, capture_output=True, check=True,
                            timeout=2400)
    (output / f"{scene}.json").write_text(result.stdout, encoding="utf-8")
    data = json.loads(result.stdout)
    perf, detections = data["perf"], data["detections"]
    if perf["abi_error"] or perf["invalid_dma_descriptors"] or len(detections) != 300:
        raise ValueError(f"{scene}: invalid output count or ABI status")
    if perf["hardware_counters_valid"] != (mode == "verilator"):
        raise ValueError(f"{scene}: wrong hardware telemetry source")
    for label, fields, dtype in (
        ("coords_q16_16", ("x1", "y1", "x2", "y2"), "i"),
        ("scores_q16_16", ("score",), "i"),
        ("classes_i64", ("class_id",), "q"),
    ):
        values = [d[field] for d in detections for field in fields]
        if dtype == "i":
            values = [round(struct.unpack("<f", struct.pack("<f", value))[0] * 65536)
                      for value in values]
        actual = struct.pack("<" + dtype * len(values), *values)
        if actual != (vectors / f"{scene}.{label}.bin").read_bytes():
            raise ValueError(f"{scene}: {label} differs from the fixed-point oracle")
    print(f"{mode} {scene} PASS detections=300 q16_i64_bytes=8400", flush=True)
    return {"scene": scene, "mode": mode, "fixed_point_bytes": 8400, **perf}


def verify_rejections(inspect, bundle, output):
    blob = bundle.read_bytes()
    crc = output / "crc.yhb"
    crc.write_bytes(blob[:-1] + bytes([blob[-1] ^ 1]))
    start = inspect_yhb(blob)["sections"]["dma"]["offset"]
    fixtures = [(crc, "YHB CRC32 mismatch")]
    for name, edits in (("half-cross", {2: 0xfffe0, 3: 64}), ("ddr-wrap", {1: 0xffffffe0, 3: 64})):
        fixtures.append((corrupt_words(blob, output, name, start, edits), "invalid YHB DMA extent"))
    for fixture, diagnosis in fixtures:
        result = subprocess.run([str(inspect), str(fixture), "--inspect"],
                                text=True, capture_output=True, timeout=15)
        if result.returncode == 0 or diagnosis not in result.stderr:
            raise ValueError(f"{fixture.name}: expected loader rejection {diagnosis!r}")
        (output / f"{fixture.stem}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        print(f"loader {fixture.stem} PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("executable", "bundle", "vectors", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--inspect", type=Path)
    parser.add_argument("--mode", choices=("reference", "verilator"), default="reference")
    parser.add_argument("--scene", action="append", help="choose a subset of the fixed-point scene vectors")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = args.scene or [path.stem for path in sorted(args.vectors.glob("*.ppm"))]
    if not cases:
        parser.error("no scene vectors")
    rows = []
    for index, scene in enumerate(cases):
        ext = ".ppm" if index % 2 == 0 else ".rgb_nhwc_u8.bin"
        rows.append(verify_scene(args.executable, args.bundle, args.vectors, args.output,
                                 args.vectors / (scene + ext), args.mode))
    if args.inspect:
        verify_rejections(args.inspect, args.bundle, args.output)
    (args.output / "result.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
