"""Compile four isolated ablations and compare frozen Z15 RTL to the integer oracle."""

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

from hetero_compiler.compiler import compile_bundle
from hetero_compiler.ir import GraphError, normalize
from hetero_compiler.onnx import parse_model
from hetero_compiler.profile import load_profile


SETTINGS = [
    ("A0", {"device_policy": "gpu-only", "fuse": False, "overlap": False, "reuse": False}),
    ("A1", {"fuse": False, "overlap": False, "reuse": False}),
    ("A2", {"overlap": False}),
    ("A3", {}),
]
FMAX_NPU_MHZ, FMAX_GPU_MHZ = 164.555, 124.239
PRIORITY = ["nice", "-n", "10", "ionice", "-c2", "-n7"]
SCRATCH = Path("/dev/shm/ysyx_compiler_c5")


def input_values(graph, seed, *, vary_defaults=False):
    values = {}
    for slot, name in enumerate(graph["inputs"]):
        tensor = graph["tensors"][name]
        if "data" in tensor and not vary_defaults:
            continue
        bias = 8 if tensor["dtype"] == "uint8" else 0
        size = math.prod(tensor["shape"])
        values[name] = [((i * (3 + 2*seed) + (i*i + 5*i) * (slot + seed) // 7 + seed*9)
                         % 17 - 8 + bias) for i in range(size)]
    return values


def run(root, model, records, seed, pipeline):
    path = Path(model)
    raw = parse_model(path.read_bytes()) if path.suffix == ".onnx" else json.loads(path.read_text())
    graph = normalize(raw, dialect="canonical" if path.suffix == ".onnx" else "auto")
    values = input_values(graph, seed)
    stem = Path(path.stem)
    data_file = root / f"{stem}.input.json"
    data_file.write_text(json.dumps(values) + "\n", encoding="ascii")
    target = load_profile("fpga_zu15eg_balanced")
    baseline = None
    for label, settings in SETTINGS:
        destination = root / f"{stem}-{label}"
        result = {"model": str(stem), "ablation": label}
        try:
            manifest = compile_bundle(raw, target, destination,
                                      dialect="canonical" if path.suffix == ".onnx" else "auto",
                                      pipeline_tiles=pipeline, vpu_lowering="off", **settings)
        except (ValueError, OSError) as exc:
            result.update(status="compile_error", reason=str(exc))
            records.append(result)
            print(f"{stem} {label}: {result['reason']}", flush=True)
            continue
        stats = manifest["estimates"]
        result.update({k: stats[k] for k in ("scratch_peak_live_bytes", "scratch_reserved_bytes",
                                             "external_transfer_bytes", "serial_compute_cycles",
                                             "scheduled_stage_count", "ir_operator_count")})
        result.update({f"{k.lower()}_stages": v for k, v in stats["device_stage_counts"].items()})
        log = destination.with_suffix(".log")
        cmd = PRIORITY + [sys.executable, "-B", "-m", "tools.verify_bundle", "--bundle", str(destination),
                         "--case", str(root / f"{stem}-{label}.rtl"), "--inputs", str(data_file),
                         "--execute-plan"]
        try:
            with log.open("w", encoding="utf-8") as stream:
                proc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT,
                                      timeout=960 if pipeline else 180)
            output = log.read_text(encoding="utf-8").splitlines()
            counters = [line.split(" ", 1)[1] for line in output if line.startswith("Z15_METRICS ")]
            if proc.returncode != 0 or len(counters) != 1 or not any(
                    line.startswith("BUNDLE_RTL_PASS ") for line in output):
                raise GraphError(f"RTL check failed. See {log.name}")
            result.update(json.loads(counters[0]))
            if pipeline and result.get("rtl_dma_bytes") != result["external_transfer_bytes"]:
                raise GraphError(f"actual AXI byte counter disagrees with scheduled bytes. See {log.name}")
            ticks = result["wall_npu_ticks"]
            if label == "A0":
                baseline = ticks
            result.update(status="byte_exact", speedup=(round(baseline/ticks, 3) if baseline else None),
                          wall_us_npu_fmax=round(ticks / FMAX_NPU_MHZ, 4),
                          tensor_busy_us_npu_fmax=round(result["npu_busy_ticks"] / FMAX_NPU_MHZ, 4),
                          gpu_busy_us_gpu_fmax=round(result["gpu_busy_edges"] / FMAX_GPU_MHZ, 4))
        except (subprocess.TimeoutExpired, GraphError) as exc:
            result.update(status="rtl_error", reason=str(exc))
        records.append(result)
        print(f"{stem} {label}: {result.get('status')} {result.get('wall_npu_ticks', '')}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--pipeline-tiles", action="store_true",
                        help="Require a four-step speed ladder and real overlap on tile pipelines")
    parser.add_argument("models", nargs="+")
    args = parser.parse_args()
    root = args.out.resolve()
    if SCRATCH not in root.parents or SCRATCH.resolve() != SCRATCH:
        parser.error("output must be under compiler C2 RAM")
    root.mkdir()
    records = []
    for path in args.models:
        run(root, path, records, args.seed, args.pipeline_tiles)
    ladder = {}
    if args.pipeline_tiles:
        for model in dict.fromkeys(rec["model"] for rec in records):
            model_rows = [rec for rec in records if rec["model"] == model]
            exact = len(model_rows) == 4 and all(rec["status"] == "byte_exact" for rec in model_rows)
            timings = [rec["wall_npu_ticks"] for rec in model_rows] if exact else []
            ladder[model] = exact and all(a > b for a, b in zip(timings, timings[1:])) and (
                model_rows[-1]["compute_dma_overlap_ticks"] > 0 or
                model_rows[-1]["npu_gpu_overlap_ticks"] > 0)
    data = {"measurement": "frozen Z15 Balanced Verilator MMIO",
            "fmax_npu_mhz": FMAX_NPU_MHZ, "fmax_gpu_mhz": FMAX_GPU_MHZ,
            "fmax_conversion": "Phase 12C routed Fmax projection, not measured board time",
            "speed_ladder": ladder, "results": records}
    (root / "ablation.json").write_text(json.dumps(data, indent=2) + "\n")
    fields = list(dict.fromkeys(key for rec in records for key in rec))
    with (root / "ablation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    if args.pipeline_tiles:
        return int(not all(ladder.values()))
    return int(any(rec["status"] != "byte_exact" and (rec["ablation"] != "A0" or
                   rec["status"] == "rtl_error") for rec in records))


if __name__ == "__main__":
    raise SystemExit(main())
