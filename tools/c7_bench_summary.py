"""Verify and summarize the full-frame C7 RTL ablation and clock sweeps."""

import argparse
from pathlib import Path
import json
import re
import shutil

from tools.c6_accounting import runtime_ledger, write_csv


BUSY = (
    "npu_engine_busy_ticks",
    "vpu_engine_busy_ticks",
    "gpu_busy_ticks",
    "dma_busy_ticks",
    "dma_compute_overlap_ticks",
    "npu_gpu_overlap_ticks",
    "npu_domain_wall_ticks",
    "dma_bytes",
    "abi_error",
    "invalid_dma_descriptors",
)


def summary(name, counters, npu, gpu, baseline, seconds=None):
    wall = counters["npu_domain_wall_ticks"]
    return {
        "case": name, "npu_mhz": npu, "gpu_mhz": gpu,
        **{counter: counters[counter] for counter in BUSY},
        "ms_per_frame": wall / (1000 * npu), "fps": 1e6 * npu / wall,
        "speedup_vs_c6_same_clock": baseline / wall if (npu, gpu) == (125, 100) else None,
        "rtl_host_seconds": seconds,
    }


def qualify(name, manifest_path, replay, target):
    manifest = json.loads(manifest_path.read_text())
    rows, metrics = runtime_ledger(manifest, replay)
    gate = json.loads(replay.with_name("qualified.json").read_text())
    if (gate["scene"] != "bus" or gate["raw_head_bytes"] != 176400
            or any(gate[key] != metrics[key] for key in (
                "npu_domain_wall_ticks", "dma_bytes", "abi_error", "invalid_dma_descriptors"))):
        raise ValueError(f"{name}: missing or mixed full-frame evidence")
    if metrics["dma_bytes"] != manifest["plan"]["dma_bytes"]:
        raise ValueError(f"{name}: physical DMA bytes differ from the manifest")
    if manifest["plan"]["scratch_peak"] > 2 << 20:
        raise ValueError(f"{name}: exceeded the C7 SRAM cap")
    target.mkdir(parents=True, exist_ok=True)
    write_csv(target / "rtl_stages.csv", rows)
    shutil.copyfile(replay, target / "replay.log")
    shutil.copyfile(replay.with_name("qualified.json"), target / "qualified.json")
    metrics.update(sram_peak_bytes=manifest["plan"]["scratch_peak"],
                   ddr_peak_bytes=manifest["plan"]["ddr_peak"],
                   rtl_host_seconds=gate["rtl_seconds"])
    (target / "counters.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", action="append", nargs=5, required=True,
                        metavar=("NAME", "NPU_MHZ", "GPU_MHZ", "MANIFEST_JSON", "REPLAY_LOG"))
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    original = baseline["npu_domain_wall_ticks"]
    results = [summary("C6", baseline, 125, 100, original)]
    args.out.mkdir(exist_ok=True, parents=True)
    for name, npu, gpu, manifest, replay in args.case:
        if not re.fullmatch(r"[\w-]+", name):
            parser.error(f"invalid case name {name!r}")
        metrics = qualify(name, Path(manifest), Path(replay), args.out / name)
        results.append(summary(name, metrics, float(npu), float(gpu), original, metrics["rtl_host_seconds"]))
    write_csv(args.out / "ablation.csv", results)
    (args.out / "ablation.json").write_text(json.dumps(results, indent=2) + "\n")
    for row in results:
        print(f"{row['case']}: {row['npu_domain_wall_ticks']:,} ticks, {row['fps']:.3f} FPS")


if __name__ == "__main__":
    main()
