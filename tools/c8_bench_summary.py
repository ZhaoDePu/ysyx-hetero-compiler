"""Collect same-clock C6/C7/C8 ablations and the C8 clock-ratio sweep."""

import argparse
import json
from pathlib import Path

from tools.c6_accounting import write_csv
from tools.c7_bench_summary import qualify, summary


def row(name, npu, gpu, counters, c6, c7):
    result = summary(name, counters, npu, gpu, c6, counters.get("rtl_host_seconds"))
    wall = counters["npu_domain_wall_ticks"]
    busy = (counters["npu_busy_ticks"] + counters["gpu_busy_ticks"] + counters["dma_busy_ticks"]
            - counters["npu_gpu_overlap_ticks"] - counters["dma_compute_overlap_ticks"])
    if not 0 <= busy <= wall:
        raise ValueError(f"{name}: invalid engine busy union")
    return {**result, "dma_descriptors": counters["dma_operations"], "idle_ticks": wall - busy,
            "slverr": 0, "raw_head_bytes": 176400,
            "speedup_vs_c7_same_clock": c7 / wall if (npu, gpu) == (125, 100) else None,
            "fixed_work_lower_bound_ticks": max(counters["npu_busy_ticks"], counters["gpu_busy_ticks"],
                                                counters["dma_busy_ticks"]),
            "sram_peak_bytes": counters.get("sram_peak_bytes"),
            "ddr_peak_bytes": counters.get("ddr_peak_bytes")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c6", required=True, type=Path)
    parser.add_argument("--c7", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--case", action="append", nargs=5, required=True,
                        metavar=("NAME", "NPU_MHZ", "GPU_MHZ", "MANIFEST_JSON", "REPLAY_LOG"))
    args = parser.parse_args()
    c6, c7 = [json.loads(path.read_text()) for path in (args.c6, args.c7)]
    denominators = [item["npu_domain_wall_ticks"] for item in (c6, c7)]
    rows = [row(name, 125, 100, data, *denominators) for name, data in (("C6", c6), ("C7", c7))]
    for name, npu, gpu, manifest, log in args.case:
        if not name.replace("-", "").isalnum():
            parser.error(f"invalid case name: {name}")
        replay = Path(log)
        if any(fragment in replay.read_text() for fragment in ("Verilator read:", "Verilator write offset=", "SLVERR")):
            raise ValueError(f"{name}: unexpected AXI error during replay")
        data = qualify(name, Path(manifest), replay, args.out / name)
        rows.append(row(name, float(npu), float(gpu), data, *denominators))
    write_csv(args.out / "ablation.csv", rows)
    (args.out / "ablation.json").write_text(json.dumps(rows, indent=2) + "\n")
    for result in rows:
        print(f"{result['case']}: {result['npu_domain_wall_ticks']:,} ticks, "
              f"{result['ms_per_frame']:.3f} ms, {result['fps']:.3f} FPS")


if __name__ == "__main__":
    main()
