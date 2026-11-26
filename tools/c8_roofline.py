"""Compute optimistic latency floors for a measured C8 command stream."""

import argparse
from collections import Counter
import json
from pathlib import Path

from tools.c6_accounting import read_csv


def critical_path(manifest, stages, *, dma, free_engine=None):
    """Keep all explicit waits but give launch and control operations zero cost."""
    cost = {stage["stage"]: int(stage["busy_ticks"]) if stage["engine"] != free_engine else 0
            for stage in stages}
    active, dma_end, current = {}, 0, 0
    for command in manifest["execution"]:
        kind = command["type"]
        if kind == "launch":
            stage = command["stage"]
            active[stage] = current + cost[stage]
        elif kind == "fence":
            current = max(current, active.pop(command["wait_for"]))
        elif kind == "dma":
            # Peak one 256-bit interface beat per NPU-domain cycle.
            dma_end = current + ((command["bytes"] + 31) // 32 if dma else 0)
        elif kind == "dma-fence":
            current = max(current, dma_end)
        else:
            raise ValueError(f"unsupported C8 schedule command: {command}")
    if active:
        raise ValueError("unterminated stage in the frozen schedule")
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "stages", "counters", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    stages = read_csv(args.stages)
    counters = json.loads(args.counters.read_text())
    native = counters["npu_busy_ticks"]
    gpu = counters["gpu_busy_ticks"]
    dma = counters["dma_busy_ticks"]
    traffic = counters["dma_bytes"]
    duration = counters["npu_domain_wall_ticks"]
    native_overlap = counters["npu_gpu_overlap_ticks"]
    active = native + gpu + dma - native_overlap - counters["dma_compute_overlap_ticks"]
    roof = {
        "scope": "fixed stage service measured at 125/100 with a deterministic issued command stream",
        "wall_cycles": duration,
        "all_idle_ticks": duration - active,
        "native_serial_service_ticks": native,
        "gpu_service_ticks": gpu,
        "dma_service_ticks": dma,
        "dma_interface_peak_bytes_per_tick": 32,
        "dma_actual_bytes_per_busy_tick": traffic / dma,
        "dma_interface_min_ticks": (traffic + 31) // 32,
        "required_native_gpu_overlap_if_free_dma_and_host_at_3_8x": max(0, native + gpu - 14_143_000),
        "required_native_gpu_overlap_if_free_dma_and_host_at_4_5x": max(0, native + gpu - 11_943_000),
        "measured_native_gpu_overlap_ticks": native_overlap,
        "command_order_floor_ignoring_dma_and_host_dispatch": critical_path(manifest, stages, dma=False),
        "command_order_floor_at_peak_dma_and_no_dispatch": critical_path(manifest, stages, dma=True),
        "command_order_counterfactual_gpu_free_ticks": critical_path(manifest, stages, dma=True, free_engine="GPU"),
        "command_order_counterfactual_vpu_free_ticks": critical_path(manifest, stages, dma=True, free_engine="VPU"),
    }
    for engine in ("NPU", "VPU", "GPU"):
        by_node = Counter()
        for stage in stages:
            if stage["engine"] == engine:
                by_node[stage["node"]] += int(stage["busy_ticks"])
        roof[f"top_{engine.lower()}_nodes"] = dict(by_node.most_common(10))
    args.out.write_text(json.dumps(roof, indent=2) + "\n")
    print(json.dumps(roof, indent=2))


if __name__ == "__main__":
    main()
