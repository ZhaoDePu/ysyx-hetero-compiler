#!/usr/bin/env python3
"""Join exact Step 2 replay counters with C6 graph and physical launch IDs."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

DRAIN = re.compile(r"^Z15_PLAN_DRAIN stage=(\S+) device=(\w+) wall_npu_ticks=(\d+)$")


def ledger(manifest, log):
    completions, metrics, checks = [], None, 0
    for line in log.splitlines():
        if line.startswith("Z15_DDR_PASS "):
            checks += 1
        elif line.startswith("Z15_METRICS "):
            if metrics is not None:
                raise ValueError("expected one completed RTL frame")
            metrics = json.loads(line.removeprefix("Z15_METRICS "))
        elif match := DRAIN.fullmatch(line):
            name, engine, end = match.groups()
            completions.append((name, engine, int(end)))
    if metrics is None or not checks:
        raise ValueError("RTL log does not prove a completed hardware verification")
    by_stage = {stage: node for node in manifest["c6"]["hardware_ops"] for stage in node["stages"]}
    begin = 0
    planned = manifest["plan"]["stages"]
    if len(planned) != len(completions):
        raise ValueError("RTL stage count differs from the selected manifest")
    rows = []
    for stage, (name, runner_engine, end) in zip(planned, completions, strict=True):
        kind = stage["device"]
        if (name != stage["id"] or (runner_engine == "GPU") != (kind == "GPU") or
                not begin <= end <= metrics["wall_npu_ticks"]):
            raise ValueError(f"invalid stage drain event {name}")
        node = by_stage[name]
        rows.append({"graph_node": node["id"], "op": node["op"], "stage": name, "engine": kind,
                     "begin": begin, "drain": end, "interval_ticks": end - begin})
        begin = end
    summary = {"checked_ddr_tensors": checks, "stage_count": len(rows), "stage_devices": Counter(
        row["engine"] for row in rows), "metrics": metrics,
        "tail_after_last_stage_ticks": metrics["wall_npu_ticks"] - begin,
        "delay_ms": {str(mhz): metrics["wall_npu_ticks"] / (1000 * mhz) for mhz in (125, 164.555)},
        "fps": {str(mhz): mhz * 1e6 / metrics["wall_npu_ticks"] for mhz in (125, 164.555)},
        "interval_note": ("Stage wall intervals include register issue, DMA and drain. "
                          "The hardware counters report combined native NPU/VPU busy ticks.")}
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = ledger(json.loads(args.manifest.read_text()), args.log.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "stage_intervals.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "rtl_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
