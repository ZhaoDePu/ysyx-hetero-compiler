"""Reconstruct C7 DMA ownership and reconcile it with the frozen RTL trace."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from unittest.mock import patch

from hetero_compiler import c6_schedule
from hetero_compiler.c6_compile import fuse_conv_edge_luts, prepare_graph
from hetero_compiler.c6_quant import QuantGraph
from hetero_compiler.c6_tensor import Transfer
from tools.c6_accounting import read_csv, runtime_ledger, write_csv


DMA = re.compile(
    r"DMA direction=(in|out) ddr=(\d+) scratch=(\d+) bytes=(\d+) issue_to_fence_ticks=(\d+)"
)


def replay_transfers(quant, manifest, scratch):
    source = QuantGraph(quant)
    graph = fuse_conv_edge_luts(source.graph)
    *_, layers, _, placement = prepare_graph(graph, source.constants, fast_reduce=True)
    by_stage = {stage: node["id"] for node in placement["hardware_ops"] for stage in node["stages"]}
    owners = {id(move): by_stage[tile.stages[0].id] for layer in layers
              for tile in layer for move in (*tile.loads, *tile.stores)}
    for layer in layers:
        for tile in layer:
            if len({by_stage[stage.id] for stage in tile.stages}) != 1:
                raise ValueError("a C7 tile spans more than one operator")

    command = Transfer.command
    attribution = []

    def traced(move):
        attribution.append(owners[id(move)])
        return command(move)

    program, stages = [], []
    with TemporaryDirectory(dir=scratch) as directory, patch.object(Transfer, "command", traced):
        c6_schedule._emit_overlap(layers, program, stages, [], [], Path(directory))
    if program != manifest["execution"] or stages != manifest["plan"]["stages"]:
        raise ValueError("reconstruction differs from the published C7 commands")
    return zip((op for op in program if op["type"] == "dma"), attribution, strict=True)


def union_size(spans):
    size, right = 0, 0
    for start, end in sorted(spans):
        size += max(0, end - max(start, right))
        right = max(end, right)
    return size


def classify(move, regions):
    start = move["ddr_offset"]
    end = start + move["bytes"]
    for region, (lower, upper) in regions.items():
        if lower <= start < end <= upper:
            return region
    raise ValueError(f"descriptor crosses a DDR region: {move}")


def profile(manifest, quant, nodes, log, scratch):
    stages, metrics = runtime_ledger(manifest, log)
    measures = defaultdict(Counter)
    spans = defaultdict(list)
    for stage in stages:
        counters = measures[stage["node"]]
        counters[stage["engine"]] += stage["busy_ticks"]
        counters["fence_windows"] += stage["fence_ticks"]
        counters["launches"] += 1

    regions = manifest["c6"]["ddr_regions"]
    partials = defaultdict(list)
    for name, slot in manifest["plan"]["allocations"].items():
        if re.search(r"/@partial\d+$", name):
            start = slot["offset"]
            partials[name.rsplit("/", 1)[0]].append((start, start + ((slot["size"] + 31) & -32)))
    actual = (DMA.fullmatch(line).groups() for line in log.read_text().splitlines()
              if line.startswith("DMA "))
    for (move, owner), event in zip(replay_transfers(quant, manifest, scratch), actual, strict=True):
        direction, ddr, start, length, duration = event
        expect = ("in" if move["direction"] == "load" else "out", move["ddr_offset"],
                  move["scratch_offset"], move["bytes"])
        if (direction, int(ddr), int(start), int(length)) != expect:
            raise ValueError(f"DMA data/order differs at {event}")
        counters = measures[owner]
        region = classify(move, regions)
        counters[f"{region}_{direction}_bytes"] += int(length)
        counters["dma_bytes"] += int(length)
        counters["dma_descriptors"] += 1
        counters["dma_issue_to_fence_ticks"] += int(duration)
        if any(low <= int(ddr) and int(ddr) + int(length) <= high
               for low, high in partials[owner]):
            counters[f"partial_{direction}_bytes"] += int(length)
        if region == "weights":
            if direction != "in":
                raise ValueError("write to constant DDR region")
            spans[owner].append((int(ddr), int(ddr) + int(length)))

    source_ops = {node["name"]: node for node in nodes}
    physical = {node["id"]: node for node in manifest["c6"]["hardware_ops"]}
    if set(physical) != set(measures):
        raise ValueError("trace missed a physical ONNX operator")
    merged_into = {edge: node["id"] for node in physical.values()
                   for edge in node.get("fused_edge_ids", ())}
    source_names = list(dict.fromkeys([row["name"] for row in nodes] + list(physical)))
    rows = []
    for name in source_names:
        source = source_ops.get(name, {})
        placement = physical.get(name, {})
        counter = measures.get(name, Counter())
        single = union_size(spans[name])
        launched = bool(placement)
        rows.append({
            "node": name, "onnx_op": source.get("op_type", "compiler-generated"),
            "lowered_op": placement.get("op", ""), "logical_macs": source.get("macs", 0),
            "fused_into": merged_into.get(name, ""), "launches": counter["launches"],
            "npu_busy_ticks": counter["NPU"], "vpu_busy_ticks": counter["VPU"],
            "gpu_busy_ticks": counter["GPU"],
            "dma_busy_ticks": "" if launched else 0,
            "dma_issue_to_fence_ticks": counter["dma_issue_to_fence_ticks"],
            "idle_ticks": "" if launched else 0,
            "compute_fence_window_ticks": counter["fence_windows"],
            "dma_bytes": counter["dma_bytes"], "dma_descriptors": counter["dma_descriptors"],
            "constant_dma_bytes": counter["weights_in_bytes"],
            "constant_unique_address_bytes": single,
            "constant_repeat_bytes": counter["weights_in_bytes"] - single,
            "working_load_bytes": counter["activation_in_bytes"],
            "working_store_bytes": counter["activation_out_bytes"],
            "partial_load_bytes": counter["partial_in_bytes"], "partial_store_bytes": counter["partial_out_bytes"],
            "frame_load_bytes": counter["frames_in_bytes"], "frame_store_bytes": counter["frames_out_bytes"],
        })

    for key, metric in (("dma_bytes", "dma_bytes"), ("dma_descriptors", "dma_operations")):
        if sum(row[key] for row in rows) != metrics[metric]:
            raise ValueError(f"{key} does not match the frozen total")
    busy_union = (metrics["npu_busy_ticks"] + metrics["gpu_busy_ticks"] + metrics["dma_busy_ticks"]
                  - metrics["npu_gpu_overlap_ticks"] - metrics["dma_compute_overlap_ticks"])
    totals = {
        **metrics,
        "idle_ticks": metrics["npu_domain_wall_ticks"] - busy_union,
        "dma_issue_to_fence_ticks": sum(row["dma_issue_to_fence_ticks"] for row in rows),
        "onnx_nodes": len(nodes), "launched_nodes": len(physical),
        "constant_dma_bytes": sum(row["constant_dma_bytes"] for row in rows),
        "constant_unique_address_bytes_per_operator": sum(row["constant_unique_address_bytes"] for row in rows),
        "constant_repeat_bytes_per_operator": sum(row["constant_repeat_bytes"] for row in rows),
        "working_load_bytes": sum(row["working_load_bytes"] for row in rows),
        "working_store_bytes": sum(row["working_store_bytes"] for row in rows),
        "partial_load_bytes": sum(row["partial_load_bytes"] for row in rows),
        "partial_store_bytes": sum(row["partial_store_bytes"] for row in rows),
        "frame_load_bytes": sum(row["frame_load_bytes"] for row in rows),
        "frame_store_bytes": sum(row["frame_store_bytes"] for row in rows),
    }
    rows.append({**dict.fromkeys(rows[0], ""), "node": "@frame-only",
                 "onnx_op": "counter-only", "dma_busy_ticks": totals["dma_busy_ticks"],
                 "idle_ticks": totals["idle_ticks"]})
    return rows, totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "quant", "nodes", "log", "scratch", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows, totals = profile(manifest, args.quant, read_csv(args.nodes), args.log, args.scratch)
    write_csv(args.out / "node-breakdown.csv", rows)
    (args.out / "accounting.json").write_text(json.dumps(totals, indent=2) + "\n")


if __name__ == "__main__":
    main()
