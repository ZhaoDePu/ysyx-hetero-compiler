"""Measure isolated, whole unique classifier operators for the GPU-only A0 baseline."""

import argparse
from collections import defaultdict
from fnmatch import fnmatchcase
from hashlib import sha256
import json
from math import prod
import os
from pathlib import Path
import subprocess

from hetero_compiler.c6_quant import QuantGraph
from tools.c9_classifier import replay


METRICS = ("npu_domain_wall_ticks", "npu_engine_busy_ticks", "vpu_engine_busy_ticks", "gpu_busy_ticks",
           "dma_busy_ticks", "npu_gpu_overlap_ticks", "dma_compute_overlap_ticks", "dma_bytes",
           "dma_operations")


def geometry(source, node):
    """Equal shape, layout and operator geometry, with actual case values held out."""
    def tensor(name):
        if name in source.constants:
            array = source.constants[name]
            return {"dtype": array.dtype.name, "shape": list(array.shape), "const": True}
        desc = source.tensors[name]
        return {key: desc[key] for key in ("dtype", "shape", "physical_shape", "layout")}

    attrs = node["attrs"]
    excluded = {"weight", "bias", "mul", "shift", "lut", "fused_edge_ids"}
    return {
        "operator": node["op"],
        "inputs": [tensor(name) for name in node["inputs"] if name],
        "outputs": [tensor(name) for name in node["outputs"]],
        "shape_attrs": {key: attrs[key] for key in sorted(attrs.keys() - excluded)},
        "has_lut": "lut" in attrs,
    }


def cases(source):
    grouped = defaultdict(list)
    for node in source.graph["nodes"]:
        signature = json.dumps(geometry(source, node), sort_keys=True)
        grouped[signature].append(node)
    rows = []
    for signature, nodes in grouped.items():
        first = nodes[0]
        macs = first.get("estimated_macs", 0)
        outputs = sum(prod(source.tensors[name]["shape"]) for name in first["outputs"])
        rows.append({
            "signature": json.loads(signature),
            "key": sha256(signature.encode()).hexdigest()[:12],
            "representative": first["id"],
            "members": [node["id"] for node in nodes],
            "output_elements": outputs,
            "logical_macs_per_instance": macs,
            "dispatch_priority": max(outputs, macs),
        })
    return sorted(rows, key=lambda row: (row["dispatch_priority"], row["representative"]))


def write_json(path, material):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(material, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def qualified(path, npu_mhz, gpu_mhz, output_elements):
    row = json.loads(path.read_text(encoding="utf-8"))
    if (row["stage"] != "A0" or (row["npu_mhz"], row["gpu_mhz"]) != (npu_mhz, gpu_mhz)
            or row["compared_bytes"] != output_elements
            or any(row[key] for key in ("npu_busy_ticks", "abi_error", "invalid_dma_descriptors", "slverr"))
            or row["gpu_busy_ticks"] <= 0):
        raise ValueError(f"{path}: replay is not a byte-exact pure GPU measurement at this clock")
    return row


def record(args, source, plan, timings, errors):
    covered = [entry for entry in plan if entry["key"] in timings]
    complete = len(covered) == len(plan)
    n = sum(len(entry["members"]) for entry in plan)
    covered_n = sum(len(entry["members"]) for entry in covered)
    totals = {key: sum(timings[entry["key"]][key] * len(entry["members"])
                       for entry in covered) for key in METRICS}
    payload = {
        "method": "whole_unique_layer_frozen_rtl_aggregate" if complete else "partial_whole_unique_layer_rtl",
        "coverage_complete": complete,
        "merged_sources": [str(path) for path in getattr(args, "merge_from", ())],
        "reference": "Each representative executes the complete selected operator with its captured real input.",
        "limitations": ("The aggregate weights geometry representatives by occurrences. Requant and "
                        "saturation may diverge on data. Graph-level warm state, graph DDR and SRAM "
                        "peak, and overlap between operators require a separate full-graph plan."),
        "gpu_program_upload_policy": upload_policy(),
        "source_onnx_sha256": source.graph.get("source_onnx_sha256"),
        "npu_mhz": args.npu_mhz,
        "gpu_mhz": args.gpu_mhz,
        "unique_measured": len(covered), "unique_total": len(plan),
        "layer_instances_measured_or_represented": covered_n, "layer_instances_total": n,
        "logical_macs_measured_or_represented": sum(
            entry["logical_macs_per_instance"] * len(entry["members"]) for entry in covered),
        "logical_macs_total": sum(entry["logical_macs_per_instance"] * len(entry["members"]) for entry in plan),
        "whole_graph_aggregate": totals if complete else None,
        "covered_work_only": totals,
        "max_isolated_sram_peak_bytes": max((row["sram_peak_bytes"] for row in timings.values()), default=None),
        "max_isolated_ddr_peak_bytes": max((row["ddr_peak_bytes"] for row in timings.values()), default=None),
        "errors": errors,
        "unique_cases": [{**entry, "verdict": timings.get(entry["key"])} for entry in plan],
    }
    write_json(args.out / "aggregation.json", payload)


def upload_policy():
    return "always" if os.getenv("YSYX_C8_GPU_UPLOAD") == "always" else "consecutive_reuse"


def read_timings(args, source, plan):
    """Only merge actual whole-operator observations with common provenance."""
    timings = {}
    for root in (args.out, *args.merge_from):
        report = root / "aggregation.json"
        if not report.is_file() or not any(root.glob("*/qualified.json")):
            continue
        prior = json.loads(report.read_text(encoding="utf-8"))
        if (prior.get("gpu_program_upload_policy") != upload_policy()
                or prior.get("source_onnx_sha256") != source.graph.get("source_onnx_sha256")):
            raise ValueError(f"{root}: study model or GPU upload policy would differ")
        for entry in plan:
            key = entry["key"]
            result = root / key / "qualified.json"
            if not result.is_file():
                continue
            try:
                row = qualified(result, args.npu_mhz, args.gpu_mhz, entry["output_elements"])
            except json.JSONDecodeError:
                if root == args.out:
                    raise
                continue  # A shard can be writing this result concurrently.
            if key in timings and row != timings[key]:
                raise ValueError(f"{result}: conflicting qualified observations for the same representative")
            timings[key] = row
    return timings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "out"):
        ap.add_argument(f"--{key}", type=Path, required=True)
    ap.add_argument("--executor", type=Path)
    ap.add_argument("--npu-mhz", type=float, default=125.0)
    ap.add_argument("--gpu-mhz", type=float, default=100.0)
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--only", action="append", help="representative ID or key glob, repeat for further cases")
    ap.add_argument("--largest-first", action="store_true", help="measure larger representative shapes first")
    ap.add_argument("--merge-from", action="append", type=Path, default=[],
                    help="read compatible whole measurements from a disjoint shard output")
    ap.add_argument("--limited-representatives", type=int,
                    help="run at most this many pending eligible representatives")
    args = ap.parse_args()
    if not args.plan_only and not args.executor:
        ap.error("--executor is required for RTL replay")
    source = QuantGraph(args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    plan = cases(source)
    write_json(args.out / "plan.json", plan)
    timings = read_timings(args, source, plan)
    errors = []
    record(args, source, plan, timings, errors)
    if args.plan_only:
        print(f"{len(plan)} unique operator geometries", flush=True)
        return
    attempted = 0
    order = reversed(plan) if args.largest_first else plan
    for entry in order:
        name, key = entry["representative"], entry["key"]
        if args.merge_from:
            timings.update(read_timings(args, source, plan))
        if key in timings or (args.only and not any(fnmatchcase(name, pattern) or fnmatchcase(key, pattern)
                                                   for pattern in args.only)):
            continue
        if args.limited_representatives is not None and attempted >= args.limited_representatives:
            break
        attempted += 1
        output = args.out / key
        try:
            replay(args.model, output, "A0", args.executor, args.npu_mhz, args.gpu_mhz,
                   args.timeout, {name})
            timings[key] = qualified(output / "qualified.json", args.npu_mhz, args.gpu_mhz,
                                     entry["output_elements"])
        except (ValueError, subprocess.SubprocessError) as err:
            errors.append({"key": key, "representative": name, "failure": str(err)})
            print(f"{name}: FAIL ({err})", flush=True)
        else:
            print(f"{name}: {timings[key]['npu_domain_wall_ticks']} wall ticks, byte-exact", flush=True)
        record(args, source, plan, timings, errors)
    record(args, source, plan, timings, errors)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
