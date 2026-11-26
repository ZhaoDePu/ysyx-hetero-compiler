"""Build the ICCD metrics ledger from qualified frozen RTL records."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path


FROZEN = "Z15 Step 2 4d410d72508a5822bc5c8014981e5b55ef8faa56"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def classifier_row(workload, metadata_path, result_path):
    source = _load(metadata_path)
    data = _load(result_path)
    manifest = result_path.parent / "manifest.json"
    plan = _load(manifest)
    if (set(plan["input_bindings"]) != {"images"} or
            not plan["input_bindings"]["images"]["needs_data"] or
            set(plan["output_bindings"]) != {"logits"} or
            plan["output_bindings"]["logits"]["size"] != 1000 or data["compared_bytes"] != 1000):
        raise ValueError(f"classifier result is not a whole image to logits chain: {result_path}")
    measured_npu = data["npu_engine_busy_ticks"]
    measured_vpu = data["vpu_engine_busy_ticks"]
    operations = source.get("logical_ops_two_per_mac") or 2 * source["logical_macs"]
    width = source.get("input_resolution", source.get("resolution"))
    if isinstance(width, list):
        resolution = " × ".join(map(str, width))
    else:
        resolution = f"{width} × {width}"
    weights = source.get("weight_name", source.get("weight_source", source.get("checkpoint_url", "")))
    checksum = source.get("weights_sha256", source.get("weight_sha256", source.get("checkpoint_sha256", "")))
    if not weights or not checksum:
        raise ValueError(f"missing evidence for frozen model weights: {metadata_path}")
    with (result_path.parent / "rtl-stages.csv").open(newline="", encoding="utf-8") as stream:
        wide_k = sum(int(r["busy_ticks"]) for r in csv.DictReader(stream)
                     if r["engine"] == "GPU" and "/reduce_" in r["stage"])
    return {
        "workload": workload, "stage": data["stage"], "variant": result_path.parent.name,
        "resolution": resolution, "weight_source": f"{weights} (SHA-256 {checksum})",
        "weight_metadata": str(metadata_path),
        "logical_ops": operations, "ops_scope": "2 × logical Conv/Gemm MACs from the exported model",
        "measurement_kind": data.get("measurement_kind", "rtl_end_to_end"),
        "frozen_core": FROZEN, "npu_mhz": data["npu_mhz"], "gpu_mhz": data["gpu_mhz"],
        "unit": "frame", "work_units": 1, "sample_count": 1,
        "source": str(result_path.parent / "rtl.log"), "source_summary": str(result_path),
        "manifest": str(manifest), "validation_scope": "whole image to all classifier logits",
        "wall_cycles": data["npu_domain_wall_ticks"], "npu_busy_cycles": measured_npu,
        "vpu_busy_cycles": measured_vpu, "native_busy_cycles": measured_npu + measured_vpu,
        "gpu_busy_cycles": data["gpu_busy_ticks"], "dma_busy_cycles": data["dma_busy_ticks"],
        "compute_dma_overlap_cycles": data["dma_compute_overlap_ticks"],
        "native_gpu_overlap_cycles": data["npu_gpu_overlap_ticks"],
        "triple_overlap_cycles": data.get("triple_overlap_ticks"),
        "sram_peak_bytes": data["sram_peak_bytes"], "ddr_peak_bytes": data["ddr_peak_bytes"],
        "dma_bytes": data["dma_bytes"], "dma_descriptors": data["dma_operations"],
        "dma_descriptors_source": "YHB1 C99 runtime DMA count",
        "compared_bytes": data["compared_bytes"], "exact": True,
        "error": data["abi_error"], "slverr": data["slverr"], "invalid_dma": data["invalid_dma_descriptors"],
        "wide_k_gpu_cycles": data.get("wide_k_gpu_cycles", None if data["stage"] == "A0" else wide_k),
        "vpu_candidate_upper_cycles": measured_vpu,
    }


def aggregate_row(workload, meta_path, aggregation, static_plan):
    source, data = _load(meta_path), _load(aggregation)
    if not data["coverage_complete"] or data["errors"] or data["unique_measured"] != data["unique_total"]:
        raise ValueError(f"unqualified A0 geometry aggregate: {aggregation}")
    results = [selected["verdict"] for selected in data["unique_cases"]]
    count = data["unique_total"]
    if not all(r and r["stage"] == "A0"
               and not any(r[key] for key in ("abi_error", "slverr", "invalid_dma_descriptors", "npu_busy_ticks"))
               for r in results):
        raise ValueError(f"an A0 measurement is missing or used a native engine: {aggregation}")
    w = source.get("input_resolution", source.get("resolution"))
    resolution = f"{w} × {w}" if isinstance(w, int) else " × ".join(map(str, w))
    k = data["whole_graph_aggregate"]
    plan = _load(static_plan)
    if (plan.get("dma_bytes", plan.get("dma_bytes_planned")) != k["dma_bytes"] or
            plan.get("dma_operations", plan.get("dma_operations_planned")) != k["dma_operations"]):
        raise ValueError(f"compiled full graph traffic differs from weighted verified layers: {aggregation}")
    weights = source.get("weight_name", source.get("weight_source"))
    digest = source.get("weights_sha256", source.get("weight_sha256"))
    ops = data["logical_macs_total"] * 2
    official_ops = source.get("logical_ops_two_per_mac", None)
    if official_ops is None:
        official_ops = 2 * source["logical_macs"]
    if ops != official_ops:
        raise ValueError(f"aggregation omitted logical MACs: {aggregation}")
    return {
        "workload": workload, "stage": "A0", "variant": "pure-gpu-geometry-aggregate",
        "resolution": resolution, "weight_source": f"{weights} (SHA-256 {digest})",
        "weight_metadata": str(meta_path), "logical_ops": ops,
        "ops_scope": "2 × logical Conv/Gemm MACs from topology multiplicities",
        "measurement_kind": "rtl_layer_aggregate", "frozen_core": FROZEN,
        "npu_mhz": data["npu_mhz"], "gpu_mhz": data["gpu_mhz"],
        "unit": "frame_equivalent_aggregate", "work_units": 1, "sample_count": count,
        "source": str(aggregation), "source_summary": str(aggregation),
        "validation_scope": "full outputs of every isolated representative operator only",
        "wall_cycles": k["npu_domain_wall_ticks"], "npu_busy_cycles": 0, "vpu_busy_cycles": 0,
        "native_busy_cycles": 0, "gpu_busy_cycles": k["gpu_busy_ticks"],
        "dma_busy_cycles": k["dma_busy_ticks"], "compute_dma_overlap_cycles": k["dma_compute_overlap_ticks"],
        "native_gpu_overlap_cycles": 0, "triple_overlap_cycles": 0,
        "sram_peak_bytes": data["max_isolated_sram_peak_bytes"],
        "ddr_peak_bytes": data["max_isolated_ddr_peak_bytes"],
        "memory_peak_scope": "maximum isolated representative peak",
        "full_graph_static_plan": str(static_plan),
        "full_graph_static_sram_peak_bytes": plan["sram_peak_bytes"],
        "full_graph_static_ddr_peak_bytes": plan["ddr_peak_bytes"],
        "dma_bytes": k["dma_bytes"], "dma_descriptors": k["dma_operations"],
        "dma_descriptors_source": "measured representatives weighted by number of typed geometry instances",
        "compared_bytes": sum(r["compared_bytes"] for r in results), "exact": True,
        "error": 0, "slverr": 0, "invalid_dma": 0, "wide_k_gpu_cycles": None, "vpu_candidate_upper_cycles": 0,
    }


def _rate(ticks, mhz, units):
    elapsed_ms = ticks / (1000 * mhz)
    return elapsed_ms, units * 1000 / elapsed_ms


def derive(r):
    row = dict(r)
    wall = row["wall_cycles"]
    units = row["work_units"]
    freq = (row["npu_mhz"], row["gpu_mhz"])
    if wall <= 0 or not row["exact"] or any(row[field] not in (None, 0) for field in ("error", "slverr", "invalid_dma")):
        raise ValueError(f"unqualified RTL record {row['workload']} {row['variant']}")
    row["latency_ms_measured_pair"], row["throughput_measured_pair"] = _rate(wall, freq[0], units)
    for npu, gpu in ((125, 100), (200, 150)):
        ms, throughput = _rate(wall, npu, units) if freq == (npu, gpu) else (None, None)
        row[f"latency_ms_{npu}_{gpu}"], row[f"throughput_{npu}_{gpu}"] = ms, throughput
    native, gpu, dma = (row[k] for k in ("native_busy_cycles", "gpu_busy_cycles", "dma_busy_cycles"))
    native_gpu, compute_dma = (row[k] for k in ("native_gpu_overlap_cycles", "compute_dma_overlap_cycles"))
    triple = row.get("triple_overlap_cycles")
    if triple is None and (native_gpu == 0 or compute_dma == 0):
        triple = 0
    if all(x is not None for x in (native, gpu, dma, native_gpu, compute_dma, triple)):
        compute_union = native + gpu - native_gpu
        union = compute_union + dma - compute_dma
        idle = wall - union
        row.update(idle_from_union_cycles=idle, single_compute_cycles=compute_union - (native_gpu - triple) - compute_dma,
                   native_gpu_without_dma_cycles=native_gpu - triple, dma_only_cycles=dma - compute_dma)
        if row["measurement_kind"] == "rtl_end_to_end" and min(row[key] for key in
                ("single_compute_cycles", "native_gpu_without_dma_cycles", "dma_only_cycles", "idle_from_union_cycles")) < 0:
            raise ValueError(f"incompatible simultaneous counter data: {row['source']}")
    for engine in ("npu", "vpu", "native", "gpu", "dma"):
        busy = row[f"{engine}_busy_cycles"]
        row[f"{engine}_wall_fraction"] = busy / wall if busy is not None else None
    row["native_gpu_overlap_fraction"] = native_gpu / wall if native_gpu is not None else None
    row["compute_dma_overlap_fraction"] = compute_dma / wall if compute_dma is not None else None
    ops, traffic = row["logical_ops"], row["dma_bytes"]
    row["ops_per_dma_byte"] = ops / traffic if ops and traffic else None
    row["gops_at_measured_pair"] = ops * freq[0] / wall / 1000 if ops else None
    row["dma_bytes_per_busy_cycle"] = traffic / dma if traffic is not None and dma else None
    for label, key in (("wide_k", "wide_k_gpu_cycles"), ("vpu", "vpu_candidate_upper_cycles")):
        saving = row.get(key)
        if saving is None:
            row[f"{label}_max_busy_share"] = row[f"{label}_optimistic_speedup"] = None
            continue
        row[f"{label}_max_busy_share"] = saving / wall
        native_fixed = (native - saving if label == "vpu" else native) if native is not None else 0
        gpu_fixed = (gpu - saving if label == "wide_k" else gpu) if gpu is not None else 0
        row[f"{label}_optimistic_speedup"] = wall / max(wall - saving, native_fixed, gpu_fixed, dma or 0, 1)
    return row


def derive_roofs(rows):
    result = {}
    for npu, gpu in ((125, 100), (200, 150)):
        points = [row for row in rows if (row["npu_mhz"], row["gpu_mhz"]) == (npu, gpu)
                  and row["measurement_kind"] == "rtl_end_to_end"]
        main = [r for r in points if r["workload"] in ("YOLO26n", "TinyLM-Chat", "ResNet-18", "MobileNetV2")]
        bandwidth = [row for row in main if row["dma_bytes_per_busy_cycle"] is not None]
        empirical = max(bandwidth, key=lambda r: r["dma_bytes_per_busy_cycle"], default=None)
        result[f"{npu}_{gpu}"] = {
            "array_peak_gops": 2 * 16 * 16 * 4 * npu / 1000,
            "interface_peak_gb_s": 32 * npu / 1000,
            "best_observed_dma_gb_s": empirical["dma_bytes_per_busy_cycle"] * npu / 1000 if empirical else None,
            "best_observed_dma_source": empirical["source"] if empirical else None,
            "scope": "old frozen Step 2 RTL at the listed frequency ratio; excludes the PS host wall time",
        }
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--history", type=Path, required=True, action="append")
    cli.add_argument("--classifier", nargs=3, action="append", default=[], metavar=("MODEL", "META", "QUALIFIED"))
    cli.add_argument("--a0-aggregate", nargs=4, action="append", default=[],
                     metavar=("MODEL", "META", "AGGREGATION", "STATIC_PLAN"))
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    rows = [derive(row) for path in args.history for row in _load(path)]
    rows.extend(derive(classifier_row(name, Path(metadata), Path(result))) for name, metadata, result in args.classifier)
    rows.extend(derive(aggregate_row(name, Path(metadata), Path(report), Path(plan)))
                for name, metadata, report, plan in args.a0_aggregate)
    best_by_baseline = defaultdict(list)
    for row in rows:
        key = (row["workload"], row["resolution"], row["npu_mhz"], row["gpu_mhz"])
        best_by_baseline[key].append(row)
    for group in best_by_baseline.values():
        for row in group:
            stage = "C6" if row["workload"] == "YOLO26n" else "A0"
            candidates = [baseline for baseline in group if baseline["stage"] == stage]
            if row.get("native_vpu_mode"):
                candidates = [r for r in candidates if r.get("native_vpu_mode") == row["native_vpu_mode"]]
            candidate = next((r for r in candidates if r["variant"].endswith("-on")), candidates[0] if candidates else None)
            row["speedup_vs_same_ratio_baseline"] = candidate["wall_cycles"] / row["wall_cycles"] if candidate else None
            row["baseline_measurement_kind"] = candidate["measurement_kind"] if candidate else None
            for name in ("dma_bytes", "dma_descriptors"):
                before = candidate.get(name) if candidate else None
                after = row.get(name)
                row[f"{name}_reduction_fraction"] = 1 - after / before if before and after is not None else None
    roofs = derive_roofs(rows)
    for row in rows:
        key = f"{row['npu_mhz']:g}_{row['gpu_mhz']:g}"
        roof = roofs.get(key)
        intensity = row["ops_per_dma_byte"]
        actual = row["gops_at_measured_pair"]
        if not roof or not intensity or not actual or row["measurement_kind"] != "rtl_end_to_end":
            continue
        for name, bandwidth in (("theoretical", roof["interface_peak_gb_s"]),
                                ("empirical", roof["best_observed_dma_gb_s"])):
            if bandwidth:
                cap = min(roof["array_peak_gops"], intensity * bandwidth)
                row[f"{name}_roof_gops"] = cap
                row[f"{name}_roof_fraction"] = actual / cap
    args.out.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.out / "iccd_master_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (args.out / "iccd_master_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump({"rows": rows, "rooflines": roofs}, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"{len(rows)} verified rows")


if __name__ == "__main__":
    main()
