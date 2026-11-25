"""Account for every source ONNX operator and frozen C99 RTL launch."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def node_roles(manifest, quant, nodes, tensors):
    c6 = manifest["c6"]
    placed = {op["id"]: op for op in c6["hardware_ops"]}
    host = {op["id"] for op in c6["host_tail"]}
    fused = {
        skip: op["id"]
        for op in quant["nodes"]
        for skip in op["attrs"].get("fused_onnx_ids", ())
    }
    inputs = {row["name"]: json.loads(row["inputs"]) for row in nodes}
    outputs = {row["name"]: json.loads(row["outputs"]) for row in nodes}
    producer = {name: row["name"] for row in nodes for name in outputs[row["name"]]}
    constants = {row["name"] for row in tensors if row["kind"] == "constant"}
    for row in nodes:
        label = row["name"]
        if row["op_type"] == "Mul" and placed.get(label, {}).get("op") == "Lookup":
            for value in inputs[label]:
                parent = producer.get(value)
                if parent is not None and parent not in placed and value not in constants:
                    fused[parent] = label

    rows = []
    for row in nodes:
        label = row["name"]
        emitted = fused.get(label, label)
        if emitted in placed:
            op = placed[emitted]
            engines = "+".join(op["engines"])
            lowering = op["op"] if emitted == label else "fused into " + op["op"]
            launches = len(op["stages"]) if emitted == label else 0
        elif label in host:
            engines, lowering, launches = "PS", "ONNX detection tail", 0
        elif all(name == "" or name in constants for name in inputs[label]):
            constants.update(outputs[label])
            engines, lowering, launches = "compile-time", "constant folding", 0
        else:
            raise ValueError(f"unaccounted ONNX node {label}")
        rows.append({"index": row["index"], "node": label, "onnx_op": row["op_type"],
                     "engines": engines, "lowering": lowering, "lowered_id": emitted,
                     "physical_launches": launches, "logical_macs": row["macs"]})
    return rows


STAGE = re.compile(
    r"STAGE id=(\d+) name=(\S+) engine=(NPU|VPU|GPU) npu_domain_busy_ticks=(\d+) fence_ticks=(\d+)"
)


def runtime_ledger(manifest, path):
    log = path.read_text(encoding="utf-8")
    stages, summary = [], None
    for line in log.splitlines():
        if match := STAGE.fullmatch(line):
            index, name, engine, busy, fence = match.groups()
            stages.append({"index": int(index), "stage": name, "engine": engine,
                           "busy_ticks": int(busy), "fence_ticks": int(fence)})
        elif line.startswith("STATS "):
            if summary is not None:
                raise ValueError("expected one frozen RTL frame")
            summary = {name: int(value, 0) for name, value in
                       re.findall(r"(\w+)=(0x[\da-f]+|\d+)", line)}
    if not summary or summary.get("abi_error") != 0 or summary.get("invalid_dma_descriptors") != 0:
        raise ValueError("missing or invalid completion report")
    planned = manifest["plan"]["stages"]
    if len(planned) != len(stages):
        raise ValueError("incomplete trace for the manifest")
    provenance = {name: node for node in manifest["c6"]["hardware_ops"] for name in node["stages"]}
    totals = Counter()
    for i, (stage, plan) in enumerate(zip(stages, planned, strict=True)):
        if (stage["index"], stage["stage"], stage["engine"]) != (i, plan["id"], plan["device"]):
            raise ValueError(f"mismatched stage {i}: {stage}")
        totals[stage["engine"]] += stage["busy_ticks"]
        stage["node"] = provenance[stage["stage"]]["id"]
        stage["op"] = provenance[stage["stage"]]["op"]
        stage["ownership_half"] = plan["zone"]
    if totals["NPU"] + totals["VPU"] != summary["npu_busy_ticks"]:
        raise ValueError("native split does not sum to its physical counter")
    if totals["GPU"] != summary["gpu_busy_ticks"]:
        raise ValueError("GPU stage sum does not match its physical counter")
    wall = summary["npu_domain_wall_ticks"]
    summary.update({"npu_engine_busy_ticks": totals["NPU"], "vpu_engine_busy_ticks": totals["VPU"],
                    "latency_ms_125_mhz": wall / 125000, "fps_125_mhz": 125e6 / wall,
                    "latency_ms_164_555_mhz": wall / 164555, "fps_164_555_mhz": 164555e3 / wall})
    return stages, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("quant", "nodes", "tensors", "rtl-log"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    if args.quant and args.nodes and args.tensors:
        roles = node_roles(manifest, json.loads(args.quant.read_text()), read_csv(args.nodes),
                           read_csv(args.tensors))
        write_csv(args.out / "onnx_operator_placement.csv", roles)
        print("ONNX operators", len(roles), dict(Counter(row["engines"] for row in roles)))
    elif args.quant or args.nodes or args.tensors:
        parser.error("--quant, --nodes and --tensors are used together")
    if args.rtl_log:
        stages, summary = runtime_ledger(manifest, args.rtl_log)
        write_csv(args.out / "rtl_stages.csv", stages)
        (args.out / "rtl_counters.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
