"""Build relocatable compiler bundles with native runtime commands and data."""

import json
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory

from . import gpu, npu
from .ir import GraphError, normalize
from .planner import make_plan


def tensor_bytes(values, dtype):
    format_code = {"int8": "b", "uint8": "B", "int32": "i"}[dtype]
    return struct.pack(f"<{len(values)}{format_code}", *values)


def _initial_image(graph, plan, profile, npu_products, directory):
    image = bytearray(profile["sram"]["capacity"])
    allocations = plan["allocations"]
    for name, spec in graph["tensors"].items():
        if "data" in spec and name in allocations:
            data = tensor_bytes(spec["data"], spec["dtype"])
            offset = allocations[name]["offset"]
            image[offset:offset + len(data)] = data
    for const in npu_products.get("constants", []):
        name, offset, data = (const[x] for x in ("tensor", "offset", "data"))
        native = allocations.get(name)
        if not native or native["offset"] != offset or native["size"] != len(data):
            raise GraphError(f"backend initializer disagrees with allocation {name}")
        image[offset:offset + len(data)] = bytes(data)
    base = npu_products.get("binary_base")
    if base is not None:
        data = (directory / "npu.bin").read_bytes()
        if base != plan["arena_end"] or base + len(data) > profile["npu"]["window_bytes"]:
            raise GraphError("descriptor image escapes the protected metadata range")
        image[base:base + len(data)] = data
    bindings = {}
    for name in graph["inputs"]:
        allocated = allocations[name]
        spec = graph["tensors"][name]
        bindings[name] = {key: allocated[key] for key in ("offset", "size")}
        bindings[name].update({key: spec[key] for key in ("dtype", "shape")})
        bindings[name]["needs_data"] = "data" not in spec
    return image, bindings


def _dma_table(graph, plan, profile):
    alignment = profile["dma"]["beat_bytes"]
    if not alignment:
        return []
    output_names = set(graph["outputs"])
    table = []
    for direction in ("load", "store"):
        selected = []
        for name, alloc in plan["allocations"].items():
            if ((direction == "load" and alloc["kind"] in ("input", "constant")) or
                    (direction == "store" and name in output_names)):
                selected.append((alloc["offset"], alloc["stored_size"], name, alloc["zone"]))
        for offset, size, name, zone in sorted(selected):
            if offset % alignment or size % alignment:
                raise GraphError(f"{name}: unaligned DMA extent")
            prev = table[-1] if table else None
            if (prev and prev["direction"] == direction and prev["zone"] == zone and
                    prev["scratch_offset"] + prev["bytes"] == offset):
                prev["bytes"] += size
                prev["tensors"].append(name)
            else:
                table.append({"direction": direction, "scratch_offset": offset, "ddr_offset": offset,
                              "bytes": size, "zone": zone, "tensors": [name]})
    return table


def compile_bundle(raw_graph, profile, output, *, dialect="auto", fuse=True, reuse=True):
    graph = normalize(raw_graph, dialect=dialect)
    plan = make_plan(graph, profile, fuse=fuse, reuse=reuse)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"bundle output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".hetero-") as tmp:
        staging = Path(tmp) / "bundle"
        staging.mkdir()
        gpu_products = gpu.lower(graph, plan, profile, staging)
        npu_products = npu.lower(graph, plan, profile, staging)
        image, bindings = _initial_image(graph, plan, profile, npu_products, staging)
        transfers = _dma_table(graph, plan, profile)
        (staging / "initial.bin").write_bytes(image)
        beat = profile["dma"]["beat_bytes"]
        estimate = {
            "scope": "static estimates; excludes load stalls, pipeline interlocks and runtime overlap",
            "serial_compute_cycles": plan["compute_cycles_serial_estimate"],
            "arena_capacity_bytes": len(image),
            "scratch_high_water_bytes": plan["high_water_bytes"],
            "scratch_peak_live_bytes": plan["peak_live_bytes"],
            "resident_edge_bytes": sum(
                plan["allocations"][name]["size"] for s in plan["stages"] for name in s["inputs"]
                if graph["tensors"][name]["kind"] == "value"),
            "external_transfer_bytes": sum(t["bytes"] for t in transfers),
            "dma_bytes_per_beat": beat,
            "required_ddr_bytes_per_estimated_compute_cycle": round(
                sum(t["bytes"] for t in transfers) / max(1, plan["compute_cycles_serial_estimate"]), 3),
            "ideal_dma_beats": sum(t["bytes"] // beat for t in transfers) if beat else 0,
        }
        manifest = {"schema": "ysyx-heterogeneous-bundle-v1", "graph": graph, "profile": profile,
                    "plan": plan, "input_bindings": bindings, "initial_image": "initial.bin",
                    "backends": {"gpu": gpu_products, "npu": npu_products}, "dma": transfers,
                    "estimates": estimate}
        # Round-trip the returned dictionary into the same JSON shape as the file.
        encoded = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        (staging / "bundle.json").write_text(encoded, encoding="utf-8")
        (staging / "memory_plan.json").write_text(
            json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (staging / "analysis.json").write_text(json.dumps(estimate, indent=2) + "\n", encoding="utf-8")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"bundle output already exists: {output}")
        os.rename(staging, output)
        return json.loads(encoded)
