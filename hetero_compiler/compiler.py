"""Build relocatable compiler bundles with native runtime commands and data."""

import json
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory

from . import asic_ip, gpu, npu
from .ir import GraphError, normalize
from .planner import make_plan
from .pipeline import make_pipeline_plan
from .streaming import make_stream_plan
from .yhb import TARGET_IDS, emit_yhb


def tensor_bytes(values, dtype):
    format_code = {"int8": "b", "uint8": "B", "int16": "h", "uint16": "H", "int32": "i", "int64": "q"}[dtype]
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


def _stream_images(graph, plan, profile, native):
    ddr = bytearray(plan["ddr_image_bytes"])
    for name, alloc in plan["ddr_allocations"].items():
        spec = graph["tensors"][name]
        if "data" in spec:
            data = tensor_bytes(spec["data"], spec["dtype"])
            ddr[alloc["offset"]:alloc["offset"] + len(data)] = data
    for entry in native["constants"]:
        name, data = entry["tensor"], bytes(entry["data"])
        start = plan["ddr_packed_offsets"][name]
        if len(data) != plan["allocations"][name]["size"]:
            raise GraphError(f"{name}: native DDR constant size mismatch")
        ddr[start:start+len(data)] = data

    def describe(names, for_inputs):
        bindings = {}
        for name in names:
            rec = plan["ddr_allocations"][name]
            spec = graph["tensors"][name]
            bindings[name] = {"offset": rec["offset"], "ddr_offset": rec["offset"], "size": rec["size"],
                              "dtype": spec["dtype"], "shape": spec["shape"]}
            if for_inputs:
                bindings[name]["needs_data"] = "data" not in spec
        return bindings
    return (bytearray(profile["sram"]["capacity"]), ddr,
            describe(graph["inputs"], True), describe(graph["outputs"], False))


def compile_bundle(raw_graph, profile, output, *, dialect="auto", fuse=True, reuse=True,
                   overlap=True, device_policy="auto", stream_ddr=False, emit_binary=None, pipeline_tiles=False):
    graph = normalize(raw_graph, dialect=dialect)
    physical, execution = graph, None
    options = {"fuse": fuse, "reuse": reuse, "overlap": overlap, "device_policy": device_policy}
    if pipeline_tiles:
        plan, physical, execution = make_pipeline_plan(graph, profile, **options)
    elif stream_ddr:
        plan, physical, execution = make_stream_plan(graph, profile, **options)
    else:
        try:
            plan = make_plan(graph, profile, **options)
        except GraphError as exc:
            if profile["npu"]["variant"] != "z15" or not any(
                    text in str(exc) for text in ("GPU address window", "scratch zone exhausted", f"no {profile['name']} lowering")):
                raise
            try:
                plan, physical, execution = make_pipeline_plan(graph, profile, **options)
            except GraphError:
                plan, physical, execution = make_stream_plan(graph, profile, **options)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"bundle output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".hetero-") as tmp:
        staging = Path(tmp) / "bundle"
        staging.mkdir()
        gpu_products = gpu.lower(physical, plan, profile, staging)
        tensor_backend = asic_ip if profile["npu"]["variant"] == "asic-ip-v3" else npu
        npu_products = tensor_backend.lower(physical, plan, profile, staging)
        if execution is None:
            image, bindings = _initial_image(graph, plan, profile, npu_products, staging)
            transfers = _dma_table(graph, plan, profile)
        else:
            image, ddr, bindings, outputs = _stream_images(graph, plan, profile, npu_products)
            (staging / "initial_ddr.bin").write_bytes(ddr)
            transfers = [c for c in execution if c["type"] == "dma"]
        (staging / "initial.bin").write_bytes(image)
        beat = profile["dma"]["beat_bytes"]
        devices = ("NPU", "VPU", "GPU")
        operators = {device: set() for device in devices}
        for stage in plan["stages"]:
            group = operators[stage["device"]]
            if "fused" in stage:
                group.update(node["id"] for node in stage["fused"])
            else:
                group.add(stage.get("logical_id", stage["id"]))
        estimate = {
            "scope": "static estimates; excludes load stalls, pipeline interlocks and runtime overlap",
            "serial_compute_cycles": plan["compute_cycles_serial_estimate"],
            "arena_capacity_bytes": len(image),
            "scratch_high_water_bytes": plan["high_water_bytes"],
            "scratch_peak_live_bytes": plan["peak_live_bytes"],
            "scratch_reserved_bytes": sum(plan["reserved_by_zone_bytes"].values()),
            "ir_operator_count": len(graph["nodes"]),
            "scheduled_stage_count": len(plan["stages"]),
            "device_stage_counts": {device: sum(s["device"] == device for s in plan["stages"])
                                    for device in devices},
            "device_operator_counts": {device: len(operators[device]) for device in devices},
            "device_operator_fractions": {device: len(operators[device])/len(graph["nodes"])
                                          for device in devices},
            "options": plan["options"],
            "resident_edge_bytes": sum(
                plan["allocations"][name]["size"] for s in plan["stages"] for name in s["inputs"]
                if physical["tensors"][name]["kind"] == "value") if execution is None else 0,
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
        if execution is not None:
            manifest.update(execution=execution, initial_ddr="initial_ddr.bin", output_bindings=outputs)
        if emit_binary is None:
            emit_binary = profile["name"] in TARGET_IDS
        if emit_binary:
            manifest["program"] = emit_yhb(staging, manifest).name
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
