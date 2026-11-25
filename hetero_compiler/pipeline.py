"""Partition independent graph tiles and schedule them on the two Z15 halves."""

import math

from .ir import DTYPES, GraphError
from .planner import _align, _allocate, _backend, _byte_count, _estimate, _placement, _resources
from .streaming import _storage

_POINTWISE = {"Add", "Mul", "Requant", "Relu", "Sigmoid", "SiLU", "GELU", "Lookup",
              "Cast", "GatedBranch"}
_BATCH_SPATIAL = {"Conv", "MaxPool", "Resize", "SPPF", "C2f"}


def _independent(graph):
    """Return disjoint input components, sharing only immutable tensors."""
    inputs = {name: key for key, name in enumerate(graph["inputs"])}
    if len(inputs) < 2:
        return []
    owner = dict(inputs)
    groups = [[] for _ in inputs]
    for node in graph["nodes"]:
        active = {owner[name] for name in node["inputs"] if name in owner}
        if len(active) != 1:
            return []
        key = active.pop()
        groups[key].append(node)
        owner[node["output"]] = key

    result = []
    for i, nodes in enumerate(groups):
        outputs = [name for name in graph["outputs"] if owner.get(name) == i]
        if not nodes or not outputs:
            return []
        sources = [name for name, key in inputs.items() if key == i]
        names = {name for node in nodes for name in node["inputs"] + [node["output"]]}
        tensors = {name: graph["tensors"][name] for name in graph["tensors"] if name in names}
        result.append(({**graph, "nodes": nodes, "tensors": tensors, "inputs": sources,
                        "outputs": outputs}, {}))
    if sum(len(tile["outputs"]) for tile, _ in result) != len(graph["outputs"]):
        return []
    return result


def _batch_tiles(graph, beat):
    """Check that one leading batch element is a closed slice of the graph."""
    if not graph["inputs"]:
        return []
    tensors = graph["tensors"]
    lead = tensors[graph["inputs"][0]]["shape"]
    if len(lead) < 2:
        return []
    batch = lead[0]
    if batch < 2 or any(tensors[name]["shape"][0] != batch or len(tensors[name]["shape"]) < 2
                        for name in graph["inputs"]):
        return []
    sliced = set(graph["inputs"])
    for node in graph["nodes"]:
        op, ins, out, attrs = (node[k] for k in ("op", "inputs", "output", "attrs"))
        moving = [i for i in ins if i in sliced]
        if not moving or tensors[out]["shape"][0] != batch:
            return []
        if op in ("Conv", "Gemm"):
            safe = moving == [ins[0]]
        elif op in _POINTWISE:
            safe = all(x in sliced for x in ins) or (
                op == "Add" and moving == [ins[0]] and attrs.get("broadcast") == "last")
        elif op in _BATCH_SPATIAL or op in ("Concat", "ArgMax", "RMSNorm", "RowSoftmax"):
            safe = all(x in sliced for x in ins)
            if op in ("Concat", "ArgMax"):
                safe = safe and attrs.get("axis", 0) % len(tensors[ins[0]]["shape"]) != 0
            if op == "RowSoftmax":
                safe = safe and not attrs.get("causal_mask")
        elif op == "Transpose":
            safe = attrs["perm"][0] == 0
        elif op == "Reshape":
            safe = tensors[ins[0]]["shape"][0] == batch
        elif op == "Slice":
            safe = (attrs["starts"][0], attrs["ends"][0], attrs["steps"][0]) == (0, batch, 1)
        else:
            safe = False
        if not safe:
            return []
        sliced.add(out)

    if any(name not in sliced or (_byte_count(tensors[name]) // batch) % beat
           for name in graph["inputs"] + graph["outputs"]):
        return []
    tiles = []
    for index in range(batch):
        spans, local = {}, dict(tensors)
        for name in sliced:
            src = tensors[name]
            count = math.prod(src["shape"][1:])
            where = index * count
            local[name] = {**src, "shape": [1, *src["shape"][1:]]}
            if "data" in src:
                local[name]["data"] = src["data"][where:where + count]
            spans[name] = where * DTYPES[src["dtype"]][0]
        nodes = []
        for node in graph["nodes"]:
            attrs = dict(node["attrs"])
            if node["op"] in ("Reshape", "Resize") and "shape" in attrs:
                attrs["shape"] = local[node["output"]]["shape"]
            if node["op"] == "Slice":
                attrs["ends"] = [1, *attrs["ends"][1:]]
            nodes.append({**node, "attrs": attrs})
        tiles.append(({**graph, "tensors": local, "nodes": nodes}, spans))
    return tiles


def _clone(tile, spans, spec, index):
    names = {name: f".__c3_{index}_{i}" for i, name in enumerate(tile["tensors"])}

    def copy(node):
        return {**node, "inputs": [names[name] for name in node["inputs"]],
                "output": names[node["output"]]}

    stages = []
    for number, stage in enumerate(spec):
        item = {**copy(stage), "id": f"{stage['id']}.__c3_{index}_{number}",
                "logical_id": stage["id"], "tile": index}
        if "fused" in stage:
            item["fused"] = [copy(part) for part in stage["fused"]]
        stages.append(item)
    phys = {**tile, "tensors": {names[name]: tensor for name, tensor in tile["tensors"].items()},
            "nodes": [], "inputs": [names[name] for name in tile["inputs"]],
            "outputs": [names[name] for name in tile["outputs"]]}
    copies = {names[name]: (name, spans.get(name, 0)) for name in tile["tensors"]}
    return phys, copies, stages


def _moves(tile, copies, allocs, ddr, packed, zone, beat):
    reads, writes = [], []
    for name, allocated in sorted(allocs.items(), key=lambda entry: entry[1]["offset"]):
        directions = ["load"] if allocated["kind"] in ("constant", "input") else []
        if name in tile["outputs"]:
            directions.append("store")
        if not directions:
            continue
        if name in packed:
            peer = packed[name]
        else:
            key, start = copies[name]
            peer = ddr[key]["offset"] + start
        for direction in directions:
            dest = reads if direction == "load" else writes
            rec = {"type": "dma", "direction": direction, "ddr_offset": peer,
                   "scratch_offset": allocated["offset"], "bytes": _align(allocated["size"], beat),
                   "zone": zone}
            if dest and (dest[-1]["ddr_offset"] + dest[-1]["bytes"] == rec["ddr_offset"] and
                         dest[-1]["scratch_offset"] + dest[-1]["bytes"] == rec["scratch_offset"]):
                dest[-1]["bytes"] += rec["bytes"]
            else:
                dest.append(rec)
    return reads, writes


def _task(stage):
    rec = {"type": "launch", "stage": stage["id"], "device": stage["device"], "zone": stage["zone"]}
    engine = "gpu" if stage["device"] == "GPU" else "tensor"
    return engine, rec, stage["estimate"]["compute_cycles"]


def _dma_task(record, beat):
    return "dma", record, 12 + record["bytes"] // beat


def _wait_for(engine, rec):
    if engine == "dma":
        return {"type": "dma-fence", "reason": "tile DMA complete"}
    return {"type": "fence", "wait_for": rec["stage"], "zone": rec["zone"], "reason": "tile dependency"}


def schedule_tiles(tiles, overlap):
    """List-schedule two ordered streams with exclusive engine and half ownership."""
    if not overlap:
        return [record for tile in tiles for engine, task, _ in tile
                for record in (task, _wait_for(engine, task))]

    # Keep per-half order. Tile buffers may be reused by the next tile in that half.
    streams = [[entry for tile in tiles[z::2] for entry in tile] for z in range(2)]
    positions = [0, 0]
    active = {}
    commands, now = [], 0
    while any(positions[z] < len(streams[z]) for z in range(2)) or active:
        choices = []
        for z, sequence in enumerate(streams):
            if z in active or positions[z] == len(sequence):
                continue
            engine, command, estimate = sequence[positions[z]]
            if any(other == engine for other, _, _ in active.values()):
                continue
            choices.append((engine == "dma", positions[z], z, engine, command, estimate))
        if choices:
            _, _, z, engine, command, estimate = min(choices)
            active[z] = (engine, command, now + estimate)
            positions[z] += 1
            commands.append(command)
        else:
            z = min(active, key=lambda k: (active[k][2], k))
            engine, command, now = active.pop(z)
            commands.append(_wait_for(engine, command))
    return commands


def make_pipeline_plan(graph, profile, *, fuse, reuse, overlap, device_policy):
    if profile["npu"]["variant"] != "z15":
        raise GraphError("graph tile pipelines require the Z15 two-half DMA ABI")
    align = profile["sram"]["alignment"]
    half = profile["sram"]["zone_bytes"]
    page = profile["gpu"]["window_bytes"]
    beat = profile["dma"]["beat_bytes"]
    parts = _independent(graph) or _batch_tiles(graph, beat)
    if not parts:
        raise GraphError("no independent leading-batch slices or closed ONNX input components")
    npu, gpu = _backend("npu"), _backend("gpu")
    all_tensors, all_copies, all_allocs, all_stages, tile_records = {}, {}, {}, [], []
    high, live = [0, 0], [0, 0]
    for i, (subgraph, spans) in enumerate(parts):
        selected = _placement(subgraph, profile, npu, gpu, fuse, device_policy)
        tile, copies, stages = _clone(subgraph, spans, selected, i)
        specs, groups, _ = _resources(tile, profile, stages, npu, gpu)
        local_specs = {name: {**rec, "zone": 0, "limit": min(rec["limit"], half)}
                       for name, rec in specs.items()}
        local, slots = _allocate(local_specs, groups, half, reuse)
        used = max(end for _, end, _, _ in slots[0])
        zone = i % 2
        base = 0 if reuse else _align(high[zone], page if any(s["device"] == "GPU" for s in stages) else align)
        high[zone] = max(high[zone], base + used)
        if high[zone] > half:
            raise GraphError(f"tile {i}: no scratch capacity in zone {zone}")
        maximum = max(sum(end - begin for begin, end, first, last in slots[0] if first <= step <= last)
                      for step in range(-1, len(stages) + 1))
        live[zone] = max(live[zone], maximum)
        for allocation in local.values():
            allocation["zone"] = zone
            allocation["offset"] += zone * half + base
        for stage in stages:
            stage["zone"] = zone
            stage["estimate"] = _estimate(stage, tile, profile)
        all_tensors.update(tile["tensors"])
        all_copies.update(copies)
        all_allocs.update(local)
        all_stages.extend(stages)
        tile_records.append((tile, stages, local, zone))

    # Only initial data and public results require DDR space. Inputs consumed
    # by multiple tiles occupy a shared immutable extent or aligned slices.
    needed = {name for name in graph["inputs"] + graph["outputs"]}
    needed.update(all_copies[key][0] for key, rec in all_allocs.items()
                  if key in all_copies and rec["kind"] == "constant")
    ddr_graph = {**graph, "tensors": {name: rec for name, rec in graph["tensors"].items() if name in needed}}
    ddr_stages = [{**s, "buffers": [key for key in s["buffers"]
                                    if key in all_copies and all_copies[key][0] in needed]} for s in all_stages]
    ddr, packed, size = _storage(ddr_graph, ddr_stages, all_copies, all_allocs, align)

    queues = []
    for tile, stages, local, zone in tile_records:
        reads, writes = _moves(tile, all_copies, local, ddr, packed, zone, beat)
        queues.append([_dma_task(move, beat) for move in reads] +
                      [_task(stage) for stage in stages] +
                      [_dma_task(move, beat) for move in writes])
    execution = schedule_tiles(queues, overlap)
    phys = {**graph, "tensors": all_tensors, "nodes": [], "inputs": [], "outputs": []}
    plan = {"schema": "hetero-plan-v1", "profile": profile["name"], "stages": all_stages,
            "commands": [x for x in execution if x["type"] in ("launch", "fence")],
            "allocations": all_allocs, "streaming": True, "pipeline_tiles": len(parts),
            "ddr_allocations": ddr, "ddr_packed_offsets": packed, "ddr_image_bytes": size,
            "descriptor_metadata_bytes": 0, "arena_end": profile["sram"]["capacity"],
            "peak_live_bytes": sum(live) if overlap else max(live),
            "high_water_bytes": max(rec["offset"] + rec["stored_size"] for rec in all_allocs.values()),
            "reserved_by_zone_bytes": {str(zone): value for zone, value in enumerate(high)},
            "compute_cycles_serial_estimate": sum(s["estimate"]["compute_cycles"] for s in all_stages),
            "options": {"device_policy": device_policy, "overlap": overlap, "fusion": fuse,
                        "mem_reuse": reuse}}
    return plan, phys, execution
