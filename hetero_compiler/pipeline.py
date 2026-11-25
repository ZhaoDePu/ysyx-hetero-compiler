"""Partition independent graph tiles and schedule them on the two Z15 halves."""

import math

from .ir import DTYPES, GraphError
from .planner import _align, _allocate, _backend, _byte_count, _estimate, _placement, _resources
from .streaming import _storage

_POINTWISE = {"Add", "Mul", "Requant", "Relu", "Sigmoid", "SiLU", "GELU", "Lookup",
              "Cast", "GatedBranch"}
_BATCH_SPATIAL = {"Conv", "MaxPool", "Resize", "SPPF", "C2f"}


def _simplify_tile_layouts(graph):
    """Share exact constants and fold inverse Transpose across private edges."""
    tensors = graph["tensors"]
    protected = set(graph["inputs"] + graph["outputs"])
    aliases, constants = {}, {}
    for name, record in tensors.items():
        if record["kind"] == "constant" and name not in protected:
            signature = (record["dtype"], tuple(record["shape"]), tuple(record["data"]))
            aliases[name] = constants.setdefault(signature, name)

    producers, nodes = {}, []
    for node in graph["nodes"]:
        out = node["output"]
        inputs = [aliases.get(name, name) for name in node["inputs"]]
        if node["op"] == "Transpose" and out not in protected:
            parent = producers.get(inputs[0])
            if parent and parent["op"] == "Transpose":
                first, second = parent["attrs"]["perm"], node["attrs"]["perm"]
                if [first[i] for i in second] == list(range(len(first))):
                    aliases[out] = parent["inputs"][0]
                    continue
        producer = {**node, "inputs": inputs}
        producers[out] = producer
        nodes.append(producer)
    names = set(protected)
    kept = []
    for node in reversed(nodes):
        if node["output"] in names:
            names.update(node["inputs"])
            kept.append(node)
    return dict(graph, tensors={name: val for name, val in tensors.items() if name in names},
                nodes=list(reversed(kept)))


def _dispatch_cost(engine, stage, graph, profile):
    """Estimated BALANCED wall-domain service cost, not a measured gate."""
    cycles = stage["estimate"]["compute_cycles"]
    if engine == "tensor":
        return cycles * 3 + 55
    output = graph["tensors"][stage["output"]]
    items = math.prod(output["shape"])
    if stage["op"] in ("QSoftmax", "RowSoftmax", "RMSNorm"):
        items //= output["shape"][-1]
    if stage["op"] == "QSoftmax":
        cycles = max(cycles, 9 * output["shape"][-1] * items)
    slots = profile["gpu"]["lanes"] * profile["gpu"]["warps"]
    setup = (items + slots - 1) // slots
    return 8 * cycles + setup * 145


def _schedule_banks(tiles):
    """Assemble independent half fences using host-aware service estimates."""
    streams = [[(index, *entry) for index in range(zone, len(tiles), 2)
                for entry in tiles[index]] for zone in range(2)]
    positions, active, now = [0, 0], {}, 0
    commands = []
    while active or any(i < len(row) for i, row in zip(positions, streams, strict=True)):
        ready = []
        for zone, row in enumerate(streams):
            if zone in active or positions[zone] >= len(row):
                continue
            index, engine, command, duration = row[positions[zone]]
            if any(other == engine for other, _, _ in active.values()):
                continue
            # Give short MMIO launches a head start before a GPU wave submission.
            priority = {"tensor": 0, "dma": 1, "gpu": 2}[engine]
            # Finish the older tile first on a shared engine so the GPU tail of the
            # oldest tile overlaps the other half's full NPU chain.
            ready.append((priority, index, zone, engine, command, duration))
        if ready:
            _, _, zone, engine, command, duration = min(ready)
            positions[zone] += 1
            active[zone] = (engine, command, now + duration)
            commands.append(command)
            continue
        zone = min(active, key=lambda k: (active[k][2], k))
        engine, command, end = active.pop(zone)
        wait = _wait_for(engine, command)
        commands.append({**wait, "zone": zone})
        now = max(now, end)
    return commands


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


def _task(stage, graph=None, profile=None):
    rec = {"type": "launch", "stage": stage["id"], "device": stage["device"], "zone": stage["zone"]}
    engine = "gpu" if stage["device"] == "GPU" else "tensor"
    duration = _dispatch_cost(engine, stage, graph, profile) if graph else stage["estimate"]["compute_cycles"]
    return engine, rec, duration


def _dma_task(record, beat, bank_aware=False):
    return "dma", record, (80 + 2 * record["bytes"] // beat if bank_aware else 12 + record["bytes"] // beat)


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


def _allocate_tile(specs, groups, stages, half, reuse, bank_aware):
    if bank_aware and reuse:
        # The frozen GPU's QSoftmax timing changes when base addresses fit a
        # single immediate. Pair the operands near the end of the GPU window
        # to preserve the faster wide-immediate code shape during reuse.
        occupied = {name for group in groups for name in group}
        softmax_pairs = []
        for stage in stages:
            if stage["device"] == "GPU" and stage["op"] == "QSoftmax":
                pair = [stage["inputs"][0], stage["output"]]
                if not occupied.intersection(pair):
                    occupied.update(pair)
                    softmax_pairs.append(pair)
        if softmax_pairs:
            try:
                return _allocate(specs, groups + softmax_pairs, half, reuse,
                                 prefer_upper={pair[0] for pair in softmax_pairs})
            except GraphError:
                # Ordinary allocation still applies when the adjacent layout
                # has insufficient room or conflicts with a native buffer.
                pass
    return _allocate(specs, groups, half, reuse)


def make_pipeline_plan(graph, profile, *, fuse, reuse, overlap, device_policy):
    if profile["npu"]["variant"] != "z15":
        raise GraphError("graph tile pipelines require the Z15 two-half DMA ABI")
    align = profile["sram"]["alignment"]
    half = profile["sram"]["zone_bytes"]
    page = profile["gpu"]["window_bytes"]
    beat = profile["dma"]["beat_bytes"]
    bank_aware = profile.get("pipeline_scheduler", "c3") == "bank-aware"
    if bank_aware:
        graph = _simplify_tile_layouts(graph)
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
        local, slots = _allocate_tile(local_specs, groups, stages, half, reuse, bank_aware)
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
        queues.append([_dma_task(move, beat, bank_aware) for move in reads] +
                      [_task(stage, tile if bank_aware else None, profile) for stage in stages] +
                      [_dma_task(move, beat, bank_aware) for move in writes])
    execution = _schedule_banks(queues) if overlap and bank_aware else schedule_tiles(queues, overlap)
    phys = {**graph, "tensors": all_tensors, "nodes": [], "inputs": [], "outputs": []}
    plan = {"schema": "hetero-plan-v1", "profile": profile["name"], "stages": all_stages,
            "commands": [x for x in execution if x["type"] in ("launch", "fence")],
            "allocations": all_allocs, "streaming": True, "pipeline_tiles": len(parts),
            "pipeline_scheduler": "bank-aware" if bank_aware else "c3",
            "ddr_allocations": ddr, "ddr_packed_offsets": packed, "ddr_image_bytes": size,
            "descriptor_metadata_bytes": 0, "arena_end": profile["sram"]["capacity"],
            "peak_live_bytes": sum(live) if overlap else max(live),
            "high_water_bytes": max(rec["offset"] + rec["stored_size"] for rec in all_allocs.values()),
            "reserved_by_zone_bytes": {str(zone): value for zone, value in enumerate(high)},
            "compute_cycles_serial_estimate": sum(s["estimate"]["compute_cycles"] for s in all_stages),
            "options": {"device_policy": device_policy, "overlap": overlap, "fusion": fuse,
                        "mem_reuse": reuse}}
    return plan, phys, execution
