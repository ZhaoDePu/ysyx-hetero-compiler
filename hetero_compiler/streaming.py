"""DDR-backed scratch tiling for the Z15 two-half ownership contract."""

import math

from .ir import DTYPES, GraphError
from .planner import _align, _allocate, _backend, _byte_count, _estimate, _placement, _resources

ELEMENTWISE = {"Add", "Mul", "Relu", "Requant", "Sigmoid", "SiLU", "GELU", "GatedBranch"}


def _gemm_rows(stage, graph):
    if stage["op"] != "Gemm":
        return set()
    parts = stage.get("fused", [stage])
    left = parts[0]["inputs"][0]
    if left in parts[0]["inputs"][1:] or any(left in n["inputs"][1:] for n in parts[1:]):
        return set()
    return {left, *(node["output"] for node in parts)}


def _row_graph(graph, rows, count):
    specs = graph["tensors"]
    return {**graph, "tensors": {**specs, **{
        name: {**specs[name], "shape": [count, *specs[name]["shape"][1:]]}
        for name in rows
    }}}


class _TilingAdmission:
    def __init__(self, backend):
        self.backend = backend

    def supports(self, stage, graph, profile):
        if self.backend.supports(stage, graph, profile):
            return True
        rows = _gemm_rows(stage, graph)
        return bool(rows) and self.backend.supports(stage, _row_graph(graph, rows, 1), profile)


def _bytes_for(stage, graph, profile, backend):
    tensors = graph["tensors"]
    names = list(dict.fromkeys(stage["inputs"] + [stage["output"]]))
    align = profile["sram"]["alignment"]
    requirements = getattr(backend, "requirements", lambda *_: {})(stage, graph, profile)
    replaced = {entry.get("replaces") for entry in requirements.values()}
    bytes_needed = sum(_align(_byte_count(tensors[name]), align) for name in names if name not in replaced)
    return bytes_needed + sum(_align(aux["size"], align) for aux in requirements.values())


def _pieces(stage, graph, profile, backend):
    tensors = graph["tensors"]
    names = list(dict.fromkeys(stage["inputs"] + [stage["output"]]))
    align = profile["sram"]["alignment"]
    capacity = (profile["gpu"]["window_bytes"] if stage["device"] == "GPU"
                else profile["sram"]["zone_bytes"])
    rows = _gemm_rows(stage, graph)
    if backend.supports(stage, graph, profile):
        bytes_needed = _bytes_for(stage, graph, profile, backend)
        if bytes_needed <= capacity:
            return [(0, None, set())]
    else:
        bytes_needed = None
    if rows:
        count = tensors[stage["output"]]["shape"][0]
        # Every nonfinal tile start must be aligned in both A and C. Equal
        # shape intermediates are register/epilogue results, never moved.
        row_bytes = [_byte_count(tensors[name]) // count for name in rows if name in names]
        unit = math.lcm(*(align // math.gcd(pitch, align) for pitch in row_bytes))
        low, high = 1, (count-1) // unit
        best = 0
        while low <= high:
            middle = (low + high) // 2
            local = _row_graph(graph, rows, middle * unit)
            if backend.supports(stage, local, profile) and _bytes_for(stage, local, profile, backend) <= capacity:
                best, low = middle * unit, middle + 1
            else:
                high = middle - 1
        if best:
            return [(start, min(best, count-start), rows) for start in range(0, count, best)]
    dst = tensors[stage["output"]]
    if (stage["op"] not in ELEMENTWISE or stage.get("fused") or
            any(tensors[name]["shape"] != dst["shape"] for name in names)):
        size = f"{bytes_needed} B" if bytes_needed is not None else "native shape"
        raise GraphError(f"{stage['id']}: {size} cannot fit into its {capacity} B "
                         f"scratch window. {stage['op']} has no legal C2 tile recipe")
    width = sum(DTYPES[tensors[name]["dtype"]][0] for name in names)
    count_per_tile = ((capacity // width) // align) * align
    if count_per_tile <= 0:
        raise GraphError(f"{stage['id']}: one aligned tile exceeds scratch capacity")
    count = math.prod(dst["shape"])
    return [(start, min(count_per_tile, count-start), set()) for start in range(0, count, count_per_tile)]


def _physical_graph(stages, graph, profile, npu, gpu):
    source = graph["tensors"]
    phys = dict(graph, tensors={}, nodes=[], inputs=[], outputs=[])
    copies, staged = {}, []
    for stage in stages:
        backend = gpu if stage["device"] == "GPU" else npu
        pieces = _pieces(stage, graph, profile, backend)
        for tile, (start, count, row_tensors) in enumerate(pieces):
            stage_id = f"{stage['id']}.__c2_tile_{tile}" if len(pieces) > 1 else stage["id"]
            clone = {}

            def rename(name):
                if name not in clone:
                    name2 = f".__c2_{len(staged)}_{len(clone)}"
                    spec = source[name]
                    first = 0
                    if count is not None and (not row_tensors or name in row_tensors):
                        tail = spec["shape"][1:] if row_tensors else []
                        pitch = math.prod(tail)
                        first = start * pitch
                        spec = {**spec, "shape": [count, *tail]}
                        if "data" in spec:
                            spec["data"] = spec["data"][first:first+count*pitch]
                    phys["tensors"][name2] = spec
                    clone[name] = name2
                    copies[name2] = (name, first * DTYPES[spec["dtype"]][0])
                return clone[name]

            def part(node):
                return {**node, "inputs": [rename(x) for x in node["inputs"]],
                        "output": rename(node["output"])}

            item = {**part(stage), "id": stage_id, "logical_id": stage["id"], "tile": tile}
            if "fused" in item:
                item["fused"] = [part(node) for node in stage["fused"]]
            if count is not None:
                if row_tensors:
                    item.update(tile_first_row=start, tile_rows=count)
                else:
                    item.update(tile_first_element=start, tile_elements=count)
            staged.append(item)
    return phys, copies, staged


def _scratch(phys, stages, profile, npu, gpu, reuse):
    half = profile["sram"]["zone_bytes"]
    alignment = profile["sram"]["alignment"]
    page_size = profile["gpu"]["window_bytes"]
    cursors, footprint, allocations, tile_sizes = [0, 0], [0, 0], {}, []
    for i, stage in enumerate(stages):
        zone = i % 2
        window = page_size if stage["device"] == "GPU" else half
        start = _align(cursors[zone], page_size if stage["device"] == "GPU" else alignment)
        specs, ordered, _ = _resources(phys, profile, [stage], npu, gpu, wide_pages=True)
        specs = {name: {**info, "zone": 0, "limit": min(window, half-start)}
                 for name, info in specs.items()}
        local, ranges = _allocate(specs, ordered, window, reuse=False)
        used = max(hi for _, hi, _, _ in ranges[0])
        tile_sizes.append(used)
        footprint[zone] = max(footprint[zone], start + used)
        if not reuse:
            cursors[zone] = start + used
        stage["zone"] = zone
        for name, alloc in local.items():
            allocations[name] = {**alloc, "offset": zone*half + start + alloc["offset"],
                                 "zone": zone}
    return allocations, footprint, tile_sizes


def _storage(graph, stages, copies, allocs, alignment):
    ddr, packed, cursor = {}, {}, 0
    used = set(graph["inputs"]) | set(graph["outputs"])
    for stage in stages:
        used.update(copies[name][0] for name in stage["buffers"] if name in copies)
    for name, spec in graph["tensors"].items():
        if name in used:
            size = _byte_count(spec)
            extent = _align(size, alignment)
            ddr[name] = {"offset": cursor, "size": size, "stored_size": extent}
            cursor += extent
    # Packed constant encodings are independent of a stage's scratch address.
    encoded = {}
    for name, alloc in allocs.items():
        if "replaces" in alloc:
            source, byte_off = copies[alloc["replaces"]]
            key = (source, byte_off, alloc["size"])
            if key not in encoded:
                encoded[key] = cursor
                cursor += alloc["stored_size"]
            packed[name] = encoded[key]
    if cursor > 1 << 29:
        raise GraphError(f"DDR pool exhausted ({cursor} B, maximum 512 MiB)")
    return ddr, packed, cursor


def _execution(stages, copies, allocs, ddr, packed, overlap):
    loads, stores = [], []
    for stage in stages:
        out = stage["output"]
        moves = []
        for name in stage["buffers"]:
            if name == out or allocs[name]["kind"] == "temp":
                continue
            if name in packed:
                at = packed[name]
            else:
                logical, offset = copies[name]
                at = ddr[logical]["offset"] + offset
            a = allocs[name]
            moves.append({"type": "dma", "direction": "load", "ddr_offset": at,
                          "scratch_offset": a["offset"], "bytes": a["stored_size"],
                          "zone": a["zone"]})
        name, offset = copies[out]
        a = allocs[out]
        stores.append({"type": "dma", "direction": "store", "ddr_offset": ddr[name]["offset"] + offset,
                       "scratch_offset": a["offset"], "bytes": a["stored_size"], "zone": a["zone"]})
        loads.append(moves)

    commands, prefetched, overlapping = [], False, []
    for i, stage in enumerate(stages):
        if not prefetched:
            commands.extend(loads[i])
        commands.append({"type": "dma-fence", "reason": "operands resident"})
        commands.append({"type": "launch", "stage": stage["id"], "device": stage["device"],
                         "zone": stage["zone"]})
        prefetched = False
        if overlap and i+1 < len(stages):
            store = stores[i]
            low, high = store["ddr_offset"], store["ddr_offset"] + store["bytes"]
            conflict = any(move["ddr_offset"] < high and low < move["ddr_offset"] + move["bytes"]
                           for move in loads[i+1])
            if not conflict:
                commands.extend(loads[i+1])
                prefetched = bool(loads[i+1])
                if prefetched:
                    overlapping.append((i, i+1))
        commands.append({"type": "fence", "wait_for": stage["id"], "reason": "compute write drain"})
        commands.append(stores[i])
    commands.append({"type": "dma-fence", "reason": "DDR outputs visible"})
    return commands, overlapping


def make_stream_plan(graph, profile, *, fuse, reuse, overlap, device_policy):
    if profile["npu"]["variant"] != "z15":
        raise GraphError("DDR streaming requires the Z15 DMA ABI")
    npu, gpu = _backend("npu"), _backend("gpu")
    logical = _placement(graph, profile, _TilingAdmission(npu), _TilingAdmission(gpu), fuse, device_policy)
    phys, copies, stages = _physical_graph(logical, graph, profile, npu, gpu)
    allocs, footprint, tile_sizes = _scratch(phys, stages, profile, npu, gpu, reuse)
    align = profile["sram"]["alignment"]
    ddr, packed, ddr_size = _storage(graph, stages, copies, allocs, align)
    commands, overlapping = _execution(stages, copies, allocs, ddr, packed, overlap)
    peak = max(tile_sizes + [tile_sizes[i] + tile_sizes[j] for i, j in overlapping])
    for stage in stages:
        stage["estimate"] = _estimate(stage, phys, profile)
    plan = {"schema": "hetero-plan-v1", "profile": profile["name"], "stages": stages,
            "commands": [c for c in commands if c["type"] in ("launch", "fence")],
            "allocations": allocs, "streaming": True, "ddr_allocations": ddr,
            "ddr_packed_offsets": packed, "ddr_image_bytes": ddr_size,
            "descriptor_metadata_bytes": 0, "arena_end": profile["sram"]["capacity"],
            "peak_live_bytes": peak, "prefetched_tile_pairs": overlapping,
            "high_water_bytes": max(a["offset"] + a["stored_size"] for a in allocs.values()),
            "reserved_by_zone_bytes": {str(i): x for i, x in enumerate(footprint)},
            "compute_cycles_serial_estimate": sum(s["estimate"]["compute_cycles"] for s in stages),
            "options": {"device_policy": device_policy, "overlap": overlap, "fusion": fuse,
                        "mem_reuse": reuse}, "logical_stage_count": len(logical)}
    return plan, phys, commands
