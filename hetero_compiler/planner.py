"""Device placement, safe fusion, scratch allocation, and launch dependencies."""

import importlib
import math
from collections import defaultdict

from .ir import DTYPES, GraphError
from .profile import validate_profile

EPILOGUES = {"Add": 0, "Requant": 1, "Relu": 3, "SiLU": 3, "GELU": 3}


def _backend(name):
    return importlib.import_module("." + name, __package__)


def _align(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def _byte_count(tensor):
    return math.prod(tensor["shape"]) * DTYPES[tensor["dtype"]][0]


def _candidate(chain, device):
    stage = dict(chain[0], device=device)
    if len(chain) > 1:
        produced = {node["output"] for node in chain}
        stage["fused"] = chain[:]
        stage["inputs"] = list(dict.fromkeys(name for node in chain
                                             for name in node["inputs"] if name not in produced))
        stage["output"] = chain[-1]["output"]
    return stage


def _placement(graph, profile, npu, gpu, fuse, device_policy):
    nodes = graph["nodes"]
    producers = {node["output"]: idx for idx, node in enumerate(nodes)}
    users = defaultdict(list)
    for idx, node in enumerate(nodes):
        for name in node["inputs"]:
            users[name].append(idx)
    stages, contracted = [], set()
    for idx, node in enumerate(nodes):
        if idx in contracted:
            continue
        asic_native = profile["npu"]["variant"] == "asic-ip-v3"
        if asic_native and node["op"] in ("RMSNorm", "RowSoftmax", "QSoftmax"):
            choices = [("VPU", npu), ("GPU", gpu)]
        elif node["op"] in ("Gemm", "Conv", "Attention", "RMSNorm", "RowSoftmax",
                            "C2f", "SPPF", "YoloDetectHead"):
            choices = [("NPU", npu), ("GPU", gpu)]
        else:
            choices = [("GPU", gpu)]
            if profile["vpu"]["enabled"] and profile["vpu"].get("status") != "reserved":
                choices.insert(0, ("VPU", npu))
            if node["op"] == "Add" and profile["npu"]["variant"] == "p8":
                choices.insert(0, ("NPU", npu))
        if device_policy == "gpu-only":
            choices = [("GPU", gpu)]
        chain, chain_idxs = [node], []
        if fuse and node["op"] in ("Gemm", "Conv"):
            tail, phase = node["output"], -1
            while tail not in graph["outputs"] and len(users[tail]) == 1:
                next_idx = users[tail][0]
                following = nodes[next_idx]
                step = EPILOGUES.get(following["op"], -1)
                if following["op"] == "Add" and phase == 1 and (
                        graph["tensors"][tail]["dtype"] == "int8" and
                        following["attrs"].get("narrowing") == "wrap" and
                        not any(part["op"] == "Add" for part in chain)):
                    step = 2
                if (step <= phase or following["inputs"][0] != tail or
                        (following["op"] == "Add" and following["attrs"].get("broadcast") != "last") or
                        any(name != tail and producers.get(name, -1) >= idx
                            for name in following["inputs"])):
                    break
                chain_idxs.append(next_idx)
                chain.append(following)
                phase, tail = step, following["output"]
        if (fuse and node["op"] == "YoloDetectHead" and node["output"] not in graph["outputs"] and
                len(users[node["output"]]) == 1):
            next_idx = users[node["output"]][0]
            if nodes[next_idx]["op"] == "YoloNMS":
                chain.append(nodes[next_idx])
                chain_idxs.append(next_idx)
        counts = range(len(chain), 0, -1)
        candidates = ((count, device, backend) for count in counts for device, backend in choices)
        if device_policy == "npu-greedy":
            candidates = ((count, device, backend) for device, backend in choices for count in counts)
        selected = None
        for count, device, backend in candidates:
            stage = _candidate(chain[:count], device)
            if backend.supports(stage, graph, profile):
                selected = stage
                contracted.update(chain_idxs[:count-1])
                stages.append(selected)
                break
        if selected is None:
            raise GraphError(f"{node['id']}: no {profile['name']} lowering for {node['op']}")
    return stages


def _resources(graph, profile, stages, npu, gpu, wide_pages=False):
    tensors = graph["tensors"]
    specs, groups = {}, []
    memory, gpu_limit = profile["sram"], profile["gpu"]["window_bytes"]
    npu_limit = profile["npu"]["window_bytes"]
    native_metadata = getattr(npu, "metadata_needed", None)
    reserve = (sum(native_metadata(stage, graph, profile) for stage in stages
                   if stage["device"] != "GPU") if native_metadata else memory["metadata_reserve"])
    metadata_end = npu_limit - _align(reserve, memory["alignment"])
    if metadata_end <= 0:
        raise GraphError(f"NPU command records require {reserve} B of the {npu_limit} B arena")
    for idx, stage in enumerate(stages):
        root = gpu if stage["device"] == "GPU" else npu
        if stage["device"] == "GPU" and not wide_pages:
            limit = min(gpu_limit, metadata_end)
        else:
            limit = metadata_end
        required = getattr(root, "requirements", lambda *_: {})(stage, graph, profile)
        replacements = {info["replaces"] for info in required.values() if "replaces" in info}
        if any(name not in stage["inputs"] or tensors[name]["kind"] != "constant" for name in replacements):
            raise GraphError(f"{stage['id']}: packed static layout requires a constant source")
        stage_names = list(dict.fromkeys([name for name in stage["inputs"] if name not in replacements] +
                                         [stage["output"]]))
        for name in stage_names:
            src = tensors[name]
            current = specs.setdefault(name, {
                "size": _byte_count(src), "kind": src["kind"], "alignment": memory["alignment"],
                "first": -1 if src["kind"] != "value" else idx, "last": idx,
                "limit": limit, "gpu": False,
            })
            current["last"] = max(current["last"], idx)
            current["limit"] = min(current["limit"], limit)
            current["gpu"] |= stage["device"] == "GPU"
        for name, info in required.items():
            if name in tensors or not isinstance(info.get("size"), int) or info["size"] <= 0:
                raise GraphError(f"{stage['id']}: invalid native buffer {name}")
            alignment = info.get("alignment", memory["alignment"])
            if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
                raise GraphError(f"{stage['id']}: invalid alignment for {name}")
            alignment = max(alignment, memory["alignment"])
            kind = info.get("kind", "temp")
            if kind not in ("constant", "temp"):
                raise GraphError(f"{stage['id']}: unknown native buffer kind {kind}")
            first = -1 if kind == "constant" else idx
            prev = specs.get(name)
            if prev is not None and (prev["size"], prev["alignment"], prev["kind"]) != (
                    info["size"], alignment, kind):
                raise GraphError(f"{stage['id']}: inconsistent buffer requirement {name}")
            specs[name] = {**info, "size": info["size"], "alignment": alignment, "kind": kind,
                           "first": min(prev["first"], first) if prev else first, "last": idx,
                           "limit": min(prev["limit"], limit) if prev else limit,
                           "gpu": stage["device"] == "GPU" or bool(prev and prev["gpu"])}
            stage_names.append(name)
        stage["buffers"] = stage_names
        ordered = getattr(root, "ordered_buffers", lambda *_: None)(stage, graph, profile)
        if ordered:
            if len(set(ordered)) != len(ordered) or any(name not in stage_names for name in ordered):
                raise GraphError(f"{stage['id']}: invalid native buffer order")
            groups.append(ordered)
        buffering = getattr(root, "buffering", lambda *_: None)(stage, graph, profile)
        if buffering:
            stage["buffering"] = buffering
    for name in graph["outputs"]:
        if name not in specs:
            src = tensors[name]
            specs[name] = {"size": _byte_count(src), "kind": src["kind"], "alignment": memory["alignment"],
                           "first": -1, "last": len(stages), "limit": metadata_end, "gpu": False}
        specs[name]["last"] = len(stages)
    for spec in specs.values():
        spec["stored_size"] = _align(spec["size"], spec["alignment"])
    return specs, groups, metadata_end


def _zone_bounds(stages, specs, profile, groups):
    zone_size = profile["sram"]["zone_bytes"]
    zones = profile["sram"]["capacity"] // zone_size
    parents = {name: name for name in specs}

    def root(name):
        while parents[name] != name:
            parents[name] = parents[parents[name]]
            name = parents[name]
        return name

    for names in [s["buffers"] for s in stages] + groups:
        leader = root(names[0])
        for name in names[1:]:
            parents[root(name)] = leader
    components = defaultdict(list)
    for name in specs:
        components[root(name)].append(name)
    load = [0] * zones
    ordering = sorted(components.values(), key=lambda ns: (
        min(specs[n]["limit"] for n in ns), -sum(specs[n]["stored_size"] for n in ns), ns[0]))
    for names in ordering:
        limit = min(specs[n]["limit"] for n in names)
        eligible = range(min(zones, (limit + zone_size - 1) // zone_size))
        selected = min(eligible, key=lambda x: (load[x], x))
        load[selected] += sum(specs[n]["stored_size"] for n in names)
        for name in names:
            rec = specs[name]
            rec["zone"] = selected
            rec["limit"] = min(rec["limit"], (selected + 1) * zone_size)
    for stage in stages:
        stage["zone"] = specs[stage["output"]]["zone"]


def _allocate(specs, groups, zone_size, reuse):
    contained, blocks = set(), []
    for names in groups:
        if tuple(names) in [tuple(old) for old in blocks]:
            continue
        if contained.intersection(names):
            raise GraphError("overlapping native ordered buffer groups")
        contained.update(names)
        blocks.append(names)
    blocks += [[name] for name in specs if name not in contained]
    records = []
    for names in blocks:
        items = [specs[name] for name in names]
        alignment = max(x["alignment"] for x in items)
        offsets, cursor = [], 0
        for item in items:
            cursor = _align(cursor, item["alignment"])
            offsets.append(cursor)
            cursor += item["stored_size"]
        records.append((names, offsets, cursor, alignment, min(x["first"] for x in items),
                        max(x["last"] for x in items), items[0]["zone"],
                        min(x["limit"] - off - x["stored_size"]
                            for x, off in zip(items, offsets, strict=True))))
    ranges, allocs = defaultdict(list), {}
    for names, offsets, size, alignment, first, last, zone, ceiling in sorted(
            records, key=lambda rec: (rec[7], rec[4], -rec[2], rec[0])):
        loc = _align(zone * zone_size, alignment)
        conflicts = sorted((lo, hi) for lo, hi, start, end in ranges[zone]
                           if last >= start and first <= end)
        if not reuse:
            loc = _align(max((r[1] for r in ranges[zone]), default=loc), alignment)
            conflicts = []
        for lo, hi in conflicts:
            if loc + size <= lo:
                break
            if loc < hi:
                loc = _align(hi, alignment)
        if loc > ceiling:
            where = "GPU address window" if any(specs[n]["gpu"] for n in names) else "scratch zone"
            raise GraphError(f"{where} exhausted while placing {', '.join(names)} (size={size})")
        ranges[zone].append((loc, loc + size, first, last))
        for name, offset in zip(names, offsets, strict=True):
            allocs[name] = {**specs[name], "offset": loc + offset,
                            "reserved_first": first, "reserved_last": last}
    return allocs, ranges


def _commands(stages, overlap=True):
    commands, active = [], {}
    for stage in stages:
        zone = stage["zone"]
        resource = "GPU" if stage["device"] == "GPU" else "TENSOR"
        blockers = [key for key in active
                    if key[0] != resource and (not overlap or key[1] == zone)]
        for key in blockers:
            previous = active.pop(key)
            commands.append({"type": "fence", "wait_for": previous["id"],
                             "from": previous["device"], "to": stage["device"], "zone": zone,
                             "reason": ("scratch ownership and write drain" if overlap
                                        else "serial ablation")})
        launch = {"type": "launch", "stage": stage["id"], "device": stage["device"], "zone": zone}
        earlier = [previous for key, previous in active.items() if key[0] == resource]
        if earlier:
            launch["engine_predecessor"] = earlier[-1]["id"]
        commands.append(launch)
        active.pop((resource, zone), None)
        active[(resource, zone)] = stage
    return commands


def _estimate(stage, graph, profile):
    t = graph["tensors"]
    op, ins, dst = stage["op"], stage["inputs"], stage["output"]
    fields = {"bytes": sum(_byte_count(t[n]) for n in ins + [dst])}
    count = math.prod(t[dst]["shape"])
    if stage["device"] == "NPU" and op in ("Conv", "Gemm", "Attention"):
        rows, cols = (profile["npu"][k] for k in ("rows", "cols"))
        w = t[ins[1]]["shape"]
        if op == "Attention":
            m, k = t[ins[0]]["shape"]
            s, n = t[ins[2]]["shape"]
            cycles = math.ceil(m/rows) * (math.ceil(s/cols)*(k + rows + cols) +
                       math.ceil(n/cols)*(s + rows + cols)) + m*s
        else:
            m, n = count // t[dst]["shape"][-1], t[dst]["shape"][-1]
            k = math.prod(w) // n
            cycles = math.ceil(m/rows) * math.ceil(n/cols) * (k + rows + cols)
        fields["compute_cycles"] = cycles
    elif stage["device"] == "GPU":
        lanes, issue = (profile["gpu"][k] for k in ("lanes", "issue_width"))
        # Nominal issued instructions per lane with no memory backpressure.
        work = 4 + len(ins) * 2
        dispatch_items = count
        if op == "Gemm":
            work = 12 + 4 * t[ins[0]]["shape"][1]
        elif op == "Conv":
            work = 25 + 7 * math.prod(t[ins[1]]["shape"][:-1])
        elif op in ("RowSoftmax", "RMSNorm"):
            width = t[ins[0]]["shape"][-1]
            work = 20 + 20 * width
            dispatch_items = math.prod(t[ins[0]]["shape"][:-1])
        elif op == "ArgMax":
            axis = stage["attrs"].get("axis", -1)
            work = 14 + 7 * t[ins[0]]["shape"][axis]
        elif op == "GatedBranch":
            work = 16 + 15 * stage["attrs"]["rounds"]
        work += sum(4 if part["op"] not in ("SiLU", "GELU") else 20
                    for part in stage.get("fused", [])[1:])
        fields["compute_cycles"] = math.ceil(dispatch_items/lanes) * math.ceil(work/issue)
    else:
        fields["compute_cycles"] = math.ceil(count/4)
    return fields


def _allocate_pages(stages, specs, groups, zone_size, page_size, reuse):
    page_sets = []
    for stage in stages:
        if stage["device"] != "GPU":
            continue
        members = set(stage["buffers"])
        matching = [other for other in page_sets if other & members]
        for other in matching:
            members.update(other)
            page_sets.remove(other)
        page_sets.append(members)

    outer = dict(specs)
    inner = {}
    for index, members in enumerate(page_sets):
        key = f".__c2_gpu_page_{index}"
        local_groups = [names for names in groups if members.intersection(names)]
        if any(not members.issuperset(names) for names in local_groups):
            raise GraphError("GPU address window overlaps a larger ordered native buffer group")
        groups = [names for names in groups if names not in local_groups]
        page_records = {name: {**specs[name], "zone": 0, "limit": page_size}
                        for name in sorted(members)}
        local, _ = _allocate(page_records, local_groups, page_size, reuse)
        first = min(rec["reserved_first"] for rec in local.values())
        last = max(rec["reserved_last"] for rec in local.values())
        zone_set = {specs[name]["zone"] for name in members}
        if len(zone_set) != 1:
            raise GraphError("GPU component spans multiple ownership halves")
        zone = zone_set.pop()
        outer[key] = {"size": page_size, "stored_size": page_size, "kind": "page",
                      "alignment": page_size, "first": first, "last": last,
                      "limit": (zone + 1) * zone_size, "zone": zone, "gpu": True}
        inner[key] = local
        for name in members:
            del outer[name]
    result, ranges = _allocate(outer, groups, zone_size, reuse)
    for key, local in inner.items():
        page = result.pop(key)
        for name, rec in local.items():
            result[name] = {**specs[name], "offset": page["offset"] + rec["offset"],
                            "reserved_first": rec["reserved_first"],
                            "reserved_last": rec["reserved_last"]}
    return result, ranges


def make_plan(graph, profile, *, fuse=True, reuse=True, overlap=True, device_policy="auto"):
    validate_profile(profile)
    if device_policy not in ("auto", "gpu-only", "npu-greedy"):
        raise GraphError(f"unknown device policy {device_policy!r}")
    tensor_module = "asic_ip" if profile["npu"]["variant"] == "asic-ip-v3" else "npu"
    npu, gpu = _backend(tensor_module), _backend("gpu")
    stages = _placement(graph, profile, npu, gpu, fuse, device_policy)
    specs, groups, metadata_end = _resources(graph, profile, stages, npu, gpu)
    _zone_bounds(stages, specs, profile, groups)
    zone_size = profile["sram"]["zone_bytes"]
    try:
        allocations, ranges = _allocate(specs, groups, zone_size, reuse)
    except GraphError:
        if profile["npu"]["variant"] != "z15":
            raise
        specs, groups, metadata_end = _resources(graph, profile, stages, npu, gpu, wide_pages=True)
        _zone_bounds(stages, specs, profile, groups)
        allocations, ranges = _allocate_pages(stages, specs, groups, zone_size,
                                              profile["gpu"]["window_bytes"], reuse)
    all_ranges = [record for zone in ranges.values() for record in zone]
    peak_live = max(sum(hi - lo for lo, hi, first, last in all_ranges if first <= step <= last)
                    for step in range(-1, len(stages) + 1))
    for stage in stages:
        stage["estimate"] = _estimate(stage, graph, profile)
    return {"schema": "hetero-plan-v1", "profile": profile["name"], "stages": stages,
            "allocations": allocations, "commands": _commands(stages, overlap),
            "options": {"device_policy": device_policy, "overlap": overlap,
                        "fusion": fuse, "mem_reuse": reuse},
            "arena_end": metadata_end,
            "descriptor_metadata_bytes": profile["npu"]["window_bytes"] - metadata_end,
            "peak_live_bytes": peak_live,
            "high_water_bytes": max(a["offset"] + a["stored_size"] for a in allocations.values()),
            "reserved_by_zone_bytes": {str(zone): max(hi for _, hi, _, _ in recs) - zone*zone_size
                                       for zone, recs in ranges.items()},
            "compute_cycles_serial_estimate": sum(s["estimate"]["compute_cycles"] for s in stages)}
