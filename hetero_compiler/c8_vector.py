"""Eliminate redundant native byte passes while preserving the quantized graph."""

from collections import Counter


def _private(name, uses, public):
    return uses[name] == 1 and name not in public


def _partition_group(inputs, first, axis, producer, tensors, uses, public):
    head = producer.get(inputs[first])
    if head is None or head["op"] != "RequantLUT":
        return None
    split = producer.get(head["inputs"][0])
    if split is None or split["op"] != "Split":
        return None
    source = split["inputs"][0]
    rank = len(tensors[source]["shape"])
    if not rank or split["attrs"].get("axis", 0) % rank != axis % rank:
        return None
    width = len(split["outputs"])
    names = inputs[first:first + width]
    if width < 2 or len(names) != width or not all(_private(name, uses, public) for name in names):
        return None
    branches = [producer.get(name, {}) for name in names]
    if any(branch.get("op") != "RequantLUT" or
           branch["inputs"] != [split_output] or
           branch["attrs"]["lut"] != head["attrs"]["lut"]
           for branch, split_output in zip(branches, split["outputs"], strict=True)):
        return None
    source_desc = tensors[source]
    key = ("dtype", "layout")
    if (source_desc["dtype"] != "int8"
            or any(any(tensors[name][field] != source_desc[field] for field in key)
                   for name in (*names, *split["outputs"]))
            or any(tensors[name]["scale"] != tensors[names[0]]["scale"] for name in names)):
        return None
    return source, branches, width


def _new_tensor(tensors, node, shaped_like, scaled_like):
    name = f"@c8/packed/{node['id']}"
    if name in tensors:
        raise ValueError(f"duplicate fused tensor {name}")
    tensors[name] = {**tensors[shaped_like], "scale": tensors[scaled_like]["scale"]}
    return name


def fuse_vector_partitions(graph):
    """Fuse equivalent byte operations and remove inverse RHS transposes."""
    tensors = dict(graph["tensors"])
    public = set(graph["outputs"])
    nodes = graph["nodes"]
    uses = Counter(name for node in nodes for name in node["inputs"])
    producer = {name: node for node in nodes for name in node["outputs"]}
    remove, before, replacements = set(), {}, {}
    for node in nodes:
        if node["op"] != "Concat":
            continue
        inputs = node["inputs"]
        remaining, fused = [], []
        first = 0
        while first < len(inputs):
            group = _partition_group(inputs, first, node["attrs"]["axis"], producer, tensors, uses, public)
            if group and group[2] < len(inputs):
                source, branches, width = group
                head, *tail = branches
                combined = _new_tensor(tensors, head, source, head["outputs"][0])
                fused.append({**head, "inputs": [source], "outputs": [combined],
                              "attrs": {**head["attrs"], "fused_edge_ids": [item["id"] for item in tail]}})
                remove.update(branch["id"] for branch in branches)
                remaining.append(combined)
                first += width
            else:
                remaining.append(inputs[first])
                first += 1
        if fused:
            if len(remaining) < 2:
                raise ValueError("fused concat must keep at least two operands")
            before[node["id"]] = fused
            replacements[node["id"]] = {**node, "inputs": remaining}

    for node in nodes:
        if node["op"] != "RequantLUT" or node["id"] in remove:
            continue
        resized, = node["inputs"]
        resize = producer.get(resized)
        if (not resize or resize["op"] != "Resize" or resize["attrs"].get("mode") != "nearest"
                or not _private(resized, uses, public)):
            continue
        source = resize["inputs"][0]
        target, = node["outputs"]
        if any(tensors[name]["dtype"] != "int8" or tensors[name]["layout"] != "NHWC"
               for name in (source, resized, target)):
            continue
        combined = _new_tensor(tensors, node, source, target)
        before[resize["id"]] = [{**node, "inputs": [source], "outputs": [combined]}]
        replacements[resize["id"]] = {**resize, "inputs": [combined, *resize["inputs"][1:]],
                                      "outputs": [target]}
        remove.add(node["id"])

    for node in nodes:
        if node["op"] != "MatMul":
            continue
        left, right = node["inputs"]
        transpose = producer.get(right)
        if (not transpose or transpose["op"] != "Transpose"
                or not _private(right, uses, public)):
            continue
        source, = transpose["inputs"]
        desc = tensors[source]
        rank = len(desc["shape"])
        if (desc["dtype"] != "uint8" or rank < 2 or
                transpose["attrs"].get("perm") != [*range(rank - 2), rank - 1, rank - 2]):
            continue
        replacements[node["id"]] = {**node, "inputs": [left, source], "attrs": {
            **node["attrs"], "rhs_pretransposed": True, "fused_edge_ids": [transpose["id"]],
        }}
        remove.add(transpose["id"])

    result = []
    for node in nodes:
        if node["id"] not in remove:
            result.extend(before.get(node["id"], ()))
            result.append(replacements.get(node["id"], node))
    return {**graph, "nodes": result, "tensors": tensors}
