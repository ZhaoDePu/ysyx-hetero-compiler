"""Normalize static integer graphs into a typed, single-assignment IR."""

import heapq
import math
from . import layout

DTYPES = {"int8": (1, -128, 127), "uint8": (1, 0, 255),
          "int32": (4, -(1 << 31), (1 << 31) - 1),
          "int64": (8, -(1 << 63), (1 << 63) - 1)}
ARITIES = {"Gemm": (2, 3), "Conv": (2, 3), "Add": (2,), "Mul": (2,),
           "Relu": (1,), "Requant": (1,), "SiLU": (1,), "GELU": (1,),
           "Attention": (3,), "GatedBranch": (1,), "RMSNorm": (1,), "QRMSNorm": (2,), "RowSoftmax": (1,),
           "C2f": (2,), "SPPF": (1,), "YoloDetectHead": (2,), "YoloNMS": (1,), **layout.ARITIES}
ALIASES = {"MatMul": "Gemm", "BiasAdd": "Add", "ReLU": "Relu",
           "Silu": "SiLU", "Gelu": "GELU", "MatMulS32": "Gemm",
           "BiasAddS32": "Add", "RequantizeS8": "Requant", "ResidualAdd": "Add",
           "Conv2D": "Conv", "SPPF_MaxPool": "SPPF", "C2f_SplitConcat": "C2f"}


class GraphError(ValueError):
    """Malformed graph or target-incompatible compilation."""


def _ints(values, count, lo, hi):
    return (isinstance(values, list) and len(values) == count and
            all(type(v) is int and lo <= v <= hi for v in values))


def _name(value):
    return (isinstance(value, str) and bool(value) and
            value.isprintable() and ".__" not in value)


def _check_tensor(name, tensor):
    if not _name(name) or not isinstance(tensor, dict):
        raise GraphError(f"invalid tensor {name!r}")
    shape = tensor.get("shape")
    if (not isinstance(shape, list) or not 1 <= len(shape) <= 4 or
            not _ints(shape, len(shape), 1, 1 << 28)):
        raise GraphError(f"{name}: expected positive static dimensions (rank 1..4)")
    dtype = tensor.get("dtype")
    if not isinstance(dtype, str) or dtype not in DTYPES:
        raise GraphError(f"{name}: unsupported dtype {dtype!r}")
    if "data" in tensor and not _ints(tensor["data"], math.prod(shape), *DTYPES[dtype][1:]):
        raise GraphError(f"{name}: tensor data has wrong length, dtype, or range")
    return dict(tensor)


def _attributes(op, a, node):
    attrs = dict(a)
    allowed = {"narrowing", "dtype"} | {
        **layout.ATTRIBUTES,
        "Gemm": {"transpose_b", "a_zero_point", "b_zero_point"},
        "Conv": {"strides", "pads", "shift", "a_zero_point", "b_zero_point"}, "Add": {"broadcast"},
        "Mul": set(), "Relu": set(), "Requant": {"multiplier", "shift", "zero_point", "input_zero_point", "divisor", "rounding"},
        "SiLU": {"mode"}, "GELU": set(), "Attention": {"causal", "exp_shift", "position"},
        "GatedBranch": {"threshold", "rounds"}, "RMSNorm": set(),
        "QRMSNorm": {"gain", "epsilon", "shift"},
        "RowSoftmax": {"causal_mask"}, "C2f": set(), "SPPF": {"kernel"},
        "YoloDetectHead": {"reg_max", "stride"}, "YoloNMS": {"score_threshold", "iou_threshold"},
    }[op]
    if unknown := attrs.keys() - allowed:
        raise GraphError(f"{node}: unsupported attributes {sorted(unknown)}")
    if attrs.get("narrowing", "saturate") not in ("wrap", "saturate"):
        raise GraphError(f"{node}: invalid narrowing")
    if "dtype" in attrs and (not isinstance(attrs["dtype"], str) or attrs["dtype"] not in DTYPES):
        raise GraphError(f"{node}: invalid output dtype")
    if op == "Gemm" and type(attrs.get("transpose_b", False)) is not bool:
        raise GraphError(f"{node}: transpose_b must be boolean")
    if op == "RowSoftmax" and type(attrs.setdefault("causal_mask", False)) is not bool:
        raise GraphError(f"{node}: causal_mask must be boolean")
    if op == "SiLU" and attrs.get("mode") not in (None, "legacy_q4_knots"):
        raise GraphError(f"{node}: unsupported SiLU mode")
    if op == "SPPF" and (type(attrs.setdefault("kernel", 3)) is not int or attrs["kernel"] not in (3, 5)):
        raise GraphError(f"{node}: expected SPPF kernel 3 or 5")
    if op == "YoloDetectHead" and any(type(attrs.setdefault(k, 4)) is not int or attrs[k] != 4
                                      for k in ("reg_max", "stride")):
        raise GraphError(f"{node}: expected four DFL bins and stride four")
    if op == "YoloNMS" and any(not _ints([attrs.setdefault(k, d)], 1, 0, 255)
                                for k, d in (("score_threshold", 160), ("iou_threshold", 100))):
        raise GraphError(f"{node}: invalid score or IoU threshold")
    if op == "Requant":
        for k, val, lo, hi in (("multiplier", 1, -(1 << 31), (1 << 31) - 1),
                               ("shift", 0, 0, 31),
                               ("zero_point", 0, -(1 << 31), (1 << 31) - 1),
                               ("input_zero_point", 0, -(1 << 31), (1 << 31) - 1)):
            v = attrs.setdefault(k, val)
            if type(v) is not int or not lo <= v <= hi:
                raise GraphError(f"{node}: invalid requant {k}")
    if op == "Requant":
        rounding = attrs.get("rounding", "floor")
        if rounding not in ("floor", "nearest_even"):
            raise GraphError(f"{node}: unknown requant rounding mode")
        if "divisor" in attrs and (rounding != "nearest_even" or
                not _ints([attrs["divisor"]], 1, 1, 1 << 31)):
            raise GraphError(f"{node}: nearest-even requires a valid positive integer divisor")
    if op in ("Gemm", "Conv"):
        for k in ("a_zero_point", "b_zero_point"):
            if k in attrs and not _ints([attrs[k]], 1, -128, 255):
                raise GraphError(f"{node}: invalid {k}")
    if op == "Add" and attrs.get("broadcast", "none") not in ("none", "last"):
        raise GraphError(f"{node}: invalid broadcast mode")
    if op == "Conv":
        if "shift" in attrs and not _ints([attrs["shift"]], 1, 0, 31):
            raise GraphError(f"{node}: invalid Conv shift")
        if not _ints(attrs.setdefault("strides", [1, 1]), 2, 1, 256):
            raise GraphError(f"{node}: expected two positive strides")
        if not _ints(attrs.setdefault("pads", [0, 0, 0, 0]), 4, 0, 256):
            raise GraphError(f"{node}: expected four nonnegative pads")
    if op == "Attention":
        if type(attrs.setdefault("causal", False)) is not bool:
            raise GraphError(f"{node}: causal must be boolean")
        if not _ints([attrs.setdefault("exp_shift", 0)], 1, 0, 31):
            raise GraphError(f"{node}: invalid attention exp_shift")
        if not _ints([attrs.setdefault("position", 0)], 1, 0, 511):
            raise GraphError(f"{node}: invalid attention position")
    return attrs


def _infer(op, inputs, attrs, legacy, output_dtype):
    shape, dtype = inputs[0]["shape"], inputs[0]["dtype"]
    if op in layout.ARITIES:
        try:
            return layout.infer(op, inputs, attrs, output_dtype)
        except ValueError as exc:
            raise GraphError(str(exc)) from exc
    if dtype == "int64" or any(t["dtype"] == "int64" for t in inputs[1:]):
        raise GraphError(f"{op}: int64 arithmetic is unsupported")
    if output_dtype == "int64":
        raise GraphError(f"{op}: int64 arithmetic output is unsupported")
    if op in ("Gemm", "Conv"):
        weight, wtype = inputs[1]["shape"], inputs[1]["dtype"]
        if not (dtype in ("int8", "uint8") and wtype == "int8"):
            raise GraphError(f"{op}: requires byte A and int8 weights")
        for label, typ in (("a_zero_point", dtype), ("b_zero_point", wtype)):
            if label in attrs and not DTYPES[typ][1] <= attrs[label] <= DTYPES[typ][2]:
                raise GraphError(f"{op}: {label} does not fit {typ}")
        if op == "Gemm":
            k_axis = 1 if attrs.get("transpose_b", False) else 0
            if len(shape) != 2 or len(weight) != 2 or shape[1] != weight[k_axis]:
                raise GraphError("Gemm: the two rank-2 reduction dimensions must match")
            out = [shape[0], weight[1 - k_axis]]
        else:
            if len(shape) != 4 or len(weight) != 4 or shape[-1] != weight[2]:
                raise GraphError("Conv: expected NHWC input and HWIO weights")
            sy, sx = attrs["strides"]
            t, l, b, r = attrs["pads"]
            oh = (shape[1] + t + b - weight[0]) // sy + 1
            ow = (shape[2] + l + r - weight[1]) // sx + 1
            if min(oh, ow) <= 0:
                raise GraphError("Conv: empty output")
            out = [shape[0], oh, ow, weight[3]]
        if len(inputs) == 3 and (inputs[2]["shape"] != [out[-1]] or inputs[2]["dtype"] != "int32"):
            raise GraphError(f"{op}: bias must be int32 with shape [N]")
        return out, output_dtype or (dtype if legacy else "int32")
    if op in ("Add", "Mul"):
        expected = [shape[-1]] if attrs.get("broadcast") == "last" else shape
        if inputs[1]["shape"] != expected:
            raise GraphError(f"{op}: expected right operand shape {expected}")
    if op in ("SiLU", "GELU", "GatedBranch") and (dtype != "int8" and
            (op != "GatedBranch" or dtype != "uint8")):
        raise GraphError(f"{op}: requires Q4 int8 input" if op != "GatedBranch"
                         else "GatedBranch: requires byte input")
    if op == "GatedBranch":
        if (not _ints([attrs.setdefault("rounds", 64)], 1, 64, 4096) or
                not _ints([attrs.setdefault("threshold", 0)], 1, *DTYPES[dtype][1:])):
            raise GraphError("GatedBranch: invalid rounds or threshold")
    if op == "QRMSNorm":
        gamma = inputs[1]
        if (len(shape) != 2 or shape[-1] > 1020 or dtype != "int8" or
                gamma["dtype"] != "int8" or gamma["shape"] != [shape[-1]] or
                output_dtype not in (None, "int8") or
                any(not _ints([attrs.get(k)], 1, lo, hi) for k,lo,hi in (
                    ("gain", 1, 4095), ("epsilon", 0, (1 << 24) - 1), ("shift", 0, 31)))):
            raise GraphError("QRMSNorm requires rank-2 signed bytes, row gain and valid fixed-point fields")
        return shape[:], "int8"
    if op in ("RMSNorm", "RowSoftmax"):
        expected = "int8" if op == "RMSNorm" else "uint8"
        if len(shape) != 2 or dtype != "int8" or output_dtype not in (None, expected):
            raise GraphError(f"{op}: requires rank-2 int8 input and {expected} output")
        return shape[:], expected
    if op in ("C2f", "SPPF"):
        if len(shape) != 4 or output_dtype not in (None, dtype) or dtype == "int32":
            raise GraphError(f"{op}: requires byte NHWC geometry")
        if op == "C2f":
            if (shape[-1] % 2 or inputs[1]["shape"] != shape[:-1] + [shape[-1] // 2] or
                    inputs[1]["dtype"] != dtype):
                raise GraphError("C2f: branch channels must be half the primary channels")
            return shape[:-1] + [shape[-1] * 3 // 2], dtype
        return shape[:-1] + [shape[-1] * 4], dtype
    if op == "YoloDetectHead":
        cls = inputs[1]
        if (len(shape) != 4 or shape[0] != 1 or shape[-1] != 16 or
                len(cls["shape"]) != 4 or cls["shape"][:-1] != shape[:-1] or
                dtype != "int8" or cls["dtype"] != "int8" or output_dtype not in (None, "uint8")):
            raise GraphError("YoloDetectHead: expected matching int8 DFL and class NHWC")
        return [shape[1] * shape[2], 6], "uint8"
    if op == "YoloNMS":
        if len(shape) != 2 or shape[1] != 6 or dtype != "uint8" or output_dtype not in (None, "uint8"):
            raise GraphError("YoloNMS: expected rows of six candidate bytes")
        return [shape[0], 7], "uint8"
    if op == "Attention":
        k, v = inputs[1:]
        if (len(shape) != 2 or len(k["shape"]) != 2 or len(v["shape"]) != 2 or
                shape[1] != k["shape"][1] or k["shape"][0] != v["shape"][0] or
                any(t["dtype"] != "int8" for t in inputs)):
            raise GraphError("Attention: expected int8 Q[M,D], K[S,D], V[S,N]")
        if attrs["causal"] and attrs["position"] + shape[0] > k["shape"][0]:
            raise GraphError("Attention: causal query positions exceed key extent")
        return [shape[0], v["shape"][1]], output_dtype or "int8"
    if op in ("SiLU", "GELU", "GatedBranch") and output_dtype not in (None, dtype):
        raise GraphError(f"{op}: output dtype must match the input")
    return shape[:], output_dtype or ("int8" if op == "Requant" else dtype)


def _expand_legacy(nodes, tensors):
    used_names = {s for n in nodes for s in (n.get("id", n.get("name")), n.get("output"))
                  if isinstance(s, str)} | set(tensors)
    expanded = []
    for idx, node in enumerate(nodes):
        if node.get("op") != "ScaledMatMul":
            expanded.append(node)
            continue
        attrs = node.get("attrs", {})
        if not isinstance(attrs, dict) or attrs.keys() - {"shift", "transpose_b"}:
            raise GraphError("ScaledMatMul: unsupported attributes")
        tag = f"@legacy/accumulator/{idx}"
        while tag in used_names:
            tag += "_"
        used_names.add(tag)
        expanded.extend((
            {"id": tag, "op": "Gemm", "inputs": node.get("inputs"), "output": tag,
             "attrs": {"dtype": "int32", "transpose_b": attrs.get("transpose_b", False)}},
            {**node, "op": "RequantizeS8", "inputs": [tag],
             "attrs": {"shift": attrs.get("shift", 0)}},
        ))
    return expanded


def normalize(graph, dialect="auto"):
    if not isinstance(graph, dict):
        raise GraphError("graph must be an object")
    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or not nodes or not all(isinstance(n, dict) for n in nodes):
        raise GraphError("graph needs nonempty nodes")
    legacy = dialect == "legacy" or (dialect == "auto" and (
        "memory_mode" in graph or any("id" not in n and "name" in n for n in nodes)))
    tensors = graph.get("tensors", {})
    if isinstance(tensors, list):
        if any(not isinstance(t, dict) or not _name(t.get("name")) for t in tensors):
            raise GraphError("invalid tensor list")
        if len({t["name"] for t in tensors}) != len(tensors):
            raise GraphError("duplicate tensor name")
        tensors = {t["name"]: {k: v for k, v in t.items() if k != "name"} for t in tensors}
    if not isinstance(tensors, dict):
        raise GraphError("tensors must be a mapping or a list")
    if legacy:
        nodes = _expand_legacy(nodes, tensors)
    tensors = {k: _check_tensor(k, v) for k, v in tensors.items()}
    producers, ids, records = {}, set(), []
    for raw in nodes:
        nid = raw.get("id", raw.get("name"))
        raw_op = raw.get("op")
        if not _name(nid) or nid in ids or not isinstance(raw_op, str):
            raise GraphError(f"invalid or duplicate node {nid!r}")
        op = ALIASES.get(raw_op, raw_op)
        ins, out = raw.get("inputs"), raw.get("output")
        if (op not in ARITIES or not isinstance(ins, list) or len(ins) not in ARITIES.get(op, ()) or
                not all(_name(i) for i in ins) or not _name(out) or out in producers):
            raise GraphError(f"{nid}: unsupported operator/arity or invalid output")
        ext = raw.get("attrs", {})
        if not isinstance(ext, dict):
            raise GraphError(f"{nid}: attrs must be an object")
        attrs = dict(ext)
        if legacy:
            attrs.setdefault("narrowing", "wrap")
            for key in ("threshold", "rounds"):
                if key in raw:
                    attrs.setdefault(key, raw[key])
        if raw_op in ("BiasAdd", "BiasAddS32"):
            attrs["broadcast"] = "last"
        if raw_op in ("MatMulS32", "BiasAddS32"):
            attrs.setdefault("dtype", "int32")
        if raw_op == "RequantizeS8":
            attrs.update(dtype="int8", narrowing="saturate")
        if raw_op in ("ResidualAdd", "RMSNorm", "RowSoftmax", "Conv2D"):
            attrs["narrowing"] = "saturate"
        if raw_op == "Conv2D":
            attrs.setdefault("dtype", "int8")
            if "stride" in attrs:
                attrs["strides"] = attrs.pop("stride")
            if "padding" in attrs:
                padding = attrs.pop("padding")
                if not _ints(padding, 4, 0, 256):
                    raise GraphError("Conv2D: invalid legacy padding")
                attrs["pads"] = [padding[i] for i in (0, 2, 1, 3)]
        if legacy and raw_op == "SiLU":
            attrs.setdefault("mode", "legacy_q4_knots")
        producers[out] = len(records)
        ids.add(nid)
        records.append({"id": nid, "op": op, "inputs": ins[:], "output": out,
                        "attrs": _attributes(op, attrs, nid)})
    outputs = graph.get("outputs", [])
    if not isinstance(outputs, list) or not all(_name(x) for x in outputs):
        raise GraphError("outputs must be a list of tensor names")
    outputs = list(dict.fromkeys(outputs + [k for k, v in tensors.items() if v.get("kind") == "output"]))
    if not outputs or any(o not in producers and o not in tensors for o in outputs):
        raise GraphError("undefined or missing graph outputs")
    followers = [[] for _ in records]
    indegree = [0] * len(records)
    for idx, node in enumerate(records):
        deps = set()
        for name in node["inputs"]:
            if name in producers:
                deps.add(producers[name])
            elif name not in tensors or tensors[name].get("kind") in ("intermediate", "output"):
                raise GraphError(f"{node['id']}: undefined input {name}")
        for dep in deps:
            followers[dep].append(idx)
        indegree[idx] = len(deps)
    queue = [i for i, deps in enumerate(indegree) if deps == 0]
    ordered = []
    while queue:
        idx = heapq.heappop(queue)
        node = records[idx]
        out = node["output"]
        decl = tensors.get(out, {})
        dtype = node["attrs"].get("dtype", decl.get("dtype"))
        if "data" in decl:
            raise GraphError(f"{out}: produced tensor may not carry initial data")
        try:
            shape, dtype = _infer(node["op"], [tensors[x] for x in node["inputs"]],
                                  node["attrs"], legacy, dtype)
        except GraphError as exc:
            raise GraphError(f"{node['id']}: {exc}") from exc
        if decl and (decl["shape"] != shape or decl["dtype"] != dtype):
            raise GraphError(f"{out}: inferred shape/dtype differs from declaration")
        tensors[out] = {"shape": shape, "dtype": dtype, "kind": "value"}
        ordered.append(node)
        for dep in followers[idx]:
            indegree[dep] -= 1
            if indegree[dep] == 0:
                heapq.heappush(queue, dep)
    if len(ordered) != len(records):
        raise GraphError("graph has a cycle")
    live, retained = set(outputs), []
    for node in reversed(ordered):
        if node["output"] in live:
            retained.append(node)
            live.update(node["inputs"])
    if not retained:
        raise GraphError("graph needs at least one output-producing operator")
    inputs = []
    for name, t in tensors.items():
        if name not in live or name in producers:
            continue
        kind = t.get("kind")
        if kind in ("constant", "weight", "bias") and "data" not in t:
            raise GraphError(f"{name}: missing constant data")
        if kind == "input" or "data" not in t:
            t["kind"] = "input"
            inputs.append(name)
        else:
            t["kind"] = "constant"
    return {"schema": "heterograph-ir-v1", "name": graph.get("name") or "graph",
            "dialect": "legacy" if legacy else "canonical",
            "tensors": {k: t for k, t in tensors.items() if k in live},
            "nodes": retained[::-1], "inputs": inputs, "outputs": outputs}
