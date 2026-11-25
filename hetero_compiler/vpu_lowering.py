"""Recognize the quantized subset implemented by the frozen Z15 vector unit."""

from collections import Counter
from fractions import Fraction
import math


def _power_shift(node, shift=None):
    if not node or node["op"] != "Requant":
        return None
    attrs = node["attrs"]
    if (attrs.get("multiplier", 1) != 1 or attrs.get("input_zero_point", 0) or
            attrs.get("zero_point", 0) or attrs.get("narrowing", "saturate") != "saturate"):
        return None
    divisor = attrs.get("divisor", 1 << attrs.get("shift", 0))
    if divisor & (divisor - 1):
        return None
    amount = divisor.bit_length() - 1
    return amount if shift in (None, amount) else None


def _exponent_shift(scale):
    return max(0, min(31, round(math.log2(math.log(2) / float(scale)))))


def lower(graph, *, fast=False):
    """Select hardware numerics only for statically representable subgraphs."""
    tensors = dict(graph["tensors"])
    nodes = graph["nodes"]
    producers = {node["output"]: node for node in nodes}
    users = Counter(name for node in nodes for name in node["inputs"])
    users.update(graph["outputs"])
    replacement, remove, ordered = {}, set(), set()

    def temporary(base, shape, dtype):
        name = base
        while name in tensors:
            name += "_"
        tensors[name] = {"kind": "value", "shape": list(shape), "dtype": dtype}
        return name

    def private(*entries):
        return all(entry and users[entry["output"]] == 1 for entry in entries)

    def matmul(entry):
        return (entry and entry["op"] == "Gemm" and len(entry["inputs"]) == 2 and
                not entry["attrs"].get("a_zero_point") and not entry["attrs"].get("b_zero_point"))

    # Prefer the complete QK/softmax/PV chain over the standalone vector opcode.
    for end in nodes:
        out = end["output"]
        if tensors[out]["dtype"] != "int8" or _power_shift(end, 8) is None:
            continue
        pv = producers.get(end["inputs"][0])
        if not matmul(pv) or pv["attrs"].get("transpose_b"):
            continue
        prob, value = pv["inputs"]
        soft = producers.get(prob)
        if not soft or soft["op"] != "QSoftmax":
            continue
        sa = soft["attrs"]
        if sa.get("axis") != 1 or sa.get("out_scale") != [1, 256] or sa.get("zero_point", 0):
            continue
        quant = producers.get(soft["inputs"][0])
        shift = _power_shift(quant)
        if shift is None or tensors[quant["output"]]["dtype"] != "int8":
            continue
        qk = producers.get(quant["inputs"][0])
        if not matmul(qk) or not private(pv, soft, quant, qk):
            continue
        query, key = qk["inputs"]
        transpose = None
        if not qk["attrs"].get("transpose_b"):
            transpose = producers.get(key)
            if not (transpose and transpose["op"] == "Transpose" and transpose["attrs"].get("perm") == [1, 0]):
                continue
            key = transpose["inputs"][0]
        q, k, v = (tensors[x] for x in (query, key, value))
        if any(t["dtype"] != "int8" or len(t["shape"]) != 2 for t in (q, k, v)):
            continue
        rows, dim = q["shape"]
        length, width = v["shape"]
        causal = "causal_offset" in sa
        position = sa.get("causal_offset", 0)
        if (len({query, key, value}) != 3 or ordered.intersection((query, key, out)) or
                k["shape"] != [length, dim] or length > 128 or max(dim, width) > 511 or
                any(n % 4 for n in (dim, width, length)) or rows > length or
                (causal and position + rows > length) or (not causal and rows != length)):
            continue
        packed = temporary(out + "/v-transposed", [width, length], "int8")
        replacement[out] = [
            {"id": out + "/v-transpose", "op": "Transpose", "inputs": [value], "output": packed,
             "attrs": {"perm": [1, 0]}},
            {**end, "op": "Attention", "inputs": [query, key, packed],
             "attrs": {"causal": causal, "position": position, "fast": fast, "packed_v": True,
                       "exp_shift": _exponent_shift(Fraction(*sa["in_scale"]) / (1 << shift))}},
        ]
        ordered.update((query, key, out))
        remove.update(row["output"] for row in (pv, soft, quant, qk))
        if private(transpose):
            remove.add(transpose["output"])

    for node in nodes:
        op, out, inputs, attrs = (node[key] for key in ("op", "output", "inputs", "attrs"))
        if out in remove or out in replacement:
            continue
        if (op == "QRMSNorm" and tensors[out]["shape"][0] <= 1023 and
                tensors[out]["shape"][-1] % 4 == 0):
            replacement[out] = [{**node, "op": "Z15RMSNorm"}]
        if op == "Requant":
            shift = _power_shift(node)
            product = producers.get(inputs[0])
            if (shift is not None and product and product["op"] == "Mul" and private(product) and
                    tensors[out]["dtype"] == "int8" and all(tensors[n]["dtype"] == "int8" for n in product["inputs"]) and
                    any(producers.get(n, {}).get("op") in ("SiLU", "Lookup") for n in product["inputs"])):
                replacement[out] = [{**node, "op": "Z15Mul", "inputs": product["inputs"],
                                     "attrs": {"shift": shift, "scale": 1}}]
                remove.add(product["output"])
        if op != "QSoftmax":
            continue
        shape = tensors[inputs[0]]["shape"]
        if (len(shape) != 2 or attrs.get("axis") != 1 or shape[0] > 511 or shape[1] > 128 or shape[1] % 4 or
                attrs.get("out_scale") != [1, 256] or attrs.get("zero_point", 0) or
                tensors[out]["dtype"] != "uint8"):
            continue
        causal = "causal_offset" in attrs
        position = attrs.get("causal_offset", 0)
        if causal and position + shape[0] > shape[1]:
            continue
        scores = temporary(out + "/scores-int32", shape, "int32")
        replacement[out] = [
            {"id": out + "/score-cast", "op": "Cast", "inputs": inputs, "output": scores,
             "attrs": {"dtype": "int32"}},
            {**node, "op": "Z15Softmax", "inputs": [scores],
             "attrs": {"causal": causal, "position": position, "fast": fast,
                       "exp_shift": _exponent_shift(Fraction(*attrs["in_scale"]))}},
        ]

    if not replacement:
        return graph
    changed = [item for node in nodes if node["output"] not in remove
               for item in replacement.get(node["output"], [node])]
    names = set(graph["inputs"] + graph["outputs"])
    for node in changed:
        names.update(node["inputs"])
        names.add(node["output"])
    return {**graph, "nodes": changed, "tensors": {k: v for k, v in tensors.items() if k in names}}
