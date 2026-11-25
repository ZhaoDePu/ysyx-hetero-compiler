"""Decode a static integer ONNX subset using only the Python standard library.

Wire schema: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3
"""

import math
import struct
from collections import defaultdict

from .ir import DTYPES, GraphError

TYPES = {2: "uint8", 3: "int8", 6: "int32"}


def _varint(data, pos):
    value = 0
    for i in range(10):
        if pos == len(data):
            raise GraphError("truncated ONNX varint")
        byte = data[pos]
        pos += 1
        if i == 9 and byte > 1:
            raise GraphError("ONNX varint overflow")
        value |= (byte & 127) << (7 * i)
        if byte < 128:
            return value, pos
    raise GraphError("ONNX varint overflow")


def _message(data):
    if not isinstance(data, (bytes, memoryview)):
        raise GraphError("invalid ONNX submessage")
    data = memoryview(data)
    pos, fields = 0, defaultdict(list)
    while pos < len(data):
        key, pos = _varint(data, pos)
        tag, wire = key >> 3, key & 7
        if not 0 < tag < (1 << 29):
            raise GraphError("invalid ONNX field number")
        if wire == 0:
            value, pos = _varint(data, pos)
        elif wire in (1, 2, 5):
            if wire == 2:
                length, pos = _varint(data, pos)
            else:
                length = 8 if wire == 1 else 4
            if length > len(data) - pos:
                raise GraphError("truncated ONNX field")
            value = data[pos:pos + length]
            pos += length
        else:
            raise GraphError(f"unsupported protobuf wire type {wire}")
        fields[tag].append((wire, value))
    return fields


def _one(msg, tag, wire, default=None):
    values = _all(msg, tag, wire)
    if len(values) > 1:
        raise GraphError(f"duplicate ONNX field {tag}")
    return values[0] if values else default


def _all(msg, tag, wire):
    if any(w != wire for w, _ in msg.get(tag, [])):
        raise GraphError(f"wrong ONNX wire type for field {tag}")
    return [v for _, v in msg.get(tag, [])]


def _text(value):
    if value is None:
        return ""
    try:
        return bytes(value).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GraphError("invalid ONNX UTF-8") from exc


def _repeated_ints(msg, tag):
    for wire, value in msg.get(tag, []):
        if wire == 0:
            yield value
        elif wire == 2:
            pos = 0
            while pos < len(value):
                x, pos = _varint(value, pos)
                yield x
        else:
            raise GraphError("invalid ONNX integer list")


def _signed32(value):
    value &= 0xffffffff
    return value if value < 0x80000000 else value - 0x100000000


def _value_info(payload):
    v = _message(payload)
    typ = _message(_one(v, 2, 2))
    t = _message(_one(typ, 1, 2))
    shape = _message(_one(t, 2, 2))
    dims = []
    for entry in _all(shape, 1, 2):
        d = _message(entry)
        value = _one(d, 1, 0, 0)
        if 2 in d or value <= 0:
            raise GraphError("ONNX dynamic dimensions are unsupported")
        dims.append(value)
    dtype = TYPES.get(_one(t, 1, 0))
    if dtype is None:
        raise GraphError("unsupported ONNX tensor dtype")
    return _text(_one(v, 1, 2)), {"shape": dims, "dtype": dtype}


def _initializer(payload):
    t = _message(payload)
    dtype = TYPES.get(_one(t, 2, 0))
    if dtype is None or 3 in t or 13 in t or _one(t, 14, 0, 0):
        raise GraphError("unsupported ONNX initializer (type, segmented, or external)")
    dims = list(_repeated_ints(t, 1))
    raw = _one(t, 9, 2)
    if raw is not None:
        if 5 in t or len(raw) != math.prod(dims) * DTYPES[dtype][0]:
            raise GraphError("invalid ONNX initializer raw_data")
        fmt = {"uint8": "B", "int8": "b", "int32": "i"}[dtype]
        data = [v for (v,) in struct.iter_unpack("<" + fmt, raw)]
    else:
        data = [_signed32(v) for v in _repeated_ints(t, 5)]
    return _text(_one(t, 8, 2)), {"shape": dims, "dtype": dtype, "data": data,
                                  "kind": "constant"}


def parse_model(data):
    model = _message(data)
    graphs = _all(model, 7, 2)
    if len(graphs) != 1 or 20 in model or 25 in model:
        raise GraphError("expected one static ONNX graph, without training or functions")
    opsets = []
    for spec in _all(model, 8, 2):
        spec = _message(spec)
        if _text(_one(spec, 1, 2)) in ("", "ai.onnx"):
            opsets.append(_one(spec, 2, 0, 0))
    if len(opsets) != 1 or not 10 <= opsets[0] <= 23:
        raise GraphError("supported main ONNX opsets are 10..23")
    graph = _message(graphs[0])
    if 15 in graph:
        raise GraphError("sparse ONNX initializers are unsupported")
    tensors, inputs, outputs, nodes = {}, set(), [], []
    for field in (11, 12, 13):
        for payload in _all(graph, field, 2):
            name, spec = _value_info(payload)
            if not name or name in tensors and tensors[name] != spec:
                raise GraphError(f"missing or conflicting ONNX ValueInfo {name}")
            tensors[name] = spec
            if field == 11:
                inputs.add(name)
            if field == 12:
                outputs.append(name)
    initializers = set()
    for payload in _all(graph, 5, 2):
        name, spec = _initializer(payload)
        if not name or name in initializers:
            raise GraphError(f"missing or duplicate ONNX initializer {name}")
        if name in tensors and any(tensors[name][k] != spec[k] for k in ("shape", "dtype")):
            raise GraphError(f"ONNX initializer differs from ValueInfo for {name}")
        initializers.add(name)
        tensors[name] = spec
    for name in inputs - initializers:
        tensors[name]["kind"] = "input"
    for idx, payload in enumerate(_all(graph, 1, 2)):
        node = _message(payload)
        op = _text(_one(node, 4, 2))
        name = _text(_one(node, 3, 2)) or f"onnx_{idx}"
        ins = [_text(i) for i in _all(node, 1, 2)]
        outs = [_text(i) for i in _all(node, 2, 2)]
        if (_text(_one(node, 7, 2)) not in ("", "ai.onnx") or
                len(outs) != 1 or 5 in node or 8 in node):
            raise GraphError(f"{name}: unsupported ONNX domain, attribute, or output count")
        if op == "MatMulInteger":
            if any(ins[2:]) or len(ins) > 4:
                raise GraphError(f"{name}: explicit MatMulInteger zero points are unsupported")
            op, ins, attrs = "Gemm", ins[:2], {"dtype": "int32"}
        elif op in ("Add", "Mul", "Relu"):
            attrs = {"narrowing": "wrap"}
        else:
            raise GraphError(f"{name}: unsupported ONNX operator {op}")
        nodes.append({"id": name, "op": op, "inputs": ins, "output": outs[0], "attrs": attrs})
    return {"name": _text(_one(graph, 2, 2)), "tensors": tensors,
            "nodes": nodes, "outputs": outputs}
