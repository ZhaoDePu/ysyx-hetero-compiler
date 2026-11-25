"""Proposed ASIC IP v3 descriptor lowering and functional simulator."""

import json
from math import prod
from pathlib import Path
from struct import pack, unpack_from

from . import ir, reference

ABI = "asic-ip-v3"
CAPACITY = 64 * 1024
MAGIC = int.from_bytes(b"IPV3", "little")
WORDS = 32
NONE = (1 << 32) - 1
OPS = {"Gemm": 1, "Conv": 2, "Requant": 3, "Relu": 4, "SiLU": 5, "GELU": 6,
       "RowSoftmax": 7, "RMSNorm": 8, "Add": 9, "Mul": 10}
OP_NAMES = {code: name for name, code in OPS.items()}
TYPE_IDS = {"int8": 1, "uint8": 2, "int32": 3}
TYPES = {code: name for name, code in TYPE_IDS.items()}
FORMAT = {"int8": "b", "uint8": "B", "int32": "i"}
NPU_OPS = {"Gemm", "Conv"}
OP_FLAGS = {"Gemm": 0x1, "Conv": 0, "Requant": 0x8, "Relu": 0, "SiLU": 0x4, "GELU": 0,
            "RowSoftmax": 0x2, "RMSNorm": 0, "Add": 0x1, "Mul": 0}


def supports(stage, graph, profile):
    op = stage["op"]
    if op not in OPS or stage.get("fused"):
        return False
    if stage["device"] != ("NPU" if op in NPU_OPS else "VPU"):
        return False
    names = stage["inputs"] + [stage["output"]]
    return all(graph["tensors"][name]["dtype"] in TYPE_IDS for name in names)


def requirements(stage, graph, profile):
    return {}


def metadata_needed(stage, graph, profile):
    # The descriptors live in YHB DATA, not in the shared data SRAM.
    return 0


def _signed(value):
    return value if value < (1 << 31) else value - (1 << 32)


def _args(op, attrs):
    flags = int(attrs.get("narrowing", "saturate") == "wrap")
    p = []
    if op in ("Gemm", "Conv"):
        p = [attrs.get("a_zero_point", 0), attrs.get("b_zero_point", 0)]
        if op == "Gemm":
            flags |= int(attrs.get("transpose_b", False)) << 1
        else:
            t, l, b, r = attrs["pads"]
            sy, sx = attrs["strides"]
            p += [attrs.get("shift", 0), sy | (sx << 16), t | (l << 16), b | (r << 16)]
    elif op == "Requant":
        shift = attrs["shift"]
        flags |= int(attrs.get("rounding") == "nearest_even") << 4
        p = [attrs["multiplier"], shift, attrs.get("divisor", 1 << shift),
             attrs["zero_point"], attrs["input_zero_point"]]
    elif op == "SiLU":
        flags |= int(attrs.get("mode") == "legacy_q4_knots") << 3
    elif op == "Add":
        flags |= int(attrs.get("broadcast") == "last") << 1
    elif op == "RowSoftmax":
        flags |= int(attrs.get("causal_mask", False)) << 2
    return flags, [value & NONE for value in p] + [0] * (7 - len(p))


def _shape(tensor):
    shape = tensor["shape"]
    return [*shape, *([0] * (4 - len(shape)))]


def _descriptor(stage, graph, plan):
    tensors, allocations = graph["tensors"], plan["allocations"]
    a, *extra = stage["inputs"]
    b = extra[0] if extra else None
    bias = extra[1] if len(extra) == 2 else None
    dst = stage["output"]
    ta, tb, td = tensors[a], tensors.get(b), tensors[dst]
    flags, params = _args(stage["op"], stage["attrs"])

    def offset(name):
        if name is None:
            return NONE
        rec = allocations[name]
        if rec["offset"] < 0 or rec["offset"] + rec["size"] > CAPACITY or rec["offset"] % 8:
            raise ir.GraphError(f"{name}: invalid ASIC IP SRAM allocation")
        return rec["offset"]

    return [MAGIC, OPS[stage["op"]], flags,
            len(ta["shape"]), len(tb["shape"]) if tb else 0, len(td["shape"]),
            offset(a), offset(b), offset(dst), offset(bias),
            TYPE_IDS[ta["dtype"]], TYPE_IDS[tb["dtype"]] if tb else 0, TYPE_IDS[td["dtype"]],
            *_shape(ta), *(_shape(tb) if tb else [0] * 4), *_shape(td), *params]


def lower(graph, plan, profile, output):
    if profile["npu"]["variant"] != ABI or profile["sram"]["capacity"] != CAPACITY:
        raise ir.GraphError("ASIC IP v3 requires a native 65536-byte data arena")
    commands = []
    for stage in plan["stages"]:
        if stage["device"] == "GPU":
            continue
        if not supports(stage, graph, profile):
            raise ir.GraphError(f"{stage['id']}: cannot emit ASIC IP v3 descriptor")
        words = _descriptor(stage, graph, plan)
        # Validate descriptor geometry before encoding.
        _decode(words, stage["device"])
        commands.append({"stage": stage["id"], "device": stage["device"], "words": words})
    result = {"abi": ABI, "commands": commands, "constants": []}
    output = Path(output)
    payload = [word for cmd in commands for word in cmd["words"]]
    (output / "npu.bin").write_bytes(pack(f"<{len(payload)}I", *payload))
    (output / "npu.hex").write_text("".join(f"{word:08x}\n" for word in payload), encoding="ascii")
    (output / "npu.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _tensor(words, rank, typeid, offset):
    kind = TYPES.get(typeid)
    if not kind or not 1 <= rank <= 4 or min(words[:rank]) <= 0 or any(words[rank:]):
        raise ValueError("invalid ASIC IP tensor type or shape")
    shape = list(words[:rank])
    size = prod(shape) * ir.DTYPES[kind][0]
    if not 0 <= offset <= CAPACITY - size or offset % 8:
        raise ValueError("invalid ASIC IP tensor extent")
    return {"shape": shape, "dtype": kind}, (offset, offset + size)


def _decode(words, device):
    if len(words) != WORDS or words[0] != MAGIC or words[1] not in OP_NAMES:
        raise ValueError("invalid ASIC IP v3 descriptor header")
    _, opcode, flags, arank, brank, crank, ao, bo, co, bias, ati, bti, cti = words[:13]
    op = OP_NAMES[opcode]
    # Flags: bit0 wrap, bit1 transpose/broadcast, bit2 causal,
    # bit3 legacy knots, bit4 nearest-even. The table is opcode-specific.
    if flags & ~(1 | (OP_FLAGS[op] << 1)) or device != ("NPU" if op in NPU_OPS else "VPU"):
        raise ValueError("invalid ASIC IP descriptor flags or engine")
    a, ar = _tensor(words[13:17], arank, ati, ao)
    c, cr = _tensor(words[21:25], crank, cti, co)
    operands, ranges = {"a": a}, {"a": ar}
    names = ["a"]
    if op in ("Gemm", "Conv", "Add", "Mul"):
        b, br = _tensor(words[17:21], brank, bti, bo)
        operands["b"], ranges["b"] = b, br
        names.append("b")
    elif bo != NONE or brank or bti or any(words[17:21]):
        raise ValueError("ASIC IP unary descriptor has a B operand")
    if bias != NONE:
        if op not in NPU_OPS:
            raise ValueError("invalid ASIC IP bias operand")
        bb, br = _tensor([c["shape"][-1], 0, 0, 0], 1, TYPE_IDS["int32"], bias)
        operands["bias"], ranges["bias"] = bb, br
        names.append("bias")
    if any(lo < cr[1] and cr[0] < hi for lo, hi in ranges.values()):
        raise ValueError("ASIC IP output aliases a live source")

    p = words[25:]
    attrs = {"dtype": c["dtype"], "narrowing": "wrap" if flags & 1 else "saturate"}
    if op in NPU_OPS:
        attrs.update(a_zero_point=_signed(p[0]), b_zero_point=_signed(p[1]))
        if op == "Gemm":
            attrs["transpose_b"] = bool(flags & 2)
            used = 2
        else:
            attrs["shift"] = p[2]
            attrs["strides"] = [p[3] & 0xffff, p[3] >> 16]
            attrs["pads"] = [p[4] & 0xffff, p[4] >> 16, p[5] & 0xffff, p[5] >> 16]
            used = 6
    elif op == "Requant":
        attrs.update(multiplier=_signed(p[0]), shift=p[1], zero_point=_signed(p[3]),
                     input_zero_point=_signed(p[4]), rounding="nearest_even" if flags & 16 else "floor")
        if flags & 16:
            attrs["divisor"] = p[2]
        elif p[1] >= 32 or p[2] != 1 << p[1]:
            raise ValueError("invalid ASIC IP arithmetic shift")
        used = 5
    else:
        if op == "SiLU" and flags & 8:
            attrs["mode"] = "legacy_q4_knots"
        if op == "Add" and flags & 2:
            attrs["broadcast"] = "last"
        if op == "RowSoftmax":
            attrs["causal_mask"] = bool(flags & 4)
        used = 0
    if any(p[used:]):
        raise ValueError("reserved ASIC IP descriptor parameters are nonzero")
    node = {"id": "op", "op": op, "inputs": names, "output": "c", "attrs": attrs}
    normalized = ir.normalize({"tensors": {**operands, "c": c}, "nodes": [node], "outputs": ["c"]})
    return normalized, {**ranges, "c": cr}


def execute_descriptor(image, words, device):
    """Functionally execute one decoded descriptor over the shared data byte array."""
    if len(image) != CAPACITY:
        raise ValueError("ASIC IP simulator requires the full 64 KiB arena")
    graph, ranges = _decode(words, device)
    inputs = {}
    for name in graph["inputs"]:
        spec = graph["tensors"][name]
        offset = ranges[name][0]
        inputs[name] = list(unpack_from(f"<{prod(spec['shape'])}{FORMAT[spec['dtype']]}", image, offset))
    dst = graph["tensors"]["c"]["dtype"]
    out = reference.evaluate(graph, inputs)["c"]
    lo, hi = ranges["c"]
    image[lo:hi] = pack(f"<{len(out)}{FORMAT[dst]}", *out)


def replay(blob, overrides=None):
    """Replay a native-only target-5 YHB using its checked binary records."""
    from .yhb import ASIC_IP_V3, inspect_yhb

    container = inspect_yhb(blob)
    if container["target"] != ASIC_IP_V3:
        raise ValueError("ASIC IP software replay requires a target-5 YHB")
    sections = container["sections"]

    def span(section, offset, size):
        record = sections[section]
        if offset < 0 or offset + size > record["bytes"]:
            raise ValueError(f"YHB {section} reference is out of range")
        start = record["offset"] + offset
        return blob[start:start + size]

    def words(section, index, stride):
        return unpack_from(f"<{stride}I", span(section, index * stride * 4, stride * 4))

    def name(offset, size):
        if not size:
            raise ValueError("empty YHB symbol")
        return span("strings", offset, size).decode("utf-8")

    if sections["dma"]["count"]:
        raise ValueError("ASIC IP software replay has no DMA aperture")
    data = bytearray(CAPACITY)
    end = 0
    for i in range(sections["initial"]["count"]):
        offset, size, source, zero = words("initial", i, 4)
        if zero or not size or offset < end or offset + size > CAPACITY or source % 8:
            raise ValueError("invalid ASIC IP initial span")
        data[offset:offset + size] = span("data", source, size)
        end = offset + size

    overrides = dict(overrides or {})
    symbols, outputs = set(), {}
    for i in range(sections["symbols"]["count"]):
        no, nl, flags, dtype, rank, *fields = words("symbols", i, 12)
        offset, size, reserved = fields[4:]
        key = name(no, nl)
        if reserved or flags not in (1, 2, 3, 5, 7) or key in symbols:
            raise ValueError("invalid ASIC IP YHB symbol")
        spec, extent = _tensor(fields[:4], rank, dtype, offset)
        if size != extent[1] - offset:
            raise ValueError("ASIC IP symbol disagrees with dtype and dimensions")
        symbols.add(key)
        if flags & 1:
            if key in overrides:
                value = overrides.pop(key)
                if len(value) != size:
                    raise ValueError(f"{key}: incorrect input byte count")
                data[offset:offset + size] = value
            elif flags & 4:
                raise ValueError(f"missing ASIC IP input {key}")
        if flags & 2:
            outputs[key] = (offset, size)
    if overrides:
        raise ValueError("unknown ASIC IP input")

    launched, active = set(), None
    for i in range(sections["commands"]["count"]):
        opcode, index, r1, r2 = words("commands", i, 4)
        if r1 or r2 or index >= sections["stages"]["count"]:
            raise ValueError("invalid ASIC IP scheduling instruction")
        if opcode == 2:
            if index not in launched:
                raise ValueError("ASIC IP waits on unlaunched stage")
            if active == index:
                active = None
            continue
        if opcode != 1 or index in launched or active is not None:
            raise ValueError("ASIC IP launches must have a fence")
        no, nl, engine, offset, count, aux, jobs, zone = words("stages", index, 8)
        if engine == 1:
            raise ValueError(f"{name(no, nl)}: v6 GPU commands need a hardware backend")
        if offset % 8 or aux % 8 or jobs or zone or engine not in (2, 3) or count != WORDS:
            raise ValueError("invalid ASIC IP YHB stage")
        execute_descriptor(data, unpack_from(f"<{WORDS}I", span("data", offset, 4 * count)),
                           "NPU" if engine == 2 else "VPU")
        launched.add(index)
        active = index
    if len(launched) != sections["stages"]["count"]:
        raise ValueError("ASIC IP command list omits stages")
    return {key: bytes(data[lo:lo + size]) for key, (lo, size) in outputs.items()}
