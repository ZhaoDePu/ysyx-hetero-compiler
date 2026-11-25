"""Native Phase 8 EXT/PROGRAM descriptors and Z15 v1 register queues."""

import json
import math
import struct
from pathlib import Path

BYTES = {"int8": 1, "uint8": 1, "int32": 4}
P8_ARENA = 4096
Z15_HALF = 1 << 20
P8_PROGRAM = 0x4E340106
P8_WIDE = 0x4E37010B
Z15_CTRL = 0x10
Z15_NPU_START = 1 << 3
Z15_ATTN_START = 1 << 5
Z15_VPU_START = 1 << 6


def _target(profile):
    npu = profile.get("npu", {})
    name = profile.get("id", profile.get("name", ""))
    variant = str(npu.get("abi", npu.get("variant", ""))).lower()
    if name == "asic_nangate45_p8" or variant in ("phase8", "p8", "p8_ext", "p8-ext-program") or npu.get("window_bytes") == P8_ARENA:
        return "p8"
    if name.startswith("fpga_zu15eg") or variant in ("phase12c", "z15", "z15_mmio_v1"):
        return "z15"
    return None


def _nodes(stage):
    fused = stage.get("fused")
    return fused if fused and all(isinstance(n, dict) for n in fused) else [stage]


def _packed_name(name):
    return f"{name}.__npu_colmajor"


def _partial_name(stage):
    return f"{stage['id']}.__npu_partial"


def _size(tensor):
    return math.prod(tensor["shape"]) * BYTES[tensor["dtype"]]


def _constant(tensor):
    return "data" in tensor and tensor.get("kind") != "input"


def _tile_k(k, n, m, wide=False):
    if not wide:
        return next((t for t in (16, 32, 64, 128) if k <= t), 128)
    if n > 16 and (k > 32 or k % 4):
        raise ValueError("Phase 8 wide multi-column tile requires K≤32 and a word-aligned pitch")
    max_slots = 16
    for t in (16, 32, 64, 128):
        if math.ceil(m / 4) * math.ceil(n / 16) * math.ceil(k / t) <= max_slots:
            return t
    raise ValueError("Phase 8 wide PROGRAM exceeds 16 slots")


def _tensor_spec(stage, graph, target):
    tensors = graph["tensors"]
    chain = _nodes(stage)
    main = chain[0]
    op, ins, attrs = main["op"], main["inputs"], main.get("attrs", {})
    a, b = (tensors[name] for name in ins[:2])
    c = tensors[stage["output"]]
    main_out = tensors[main["output"]]
    n = c["shape"][-1]
    transpose = attrs.get("transpose_b", False)
    if op == "Gemm":
        m, k = a["shape"]
    else:
        kh, kw, ci, _ = b["shape"]
        batch, h, w, _ = a["shape"]
        _, oh, ow, _ = c["shape"]
        m, k = batch * oh * ow, kh * kw * ci
    if a["dtype"] not in ("int8", "uint8") or b["dtype"] != "int8":
        raise ValueError("tensor engines require int8 weights and int8 or uint8 activations")
    if ((target == "p8" and max(m, n) > 255) or
            (target == "z15" and max(m, n, k) > 511)):
        raise ValueError("tensor engine dimension limit exceeded")
    if transpose and (op != "Gemm" or k % 4 or n % 4 or k > 255 or target != "p8"):
        raise ValueError("Phase 8 linear transpose requires K and N divisible by four")
    if target == "z15" and (k % 4 or (c["dtype"] == "int8" and n % 4) or not _constant(b)):
        raise ValueError("Z15 tensor needs constant packed weights, K divisible by four and word-packed output rows")
    bias = ins[2] if len(ins) == 3 else None
    bias_mode = "wide" if bias else None
    if bias and (tensors[bias]["dtype"] != "int32" or tensors[bias]["shape"] != [n]):
        raise ValueError("tensor engine int32 bias must match the output channel count")
    activation = None
    activation_mode = None
    shift = attrs.get("shift", 0)
    narrowing = attrs.get("narrowing", "saturate")
    current = main["output"]
    seen_requant = False
    for part in chain[1:]:
        part_attrs = part.get("attrs", {})
        if part["inputs"][0] != current or activation is not None:
            raise ValueError("nonlinear or branched tensor fusion")
        if part["op"] == "Add" and bias is None:
            bias = part["inputs"][1]
            bt = tensors[bias]
            if part_attrs.get("broadcast") != "last" or bt["shape"] != [n]:
                raise ValueError("tensor bias fusion requires last-axis bias")
            if not seen_requant and (tensors[current]["dtype"] == bt["dtype"] == tensors[part["output"]]["dtype"] == "int32"):
                if shift:
                    raise ValueError("int32 bias cannot be hoisted before an existing tensor shift")
                bias_mode = "wide"
            elif (target == "p8" and
                  tensors[current]["dtype"] == bt["dtype"] == tensors[part["output"]]["dtype"] == "int8" and
                  part_attrs.get("narrowing") == "wrap"):
                bias_mode = "legacy"
            else:
                raise ValueError("unsupported tensor bias narrowing")
        elif part["op"] == "Requant" and not seen_requant and main_out["dtype"] == "int32":
            if (part_attrs.get("multiplier", 1) != 1 or part_attrs.get("zero_point", 0) != 0 or
                    part_attrs.get("narrowing", "saturate") != "saturate" or
                    tensors[part["output"]]["dtype"] != "int8"):
                raise ValueError("native requant requires a signed, saturating shift")
            shift = min(31, shift + part_attrs.get("shift", 0))
            narrowing = part_attrs.get("narrowing", "saturate")
            seen_requant = True
        elif part["op"] in ("Relu", "SiLU", "GELU"):
            if tensors[current]["dtype"] != tensors[part["output"]]["dtype"]:
                raise ValueError("native activation requires unchanged dtype")
            activation = part["op"]
            activation_mode = part_attrs.get("mode")
        else:
            raise ValueError("unsupported tensor fusion sequence")
        current = part["output"]
    if c["dtype"] not in ("int8", "int32") or not 0 <= shift <= 31:
        raise ValueError("unsupported output dtype or shift")
    if target == "p8":
        wide = bias_mode == "wide" or k > 255
        if (c["dtype"] != "int8" or activation == "GELU" or
                (activation == "SiLU" and activation_mode != "legacy_q4_knots")):
            raise ValueError("Phase 8 native activation supports ReLU and the legacy signed Q4 knot approximation")
        if wide and bias_mode == "legacy":
            raise ValueError("Phase 8 wide PROGRAM requires int32 pre-requant bias")
        if wide and (op != "Gemm" or transpose or a["dtype"] != "int8" or narrowing == "wrap" or k > 4096):
            raise ValueError("Phase 8 signed-wide bias/large-K path requires saturated signed GEMM")
    else:
        wide = False
        if narrowing == "wrap" and c["dtype"] == "int8":
            raise ValueError("Z15 packed tensor epilogue saturates")
        if c["dtype"] == "int32" and (shift or activation in ("SiLU", "GELU")):
            raise ValueError("Z15 int32 tensor output has no requantization or lookup")
    spec = dict(op=op, a=ins[0], b=ins[1], c=stage["output"], bias=bias, bias_mode=bias_mode,
                m=m, n=n, k=k, shift=shift, activation=activation, unsigned_a=a["dtype"] == "uint8",
                activation_mode=activation_mode,
                wrap=narrowing == "wrap", packed=c["dtype"] == "int8", wide=wide, transpose=transpose)
    spec["tile_k"] = _tile_k(k, n, m, wide)
    if op == "Conv":
        sy, sx = attrs.get("strides", [1, 1])
        t, l, bot, r = attrs.get("pads", [0, 0, 0, 0])
        if target == "p8":
            if (max(batch, h, w, ci, oh, ow) > 255 or max(kh, kw) > 7 or
                    max(sy, sx) > 8 or max(t, bot, l, r) > 7 or spec["wrap"]):
                raise ValueError("Phase 8 Conv geometry or epilogue exceeds native ABI")
        elif (batch != 1 or kh != kw or kh not in (1, 3) or ci % 4 or
              sy != sx or sy not in (1, 2) or t != bot or t != l or t != r or t not in (0, 1) or
              (kh == 1 and t) or max(h, w, ci) > 511):
            raise ValueError("Z15 Conv requires batch 1, symmetric 1/3 kernel, stride 1/2 and pad 0/1")
        spec.update(h=h, w=w, ci=ci, batch=batch, oh=oh, ow=ow, kh=kh, kw=kw,
                    sy=sy, sx=sx, t=t, bot=bot, l=l, r=r)
    return spec


def _ewise_spec(stage, graph, target, profile):
    tensors = graph["tensors"]
    stage = _nodes(stage)[0]
    op, names, out = stage["op"], stage["inputs"], stage["output"]
    attrs = stage.get("attrs", {})
    if attrs.get("broadcast", "none") != "none" or attrs.get("narrowing", "saturate") != "saturate":
        raise ValueError("native elementwise requires shape equality and saturation")
    expected_dtype = "uint8" if op == "RowSoftmax" else "int8"
    if (tensors[out]["dtype"] != expected_dtype or
            any(tensors[n]["dtype"] != "int8" or tensors[n]["shape"] != tensors[out]["shape"] for n in names)):
        raise ValueError("native elementwise requires equal signed-byte inputs")
    if target == "p8":
        shape = tensors[out]["shape"]
        opcode = {"RMSNorm": 8, "RowSoftmax": 9, "Add": 10, "ResidualAdd": 10}.get(op)
        if opcode is None or len(shape) != 2 or not 1 <= shape[0] <= 8 or shape[1] not in (4, 8, 16):
            raise ValueError("Phase 8 row operations use rows 1..8 and columns 4/8/16")
        return dict(a=names[0], b=names[1] if opcode == 10 else None, c=out, op=opcode,
                    rows=shape[0], cols=shape[1], causal=attrs.get("causal_mask", False))
    if not profile.get("vpu", {}).get("enabled"):
        raise ValueError("target VPU unavailable")
    opcode = {"Add": 0, "ResidualAdd": 0, "Mul": 6, "Relu": 11, "SiLU": 11, "GELU": 11}.get(op)
    if opcode is None:
        raise ValueError("unsupported VPU operation")
    return dict(a=names[0], b=names[1] if len(names) > 1 else None, c=out,
                num_bytes=_size(tensors[out]), op=opcode, activation=op if opcode == 11 else None,
                activation_mode=attrs.get("mode"))


def _attention_spec(stage, graph, target):
    tensors = graph["tensors"]
    q, k, v = (tensors[name] for name in stage["inputs"])
    out = tensors[stage["output"]]
    attrs = stage.get("attrs", {})
    rows, dim = q["shape"]
    length, n = v["shape"]
    causal, position = attrs.get("causal", False), attrs.get("position", 0)
    exp_shift = attrs.get("exp_shift", 0)
    if (target != "z15" or any(x["dtype"] != "int8" for x in (q, k, v, out)) or
            not _constant(v) or dim > 511 or n > 511 or length > 128 or
            any(x % 4 for x in (dim, n, length)) or not 0 <= exp_shift < 32 or
            attrs.get("narrowing", "saturate") != "saturate" or
            (causal and not 0 <= position <= length - rows) or
            (not causal and (rows != length or position))):
        raise ValueError("Z15 attention requires signed compact tensors, constant V, aligned row widths and valid query geometry")
    return dict(op="Attention", q=stage["inputs"][0], key=stage["inputs"][1],
                v=stage["inputs"][2], c=stage["output"], m=rows, n=n, k=dim, length=length,
                tile_k=_tile_k(max(dim, length), n, rows), causal=causal, position=position, shift=exp_shift)


def _scores(stage):
    return f"{stage['id']}.__npu_scores"


def _probabilities(stage):
    return f"{stage['id']}.__npu_probabilities"


def ordered_buffers(stage, graph, profile):
    """Buffers required to be ordered, disjoint, and inside one Z15 scratch half."""
    spec = _spec(stage, graph, profile)
    if spec["op"] != "Attention":
        return []
    return [spec["q"], spec["key"], _packed_name(spec["v"]), _scores(stage), _probabilities(stage), spec["c"]]


def _p8_detection(stage, graph):
    nodes = _nodes(stage)
    first = nodes[0]
    op, names = first["op"], first["inputs"]
    tensors = graph["tensors"]
    batch, h, w, channels = tensors[names[0]]["shape"]
    a, c = tensors[names[0]], tensors[stage["output"]]
    attrs = first.get("attrs", {})
    if not all(1 <= d <= 255 for d in (batch, h, w, channels)):
        raise ValueError("Phase 8 detection spatial dimensions must fit one byte")
    if op == "YoloDetectHead":
        if (len(nodes) != 2 or nodes[1]["op"] != "YoloNMS" or len(names) != 2 or
                nodes[1]["inputs"] != [first["output"]] or nodes[1]["output"] != stage["output"] or
                first["output"] in graph.get("outputs", ())):
            raise ValueError("Phase 8 POST requires a private detection head consumed by one NMS")
        nms = nodes[1].get("attrs", {})
        threshold, iou = nms.get("score_threshold", 0), nms.get("iou_threshold", 0)
        b = tensors[names[1]]
        n = b["shape"][-1]
        if (a["dtype"] != "int8" or channels != 16 or batch != 1 or h*w > 8 or
                attrs.get("reg_max", 4) != 4 or attrs.get("stride", 4) != 4 or
                b["dtype"] != "int8" or b["shape"][:3] != [1, h, w] or n not in (1, 2) or
                c["dtype"] != "uint8" or c["shape"] != [h*w, 7] or
                not 0 <= threshold <= 255 or not 0 <= iou <= 255):
            raise ValueError("Phase 8 POST requires batch1, 1..8 cells, 1..2 classes, reg_max=stride=4")
        return dict(op=5, a=names[0], b=names[1], c=stage["output"], h=h, w=w, classes=n,
                    threshold=threshold, iou=iou)
    if len(nodes) != 1 or a["dtype"] not in ("int8", "uint8") or a["dtype"] != c["dtype"]:
        raise ValueError("Phase 8 view/pool requires matching byte tensors")
    base = dict(a=names[0], c=stage["output"], batch=batch, h=h, w=w, channels=channels)
    if op == "SPPF":
        kernel = attrs.get("kernel", 3)
        if kernel not in (3, 5) or c["shape"] != [batch, h, w, channels*4]:
            raise ValueError("Phase 8 SPPF supports sequential kernel 3 or 5")
        return dict(base, op=4, kernel=kernel, unsigned_a=a["dtype"] == "uint8")
    b = tensors[names[1]]
    if (channels % 2 or b["shape"] != [batch, h, w, channels//2] or b["dtype"] != a["dtype"] or
            c["shape"] != [batch, h, w, channels*3//2]):
        raise ValueError("Phase 8 C2f requires NHWC and two branches in a 2:1 channel ratio")
    return dict(base, op=3, b=names[1])


def _spec(stage, graph, profile):
    target = _target(profile)
    if target is None:
        raise ValueError("no executable NPU ABI for profile")
    op = _nodes(stage)[0]["op"]
    if op in ("Gemm", "Conv"):
        return _tensor_spec(stage, graph, target)
    if op in ("C2f", "SPPF", "YoloDetectHead"):
        if target != "p8":
            raise ValueError("target lacks the Phase 8 view/pool/POST descriptor ABI")
        return _p8_detection(stage, graph)
    if len(_nodes(stage)) != 1:
        raise ValueError("non-tensor fusion unavailable")
    if op == "Attention":
        return _attention_spec(stage, graph, target)
    return _ewise_spec(stage, graph, target, profile)


def supports(stage, graph, profile):
    """Return whether a normalized stage preserves its semantics on this native target."""
    try:
        _spec(stage, graph, profile)
    except (KeyError, TypeError, ValueError):
        return False
    return True


def metadata_needed(stage, graph, profile):
    """Aligned bytes of immutable metadata allocated in the Phase 8 arena."""
    if _target(profile) != "p8":
        return 0
    spec = _spec(stage, graph, profile)
    if spec.get("wide"):
        m, n, k, tile = (spec[key] for key in ("m", "n", "k", "tile_k"))
        return 16 + 72 * math.ceil(m/4) * math.ceil(n/16) * math.ceil(k/tile)
    return {"Conv": 40, "Gemm": 24, 3: 32, 4: 32, 5: 24, 8: 16, 9: 16, 10: 16}[spec["op"]]


def requirements(stage, graph, profile):
    """Auxiliary scratch/constant allocations needed by the native ABI."""
    spec = _spec(stage, graph, profile)
    if spec["op"] == "Attention":
        rows, length, cols = (spec[x] for x in ("m", "length", "n"))
        return {
            _packed_name(spec["v"]): {"size": length * cols, "alignment": 32, "kind": "constant", "replaces": spec["v"]},
            _scores(stage): {"size": rows * length * 4, "alignment": 32, "kind": "temp"},
            _probabilities(stage): {"size": rows * length, "alignment": 32, "kind": "temp"},
        }
    if _target(profile) != "z15" or spec["op"] not in ("Gemm", "Conv"):
        return {}
    req = {_packed_name(spec["b"]): {"size": spec["k"] * spec["n"], "alignment": 32,
                                         "kind": "constant", "replaces": spec["b"]}}
    if spec["packed"] and spec["k"] > spec["tile_k"]:
        req[_partial_name(stage)] = {"size": 4 * spec["m"] * spec["n"], "alignment": 32, "kind": "temp"}
    return req


def buffering(stage, graph, profile):
    """Describe hardware-managed ping-pong banks. Banks are internal to the RTL."""
    spec = _spec(stage, graph, profile)
    if _target(profile) != "z15" or spec["op"] not in ("Gemm", "Conv", "Attention"):
        return None
    if spec["op"] == "Attention":
        return {"kind": "native_attention", "tile_k": spec["tile_k"],
                "phases": ["QK", "VPU_ROW_SOFTMAX", "PV"],
                "causal": spec["causal"], "materialized_scores": True,
                "a_buffer_count": 2, "b_buffer_count": 2, "output_buffer_count": 2}
    rows = profile["npu"].get("rows", 16)
    cols = profile["npu"].get("cols", 16)
    return {"kind": "native_tensor_pipeline", "tile_k": spec["tile_k"],
            "m_tiles": math.ceil(spec["m"] / rows), "n_tiles": math.ceil(spec["n"] / cols),
            "k_slabs": math.ceil(spec["k"] / spec["tile_k"]),
            "a_buffer_count": 2, "b_buffer_count": 2, "output_buffer_count": 2,
            "partial_accumulator": (_partial_name(stage) if spec["packed"] and
                                    spec["k"] > spec["tile_k"] else None)}


def _lookup(op, mode):
    values = []
    for i in range(256):
        x = (i if i < 128 else i - 256) / 16
        if op == "Relu":
            y = max(x, 0)
        elif op == "SiLU":
            if mode == "legacy_q4_knots":
                q = i if i < 128 else i - 256
                probability = int(256 / (1 + math.exp(-((q+8)//16))) + .5)
                values.append(int(q * probability / 256) & 255)
                continue
            y = x / (1 + math.exp(-x))
        else:
            y = x * (1 + math.erf(x / math.sqrt(2))) / 2
        values.append((round(y * 16) & 255))
    return values


def _span(plan, name, size, limit, alignment=4):
    obj = plan["allocations"].get(name)
    if obj is None:
        raise ValueError(f"missing NPU allocation {name}")
    addr = obj["offset"]
    if addr < 0 or addr % alignment or size > obj["size"] or addr + size > limit:
        raise ValueError(f"invalid NPU allocation for {name}, size {size}, alignment {alignment}")
    return addr, addr + size


def _overlap(left, right):
    return max(left[0], right[0]) < min(left[1], right[1])


def _guard(named, output, *, half=False, strict=False):
    spans = list(named.values())
    if half and len({byte // Z15_HALF for lo, hi in spans for byte in (lo, hi - 1)}) != 1:
        raise ValueError("Z15 operands cross a scratch ownership half")
    for name, span in named.items():
        if name != output and _overlap(span, named[output]):
            raise ValueError(f"NPU output aliases {name}")
    if strict and any(_overlap(a, b) for i, a in enumerate(spans) for b in spans[i + 1:]):
        raise ValueError("Phase 8 wide operands must be pairwise disjoint")


def _operand_spans(spec, tensors, plan, limit, *, packed=False):
    spans = {}
    for key in ("a", "b", "c", "bias"):
        if name := spec.get(key):
            physical = _packed_name(name) if packed and key == "b" else name
            spans[key] = _span(plan, physical, _size(tensors[name]), limit)
    return spans


def _p8_words(spec, spans):
    op, c = spec["op"], spans["c"][0]
    if op in (8, 9, 10):
        header = 0x4E360109 if spec["causal"] else 0x4E350100 | op
        return [header, spans["b"][0] if op == 10 else 0, c,
                spec["rows"] | spec["cols"] << 8 | int(spec["causal"]) << 16]
    if op in (3, 4):
        spatial = spec["h"] | spec["w"] << 8 | spec["channels"] << 16 | spec["batch"] << 24
        if op == 3:
            return [0x4E320103, spans["b"][0], c, 0xFFFFFFFF, 0, 0, spatial, 0]
        kernel = spec["kernel"]
        dims = spec["h"] | spec["w"] << 8 | kernel << 16 | kernel << 24
        return [0x4E320102, 0, c, 0xFFFFFFFF, 0, int(spec["unsigned_a"]) << 8, spatial, dims]
    if op == 5:
        dims = spec["h"] | spec["w"] << 8 | 4 << 16 | spec["classes"] << 24
        thresholds = 4 | spec["threshold"] << 8 | spec["iou"] << 16
        return [0x4E330104, spans["b"][0], c, dims, thresholds, 0]
    bias = spans["bias"][0] if "bias" in spans else 0xFFFFFFFF
    dim = spec["m"] | spec["n"] << 8 | spec["k"] << 16
    flags = spec["shift"] | int(spec["unsigned_a"]) << 8 | int(spec["wrap"]) << 9
    flags |= {None: 0, "Relu": 1, "SiLU": 2}[spec["activation"]] << 10
    flags |= int(spec["transpose"]) << 13
    header = 0x4E350100 if spec["transpose"] else 0x4E320100 | int(op == "Conv")
    words = [header, spans["b"][0], c, bias, dim, flags]
    if op == "Conv":
        s = spec
        words.extend([s["h"] | s["w"] << 8 | s["ci"] << 16 | s["batch"] << 24,
                      s["oh"] | s["ow"] << 8 | s["kh"] << 16 | s["kw"] << 24,
                      s["sy"] | s["sx"] << 8,
                      s["t"] | s["bot"] << 8 | s["l"] << 16 | s["r"] << 24])
    return words


def _p8_program(spec, spans, cursor):
    m, n, k = (spec[key] for key in ("m", "n", "k"))
    a, b, c = (spans[key][0] for key in ("a", "b", "c"))
    payloads, slots = [], []
    for row in range(0, m, 4):
        for col in range(0, n, 16):
            for red in range(0, k, spec["tile_k"]):
                tm, tn, tk = min(4, m - row), min(16, n - col), min(spec["tile_k"], k - red)
                first, last = red == 0, red + tk == k
                bias = spans["bias"][0] + 4 * col if "bias" in spans else 0
                flags = int(first) | int(last) << 1 | int("bias" in spans) << 2
                flags |= {None: 0, "Relu": 1, "SiLU": 2}[spec["activation"]] << 3
                dest = c + row * n + col
                descriptor = cursor + 16 + 48 * len(payloads)
                payloads.append({"offset": descriptor, "words": [
                    P8_WIDE, b + red * n + col, dest, tm | tn << 8 | tk << 16 | spec["shift"] << 24,
                    k | n << 16, n, red | k << 16, bias, flags, b, c,
                    row | col << 8 | m << 16 | n << 24]})
                slots.append([11, a + row * k + red, descriptor, dest, (tm - 1) * n + tn if last else 0, 0])
    table = cursor + 16 + 48 * len(payloads)
    header = {"offset": cursor, "words": [P8_PROGRAM, len(slots), table, 0]}
    return [header, *payloads, *({"offset": table + 24 * i, "words": w} for i, w in enumerate(slots))], len(slots)


def _lower_p8(graph, stages, plan, profile, out):
    limit = profile.get("npu", {}).get("window_bytes", profile["sram"]["capacity"])
    if limit != P8_ARENA or limit > profile["sram"]["capacity"]:
        raise ValueError("Phase 8 descriptor ABI requires a 4096-byte arena")
    base = plan.get("npu_arena_end", plan.get("arena_end", limit - profile["sram"].get("metadata_reserve", 1024)))
    if base % 8 or not 0 < base <= limit:
        raise ValueError("invalid Phase 8 descriptor metadata boundary")
    cursor = base
    records, commands = [], []
    for stage, spec in stages:
        spans = _operand_spans(spec, graph["tensors"], plan, base)
        _guard(spans, "c", strict=spec.get("wide", False))
        if spec.get("wide"):
            new, slots = _p8_program(spec, spans, cursor)
            a_base = 0
        else:
            new, slots = [{"offset": cursor, "words": _p8_words(spec, spans)}], 0
            a_base = spans["a"][0]
        commands.append({"stage": stage["id"], "device": "NPU", "descriptor": cursor,
                         "a_base": a_base, "program_slots": slots})
        records.extend(new)
        end = records[-1]["offset"] + 4 * len(records[-1]["words"])
        if end > limit:
            raise ValueError(f"Phase 8 metadata capacity exceeded at stage {stage['id']}")
        cursor = (end + 7) & ~7
    payload = bytearray(cursor - base)
    for r in records:
        struct.pack_into(f"<{len(r['words'])}I", payload, r["offset"] - base, *r["words"])
    out.joinpath("npu.bin").write_bytes(payload)
    out.joinpath("npu.hex").write_text("\n".join(f"{w[0]:08x}" for w in struct.iter_unpack("<I", payload)) +
                                      ("\n" if payload else ""), encoding="ascii")
    return {"abi": "p8-ext-program", "binary_base": base, "binary_format": "little-endian descriptor image",
            "records": records, "commands": commands, "constants": []}


def _z15_activation(op, mode):
    if not op:
        return []
    return [pair for i, value in enumerate(_lookup(op, mode))
            for pair in ((0xCC, i | value << 8), (Z15_CTRL, 1 << 8))]


def _z15_tensor(spec, spans, table):
    a, b, c = (spans[key][0] for key in ("a", "b", "c"))
    has_bias = "bias" in spans
    flags = (int(spec["unsigned_a"]) | int(has_bias) << 1 | int(spec["packed"]) << 2 |
             int(spec["activation"] == "Relu") << 3 | spec["shift"] << 4 |
             int(table is not None) << 16 | int(spec["packed"]) << 17)
    shape = 0
    if spec["op"] == "Conv":
        flags |= (1 << 9 | int(spec["kh"] == 3) << 10 | int(spec["sy"] == 2) << 11 |
                  int(spec["t"] == 1) << 12 | 1 << 14)
        shape = spec["h"] | spec["w"] << 9 | spec["ci"] << 18
    return _z15_activation(table, spec["activation_mode"]) + [
        (0x60, spec["m"] | spec["n"] << 16), (0x64, spec["k"] | spec["tile_k"] << 16),
        (0x68, a), (0x6C, b), (0x70, c), (0x74, spans["bias"][0] if has_bias else 0),
        (0x78, flags), (0x7C, shape), (0x80, 0), (0x84, spans.get("partial", (0, 0))[0]),
        (Z15_CTRL, Z15_NPU_START)]


def _z15_ewise(spec, spans):
    return _z15_activation(spec["activation"], spec["activation_mode"]) + [
        (0xA0, spans["a"][0]), (0xA4, spans.get("b", (0, 0))[0]), (0xA8, spans["c"][0]),
        (0xAC, 0), (0xB0, spec["num_bytes"]), (0xB4, 0), (0xB8, 0),
        (0xBC, spec["op"] | 1 << 8), (0xC0, 0), (Z15_CTRL, Z15_VPU_START)]


def _packed_constant(values, name, rows, cols, address):
    return {"tensor": _packed_name(name), "offset": address,
            "data": [values[row * cols + col] & 255 for col in range(cols) for row in range(rows)]}


def _z15_attention(stage, spec, graph, profile, plan, limit, constants):
    tensors = graph["tensors"]
    order = ordered_buffers(stage, graph, profile)
    req = requirements(stage, graph, profile)
    sizes = [req[name]["size"] if name in req else _size(tensors[name]) for name in order]
    spans = [_span(plan, name, size, limit) for name, size in zip(order, sizes)]
    q, k, v, score, probability, c = (lo for lo, _ in spans)
    if (any(spans[i][1] > spans[i + 1][0] for i in range(len(spans) - 1)) or
            q // Z15_HALF != (spans[-1][1] - 1) // Z15_HALF):
        raise ValueError("Z15 attention requires six ordered, disjoint buffers inside one ownership half")
    vv = spec["v"]
    constants[_packed_name(vv)] = _packed_constant(tensors[vv]["data"], vv, spec["length"], spec["n"], v)
    control = 4 | int(spec["causal"]) << 1 | spec["m"] << 8 | spec["shift"] << 20
    return [
        (0x60, spec["length"] | spec["n"] << 16), (0x64, spec["k"] | spec["tile_k"] << 16),
        (0x68, q), (0x6C, k), (0x70, c), (0x88, control), (0x8C, v),
        (0x90, score), (0x94, probability), (0x98, spec["position"]),
        (Z15_CTRL, Z15_ATTN_START)]


def _lower_z15(graph, stages, plan, profile, out):
    limit = profile["sram"]["capacity"]
    if limit != 2 * Z15_HALF:
        raise ValueError("Z15 v1 ABI requires two 1-MiB ownership halves")
    commands, constants = [], {}
    for stage, spec in stages:
        tensor_job = spec["op"] in ("Gemm", "Conv")
        if spec["op"] == "Attention":
            writes = _z15_attention(stage, spec, graph, profile, plan, limit, constants)
            commands.append({"stage": stage["id"], "device": "NPU", "writes": writes,
                             "wait": {"register": 0xC, "done_mask": 16, "error_mask": 128}})
            continue
        spans = _operand_spans(spec, graph["tensors"], plan, limit, packed=tensor_job)
        if tensor_job:
            name, k, n = spec["b"], spec["k"], spec["n"]
            values = graph["tensors"][name]["data"]
            packed = _packed_name(name)
            constants[packed] = _packed_constant(values, name, k, n, spans["b"][0])
            if spec["packed"] and k > spec["tile_k"]:
                spans["partial"] = _span(plan, _partial_name(stage), 4 * spec["m"] * n, limit)
                for key, span in spans.items():
                    if key != "partial" and _overlap(spans["partial"], span):
                        raise ValueError(f"Z15 partial accumulator buffer aliases {key}")
            activation = spec["activation"]
            writes = _z15_tensor(spec, spans, activation if activation in ("SiLU", "GELU") else None)
        else:
            writes = _z15_ewise(spec, spans)
        _guard(spans, "c", half=True)
        commands.append({"stage": stage["id"], "device": "NPU" if tensor_job else "VPU",
                         "writes": writes, "wait": {"register": 0xC, "done_mask": 16, "error_mask": 128}})
    payload = [w for cmd in commands for pair in cmd["writes"] for w in pair]
    out.joinpath("npu.bin").write_bytes(struct.pack(f"<{len(payload)}I", *payload))
    out.joinpath("npu.hex").write_text("".join(f"{w:08x}\n" for w in payload), encoding="ascii")
    return {"abi": "z15-mmio-v1", "binary_format": "little-endian (register offset, data) pairs",
            "commands": commands, "constants": list(constants.values())}


def lower(graph, plan, profile, output):
    """Write native NPU/VPU files and return their commands and startup constants."""
    target = _target(profile)
    stages = [(s, _spec(s, graph, profile)) for s in plan["stages"] if s["device"] in ("NPU", "VPU")]
    output = Path(output)
    if target == "p8":
        result = _lower_p8(graph, stages, plan, profile, output)
    elif target == "z15":
        result = _lower_z15(graph, stages, plan, profile, output)
    else:
        result = {"abi": None, "commands": [], "constants": []}
        output.joinpath("npu.bin").write_bytes(b"")
        output.joinpath("npu.hex").write_text("", encoding="ascii")
    output.joinpath("npu.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
