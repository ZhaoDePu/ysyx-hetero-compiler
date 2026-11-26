"""Compile the C6 calibrated graph into a static Balanced YHB1 program."""

from collections import Counter
from dataclasses import dataclass
from math import prod

import numpy as np

from .c6_attention import lower_matmul, lower_q23_softmax, lower_vpu_softmax, lower_vpu_softmax_words
from .c6_depthwise import depthwise_tiles, depthwise_weights, depthwise_widths
from .c6_layout import plan_layout
from .c6_memory import Lifetime, reserve_ddr
from .c6_reduce_conv import combine_tiles, raw_conv_tiles, reduction_widths
from .c6_schedule import emit_program
from .c6_tensor import Conv, conv_tiles, pack_weights
from .c6_vector import vector_tiles
from .c7_reduce_conv import (
    native_combine_tiles, native_combine_width, scratch_combine_tiles, scratch_combine_width,
)
from .c7_softmax import log_softmax_tables, lower_log_softmax


WIDTH = {"int8": 1, "uint8": 1, "int16": 2, "uint16": 2, "int32": 4, "int64": 8}
_MATRIX_OPS = {"MatMul", "Q23Softmax", "VpuSoftmax", "C7LogSoftmax", "Transpose"}
_LAYOUT_CHAIN = {"EWAdd", "EWMul", "RequantLUT", "Lookup", "Split", "Concat", "Transpose"}
_VECTOR_OPS = {"EWAdd", "EWMul", "RequantLUT", "Lookup", "Split", "Concat", "Resize", "MaxPool"}


def matrix_names(graph):
    """Use plain contiguous matrix storage for attention views and their branches."""
    names = {name for node in graph["nodes"] if node["op"] in _MATRIX_OPS
             and node["device"] != "PS" for name in node["inputs"] + node["outputs"] if name}
    while True:
        before = len(names)
        for node in graph["nodes"]:
            if node["device"] != "PS" and node["op"] in _LAYOUT_CHAIN:
                tensors = set(node["inputs"] + node["outputs"])
                if names & tensors:
                    names.update(name for name in tensors if name and
                                 len(graph["tensors"][name]["shape"]) >= 2)
        if len(names) == before:
            return names


def layouts(graph):
    matrices = matrix_names(graph)
    return {name: {**desc,
                   "physical_shape": ([*desc["physical_shape"][:-1], 4]
                                      if name in graph["inputs"] and desc["physical_shape"][-1:] == [3]
                                      else desc["shape"] if name in matrices else desc["physical_shape"]),
                   "layout": "row-major" if name in matrices else desc["layout"]}
            for name, desc in graph["tensors"].items()}


def physical_array(name, array, tensors):
    """Encode a logical ONNX reference tensor in the scheduled DDR layout."""
    desc = tensors[name]
    data = np.asarray(array, dtype=np.dtype(desc["dtype"]).newbyteorder("<"))
    if desc["layout"] == "NHWC":
        data = data.transpose(0, 2, 3, 1)
        if data.shape[-1] == 3 and desc["physical_shape"][-1] == 4:
            data = np.pad(data, ((0, 0), (0, 0), (0, 0), (0, 1)))
    if list(data.shape) != desc["physical_shape"]:
        raise ValueError(f"{name}: physical shape does not match compiled frame")
    return data.tobytes()


@dataclass
class _Preparation:
    index: int
    tensors: dict
    constants: dict
    blobs: dict
    auxiliary: dict

    def constant(self, name, data):
        if name in self.auxiliary:
            raise ValueError(f"duplicate C6 data region {name}")
        self.blobs[name] = data
        self.auxiliary[name] = (len(data), 0, self.index)
        return name

    def temporary(self, name, nbytes):
        if name in self.auxiliary:
            raise ValueError(f"duplicate C6 workspace {name}")
        self.auxiliary[name] = (nbytes, self.index, self.index)
        return name


def _conv(node, ctx, *, fast_reduce=False, reduce_bank_skew=False, reduce_in_scratch=False):
    name, attrs = node["id"], node["attrs"]
    src, dst = node["inputs"][0], node["outputs"][0]
    a, c = ctx.tensors[src], ctx.tensors[dst]
    if (a["layout"] != "NHWC" or c["layout"] != "NHWC" or a["dtype"] != "int8" or c["dtype"] != "int8"
            or a["shape"][0] != 1 or c["shape"][0] != 1):
        raise ValueError(f"{name}: native Conv expects one batch of signed NHWC images")
    _, ci, h, w = a["shape"]
    _, co, oh, ow = c["shape"]
    kh, kw = attrs["kernel"]
    sy, sx = attrs["strides"]
    pt, pl, pb, pr = attrs["pads"]
    if kh != kw or sy != sx or not pt == pl == pb == pr or attrs["dilations"] != [1, 1]:
        raise ValueError(f"{name}: unsupported native Conv attributes")
    spec = Conv(name, h, w, ci, co, kh, sy, pt)
    if spec.output_hw != (oh, ow) or a["physical_shape"][-1] != spec.padded_channels:
        raise ValueError(f"{name}: Conv geometry requires a correctly channel-padded NHWC input")
    weights = ctx.constants[attrs["weight"]]
    bias = ctx.constants[attrs["bias"]].astype("<i4")
    mul, shift, lut = attrs["mul"], attrs["shift"], attrs.get("lut")
    groups = attrs["groups"]

    if groups == ci == co and groups > 1:
        widths = depthwise_widths(spec)
        blocks = depthwise_weights(spec, weights)
        b_names = [ctx.constant(f"{name}/@kernel{idx}", block) for idx, block in enumerate(blocks)]
        bias_names = []
        first = 0
        for idx, count in enumerate(widths):
            bias_names.append(ctx.constant(f"{name}/@bias{idx}", bias[first:first+count].tobytes()))
            first += count

        def schedule(slots):
            return (depthwise_tiles(spec, input_ddr=slots[src].offset, output_ddr=slots[dst].offset,
                                    weights_ddr=[slots[b].offset for b in b_names],
                                    bias_ddr=[slots[b].offset for b in bias_names],
                                    mul=mul, shift=shift, lut=lut),)

        return schedule
    if groups != 1:
        raise ValueError(f"{name}: unsupported non-depthwise Conv group count {groups}")

    bias_name = ctx.constant(f"{name}/@bias", bias.tobytes())
    if spec.depth <= 511:
        b_name = ctx.constant(f"{name}/@kernel", pack_weights(spec, weights))

        def schedule(slots):
            return (conv_tiles(spec, input_ddr=slots[src].offset, output_ddr=slots[dst].offset,
                               weights_ddr=slots[b_name].offset, bias_ddr=slots[bias_name].offset,
                               mul=mul, shift=shift, lut=lut),)

        return schedule

    direct_width = scratch_combine_width(spec) if reduce_in_scratch else 0
    parts = reduction_widths(spec)
    b_names, partials = [], []
    first = 0
    for idx, count in enumerate(parts):
        child = Conv(spec.name, h, w, count, co, kh, sy, pt)
        src_part = weights[:, :, first:first + count, :]
        if src_part.shape[2] != count:
            src_part = np.pad(src_part, ((0, 0), (0, 0), (0, count - src_part.shape[2]), (0, 0)))
        b_names.append(ctx.constant(f"{name}/@kernel{idx}", pack_weights(child, src_part)))
        if not direct_width:
            partials.append(ctx.temporary(f"{name}/@partial{idx}", 4 * oh * ow * co))
        first += count
    width = direct_width or (native_combine_width(spec) if fast_reduce else 0)
    if width:
        zero = ctx.constant(f"{name}/@epilogue_zero", bytes(4 * width + 4))
    else:
        words = (np.asarray(mul, dtype=np.uint32) | np.asarray(shift, dtype=np.uint32) << 16)
        params = ctx.constant(f"{name}/@scales", words.astype("<u4").tobytes())
        lut_name = ctx.constant(f"{name}/@lut", np.asarray(lut, dtype=np.int8).tobytes()) if lut is not None else None

    def schedule(slots):
        weights_ddr = [slots[b].offset for b in b_names]
        if direct_width:
            return (scratch_combine_tiles(
                spec, input_ddr=slots[src].offset, output_ddr=slots[dst].offset,
                weights_ddr=weights_ddr, bias_ddr=slots[bias_name].offset,
                zero_ddr=slots[zero].offset, mul=mul, shift=shift, lut=lut,
                warp_skew=reduce_bank_skew),)
        locations = [slots[part].offset for part in partials]
        raw = raw_conv_tiles(spec, input_ddr=slots[src].offset, weights_ddr=weights_ddr,
                             bias_ddr=slots[bias_name].offset, partials_ddr=locations)
        if width:
            combined = native_combine_tiles(spec, output_ddr=slots[dst].offset, partials_ddr=locations,
                                            zero_ddr=slots[zero].offset, mul=mul, shift=shift, lut=lut,
                                            warp_skew=reduce_bank_skew)
        else:
            combined = combine_tiles(spec, output_ddr=slots[dst].offset, partials_ddr=locations,
                                     params_ddr=slots[params].offset,
                                     lut_ddr=slots[lut_name].offset if lut_name else None)
        return raw, combined

    return schedule


def _vector(node, ctx):
    op, attrs = node["op"], node["attrs"]
    ins = [name for name in node["inputs"] if name and name not in ctx.constants]
    outs = node["outputs"]
    tables, mul, shift = attrs.get("lut"), attrs.get("mul", 1), attrs.get("shift", 0)
    input_shapes = [tuple(ctx.tensors[name]["physical_shape"]) for name in ins]
    output_shapes = [tuple(ctx.tensors[name]["physical_shape"]) for name in outs]
    axis = attrs.get("axis")
    if axis is not None:
        axis %= len(input_shapes[0])
        if ctx.tensors[ins[0]]["layout"] == "NHWC":
            axis = (0, 3, 1, 2)[axis]
    if op == "EWMul" and len(ins) == 1:
        from .c6_quant import ew_mul
        scalar = ctx.constants[next(name for name in node["inputs"] if name in ctx.constants)]
        if scalar.ndim:
            raise ValueError(f"{node['id']}: only scalar constant EWMul can be lowered to a VPU LUT")
        values = np.arange(256, dtype=np.uint8).view(np.int8)
        tables = list(ew_mul(values, scalar, mul, shift))
        op = "RequantLUT"
    if op == "MaxPool" and (attrs.get("kernel_shape") != [5, 5] or
            attrs.get("pads") != [2, 2, 2, 2] or attrs.get("strides") != [1, 1]):
        raise ValueError(f"{node['id']}: VPU max pool requires 5x5 stride 1 padding 2")

    def schedule(slots):
        return (vector_tiles(node["id"], op, input_shapes, output_shapes,
                             tuple(slots[name].offset for name in ins),
                             tuple(slots[name].offset for name in outs),
                             axis=axis, table=tables, mul=mul, shift=shift),)

    return schedule


def _layout(node, ctx):
    name = node["id"]
    src, dst = node["inputs"][0], node["outputs"][0]
    a, c = ctx.tensors[src], ctx.tensors[dst]
    if a["dtype"] not in {"uint8", "int8"} or a["dtype"] != c["dtype"]:
        raise ValueError(f"{name}: layout map expects byte activations")
    layout = plan_layout(name, node["op"], a["shape"], c["shape"],
                         input_layout=a["layout"], output_layout=c["layout"],
                         perm=node["attrs"].get("perm"))
    idx = (ctx.constant(f"{name}/@reindex", layout.constant_data) if layout.constant_data else None)

    def schedule(slots):
        return (layout.tiles(slots[src].offset, slots[dst].offset, slots[idx].offset if idx else None),)

    return schedule


def _matrix(node, ctx):
    name, attrs = node["id"], node["attrs"]
    src, dst = node["inputs"][0], node["outputs"][0]
    a, c = ctx.tensors[src], ctx.tensors[dst]
    if a["layout"] != "row-major" or c["layout"] != "row-major":
        raise ValueError(f"{name}: attention needs contiguous matrix rows")
    if node["op"] == "MatMul":
        b_name = node["inputs"][1]
        b = ctx.tensors[b_name]
        raw = attrs.get("raw_output", False)
        result_dtype = "int32" if raw else "int8"
        if (b["layout"] != "row-major" or a["dtype"] != "int8" or c["dtype"] != result_dtype
                or b["dtype"] not in {"uint8", "int8"}):
            raise ValueError(f"{name}: invalid Z15 Q/K/V matrix dtype or layout")

        def schedule(slots):
            return (lower_matmul(name, slots[src].offset, slots[b_name].offset, slots[dst].offset,
                                 a["shape"], b["shape"], attrs["mul"], attrs["shift"],
                                 unsigned_rhs=b["dtype"] == "uint8", raw_output=raw,
                                 rhs_pretransposed=attrs.get("rhs_pretransposed", False)),)

        return schedule
    if attrs["axis"] % len(a["shape"]) != len(a["shape"]) - 1:
        raise ValueError(f"{name}: C6 softmax requires the row-contiguous last axis")
    if node["op"] == "VpuSoftmax":
        lower = lower_vpu_softmax_words if a["dtype"] == "int32" else lower_vpu_softmax

        def native(slots):
            return (lower(name, slots[src].offset, slots[dst].offset, a["shape"], attrs["shift"]),)
        return native
    if node["op"] == "C7LogSoftmax":
        table = log_softmax_tables(attrs["lut"], attrs["scale"], attrs["steps"], a["shape"][-1])
        lut_name = ctx.constant(f"{name}/@log-softmax", table.payload)

        def packed(slots):
            return (lower_log_softmax(name, slots[src].offset, slots[dst].offset,
                                      slots[lut_name].offset, a["shape"], table),)

        return packed
    lut_name = ctx.constant(f"{name}/@exp", np.asarray(attrs["lut"], dtype="<u4").tobytes())

    def schedule(slots):
        return (lower_q23_softmax(name, slots[src].offset, slots[dst].offset, slots[lut_name].offset, a["shape"]),)

    return schedule


def _head_contract(graph, tensors, constants, ps, outputs):
    """Check that the six frozen raw feature maps feed the static YOLO26n host tail."""
    if len(graph["inputs"]) != 1 or len(graph["outputs"]) != 1 or len(outputs) != 6:
        raise ValueError("C6 board export expects one RGB input and six head feature maps")
    _, _, height, width = tensors[graph["inputs"][0]]["shape"]
    symbols, anchors, strides, shapes = {}, [], [], {}
    for stride in (8, 16, 32):
        p, q = height // stride, width // stride
        grid_y, grid_x = np.indices((p, q), dtype=np.float32)
        anchors.append(np.stack((grid_x.ravel() + .5, grid_y.ravel() + .5)))
        strides.append(np.full(p * q, stride, dtype=np.float32))
        for role, channels in (("boxes", 4), ("scores", 80)):
            symbol = f"{role}_stride{stride}"
            match = [source for source in outputs if tensors[source]["shape"] == [1, channels, p, q]
                     and tensors[source]["dtype"] == "int8" and tensors[source]["layout"] == "NHWC"]
            if len(match) != 1:
                raise ValueError(f"{symbol}: expected one matching raw NHWC feature map")
            symbols[symbol] = match[0]
            shapes[symbol] = [1, p, q, channels]
    anchor = np.concatenate(anchors, axis=1)[None]
    stride_vector = np.concatenate(strides)[None]
    for op, expected in (("Sub", anchor), ("Add", anchor), ("Mul", stride_vector)):
        options = [constants[name] for node in ps if node["op"] == op for name in node["inputs"]
                   if name in constants and constants[name].shape == expected.shape]
        if len(options) != 1 or not np.array_equal(options[0], expected):
            raise ValueError(f"C6 board head has an unexpected {op} anchor or stride map")
    topks = [int(np.asarray(constants[node["inputs"][1]]).item()) for node in ps if node["op"] == "TopK"]
    if len(topks) != 2 or topks[0] != topks[1] or not 0 < topks[0] <= anchor.shape[-1]:
        raise ValueError("C6 board head Top-K contract is incompatible")
    scales = np.array([tensors[symbols[f"{role}_stride{stride}"]]["scale"]
                       for role in ("boxes", "scores") for stride in (8, 16, 32)], dtype="<f4")
    dx = np.arange(256, dtype=np.uint8).view(np.int8).astype(np.float32)
    logits = scales[3:, None] * dx
    lookup = 1. / (1. + np.exp(-np.clip(logits, -80, 80)))
    initial = {"c6_head_scales": scales.tobytes(), "c6_score_lut": lookup.astype("<f4").tobytes()}
    descs = {"c6_head_scales": {"shape": [6], "dtype": "int32"},
             "c6_score_lut": {"shape": [3, 256], "dtype": "int32"}}
    return symbols, initial, descs, {"top_k": topks[0], "xyxy": True, "grid_offset": .5,
                                      "classes": 80, "symbols": symbols, "head_shapes": shapes,
                                      "scales": list(map(float, scales)), "host_tail": ps}

def fuse_conv_edge_luts(graph):
    """Compose a private byte requant with the preceding native Conv epilogue."""
    uses = Counter(name for node in graph["nodes"] for name in node["inputs"])
    public = set(graph["outputs"])
    producer = {}
    nodes = []
    for node in graph["nodes"]:
        if node["op"] == "RequantLUT":
            source, = node["inputs"]
            index = producer.get(source)
            if (index is not None and uses[source] == 1 and source not in public
                    and nodes[index]["op"] == "Conv" and nodes[index]["outputs"] == [source]):
                conv = nodes[index]
                outer = node["attrs"]["lut"]
                inner = conv["attrs"].get("lut", range(256))
                attrs = {**conv["attrs"], "lut": [outer[code & 255] for code in inner],
                         "fused_edge_ids": [*conv["attrs"].get("fused_edge_ids", ()), node["id"]]}
                nodes[index] = {**conv, "attrs": attrs, "outputs": node["outputs"]}
                for output in node["outputs"]:
                    producer[output] = index
                continue
        for output in node["outputs"]:
            producer[output] = len(nodes)
        nodes.append(node)
    return {**graph, "nodes": nodes}


def prepare_graph(graph, constants, *, selected=None, lowerers=None, fast_reduce=False,
                  reduce_bank_skew=False, reduce_in_scratch=False):
    """Resolve graph boundaries and return physical layer jobs with DDR liveness."""
    tensors = layouts(graph)
    active = [node for node in graph["nodes"] if node["device"] != "PS" and
              (selected is None or node["id"] in selected)]
    if not active:
        raise ValueError("C6 graph contains no selected hardware nodes")
    producers = {name: i for i, node in enumerate(active, start=1) for name in node["outputs"]}
    last_use = dict(producers)
    inputs, launchers, outputs = [], [], []
    initial, auxiliary = {}, {}
    ps = [node for node in graph["nodes"] if node["device"] == "PS"]
    boundary = (list(dict.fromkeys(name for node in ps for name in node["inputs"] if name)) +
                graph["outputs"]) if selected is None else (
                    [name for node in active for name in node["outputs"]])
    for name in dict.fromkeys(boundary):
        if name in producers:
            outputs.append(name)
            last_use[name] = len(active) + 1
    if not outputs:
        raise ValueError("C6 graph has no outputs at the hardware boundary")

    def conv(node, context):
        return _conv(node, context, fast_reduce=fast_reduce or reduce_bank_skew or reduce_in_scratch,
                     reduce_bank_skew=reduce_bank_skew, reduce_in_scratch=reduce_in_scratch)

    lower = {**dict.fromkeys(_VECTOR_OPS, _vector), "Conv": conv, "Reshape": _layout,
             "Transpose": _layout, "MatMul": _matrix, "Q23Softmax": _matrix, "VpuSoftmax": _matrix, "C7LogSoftmax": _matrix,
             **(lowerers or {})}
    for index, node in enumerate(active, start=1):
        for name in node["inputs"]:
            if not name or name in constants:
                continue
            if name not in producers:
                producers[name] = 0
                inputs.append(name)
            if producers[name] >= index:
                raise ValueError(f"{node['id']}: non-topological dependency {name}")
            last_use[name] = max(index, last_use.get(name, 0))
        if node["op"] not in lower:
            raise ValueError(f"{node['id']}: unsupported C6 hardware op {node['op']}")
        context = _Preparation(index, tensors, constants, initial, auxiliary)
        launchers.append(lower[node["op"]](node, context))

    lives = [Lifetime(name, prod(tensors[name]["physical_shape"]) * WIDTH[tensors[name]["dtype"]],
                      first, last_use[name]) for name, first in producers.items()]
    c6 = {}
    aliases, const_outputs = {}, {}
    if selected is None:
        aliases, metadata, const_outputs, c6 = _head_contract(graph, tensors, constants, ps, outputs)
        initial.update(metadata)
    persistent = [Lifetime(name, len(data), 0, 0) for name, data in initial.items()]
    weights, weights_end = reserve_ddr(persistent)
    public = set(inputs) | set(outputs)
    frames, frames_end = reserve_ddr(
        (life for life in lives if life.name in public), base=weights_end)
    working = [life for life in lives if life.name not in public]
    working += [Lifetime(name, *lifetime) for name, lifetime in auxiliary.items() if name not in initial]
    workspace, peak = reserve_ddr(working, base=frames_end)
    slots = weights | frames | workspace
    layer_groups = [make(slots) for make in launchers]
    layers = [tiles for layer in layer_groups for tiles in layer]
    c6["hardware_ops"] = [{
        "id": node["id"], "op": node["op"], "inputs": node["inputs"], "outputs": node["outputs"],
        "passes": [len(tiles) for tiles in scheduled],
        "stages": [stage.id for tiles in scheduled for tile in tiles for stage in tile.stages],
        "engines": sorted({stage.device for tiles in scheduled for tile in tiles for stage in tile.stages}),
        **({"fused_edge_ids": node["attrs"]["fused_edge_ids"]} if "fused_edge_ids" in node["attrs"] else {}),
    } for node, scheduled in zip(active, layer_groups, strict=True)]
    c6["source_onnx_sha256"] = graph.get("source_onnx_sha256")
    c6["static_placement_estimate"] = graph.get("placement_estimate")
    c6["ddr_regions"] = {"weights": [0, weights_end], "frames": [weights_end, frames_end],
                          "activation": [frames_end, peak]}
    for alias, source in aliases.items():
        slots[alias] = slots[source]
    return tensors, inputs, outputs, slots, initial, layers, const_outputs, c6


def compile_graph(directory, graph, constants, *, lowerers=None, selected=None, seeded=None,
                  fast_reduce=False, overlap=False, reduce_bank_skew=False, reduce_in_scratch=False,
                  stream_overlap=False):
    """Write a guarded legacy-compatible YHB1 bundle with C6 DDR/scratch jobs."""
    tensors, inputs, outputs, slots, initial, layers, const_outputs, c6 = prepare_graph(
        graph, constants, selected=selected, lowerers=lowerers, fast_reduce=fast_reduce,
        reduce_bank_skew=reduce_bank_skew, reduce_in_scratch=reduce_in_scratch)
    if seeded is not None:
        if set(seeded) != set(inputs):
            raise ValueError("replay seed tensors must exactly match compiled input symbols")
        initial.update({name: physical_array(name, value, tensors) for name, value in seeded.items()})
    yhb_tensors = {name: {"shape": tensors[name]["physical_shape"], "dtype": tensors[name]["dtype"],
                         "scale": tensors[name].get("scale")} for name in inputs + outputs}
    aliases = c6.get("symbols", {})
    if aliases:
        outputs = list(aliases) + list(const_outputs)
        yhb_tensors.update({alias: yhb_tensors[source] for alias, source in aliases.items()})
        yhb_tensors.update(const_outputs)
    manifest = emit_program(directory, layers, yhb_tensors, inputs, outputs, slots, initial,
                            external_inputs=seeded is None, c6=c6, overlap=overlap, stream_overlap=stream_overlap)
    return manifest, tensors, initial
