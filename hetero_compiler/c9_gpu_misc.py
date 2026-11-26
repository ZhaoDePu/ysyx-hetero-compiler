"""Exact pure-GPU classifier operators for C9 A0 RTL replay."""

from math import gcd, prod

import numpy as np

from .c6_gpu import DATA_WINDOW_BYTES, PAGE_BYTES, build_gpu_stage, emit_requant
from .c6_memory import ALIGNMENT, Scratch, align_dma
from .c6_tensor import Tile, Transfer
from .gpu import _clip


# Keep one GPU submission bounded even before checking the data window.
MAX_OUTPUT_BYTES = 8192
LANES = 64


def _finish(name, scratch, loads, stage_list, stores):
    if scratch.used > DATA_WINDOW_BYTES:
        raise ValueError(f"{name}: GPU tile exceeds the verified C8 data window")
    return Tile(scratch.base // (1 << 20), tuple(loads), tuple(stage_list), tuple(stores), scratch.used)


def _stage(name, scratch, count, kernel):
    lanes = min(count, LANES)
    return build_gpu_stage(name, scratch.base // PAGE_BYTES, lanes, lambda p: kernel(p, lanes))


def _load(scratch, loads, address, size):
    if address % ALIGNMENT:
        raise ValueError("unaligned GPU load address")
    slot = scratch.reserve(size)
    loads.append(Transfer("load", address, slot.offset, slot.capacity))
    return slot.offset - scratch.base


def _store(scratch, stores, address, size):
    if address % ALIGNMENT:
        raise ValueError("unaligned GPU store address")
    slot = scratch.reserve(size)
    stores.append(Transfer("store", address, slot.offset, slot.capacity))
    return slot.offset - scratch.base


def _shape(ctx, name):
    value = ctx.tensors[name]
    shape = tuple(value["physical_shape"])
    if (value["dtype"] != "int8" or value["layout"] != "NHWC" or len(shape) != 4
            or shape[0] != 1 or any(type(extent) is not int or extent < 1 for extent in shape)):
        raise ValueError(f"{name}: A0 misc expects positive batch-one INT8 NHWC dimensions")
    return shape


def _lookup_table(values):
    if len(values) != 256 or any(not isinstance(v, int) or not -128 <= v <= 127 for v in values):
        raise ValueError("expected signed INT8 LUT with 256 entries")
    return np.asarray(values, dtype=np.int8).tobytes()


def _vector_program(p, sources, destination, count, table, lanes):
    for index, addr in enumerate(sources):
        p.li(11 + index, addr)
    p.li(13, destination)
    p.li(14, count)
    if table is not None:
        p.li(15, table)
    p.label("element")
    p.arithmetic(6, 5, 11)
    p.load(7, 6, "uint8" if table is not None else "int8")
    if table is not None:
        p.arithmetic(6, 7, 15)
        p.load(7, 6, "int8")
    else:
        p.arithmetic(6, 5, 12)
        p.load(8, 6, "int8")
        p.arithmetic(7, 7, 8)
        _clip(p, "int8", "saturate", False)
    p.arithmetic(6, 5, 13)
    p.store(7, 6, "int8")
    p.addi(5, 5, lanes)
    p.branch(6, 5, 14, "element")


def lower_gpu_vector(node, ctx):
    """Emit table lookup or saturating elementwise add on G3."""
    name, op = node["id"], node["op"]
    inputs = node["inputs"]
    output, = node["outputs"]
    shape = _shape(ctx, output)
    uses_lut = op in ("RequantLUT", "Lookup")
    if (not uses_lut and op != "EWAdd") or len(inputs) != (1 if uses_lut else 2):
        raise ValueError(f"{name}: invalid A0 vector operation")
    if any(_shape(ctx, item) != shape for item in inputs):
        raise ValueError(f"{name}: incompatible A0 elementwise shapes")
    lut = ctx.constant(name + "/@a0-lut", _lookup_table(node["attrs"]["lut"])) if uses_lut else None

    def schedule(slots):
        elements = prod(shape)
        tiles = []
        for first in range(0, elements, MAX_OUTPUT_BYTES):
            count = min(MAX_OUTPUT_BYTES, elements - first)
            scratch = Scratch(len(tiles) % 2)
            loads, stores = [], []
            table = _load(scratch, loads, slots[lut].offset, 256) if lut else None
            sources = [_load(scratch, loads, slots[item].offset + first, count) for item in inputs]
            destination = _store(scratch, stores, slots[output].offset + first, count)
            label = f"{name}/a0-vector-{first}"
            stage = _stage(label, scratch, count, lambda p, lanes: _vector_program(
                p, sources, destination, count, table, lanes))
            tiles.append(_finish(label, scratch, loads, [stage], stores))
        return (tuple(tiles),)

    return schedule


def _partition_program(p, source, destination, input_words, output_words, group_words, count, lanes):
    p.li(11, source)
    p.li(12, destination)
    p.li(13, group_words)
    p.li(14, input_words * 4)
    p.li(15, output_words * 4)
    p.label("word")
    p.arithmetic(6, 5, 13, f3=5, f7=1)  # pixel index
    p.arithmetic(7, 5, 13, f3=7, f7=1)  # channel word in the current group
    p.addi(7, 7, 2, f3=1)
    p.arithmetic(8, 6, 14, f7=1)
    p.arithmetic(8, 8, 7)
    p.arithmetic(8, 8, 11)
    p.load(8, 8, "int32")
    p.arithmetic(9, 6, 15, f7=1)
    p.arithmetic(9, 9, 7)
    p.arithmetic(9, 9, 12)
    p.store(8, 9, "int32")
    p.addi(5, 5, lanes)
    p.li(10, count)
    p.branch(6, 5, 10, "word")


def lower_gpu_partition(node, ctx):
    """Move aligned channel words for standard NHWC Concat and Split."""
    name, op = node["id"], node["op"]
    join = op == "Concat"
    if op not in ("Concat", "Split") or node["attrs"].get("axis", 0) not in (1, -3):
        raise ValueError(f"{name}: expected an A0 channel-axis partition")
    group_symbols, (whole,) = (node["inputs"], node["outputs"]) if join else (node["outputs"], node["inputs"])
    shape = _shape(ctx, whole)
    parts = [_shape(ctx, item) for item in group_symbols]
    widths = [part[-1] for part in parts]
    total = sum(widths)
    if len(parts) < 2 or any(part[:-1] != shape[:-1] for part in parts) or total != shape[-1]:
        raise ValueError(f"{name}: incompatible A0 partition shapes")
    if any(width % 4 for width in widths):
        raise ValueError(f"{name}: group widths must be scratch-word aligned")
    # A nonfinal tile ends on a DMA boundary for every group and the joined tensor.
    unit = ALIGNMENT // gcd(ALIGNMENT, *widths)
    limit = MAX_OUTPUT_BYTES // total // unit * unit
    if not limit:
        raise ValueError(f"{name}: aligned A0 partition does not fit")

    def schedule(slots):
        pixels = prod(shape[1:3])
        tiles = []
        for first in range(0, pixels, limit):
            count = min(limit, pixels - first)
            scratch = Scratch(len(tiles) % 2)
            loads, stages, stores = [], [], []
            joined = (_store if join else _load)(
                scratch, stores if join else loads, slots[whole].offset + first * total, count * total)
            prefix = 0
            for index, (item, width) in enumerate(zip(group_symbols, widths, strict=True)):
                group = (_load if join else _store)(
                    scratch, loads if join else stores, slots[item].offset + first * width, count * width)
                source, destination = (group, joined + prefix) if join else (joined + prefix, group)
                input_width, output_width = (width, total) if join else (total, width)
                group_words = width // 4
                words = count * group_words
                label = f"{name}/a0-partition-{first}-{index}"
                stages.append(_stage(label, scratch, words, lambda p, lanes: _partition_program(
                    p, source, destination, input_width // 4, output_width // 4, group_words, words, lanes)))
                prefix += width
            tiles.append(_finish(name, scratch, loads, stages, stores))
        return (tuple(tiles),)

    return schedule


def _depthwise_program(p, weights, bias, params, index, output, channels, count, table, lanes):
    p.li(15, channels)
    p.label("values")
    p.arithmetic(6, 5, 15, f3=5, f7=1)
    p.arithmetic(8, 5, 15, f3=7, f7=1)
    p.li(10, 36)
    p.arithmetic(6, 6, 10, f7=1)
    p.li(10, index)
    p.arithmetic(6, 6, 10)

    p.li(11, bias)
    p.indexed(9, 8, 11, "int32")
    p.load(7, 9, "int32")
    p.li(11, weights)
    p.arithmetic(11, 11, 8)
    for tap in range(9):
        p.load(9, 6, "int32", 4 * tap)
        p.arithmetic(9, 9, 8)
        p.load(9, 9, "int8")
        p.load(10, 11, "int8")
        p.arithmetic(10, 10, 9, f7=1)
        p.arithmetic(7, 7, 10)
        if tap < 8:
            p.addi(11, 11, channels)
    p.li(10, params)
    p.indexed(8, 8, 10, "int32")
    p.load(8, 8, "int32")
    emit_requant(p)
    if table is not None:
        p.addi(7, 7, 255, f3=7)
        p.li(8, table)
        p.arithmetic(7, 7, 8)
        p.load(7, 7, "int8")
    p.li(8, output)
    p.arithmetic(8, 8, 5)
    p.store(7, 8, "int8")
    p.addi(5, 5, lanes)
    p.li(8, count)
    p.branch(6, 5, 8, "values")


def lower_gpu_depthwise(node, ctx):
    """Map exact signed 3×3 depthwise with halo DMA and the C7 wide requant."""
    name, attrs = node["id"], node["attrs"]
    source, = node["inputs"]
    output, = node["outputs"]
    _, h, w, channels = _shape(ctx, source)
    _, oh, ow, output_channels = _shape(ctx, output)
    stride = attrs["strides"][0]
    pad = attrs["pads"][0]
    groups = attrs["groups"]
    if (attrs["kernel"] != [3, 3] or attrs["dilations"] != [1, 1] or stride not in (1, 2)
            or pad not in (0, 1) or attrs["strides"] != [stride] * 2 or attrs["pads"] != [pad] * 4
            or not 1 <= channels < 512 or groups != channels or output_channels != channels
            or (oh, ow) != ((h + 2 * pad - 3) // stride + 1, (w + 2 * pad - 3) // stride + 1)):
        raise ValueError(f"{name}: invalid A0 depthwise geometry")
    weight = ctx.constants[attrs["weight"]]
    bias = ctx.constants[attrs["bias"]].astype("<i4")
    if weight.shape != (3, 3, 1, channels) or weight.dtype != np.int8 or bias.shape != (channels,):
        raise ValueError(f"{name}: invalid signed A0 depthwise constants")
    mul, shift = (np.asarray(attrs[key], dtype=np.uint32) for key in ("mul", "shift"))
    if mul.shape != (channels,) or shift.shape != (channels,) or np.any(mul > 65535) or np.any(shift > 31):
        raise ValueError(f"{name}: invalid A0 depthwise requantization")
    params = (mul | shift << 16).astype("<u4")
    weight_id = ctx.constant(name + "/@a0-kernel", weight.tobytes())
    bias_id = ctx.constant(name + "/@a0-bias", bias.tobytes())
    param_id = ctx.constant(name + "/@a0-params", params.tobytes())
    pad_id = ctx.constant(name + "/@a0-pad", bytes(align_dma(channels)))
    table_id = (ctx.constant(name + "/@a0-lut", _lookup_table(attrs["lut"]))
                if "lut" in attrs else None)

    source_pitch = w * channels
    output_pitch = ow * channels
    row_unit = ALIGNMENT // gcd(ALIGNMENT, output_pitch)
    row_cap = max(row_unit, MAX_OUTPUT_BYTES // output_pitch // row_unit * row_unit)
    tables = []
    for first in range(0, oh, row_cap):
        rows = min(row_cap, oh - first)
        begin = max(0, first * stride - pad)
        end = min(h, (first + rows - 1) * stride - pad + 3)
        phase = begin * source_pitch % ALIGNMENT
        start = begin * source_pitch - phase
        span = phase + (end - begin) * source_pitch
        pad_addr = align_dma(span)
        index_rows = np.empty((rows * ow, 9), dtype="<u4")
        for row in range(rows):
            for col in range(ow):
                for dy in range(3):
                    for dx in range(3):
                        y, x = (first + row) * stride - pad + dy, col * stride - pad + dx
                        index_rows[row * ow + col, dy * 3 + dx] = (
                            phase + ((y - begin) * w + x) * channels if begin <= y < end and 0 <= x < w
                            else pad_addr)
        index_id = ctx.constant(f"{name}/@a0-index-{first}", index_rows.tobytes())
        tables.append((first, rows, start, span, index_id, index_rows.nbytes))

    def schedule(slots):
        tiles = []
        for first, rows, start, span, index_id, index_bytes in tables:
            scratch = Scratch(len(tiles) % 2)
            loads, stores = [], []
            # The local address map is relative to the first source region.
            _load(scratch, loads, slots[source].offset + start, span)
            _load(scratch, loads, slots[pad_id].offset, channels)
            indices = _load(scratch, loads, slots[index_id].offset, index_bytes)
            weights = _load(scratch, loads, slots[weight_id].offset, weight.nbytes)
            biases = _load(scratch, loads, slots[bias_id].offset, bias.nbytes)
            params_at = _load(scratch, loads, slots[param_id].offset, params.nbytes)
            table = (_load(scratch, loads, slots[table_id].offset, 256) if table_id else None)
            count = rows * output_pitch
            dest = _store(scratch, stores, slots[output].offset + first * output_pitch, count)
            label = f"{name}/a0-depthwise-{first}"
            stage = _stage(label, scratch, count, lambda p, lanes: _depthwise_program(
                p, weights, biases, params_at, indices, dest, channels, count, table, lanes))
            tiles.append(_finish(label, scratch, loads, [stage], stores))
        return (tuple(tiles),)

    return schedule
