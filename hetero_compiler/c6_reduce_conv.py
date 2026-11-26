"""Exact reduction-channel tiling for INT8 Conv with native K above 511."""

from .c6_gpu import DATA_WINDOW_BYTES, PAGE_BYTES, build_gpu_stage, emit_requant
from .c6_memory import ALIGNMENT, HALF_BYTES, Scratch, align_dma
from .c6_tensor import NativeStage, Tile, Transfer, _halo
from .npu import Z15_CTRL, Z15_NPU_START, Z15_VPU_START


def reduction_widths(spec):
    """Return word-aligned channel spans that fit the native K dimension."""
    width = 4 * (511 // (spec.kernel ** 2 * 4))
    return tuple(min(width, spec.padded_channels - start)
                 for start in range(0, spec.padded_channels, width))


def _split(stage_id, a, c, d, h, w, keep, rest):
    return NativeStage(stage_id, "VPU", (
        (0xA0, a), (0xA4, 0), (0xA8, c), (0xAC, d), (0xB0, 0),
        (0xB4, h | w << 16), (0xB8, keep | rest << 16), (0xBC, 2),
        (0xC0, 0), (Z15_CTRL, Z15_VPU_START)))


def _raw_tile(spec, group_widths, group_weights, group_outputs, input_ddr, bias_ddr, first, rows, index, finish):
    zone = index % 2
    start, height, local_first = _halo(spec, first, rows)
    m = rows * spec.output_hw[1]
    n = spec.output_channels
    scratch = Scratch(zone)
    input_at = input_ddr + start * spec.input_pitch
    prefix = input_at % ALIGNMENT
    npx = height * spec.width
    # Coefficients and bias live at a fixed offset in each half across the layer.
    bias = scratch.reserve(n * 4)
    weights = [scratch.reserve(part * spec.kernel ** 2 * n) for part in group_widths]
    source = scratch.reserve(prefix + height * spec.input_pitch)
    loads = [Transfer("load", input_at - prefix, source.offset, source.capacity)]
    if index < 2:
        loads.append(Transfer("load", bias_ddr, bias.offset, bias.capacity))
        loads.extend(Transfer("load", ddr_at, buf.offset, buf.capacity)
                     for ddr_at, buf in zip(group_weights, weights))
    if finish is not None:
        padding = -scratch.used % PAGE_BYTES
        if padding:
            scratch.reserve(padding)
        partials = [scratch.reserve(4 * m * n) for _ in group_widths]
        if sum(part.capacity for part in partials) > DATA_WINDOW_BYTES:
            raise ValueError(f"{spec.name}: partial words exceed the G3 data window")
    a = source.offset + prefix
    remaining = spec.padded_channels
    stages, stores = [], []
    for group, (channels, b) in enumerate(zip(group_widths, weights)):
        group_name = f"{spec.name}/rows_{first}_{first + rows}/k{group}"
        if channels < remaining:
            rest = remaining - channels
            cropped = scratch.reserve(npx * channels)
            tail = scratch.reserve(npx * rest)
            stages.append(_split(f"{group_name}/split", a, cropped.offset, tail.offset, height,
                                 spec.width, channels, rest))
            operand = cropped.offset
            a = tail.offset
            remaining = rest
        else:
            operand = a
        c = scratch.reserve(4 * m * n) if finish is None else partials[group]
        k = channels * spec.kernel ** 2
        flags = (int(group == 0) << 1 | 1 << 9 | int(spec.kernel == 3) << 10 |
                 int(spec.stride == 2) << 11 | int(spec.pad == 1) << 12 | 1 << 13 | 1 << 14)
        stages.append(NativeStage(group_name, "NPU", (
            (0x60, m | n << 16), (0x64, k | 128 << 16), (0x68, operand), (0x6C, b.offset),
            (0x70, c.offset), (0x74, bias.offset), (0x78, flags),
            (0x7C, height | spec.width << 9 | channels << 18),
            (0x80, local_first | rows << 16), (0x84, 0), (Z15_CTRL, Z15_NPU_START))))
        if finish is None:
            stores.append(Transfer("store", group_outputs[group] + first * spec.output_pitch * 4,
                                   c.offset, c.capacity))
    if finish is not None:
        extra_loads, extra_stages, store = finish(scratch, first, rows, index, partials)
        loads.extend(extra_loads)
        stages.extend(extra_stages)
        stores.append(store)
    return Tile(zone, tuple(loads), tuple(stages), tuple(stores), scratch.used)


def raw_conv_tiles(spec, *, input_ddr, weights_ddr, bias_ddr, partials_ddr=None, finish=None):
    """Emit raw INT32 NPU Conv pieces and drain them to DDR or a tile-local epilogue."""
    groups = reduction_widths(spec)
    if len(groups) < 2 or len(weights_ddr) != len(groups):
        raise ValueError(f"{spec.name}: expected a coefficient span for each reduction group")
    if (finish is None) == (partials_ddr is None) or (
            partials_ddr is not None and len(partials_ddr) != len(groups)):
        raise ValueError(f"{spec.name}: expected DDR partial spans or a local epilogue")
    item_bytes = 4 if finish is None else 1
    result = []
    first = 0
    oh, ow = spec.output_hw
    while first < oh:
        rows = min(oh - first, 511 // ow)
        while rows:
            # Intermediate tile boundaries are also aligned DMA boundaries.
            if item_bytes * rows * spec.output_pitch % ALIGNMENT and first + rows < oh:
                rows -= 1
                continue
            try:
                tile = _raw_tile(spec, groups, weights_ddr, partials_ddr, input_ddr, bias_ddr,
                                 first, rows, len(result), finish)
            except ValueError:
                rows -= 1
                continue
            break
        if not rows:
            raise ValueError(f"{spec.name}: no reduction-channel tile fits the 1 MiB ownership half")
        result.append(tile)
        first += rows
    return tuple(result)


def _combine_kernel(p, offsets, params, output, channels, phase, lut):
    p.addi(6, 5, 2, f3=1)
    for index, offset in enumerate(offsets):
        p.li(9, offset)
        p.arithmetic(9, 6, 9)
        p.load(7 if index == 0 else 8, 9, "int32")
        if index:
            p.arithmetic(7, 7, 8)
    p.addi(6, 5, phase)
    if channels & (channels - 1):
        p.li(9, channels)
        p.arithmetic(6, 6, 9, f3=7, f7=1)
    else:
        p.addi(6, 6, channels - 1, f3=7)
    p.addi(6, 6, 2, f3=1)
    p.addi(6, 6, params)
    p.load(8, 6, "int32")
    emit_requant(p)
    if lut is not None:
        p.addi(8, 7, 255, f3=7)
        p.addi(8, 8, lut)
        p.load(7, 8, "int8")
    p.li(6, output)
    p.arithmetic(6, 6, 5)
    p.store(7, 6, "int8")


def combine_tiles(spec, *, output_ddr, partials_ddr, params_ddr, lut_ddr=None):
    """Sum raw partials and apply one native-equivalent requant + optional LUT."""
    n = spec.output_channels
    # The scale table is shared by all pages at invariant offsets.
    params_size = align_dma(n * 4)
    lut_size = 256 if lut_ddr is not None else 0
    unit = (PAGE_BYTES - params_size - lut_size) // (4 * len(partials_ddr) + 1)
    unit -= unit % ALIGNMENT
    if not unit:
        raise ValueError(f"{spec.name}: GPU reduction tile cannot fit a 4 KiB page")
    count = spec.output_hw[0] * spec.output_pitch
    result = []
    for first in range(0, count, unit):
        items = min(unit, count - first)
        zone = len(result) % 2
        scratch = Scratch(zone)
        params = scratch.reserve(params_size)
        lut = scratch.reserve(lut_size) if lut_size else None
        buf = [scratch.reserve(4 * items) for _ in partials_ddr]
        out = scratch.reserve(items)
        loads = [Transfer("load", ddr_at + 4 * first, place.offset, place.capacity)
                 for place, ddr_at in zip(buf, partials_ddr)]
        if len(result) < 2:
            loads += [Transfer("load", params_ddr, params.offset, params.capacity)]
            if lut is not None:
                loads += [Transfer("load", lut_ddr, lut.offset, lut.capacity)]
        stage = build_gpu_stage(f"{spec.name}/reduce_{first}_{first + items}",
                                zone * HALF_BYTES // PAGE_BYTES, items,
                                lambda p: _combine_kernel(p, [place.offset - scratch.base for place in buf], 0,
                                    out.offset - scratch.base, n, first % n,
                                    lut.offset - scratch.base if lut is not None else None))
        store = Transfer("store", output_ddr + first, out.offset, out.capacity)
        result.append(Tile(zone, tuple(loads), (stage,), (store,), scratch.used))
    return tuple(result)
