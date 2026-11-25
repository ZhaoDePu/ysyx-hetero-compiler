"""Balanced VPU lowering for C6 byte activations and their physical layouts."""

from itertools import accumulate
from math import gcd, lcm, prod

from .c6_memory import ALIGNMENT, HALF_BYTES, Scratch, align_dma
from .c6_tensor import NativeStage, Tile, Transfer, channel_configuration
from .npu import Z15_CTRL, Z15_VPU_START
from .yhb import POOL_BYTES

EW_ADD = 0
EW_CONCAT = 1
EW_SPLIT = 2
EW_UPSAMPLE = 3
EW_MAXPOOL5 = 5
EW_MUL = 6
EW_COPY = 8
EW_LOOKUP = 11


def _ew(name, opcode, a, c, *, b=0, d=0, length=0, h=0, w=0, ca=0, cb=0,
        scale=1, shift=0, setup=()):
    return NativeStage(name, "VPU", setup + (
        (0xA0, a), (0xA4, b), (0xA8, c), (0xAC, d), (0xB0, length),
        (0xB4, h | (w << 16)), (0xB8, ca | (cb << 16)),
        (0xBC, opcode | (scale << 8) | (shift << 24)), (0xC0, 0),
        (Z15_CTRL, Z15_VPU_START),
    ))


class _TileBuilder:
    def __init__(self, index):
        self.zone = index % 2
        self.scratch = Scratch(self.zone)
        self.loads = []
        self.stores = []
        self.stages = []

    def load(self, ddr, size):
        slot = self.scratch.reserve(size)
        self.loads.append(Transfer("load", ddr, slot.offset, slot.capacity))
        return slot.offset

    def store(self, ddr, scratch, size):
        self.stores.append(Transfer("store", ddr, scratch, align_dma(size)))

    def end(self):
        return Tile(self.zone, tuple(self.loads), tuple(self.stages), tuple(self.stores), self.scratch.used)


def _validate_operand(offset, shape):
    if (not shape or any(type(extent) is not int or extent <= 0 for extent in shape)
            or type(offset) is not int or offset < 0 or offset % ALIGNMENT
            or offset + align_dma(prod(shape)) > POOL_BYTES):
        raise ValueError("Z15 vector operands require positive extents and aligned DDR ranges")


def _alignment_unit(pitches):
    return lcm(*(ALIGNMENT // gcd(pitch, ALIGNMENT) for pitch in pitches))


def _row_windows(rows, pitches, capacity, size, tile_rows=None):
    unit = _alignment_unit(pitches)
    limit = min(capacity, tile_rows) if tile_rows is not None else capacity
    if limit < rows and limit < unit:
        raise ValueError("one DDR-aligned row block exceeds the configured tile limit")
    first = 0
    while first < rows:
        count = min(rows - first, limit)
        if first + count < rows:
            count = count // unit * unit
        while count > 0 and size(first, count) > HALF_BYTES:
            count = (count - 1) // unit * unit
        if count <= 0:
            raise ValueError("one DDR-aligned row block exceeds a Z15 scratch half")
        yield first, count
        first += count


def _linear_tiles(name, op, shape, inputs, output, table, mul, shift, tile_rows):
    opcode = {"Add": EW_ADD, "EWAdd": EW_ADD, "EWMul": EW_MUL, "RequantLUT": EW_LOOKUP,
              "Lookup": EW_LOOKUP, "Copy": EW_COPY}[op]
    setup = channel_configuration(0, (), (), table) if opcode == EW_LOOKUP else ()
    if opcode == EW_MUL and (not 0 <= mul < 65536 or not 0 <= shift < 32):
        raise ValueError("EWMul requires an unsigned 16-bit multiplier and 5-bit shift")
    elements = prod(shape)
    maximum = ((HALF_BYTES // (len(inputs) + 1)) // ALIGNMENT) * ALIGNMENT
    if tile_rows is not None:
        pitch = prod(shape[2:]) if len(shape) == 4 else 1
        limit = tile_rows * pitch
        if limit < elements and limit % ALIGNMENT:
            raise ValueError("vector chunks must start at DDR-aligned positions")
        maximum = min(maximum, limit)
    tiles = []
    for first in range(0, elements, maximum):
        count = min(elements - first, maximum)
        tile = _TileBuilder(len(tiles))
        addresses = [tile.load(offset + first, count) for offset in inputs]
        dst = tile.scratch.reserve(count).offset
        tile.stages.append(_ew(f"{name}/bytes_{first}_{first + count}", opcode, addresses[0], dst,
                               b=addresses[-1] if len(addresses) == 2 else 0, length=count,
                               scale=mul, shift=shift, setup=setup if first == 0 else ()))
        tile.store(output + first, dst, count)
        tiles.append(tile.end())
    return tuple(tiles)


def _spatial_tiles(name, op, shape, src, dst, tile_rows):
    _, height, width, channels = shape
    if width > 1023 or channels > 1023:
        raise ValueError("VPU spatial shape exceeds 10-bit width/channel fields")
    pool = op == "MaxPool"
    pitch = width * channels
    if pool:
        unit = _alignment_unit([pitch])
        def span(first, rows):
            start = max(0, first - 2)
            start -= start % unit
            return start, min(height, first + rows + 2) - start

        capacity = 1023 - (unit + 3)
        def size(first, rows):
            return 2 * align_dma(span(first, rows)[1] * pitch)
    else:
        capacity = 1023
        def size(_, rows):
            return align_dma(rows * pitch) + align_dma(4 * rows * pitch)

    tiles = []
    for first, rows in _row_windows(height, [pitch], capacity, size, tile_rows):
        tile = _TileBuilder(len(tiles))
        start, count = span(first, rows) if pool else (first, rows)
        a = tile.load(src + start * pitch, count * pitch)
        dst_pitch = pitch if pool else 4 * pitch
        c = tile.scratch.reserve(count * dst_pitch).offset
        opcode = EW_MAXPOOL5 if pool else EW_UPSAMPLE
        tile.stages.append(_ew(f"{name}/rows_{first}_{first + rows}", opcode, a, c,
                               h=count, w=width, ca=channels))
        tile.store(dst + first * dst_pitch, c + (first - start) * dst_pitch, rows * dst_pitch)
        tiles.append(tile.end())
    return tuple(tiles)


def _partition(name, op, inputs, outputs, axis):
    concat = op == "Concat"
    parts, whole = (inputs, outputs[0]) if concat else (outputs, inputs[0])
    rank = len(whole)
    if type(axis) is not int or not -rank <= axis < rank:
        raise ValueError(f"{name}: tensor partition needs a valid physical axis")
    axis %= rank
    same = whole[:axis] + whole[axis + 1:]
    if (any(len(s) != rank or s[:axis] + s[axis + 1:] != same for s in parts)
            or sum(s[axis] for s in parts) != whole[axis]):
        raise ValueError(f"{name}: inconsistent tensor partition shapes")
    width = whole[axis - 1] if axis else 1
    rows = prod(whole[:axis]) // width
    channels = [prod(s[axis:]) for s in parts]
    return rows, width, channels


def _partition_tiles(name, op, shape_in, shape_out, inputs, outputs, axis, tile_rows):
    concat = op == "Concat"
    height, width, chan = _partition(name, op, shape_in, shape_out, axis)
    total = sum(chan)
    if width > 1023 or max(max(chan), total - chan[-1] if concat else total - chan[0]) > 1023:
        return _copy_partition(name, concat, height * width, chan, inputs, outputs)

    # Concatenate left-to-right. Split right-to-left so every output is packed.
    prefix = list(accumulate(chan))
    temporary = prefix[:-1][1:] if concat else [total - used for used in prefix[:-2]]
    pitches = [c * width for c in (*chan, total)]
    def size(_, rows):
        return sum(align_dma(rows * p) for p in (*pitches, *(c * width for c in temporary)))

    tiles = []
    for first, rows in _row_windows(height, pitches, 1023, size, tile_rows):
        tile = _TileBuilder(len(tiles))
        suffix = f"{name}/rows_{first}_{first + rows}"
        count = rows * width
        if concat:
            src = [tile.load(offset + first * width * ch, count * ch)
                   for offset, ch in zip(inputs, chan, strict=True)]
            combined = src[0]
            for step, (address, channels) in enumerate(zip(src[1:], chan[1:], strict=True)):
                out = tile.scratch.reserve(count * prefix[step + 1]).offset
                tile.stages.append(_ew(f"{suffix}/concat_{step}", EW_CONCAT, combined, out,
                                       b=address, h=rows, w=width, ca=prefix[step], cb=channels))
                combined = out
            tile.store(outputs[0] + first * width * total, combined, count * total)
        else:
            current = tile.load(inputs[0] + first * width * total, count * total)
            for step, ch in enumerate(chan[:-1]):
                remaining = total - prefix[step]
                left = tile.scratch.reserve(count * ch).offset
                right = tile.scratch.reserve(count * remaining).offset
                tile.stages.append(_ew(f"{suffix}/split_{step}", EW_SPLIT, current, left,
                                       d=right, h=rows, w=width, ca=ch, cb=remaining))
                tile.store(outputs[step] + first * width * ch, left, count * ch)
                current = right
            tile.store(outputs[-1] + first * width * chan[-1], current, count * chan[-1])
        tiles.append(tile.end())
    return tuple(tiles)


def _copy_partition(name, concat, records, channels, inputs, outputs):
    if any(v % ALIGNMENT for v in channels):
        raise ValueError(f"{name}: oversized partition records need aligned DDR slices")
    total = sum(channels)
    maximum = HALF_BYTES // 2
    tiles = []
    for record in range(records):
        prefix = 0
        for part, channels_in_part in enumerate(channels):
            for start in range(0, channels_in_part, maximum):
                count = min(maximum, channels_in_part - start)
                packed = record * channels_in_part + start
                joined = record * total + prefix + start
                a, c = (inputs[part] + packed, outputs[0] + joined) if concat else (
                    inputs[0] + joined, outputs[part] + packed)
                tile = _TileBuilder(len(tiles))
                source = tile.load(a, count)
                result = tile.scratch.reserve(count).offset
                stage = f"{name}/record_{record}/part_{part}/bytes_{start}_{start + count}"
                tile.stages.append(_ew(stage, EW_COPY, source, result, length=count))
                tile.store(c, result, count)
                tiles.append(tile.end())
            prefix += channels_in_part
    return tuple(tiles)


def vector_tiles(name, op, input_shapes, output_shapes, input_ddr, output_ddr, *,
                 axis=None, table=None, mul=1, shift=0, tile_rows=None):
    """Plan signed-byte operations over physical shapes; DDR strides contain no padding."""
    ins, outs = [tuple(s) for s in input_shapes], [tuple(s) for s in output_shapes]
    if len(ins) != len(input_ddr) or len(outs) != len(output_ddr):
        raise ValueError(f"{name}: input and output shapes must match their DDR bindings")
    for offset, shape in zip((*input_ddr, *output_ddr), (*ins, *outs), strict=True):
        _validate_operand(offset, shape)
    if tile_rows is not None and (type(tile_rows) is not int or tile_rows <= 0):
        raise ValueError("requested tile_rows must be positive")
    if op == "Concat" and len(ins) >= 2 and len(outs) == 1:
        return _partition_tiles(name, op, ins, outs, input_ddr, output_ddr, axis, tile_rows)
    if op == "Split" and len(ins) == 1 and len(outs) >= 2:
        return _partition_tiles(name, op, ins, outs, input_ddr, output_ddr, axis, tile_rows)
    if op in {"Resize", "MaxPool"} and len(ins) == len(outs) == 1:
        shape = ins[0]
        if len(shape) != 4 or shape[0] != 1:
            raise ValueError(f"{name}: spatial vector operators require batch-one NHWC")
        expected = (1, shape[1] * 2, shape[2] * 2, shape[3]) if op == "Resize" else shape
        if outs[0] != expected:
            raise ValueError(f"{name}: native spatial vector geometry is 2x nearest resize or same-size pooling")
        return _spatial_tiles(name, op, shape, input_ddr[0], output_ddr[0], tile_rows)
    binary = {"Add", "EWAdd", "EWMul"}
    unary = {"Lookup", "RequantLUT", "Copy"}
    if (op in binary | unary and len(ins) == (2 if op in binary else 1)
            and len(outs) == 1 and all(shape == outs[0] for shape in ins)):
        if op in {"Lookup", "RequantLUT"} and table is None:
            raise ValueError(f"{name}: missing signed-byte lookup table")
        return _linear_tiles(name, op, outs[0], input_ddr, output_ddr[0], table, mul, shift, tile_rows)
    raise ValueError(f"{name}: unsupported Z15 vector operation or arity {op}")
