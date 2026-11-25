"""Lower depthwise 3x3 Conv to block-diagonal native DOT4 tensor jobs."""

import numpy as np

from .c6_memory import ALIGNMENT, Scratch
from .c6_reduce_conv import _split
from .c6_tensor import Conv, NativeStage, Tile, Transfer, _halo, channel_configuration, pack_weights
from .npu import Z15_CTRL, Z15_NPU_START, Z15_VPU_START

BLOCK_CHANNELS = 16


def depthwise_widths(spec):
    """Use native lane blocks and keep the final short block word aligned."""
    if spec.input_channels != spec.output_channels or spec.input_channels % 4:
        raise ValueError(f"{spec.name}: native depthwise requires one-to-one word-aligned channels")
    return tuple(min(BLOCK_CHANNELS, spec.input_channels - lo)
                 for lo in range(0, spec.input_channels, BLOCK_CHANNELS))


def depthwise_weights(spec, weights):
    if weights.dtype != np.int8 or weights.shape != (
            spec.kernel, spec.kernel, 1, spec.input_channels):
        raise ValueError(f"{spec.name}: expected signed INT8 depthwise HWIO filter")
    data = []
    lo = 0
    for channels in depthwise_widths(spec):
        part = Conv(spec.name, spec.height, spec.width, channels, channels,
                    spec.kernel, spec.stride, spec.pad)
        eye = np.eye(channels, dtype=np.int8)
        block = weights[:, :, :, lo:lo + channels] * eye
        data.append(pack_weights(part, block))
        lo += channels
    return tuple(data)


def _concat(stage_id, a, b, c, h, w, ca, cb):
    return NativeStage(stage_id, "VPU", (
        (0xA0, a), (0xA4, b), (0xA8, c), (0xAC, 0), (0xB0, 0),
        (0xB4, h | w << 16), (0xB8, ca | cb << 16), (0xBC, 1),
        (0xC0, 0), (Z15_CTRL, Z15_VPU_START)))


def _depthwise_tile(spec, index, first, rows, input_ddr, output_ddr, weights_ddr, bias_ddr, mul, shift, lut):
    zone = index % 2
    start, height, local_first = _halo(spec, first, rows)
    width = spec.width
    out_width = spec.output_hw[1]
    pixels = height * width
    m = rows * out_width
    group_widths = depthwise_widths(spec)
    scratch = Scratch(zone)
    weights = [scratch.reserve(spec.kernel ** 2 * channels ** 2) for channels in group_widths]
    biases = [scratch.reserve(channels * 4) for channels in group_widths]
    input_at = input_ddr + start * spec.input_pitch
    prefix = input_at % ALIGNMENT
    source = scratch.reserve(prefix + height * spec.input_pitch)
    loads = [Transfer("load", input_at - prefix, source.offset, source.capacity)]
    if index < 2:
        loads.extend(Transfer("load", address, allocation.offset, allocation.capacity)
                     for allocations, addresses in ((weights, weights_ddr), (biases, bias_ddr))
                     for allocation, address in zip(allocations, addresses))
    stages = []
    name = f"{spec.name}/rows_{first}_{first + rows}"

    def split(a, lo, hi):
        if hi - lo == 1:
            return [a]
        mid = (lo + hi) // 2
        ca, cb = sum(group_widths[lo:mid]), sum(group_widths[mid:hi])
        left, right = scratch.reserve(pixels * ca), scratch.reserve(pixels * cb)
        stages.append(_split(f"{name}/split_{lo}_{hi}", a, left.offset, right.offset,
                             height, width, ca, cb))
        return split(left.offset, lo, mid) + split(right.offset, mid, hi)

    operands = split(source.offset + prefix, 0, len(group_widths))
    values = []
    lo = 0
    for group, (channels, a, b, bias) in enumerate(zip(group_widths, operands, weights, biases)):
        k = channels * spec.kernel ** 2
        c = scratch.reserve(m * channels)
        partial = scratch.reserve(4 * m * channels) if k > 128 else None
        # Scale RAM is shared and indexed from zero at each block launch.
        # The SiLU table is invariant for the layer and is written once.
        config = channel_configuration(channels, mul[lo:lo + channels], shift[lo:lo + channels],
                                       lut if index == group == 0 else None)
        flags = (1 << 1 | 1 << 2 | 1 << 9 | int(spec.kernel == 3) << 10 |
                 int(spec.stride == 2) << 11 | int(spec.pad == 1) << 12 |
                 1 << 13 | 1 << 14 | 1 << 15 | int(lut is not None) << 16 | 1 << 17)
        stages.append(NativeStage(f"{name}/group{group}", "NPU", config + (
            (0x60, m | channels << 16), (0x64, k | 128 << 16), (0x68, a),
            (0x6C, b.offset), (0x70, c.offset), (0x74, bias.offset), (0x78, flags),
            (0x7C, height | width << 9 | channels << 18), (0x80, local_first | rows << 16),
            (0x84, partial.offset if partial else 0), (Z15_CTRL, Z15_NPU_START))))
        values.append(c)
        lo += channels

    def join(lo, hi):
        if hi - lo == 1:
            return values[lo]
        mid = (lo + hi) // 2
        ca, cb = sum(group_widths[lo:mid]), sum(group_widths[mid:hi])
        a, b = join(lo, mid), join(mid, hi)
        c = scratch.reserve(m * (ca + cb))
        stages.append(_concat(f"{name}/join_{lo}_{hi}", a.offset, b.offset, c.offset,
                              rows, out_width, ca, cb))
        return c

    got = join(0, len(group_widths))
    return Tile(zone, tuple(loads), tuple(stages),
                (Transfer("store", output_ddr + first * spec.output_pitch, got.offset, got.capacity),),
                scratch.used)


def depthwise_tiles(spec, *, input_ddr, output_ddr, weights_ddr, bias_ddr, mul, shift, lut=None):
    """Partition space and channels with native VPU DMA-safe NHWC split/join."""
    groups = depthwise_widths(spec)
    if not len(groups) == len(weights_ddr) == len(bias_ddr):
        raise ValueError(f"{spec.name}: missing a depthwise group's initializer")
    first = 0
    result = []
    oh, ow = spec.output_hw
    while first < oh:
        rows = min(oh - first, 511 // ow)
        while rows:
            if rows * spec.output_pitch % ALIGNMENT and first + rows < oh:
                rows -= 1
                continue
            try:
                tile = _depthwise_tile(spec, len(result), first, rows, input_ddr, output_ddr,
                                       weights_ddr, bias_ddr, mul, shift, lut)
            except ValueError:
                rows -= 1
                continue
            break
        if not rows:
            raise ValueError(f"{spec.name}: one depthwise row does not fit the 1 MiB ownership half")
        result.append(tile)
        first += rows
    return tuple(result)
