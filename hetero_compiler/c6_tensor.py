"""Native INT8 Conv tiling for the frozen Phase 12C Balanced tensor port."""

from dataclasses import dataclass
from math import gcd

import numpy as np

from .c6_memory import ALIGNMENT, HALF_BYTES, Allocation, Scratch, align_dma
from .npu import Z15_CTRL, Z15_NPU_START


@dataclass(frozen=True)
class NativeStage:
    id: str
    device: str
    writes: tuple


@dataclass(frozen=True)
class Transfer:
    direction: str
    ddr_offset: int
    scratch_offset: int
    size: int

    def command(self):
        return {"type": "dma", "direction": self.direction, "ddr_offset": self.ddr_offset,
                "scratch_offset": self.scratch_offset, "bytes": self.size}


@dataclass(frozen=True)
class Tile:
    zone: int
    loads: tuple
    stages: tuple
    stores: tuple
    scratch_bytes: int


@dataclass(frozen=True)
class Conv:
    name: str
    height: int
    width: int
    input_channels: int
    output_channels: int
    kernel: int
    stride: int
    pad: int

    def __post_init__(self):
        if (any(type(dim) is not int or not 1 <= dim <= 511 for dim in
                (self.height, self.width, self.output_channels))
                or type(self.input_channels) is not int
                or not 1 <= self.input_channels <= (1528 if self.kernel == 1 else 1023)
                or self.kernel not in (1, 3) or self.stride not in (1, 2)
                or self.pad not in (0, 1) or min(self.output_hw) < 1):
            raise ValueError(f"{self.name}: unsupported native Conv geometry")
        if self.output_channels % 4:
            raise ValueError(f"{self.name}: packed Conv requires a multiple of four output channels")

    @property
    def padded_channels(self):
        return (self.input_channels + 3) & -4

    @property
    def depth(self):
        return self.kernel ** 2 * self.padded_channels

    @property
    def input_pitch(self):
        return self.width * self.padded_channels

    @property
    def output_hw(self):
        return tuple((extent + 2 * self.pad - self.kernel) // self.stride + 1
                     for extent in (self.height, self.width))

    @property
    def output_pitch(self):
        return self.output_hw[1] * self.output_channels


def pack_weights(spec, weights):
    """Pack ONNX HWIO into the native per-output-channel K-major format."""
    if weights.dtype != np.int8 or weights.shape != (
            spec.kernel, spec.kernel, spec.input_channels, spec.output_channels):
        raise ValueError(f"{spec.name}: expected signed INT8 HWIO coefficients")
    packed = np.zeros((spec.output_channels, spec.kernel, spec.kernel, spec.padded_channels),
                      dtype=np.int8)
    packed[..., :spec.input_channels] = weights.transpose(3, 0, 1, 2)
    return packed.tobytes()


def channel_configuration(channels, mul, shift, lut=None):
    if len(mul) != channels or len(shift) != channels:
        raise ValueError("one multiplier and shift are required per output channel")
    writes = []
    for index, (m, s) in enumerate(zip(mul, shift)):
        if not 0 <= m < 65536 or not 0 <= s < 32 or int(m) != m or int(s) != s:
            raise ValueError("invalid u16 multiplier or u5 shift")
        writes += [(0xC4, index), (0xC8, int(m) | int(s) << 16), (Z15_CTRL, 1 << 7)]
    if lut is not None:
        if len(lut) != 256 or any(not -128 <= value <= 127 for value in lut):
            raise ValueError("expected a 256-element signed INT8 activation LUT")
        for index, value in enumerate(lut):
            writes += [(0xCC, index | (int(value) & 255) << 8), (Z15_CTRL, 1 << 8)]
    return tuple(writes)


def _halo(spec, first, rows):
    # Preserve the global stride phase. Include a real row before an interior
    # halo when needed so the local slice can skip its implicit top padding.
    start = max(0, first * spec.stride - spec.pad)
    start -= start % spec.stride
    stop = min(spec.height, (first + rows - 1) * spec.stride - spec.pad + spec.kernel)
    return start, stop - start, first - start // spec.stride


def _tile(spec, zone, first, rows, input_ddr, output_ddr, weights_ddr, bias_ddr, config, use_lut,
          load_constants):
    start, height, slice_first = _halo(spec, first, rows)
    m = rows * spec.output_hw[1]
    n, k = spec.output_channels, spec.depth
    scratch = Scratch(zone)
    # Keep weights/bias at invariant offsets in each half across spatial tiles.
    b = scratch.reserve(n * k)
    bias = scratch.reserve(4 * n)
    input_at = input_ddr + start * spec.input_pitch
    prefix = input_at % ALIGNMENT
    a = scratch.reserve(prefix + height * spec.input_pitch)
    c = scratch.reserve(rows * spec.output_pitch)
    partial = scratch.reserve(4 * m * n) if k > 128 else Allocation(0, 0)
    shape = height | spec.width << 9 | spec.input_channels << 18
    flags = (1 << 1 | 1 << 2 | 1 << 9 | int(spec.kernel == 3) << 10 |
             int(spec.stride == 2) << 11 | int(spec.pad == 1) << 12 |
             1 << 13 | 1 << 14 | 1 << 15 | int(use_lut) << 16 | 1 << 17)
    writes = config + (
        (0x60, m | n << 16), (0x64, k | 128 << 16), (0x68, a.offset + prefix), (0x6C, b.offset),
        (0x70, c.offset), (0x74, bias.offset), (0x78, flags), (0x7C, shape),
        (0x80, slice_first | rows << 16), (0x84, partial.offset), (Z15_CTRL, Z15_NPU_START))
    loads = (Transfer("load", input_at - prefix, a.offset, a.capacity),)
    if load_constants:
        loads += (Transfer("load", weights_ddr, b.offset, b.capacity),
                  Transfer("load", bias_ddr, bias.offset, bias.capacity))
    store = Transfer("store", output_ddr + first * spec.output_pitch, c.offset, c.capacity)
    launch = NativeStage(f"{spec.name}/rows_{first}_{first + rows}", "NPU", writes)
    return Tile(zone, loads, (launch,), (store,), scratch.used)


def conv_tiles(spec, *, input_ddr, output_ddr, weights_ddr, bias_ddr, mul, shift, lut=None):
    """Plan complete output raster with halo DMA, alternating ownership halves."""
    if spec.depth > 511:
        raise ValueError(f"{spec.name}: Conv K={spec.depth} requires reduction tiling")
    config = channel_configuration(spec.output_channels, mul, shift, lut)
    result = []
    count, width = spec.output_hw
    first = 0
    row_unit = ALIGNMENT // gcd(spec.output_pitch, ALIGNMENT)
    while first < count:
        rows = min(count - first, 511 // width)
        while rows:
            if first + rows == count or rows % row_unit == 0:
                start, height, _ = _halo(spec, first, rows)
                n = spec.output_channels
                prefix = (input_ddr + start * spec.input_pitch) % ALIGNMENT
                minimum = sum(align_dma(size) for size in (n * spec.depth, 4 * n,
                    prefix + height * spec.input_pitch, rows * spec.output_pitch,
                    4 * rows * width * n if spec.depth > 128 else 0))
                if minimum <= HALF_BYTES:
                    break
            rows -= 1
        if not rows:
            raise ValueError(f"{spec.name}: one Conv output row does not fit the native tile")
        result.append(_tile(spec, len(result) % 2, first, rows, input_ddr, output_ddr, weights_ddr,
                            bias_ddr, config if not result else (), lut is not None, len(result) < 2))
        first += rows
    return tuple(result)
