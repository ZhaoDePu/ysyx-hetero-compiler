"""Phase 17 K continuation for implicit Conv that fits one tensor tile."""

import numpy as np

from .c6_memory import HALF_BYTES, Scratch, align_dma
from .c6_tensor import NativeStage, Tile, Transfer, channel_configuration, pack_weights
from .npu import Z15_CTRL, Z15_NPU_START

CONTINUE_K = 1 << 18
DEFER_K = 1 << 19
K_CHUNK = 508


def can_chain_conv(spec):
    """Implicit Conv slices complete rows and retains at most 16x16 accumulators."""
    oh, width = spec.output_hw
    n = spec.output_channels
    scratch = (align_dma(4 * n) + align_dma(spec.height * spec.input_pitch) +
               align_dma(oh * spec.output_pitch) +
               sum(align_dma(n * depth) for depth in chain_depths(spec.depth)))
    return (511 < spec.depth <= 8192 and spec.input_channels <= 511 and oh <= 511
            and width <= 16 and n <= 16 and scratch <= HALF_BYTES)


def chain_depths(k):
    return tuple(min(K_CHUNK, k - start) for start in range(0, k, K_CHUNK))


def pack_chain_weights(spec, weights):
    rows = np.frombuffer(pack_weights(spec, weights), dtype=np.int8)
    rows = rows.reshape(spec.output_channels, spec.depth)
    return tuple(rows[:, start:start + K_CHUNK].tobytes()
                 for start in range(0, spec.depth, K_CHUNK))


def chain_tiles(spec, *, input_ddr, output_ddr, weights_ddr, bias_ddr, mul, shift, lut):
    """Feed adjacent K slices of the same patch and write packed bytes after the last."""
    if not can_chain_conv(spec):
        raise ValueError(f"{spec.name}: geometry requires an extra spatial or channel tiling pass")
    depths = chain_depths(spec.depth)
    if len(weights_ddr) != len(depths):
        raise ValueError(f"{spec.name}: one packed buffer is required per K chunk")
    oh, ow = spec.output_hw
    n = spec.output_channels
    row_step = 16 // ow
    config = channel_configuration(n, mul, shift, lut)
    scratch = Scratch(0)
    bias = scratch.reserve(n * 4)
    weights = tuple(scratch.reserve(n * depth) for depth in depths)
    a = scratch.reserve(spec.height * spec.input_pitch)
    c = scratch.reserve(oh * spec.output_pitch)
    loads = [Transfer("load", input_ddr, a.offset, a.capacity),
             Transfer("load", bias_ddr, bias.offset, bias.capacity)]
    loads.extend(Transfer("load", address, target.offset, target.capacity)
                 for address, target in zip(weights_ddr, weights, strict=True))

    stages = []
    conv = (1 << 9 | int(spec.kernel == 3) << 10 | int(spec.stride == 2) << 11 |
            int(spec.pad == 1) << 12 | 1 << 13 | 1 << 14)
    for first in range(0, oh, row_step):
        rows = min(row_step, oh - first)
        tile_output = c.offset + first * spec.output_pitch
        position = 0
        for index, (depth, b) in enumerate(zip(depths, weights, strict=True)):
            final = index + 1 == len(depths)
            flags = conv | (CONTINUE_K if index else 1 << 1)
            flags |= ((1 << 2 | 1 << 15 | int(lut is not None) << 16 | 1 << 17)
                      if final else DEFER_K)
            table = config if not stages else ()
            stages.append(NativeStage(f"{spec.name}/rows_{first}_{first + rows}/chain{index}", "NPU",
                table + ((0x60, rows * ow | n << 16), (0x64, depth | 128 << 16),
                         (0x68, a.offset), (0x6C, b.offset), (0x70, tile_output),
                         (0x74, bias.offset), (0x78, flags),
                         (0x7C, spec.height | spec.width << 9 | spec.input_channels << 18),
                         (0x80, first | rows << 16), (0x84, position), (Z15_CTRL, Z15_NPU_START))))
            position += depth
    return (Tile(0, tuple(loads), tuple(stages),
                 (Transfer("store", output_ddr, c.offset, c.capacity),), scratch.used),)
