"""Additional exact spatial operators for the frozen Z15 classifier path."""

from itertools import product

import numpy as np

from .c6_gpu import DATA_WINDOW_BYTES, PAGE_BYTES, build_gpu_stage
from .c6_memory import HALF_BYTES, Scratch, align_dma
from .c6_tensor import NativeStage, Tile, Transfer, channel_configuration
from .npu import Z15_CTRL, Z15_NPU_START


def pixel_map(height, width, oh, ow, channels, kernel, stride, pad):
    """Return byte addresses, mapping out-of-range pixels to a padded vector."""
    padding = height * width * channels
    indices = [
        ((y * width + x) * channels if 0 <= y < height and 0 <= x < width else padding)
        for oy, ox in product(range(oh), range(ow))
        for y, x in product(range(oy * stride - pad, oy * stride - pad + kernel),
                            range(ox * stride - pad, ox * stride - pad + kernel))
    ]
    return np.asarray(indices, dtype="<u4").reshape(oh * ow, kernel * kernel)


def _check_window(scratch, end):
    if end - scratch.base > DATA_WINDOW_BYTES:
        raise ValueError("classifier GPU tile exceeds the verified G3 data window")


def _gather_words(p, indices, output, words, lanes):
    p.addi(5, 5, 2, f3=1)
    p.li(11, indices)
    p.li(12, output)
    p.li(13, 4 * words)
    p.label("pixels")
    p.arithmetic(6, 5, 11)
    p.load(6, 6, "int32")
    p.load(7, 6, "int32")
    p.arithmetic(8, 5, 12)
    p.store(7, 8, "int32")
    p.addi(5, 5, 4 * lanes)
    p.branch(6, 5, 13, "pixels")


def stem_tiles(name, source, weights, bias, mapping, output, *, height, width, oh, ow,
               channels, mul, shift, lut=None, tile_pixels=64):
    """Map a padded 7×7 image stem to GPU word gather and native NPU GEMM."""
    if channels % 32 or not 1 <= channels <= 511 or not 1 <= tile_pixels <= 511:
        raise ValueError(f"{name}: expected a word-padded NPU stem")
    depth = 7 * 7 * 4
    count = oh * ow
    if height * width * 4 % 32 or tile_pixels * depth % 32:
        raise ValueError(f"{name}: stem planes must use DMA-aligned boundaries")
    source_bytes = align_dma(height * width * 4 + 4)
    config = channel_configuration(channels, mul, shift, lut)
    tiles = []
    for start in range(0, count, tile_pixels):
        size = min(tile_pixels, count - start)
        scratch = Scratch(len(tiles) % 2)
        image = scratch.reserve(source_bytes)
        indices = scratch.reserve(size * depth)
        gathered = scratch.reserve(size * depth)
        _check_window(scratch, scratch.base + scratch.used)
        w = scratch.reserve(channels * depth)
        b = scratch.reserve(channels * 4)
        c = scratch.reserve(size * channels)
        partial = scratch.reserve(4 * size * channels)
        # The map's out-of-image pixels point to a dedicated zero vector.
        loads = [Transfer("load", source, image.offset, source_bytes - 32),
                 Transfer("load", mapping + align_dma(count * depth), image.offset + source_bytes - 32, 32),
                 Transfer("load", mapping + start * depth, indices.offset, indices.capacity)]
        if len(tiles) < 2 or size != tile_pixels:
            loads.extend((Transfer("load", weights, w.offset, w.capacity),
                          Transfer("load", bias, b.offset, b.capacity)))
        words = size * 49
        lanes = min(words, 64)
        label = f"{name}/stem_{start}_{start+size}"
        gpu = build_gpu_stage(label + "/gather", scratch.base // PAGE_BYTES, lanes,
                              lambda p: _gather_words(p, indices.offset - scratch.base,
                                                      gathered.offset - scratch.base, words, lanes))
        flags = 1 << 1 | 1 << 2 | 1 << 15 | int(lut is not None) << 16 | 1 << 17
        npu = NativeStage(label + "/dot", "NPU", (config if not tiles else ()) + (
            (0x60, size | channels << 16), (0x64, depth | 128 << 16), (0x68, gathered.offset),
            (0x6C, w.offset), (0x70, c.offset), (0x74, b.offset), (0x78, flags),
            (0x7C, 0), (0x80, 0), (0x84, partial.offset), (Z15_CTRL, Z15_NPU_START),
        ))
        tiles.append(Tile(scratch.base // HALF_BYTES, tuple(loads), (gpu, npu),
                          (Transfer("store", output + start * channels, c.offset, c.capacity),),
                          scratch.used))
    return tuple(tiles)


def _max_pool(p, channels, indices, output, count, lanes):
    p.li(11, output)
    p.li(12, count)
    p.li(13, channels)
    p.label("output")
    p.arithmetic(6, 5, 13, f3=5, f7=1)
    p.arithmetic(8, 5, 13, f3=7, f7=1)
    p.li(9, 36)
    p.arithmetic(6, 6, 9, f7=1)
    p.li(9, indices)
    p.arithmetic(6, 6, 9)
    p.li(7, -128)
    for index in range(9):
        p.load(9, 6, "int32", 4 * index)
        p.arithmetic(9, 9, 8)
        p.load(9, 9, "int8")
        p.branch(5, 7, 9, f"tap_{index}")
        p.addi(7, 9, 0)
        p.label(f"tap_{index}")
    p.arithmetic(10, 5, 11)
    p.store(7, 10, "int8")
    p.addi(5, 5, lanes)
    p.branch(6, 5, 12, "output")


def max_pool_tiles(name, source, mapping, output, *, height, width, oh, ow, channels):
    """Run the signed 3×3 stride-two pool with a fixed address map."""
    scratch = Scratch(0)
    x = scratch.reserve(height * width * channels)
    pad = scratch.reserve(channels)
    index = scratch.reserve(oh * ow * 36)
    y = scratch.reserve(oh * ow * channels)
    if x.size != x.capacity:
        raise ValueError(f"{name}: source shape requires aligned pixels")
    _check_window(scratch, scratch.used)
    size = oh * ow * channels
    lanes = min(size, 64)
    loads = (Transfer("load", source, x.offset, x.capacity),
             Transfer("load", mapping + index.capacity, pad.offset, pad.capacity),
             Transfer("load", mapping, index.offset, index.capacity))
    stage = build_gpu_stage(name + "/pool", 0, lanes,
                            lambda p: _max_pool(p, channels, index.offset, y.offset, size, lanes))
    return (Tile(0, loads, (stage,), (Transfer("store", output, y.offset, y.capacity),), scratch.used),)


def _average(p, channels, pixels, output, lanes):
    p.li(11, channels)
    p.li(12, pixels)
    p.li(13, output)
    p.label("channel")
    p.addi(6, 5, 0)
    p.addi(7, 0, 0)
    p.addi(10, 12, 0)
    p.label("pixels")
    p.load(8, 6, "int8")
    p.arithmetic(7, 7, 8)
    p.arithmetic(6, 6, 11)
    p.addi(10, 10, -1)
    p.branch(1, 10, 0, "pixels")
    p.arithmetic(8, 7, 12, f3=4, f7=1)
    p.arithmetic(9, 7, 12, f3=6, f7=1)
    p.branch(5, 9, 0, "write")
    p.addi(8, 8, -1)
    p.label("write")
    p.arithmetic(6, 5, 13)
    p.store(8, 6, "int8")
    p.addi(5, 5, lanes)
    p.branch(6, 5, 11, "channel")


def average_tiles(name, source, output, *, channels, pixels):
    scratch = Scratch(0)
    x = scratch.reserve(channels * pixels)
    y = scratch.reserve(channels)
    _check_window(scratch, scratch.used)
    lanes = min(64, channels)
    stage = build_gpu_stage(name + "/gap", 0, lanes,
                            lambda p: _average(p, channels, pixels, y.offset, lanes))
    loads = (Transfer("load", source, x.offset, x.capacity),)
    stores = (Transfer("store", output, y.offset, y.capacity),)
    return (Tile(0, loads, (stage,), stores, scratch.used),)
