"""Dense signed INT8 classifier convolution for the frozen G3 GPU."""

from math import gcd

import numpy as np

from .c6_gpu import DATA_WINDOW_BYTES, PAGE_BYTES, build_gpu_stage, emit_requant
from .c6_memory import Scratch, align_dma
from .c6_tensor import Tile, Transfer
from .c9_spatial import pixel_map


def _divmod(p, quotient, remainder, source, divide, work):
    if divide & (divide - 1) == 0 and divide <= 2048:
        p.addi(quotient, source, divide.bit_length() - 1, f3=5)
        p.addi(remainder, source, divide - 1, f3=7)
    else:
        p.li(work, divide)
        p.arithmetic(quotient, source, work, f3=5, f7=1)
        p.arithmetic(remainder, source, work, f3=7, f7=1)


def _gather(p, mapping, gathered, channels, words, lanes):
    # x5 indexes output words, mapped in pixel/kernel/channel order.
    p.li(10, mapping)
    p.li(11, gathered)
    p.li(12, words)
    p.label("gather")
    _divmod(p, 6, 7, 5, channels // 4, 9)
    p.addi(6, 6, 2, f3=1)
    p.arithmetic(6, 6, 10)
    p.load(6, 6, "int32")
    p.addi(7, 7, 2, f3=1)
    p.arithmetic(6, 6, 7)
    p.load(8, 6, "int32")
    p.addi(6, 5, 2, f3=1)
    p.arithmetic(6, 6, 11)
    p.store(8, 6, "int32")
    p.addi(5, 5, lanes)
    p.branch(6, 5, 12, "gather")


def _dot(p, activations, weights, params, result, depth, co, first, stop, lanes, lut):
    """Interpret x5 as an independent (pixel, channel) output index."""
    if first:
        p.li(6, first)
        p.arithmetic(5, 5, 6)
    p.label("output")
    _divmod(p, 15, 6, 5, co, 8)
    p.li(8, depth)
    p.arithmetic(11, 15, 8, f7=1)
    p.li(9, activations)
    p.arithmetic(11, 11, 9)
    p.arithmetic(10, 6, 8, f7=1)
    p.li(9, weights)
    p.arithmetic(10, 10, 9)
    p.arithmetic(9, 10, 8)
    p.addi(6, 6, 3, f3=1)
    p.li(8, params)
    p.arithmetic(6, 6, 8)
    p.load(7, 6, "int32")
    # Operands are little-endian packs of four signed INT8 values.
    unroll = 4 if depth % 16 == 0 else 1
    p.label("k")
    for word in range(unroll):
        p.load(12, 11, "int32", 4 * word)
        p.load(13, 10, "int32", 4 * word)
        p.emit(13 << 20 | 12 << 15 | 1 << 12 | 14 << 7 | 0x2B)
        p.arithmetic(7, 7, 14)
    p.addi(10, 10, 4 * unroll)
    p.addi(11, 11, 4 * unroll)
    p.branch(1, 10, 9, "k")
    p.load(8, 6, "int32", 4)
    emit_requant(p)
    if lut is not None:
        p.addi(9, 7, 255, f3=7)
        p.li(6, lut)
        p.arithmetic(6, 6, 9)
        p.load(7, 6, "int8")
    p.li(6, result)
    p.arithmetic(6, 6, 5)
    p.store(7, 6, "int8")
    p.addi(5, 5, lanes)
    p.li(6, stop)
    p.branch(6, 5, 6, "output")


def tile_shape(pixels, out_channels, image_bytes, channels, taps, gather, lut):
    """Find a spatial tile and an output-channel block inside a G3 data window."""
    depth = channels * taps
    source = align_dma(image_bytes)
    base = source + (align_dma(channels) if gather else 0) + (256 if lut else 0)
    unit = 32 // gcd(32, out_channels)
    if gather:
        unit = max(unit, 32 // gcd(32, taps * 4))
    spatial_limit = max(unit, min(32, align_dma(pixels)))
    for pixel_count in range(spatial_limit, unit - 1, -unit):
        occupied = base
        if gather:
            occupied += align_dma(pixel_count * depth) + align_dma(pixel_count * taps * 4)
        block = min(out_channels, (DATA_WINDOW_BYTES - occupied) // (depth + 8 + pixel_count))
        if block < out_channels:
            if out_channels % 32 and pixels != 1:
                continue
            block = block // 32 * 32
        if block > 0 and (occupied + align_dma(block * depth) + align_dma(block * 8) +
                          align_dma(block * pixel_count) <= DATA_WINDOW_BYTES):
            return pixel_count, block
    raise ValueError("dense classifier convolution does not fit a G3 data window")


def dot_slices(outputs, depth):
    """Fence long DOT4 waves before the fixed C99 compute poll limit."""
    outputs_per_wave = 64
    budget_macs = 1_200_000
    chunk = max(outputs_per_wave, budget_macs // depth // outputs_per_wave * outputs_per_wave)
    for first in range(0, outputs, chunk):
        yield first, min(first + chunk, outputs)


def _tiles(name, src, coef, params, map_source, lut_source, dst, *,
           height, width, output_pixels, ci, co, taps, tile_pixels, tile_channels):
    depth, image_bytes = taps * ci, height * width * ci
    tiles, resident = [], [None, None]
    for channel in range(0, co, tile_channels):
        channels = min(tile_channels, co - channel)
        for pixel in range(0, output_pixels, tile_pixels):
            pixels = min(tile_pixels, output_pixels - pixel)
            zone = len(tiles) % 2
            scratch = Scratch(zone)
            x = scratch.reserve(image_bytes)
            if map_source is not None:
                zero = scratch.reserve(ci)
                mapping = scratch.reserve(tile_pixels * taps * 4)
                gathered = scratch.reserve(tile_pixels * depth)
                a = gathered.offset
            else:
                a = x.offset + pixel * ci
            lut = scratch.reserve(256).offset if lut_source is not None else None
            w = scratch.reserve(tile_channels * depth)
            b = scratch.reserve(tile_channels * 8)
            c = scratch.reserve(tile_pixels * tile_channels)
            if scratch.used > DATA_WINDOW_BYTES:
                raise ValueError(f"{name}: tile exceeded the verified G3 local window")
            loads = []
            if resident[zone] is None:
                loads.append(Transfer("load", src, x.offset, x.capacity))
                if map_source is not None:
                    map_bytes = align_dma(output_pixels * taps * 4)
                    loads.append(Transfer("load", map_source + map_bytes, zero.offset, zero.capacity))
                if lut is not None:
                    loads.append(Transfer("load", lut_source, lut, 256))
            if resident[zone] != channel:
                loads.extend((Transfer("load", coef + channel * depth, w.offset, align_dma(channels * depth)),
                              Transfer("load", params + channel * 8, b.offset, align_dma(channels * 8))))
                resident[zone] = channel
            label = f"{name}/p{pixel}_{pixel+pixels}/c{channel}_{channel+channels}"
            stages = []
            if map_source is not None:
                bytes_ = align_dma(pixels * taps * 4)
                loads.append(Transfer("load", map_source + pixel * taps * 4, mapping.offset, bytes_))
                words = pixels * depth // 4
                lanes = min(64, words)
                stages.append(build_gpu_stage(
                    label + "/gather", scratch.base // PAGE_BYTES, lanes,
                    lambda p: _gather(p, mapping.offset - scratch.base, a - scratch.base, ci, words, lanes),
                ))
            size = pixels * channels
            for first, stop in dot_slices(size, depth):
                lanes = min(64, stop - first)
                suffix = "/dot4" if stop - first == size else f"/dot4-{first}-{stop}"
                stages.append(build_gpu_stage(
                    label + suffix, scratch.base // PAGE_BYTES, lanes,
                    lambda p: _dot(p, a - scratch.base, w.offset - scratch.base, b.offset - scratch.base,
                                   c.offset - scratch.base, depth, channels, first, stop, lanes,
                                   lut - scratch.base if lut is not None else None),
                ))
            if channels == co or output_pixels == 1:
                stores = (Transfer("store", dst + pixel * co + channel, c.offset, align_dma(size)),)
            else:
                stores = tuple(Transfer("store", dst + (pixel + i) * co + channel,
                                        c.offset + i * channels, channels) for i in range(pixels))
            tiles.append(Tile(zone, tuple(loads), tuple(stages), stores, scratch.used))
    return tuple(tiles)


def lower_gpu_conv(node, ctx):
    """Prepare dense C6 Conv operands and a classifier-compatible pure-GPU schedule."""
    name, attrs = node["id"], node["attrs"]
    source, target = node["inputs"][0], node["outputs"][0]
    _, height, width, ci = ctx.tensors[source]["physical_shape"]
    _, oh, ow, co = ctx.tensors[target]["physical_shape"]
    kh, kw = attrs["kernel"]
    sy, sx = attrs["strides"]
    pad = attrs["pads"][0]
    if (attrs["groups"] != 1 or kh != kw or kh not in (1, 3, 7) or sx != sy or
            attrs["pads"] != [pad] * 4 or attrs["dilations"] != [1, 1] or ci % 4):
        raise ValueError(f"{name}: unsupported dense G3 convolution")
    weight = ctx.constants[attrs["weight"]].transpose(3, 0, 1, 2)
    packed = np.zeros((co, kh, kw, ci), dtype=np.int8)
    packed[..., :weight.shape[-1]] = weight
    weights = ctx.constant(f"{name}/@a0-weight", packed.tobytes())
    raw_bias = ctx.constants[attrs["bias"]].astype("<i4")
    rq = np.asarray(attrs["mul"], dtype="<u4") | (np.asarray(attrs["shift"], dtype="<u4") << 16)
    metadata = np.stack((raw_bias.view("<u4"), rq), axis=-1).astype("<u4")
    params = ctx.constant(f"{name}/@a0-requant", metadata.tobytes())
    lut = (ctx.constant(f"{name}/@a0-lut", np.asarray(attrs["lut"], dtype=np.int8).tobytes())
           if "lut" in attrs else None)
    gather = (kh, sy, pad) != (1, 1, 0)
    length = height * width * ci
    if gather:
        mapping = pixel_map(height, width, oh, ow, ci, kh, sy, pad)
        mapping[mapping == length] = align_dma(length)
        raw = mapping.tobytes()
        map_tensor = ctx.constant(f"{name}/@a0-gather", raw.ljust(align_dma(len(raw)) + align_dma(ci), b"\0"))
    else:
        map_tensor = None
    pixels, channels = tile_shape(oh * ow, co, length, ci, kh * kw, gather, lut is not None)

    def schedule(slots):
        return (_tiles(name, slots[source].offset, slots[weights].offset, slots[params].offset,
                       slots[map_tensor].offset if map_tensor is not None else None,
                       slots[lut].offset if lut is not None else None, slots[target].offset,
                       height=height, width=width, output_pixels=oh * ow, ci=ci, co=co,
                       taps=kh * kw, tile_pixels=pixels, tile_channels=channels),)
    return schedule
