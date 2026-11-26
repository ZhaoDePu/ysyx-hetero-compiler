"""Use the frozen tensor epilogue to finish deep Conv reductions exactly."""

from math import gcd, lcm

from .c6_gpu import DATA_WINDOW_BYTES, PAGE_BYTES, build_gpu_stage
from .c6_memory import ALIGNMENT, HALF_BYTES, Scratch, align_dma
from .c6_reduce_conv import raw_conv_tiles, reduction_widths
from .c6_tensor import NativeStage, Tile, Transfer, channel_configuration
from .npu import Z15_CTRL, Z15_NPU_START


def native_combine_width(spec):
    """Return a feasible width with the same channel phase in every tile."""
    parts = len(reduction_widths(spec))
    if parts < 2:
        return 0
    granularity = lcm(spec.output_channels, ALIGNMENT)
    limit = min(511, PAGE_BYTES // (4 * parts))
    return limit // granularity * granularity


def _sum_partials(p, offsets, blocks):
    p.addi(6, 5, 2, f3=1)
    stride = 16 * 4
    # Bias the base just enough to reach the last word using a signed
    # 12-bit immediate. Mask the encoded field before calling Program.
    base = max(0, offsets[-1] + (blocks - 1) * stride - 2044)
    if base:
        p.addi(6, 6, base)
    for block in range(blocks):
        adjustment = block * stride - base
        p.load(7, 6, "int32", adjustment & 4095)
        for offset in offsets[1:]:
            p.load(8, 6, "int32", (adjustment + offset) & 4095)
            p.arithmetic(7, 7, 8)
        # Different lanes own disjoint words. The first partial is dead
        # after the corresponding lane reads it, so overwrite it in place.
        p.store(7, 6, "int32", adjustment & 4095)


def native_combine_tiles(spec, *, output_ddr, partials_ddr, zero_ddr, mul, shift, lut=None, warp_skew=False):
    """Reduce i32 partials on SIMT and reuse the NPU's exact packed epilogue.

    M=1 and K=4 with zero operands make the tensor product zero. The
    accumulated words become the bias row, so the native scale, shift,
    clipping and optional lookup are identical to the original convolution.
    """
    width = native_combine_width(spec)
    if not width or len(partials_ddr) != len(reduction_widths(spec)):
        raise ValueError(f"{spec.name}: native epilogue cannot fit the reduction")
    co = spec.output_channels
    if len(mul) != co or len(shift) != co:
        raise ValueError("one multiplier and shift required per output channel")
    copies = width // co
    config = channel_configuration(width, tuple(mul) * copies, tuple(shift) * copies, lut)
    flags = 1 << 1 | 1 << 2 | 1 << 15 | int(lut is not None) << 16 | 1 << 17
    count = spec.output_hw[0] * spec.output_pitch
    result = []
    for first in range(0, count, width):
        items = min(width, count - first)
        zone = len(result) % 2
        scratch = Scratch(zone)
        bias_words = [scratch.reserve(4 * items) for _ in partials_ddr]
        zeros = scratch.reserve(4 * width + 4)
        output = scratch.reserve(items)
        loads = [
            Transfer("load", ddr_at + 4 * first, span.offset, span.capacity)
            for ddr_at, span in zip(partials_ddr, bias_words)
        ]
        # A/B stay at invariant addresses within a half for all full tiles.
        # A short final tile repacks its partials, so reload the zero operands.
        if len(result) < 2 or items != width:
            loads.append(Transfer("load", zero_ddr, zeros.offset, zeros.capacity))
        prefix = f"{spec.name}/reduce_{first}_{first + items}"
        offsets = tuple(span.offset - scratch.base for span in bias_words)
        # Process contiguous sixteen-word blocks without losing one of the
        # four warps on a full tile. Preserve the one-block narrow tail.
        if items >= 256 and items % 64 == 0:
            blocks = 4
        else:
            blocks = 2 if items % 32 == 0 else 1
        gpu = build_gpu_stage(f"{prefix}/sum", scratch.base // PAGE_BYTES, items // blocks,
                              lambda program: _sum_partials(program, offsets, blocks),
                              index_stride=blocks, warp_skew=warp_skew)
        native = NativeStage(f"{prefix}/epilogue", "NPU", (config if not result else ()) + (
            (0x60, 1 | items << 16), (0x64, 4 | 16 << 16),
            (0x68, zeros.offset), (0x6C, zeros.offset + 4), (0x70, output.offset),
            (0x74, bias_words[0].offset), (0x78, flags), (0x7C, 0), (0x80, 0),
            (0x84, 0), (Z15_CTRL, Z15_NPU_START),
        ))
        stores = (Transfer("store", output_ddr + first, output.offset, output.capacity),)
        result.append(Tile(zone, tuple(loads), (gpu, native), stores, scratch.used))
    return tuple(result)



def scratch_combine_width(spec):
    """Return an aligned bias width when every minimum-size raw tile can fit."""
    co = spec.output_channels
    parts = reduction_widths(spec)
    granularity = lcm(co, ALIGNMENT)
    width = 511 // granularity * granularity
    oh, ow = spec.output_hw
    rows = min(oh, ALIGNMENT // gcd(spec.output_pitch, ALIGNMENT))
    partials = len(parts) * align_dma(4 * rows * spec.output_pitch)
    if not width or len(parts) < 2 or rows * ow > 511 or partials > DATA_WINDOW_BYTES:
        return 0
    # Bound the halo and address prefix for any aligned minimum-size tile.
    height = min(spec.height, rows * spec.stride + spec.kernel - 1)
    pixels = height * spec.width
    source = align_dma(ALIGNMENT - 1 + height * spec.input_pitch)
    weights = sum(align_dma(part * spec.kernel ** 2 * co) for part in parts)
    partial_base = (align_dma(4 * co) + weights + source + PAGE_BYTES - 1) // PAGE_BYTES * PAGE_BYTES
    remaining = spec.padded_channels
    split = 0
    for part in parts[:-1]:
        remaining -= part
        split += align_dma(pixels * part) + align_dma(pixels * remaining)
    total = partial_base + partials + split + align_dma(4 * width + 4) + align_dma(rows * spec.output_pitch)
    return width if total <= HALF_BYTES else 0


def _sum_scratch(p, parts, stride, items, lanes):
    p.li(9, stride)
    p.li(11, 4 * items)
    p.addi(6, 5, 2, f3=1)
    p.label("sum")
    p.load(7, 6, "int32")
    p.arithmetic(10, 6, 9)
    for part in range(1, parts):
        p.load(8, 10, "int32")
        p.arithmetic(7, 7, 8)
        if part + 1 < parts:
            p.arithmetic(10, 10, 9)
    p.store(7, 6, "int32")
    p.addi(6, 6, 4 * lanes)
    p.branch(6, 6, 11, "sum")  # BLTU: skip partial final warp iterations.


def scratch_combine_tiles(spec, *, input_ddr, output_ddr, weights_ddr, bias_ddr,
                          zero_ddr, mul, shift, lut=None, warp_skew=False):
    """Keep raw planes in one ownership half and sum directly into the bias plane."""
    width = scratch_combine_width(spec)
    co = spec.output_channels
    if not width or len(mul) != co or len(shift) != co:
        raise ValueError(f"{spec.name}: invalid SRAM reduction epilogue")
    copies = width // co
    config = channel_configuration(width, tuple(mul) * copies, tuple(shift) * copies, lut)
    flags = 1 << 1 | 1 << 2 | 1 << 15 | int(lut is not None) << 16 | 1 << 17

    def finish(scratch, first, rows, index, parts):
        count = rows * spec.output_pitch
        start = first * spec.output_pitch
        bias = parts[0]
        zeros = scratch.reserve(4 * width + 4)
        output = scratch.reserve(count)
        loads = (Transfer("load", zero_ddr, zeros.offset, zeros.capacity),)
        lanes = min(count, 64)
        gpu = build_gpu_stage(f"{spec.name}/reduce_{start}_{start + count}/sum",
                              bias.offset // PAGE_BYTES, lanes,
                              lambda p: _sum_scratch(p, len(parts), bias.capacity, count, lanes),
                              warp_skew=warp_skew)
        stages = [gpu]
        for begin in range(0, count, width):
            items = min(width, count - begin)
            writes = config if index == 0 and begin == 0 else ()
            stages.append(NativeStage(f"{spec.name}/reduce_{start + begin}_{start + begin + items}/epilogue",
                                     "NPU", writes + (
                (0x60, 1 | items << 16), (0x64, 4 | 16 << 16),
                (0x68, zeros.offset), (0x6C, zeros.offset + 4), (0x70, output.offset + begin),
                (0x74, bias.offset + 4 * begin), (0x78, flags), (0x7C, 0), (0x80, 0),
                (0x84, 0), (Z15_CTRL, Z15_NPU_START),
            )))
        store = Transfer("store", output_ddr + start, output.offset, output.capacity)
        return loads, stages, store

    return raw_conv_tiles(spec, input_ddr=input_ddr, weights_ddr=weights_ddr,
                          bias_ddr=bias_ddr, finish=finish)
