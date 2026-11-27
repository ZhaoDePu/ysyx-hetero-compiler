"""Phase 17 implicit-Conv chains with spatial and output-channel tiling."""

from itertools import accumulate

from .c14_conv import CONTINUE_K, DEFER_K, chain_depths, pack_chain_weights
from .c6_memory import ALIGNMENT, HALF_BYTES, Scratch, align_dma
from .c6_tensor import Conv, NativeStage, Tile, Transfer, _halo, channel_configuration
from .c6_vector import EW_CONCAT, EW_COPY, EW_SPLIT, _ew
from .npu import Z15_CTRL, Z15_NPU_START


def panels(spec):
    return tuple(min(16, spec.output_channels - start)
                 for start in range(0, spec.output_channels, 16))


def _extents(spec, first, rows):
    _, ow = spec.output_hw
    if ow <= 16:
        y, h, roi = _halo(spec, first, rows)
        return y, h, roi, 0
    y0 = first * spec.stride - spec.pad
    height = (rows - 1) * spec.stride + spec.kernel
    start = max(y0, 0)
    stop = min(y0 + height, spec.height)
    return start, stop - start, 0, height


def split_1x1(spec):
    return (spec.kernel == 1 and spec.stride == 1 and not spec.pad
            and spec.output_hw[1] <= 16 and spec.input_channels > 511
            and len(chain_depths(spec.depth)) == 2)


def _regions(spec, first, rows, weight_bytes, zone):
    _, ow = spec.output_hw
    out_pitch = spec.output_pitch
    max_w = min(16, ow)
    _, orig_rows, roi, crop_rows = _extents(spec, first, rows)
    if orig_rows > 511 or crop_rows > 511 or (not crop_rows and roi + rows > 511):
        raise ValueError(f"{spec.name}: shape or slice exceeds the nine-bit geometry fields")
    ci = spec.padded_channels
    crop_pitch = ((max_w - 1) * spec.stride + spec.kernel) * ci
    scratch = Scratch(zone)
    weights = scratch.reserve(weight_bytes)
    source = scratch.reserve(orig_rows * spec.input_pitch)
    split = scratch.reserve(source.size) if split_1x1(spec) else None
    crop = scratch.reserve(crop_rows * crop_pitch) if crop_rows else None
    zero = scratch.reserve(crop_pitch) if crop_rows and spec.pad else None
    processed = rows * max_w * spec.output_channels
    a = scratch.reserve(processed)
    b = scratch.reserve(processed) if spec.output_channels > 16 else None
    output = scratch.reserve(rows * out_pitch) if ow > 16 else None
    return scratch, weights, source, crop, zero, a, b, output, split


def pack_spatial_weights(spec, weights, bias):
    """Store each 16-column panel's original bias and contiguous K chunks."""
    blob = bytearray()
    descriptors = []
    first = 0
    for n in panels(spec):
        child = Conv(spec.name, spec.height, spec.width, spec.input_channels, n,
                     spec.kernel, spec.stride, spec.pad)
        bias_base = len(blob)
        bias_data = bias[first:first + n].astype("<i4", copy=False).tobytes()
        blob.extend(bias_data)
        chunks = []
        for data in pack_chain_weights(child, weights[..., first:first + n]):
            chunks.append(len(blob))
            blob.extend(data)
        descriptors.append((bias_base, tuple(chunks)))
        first += n
    return bytes(blob), tuple(descriptors)


def row_plan(spec, weight_bytes):
    """Pick the largest row slab that fits one half and has aligned DMA boundaries."""
    oh, _ = spec.output_hw
    if (spec.input_pitch % ALIGNMENT or spec.output_pitch % ALIGNMENT or spec.depth > 8192
            or (spec.input_channels > 511 and not split_1x1(spec))):
        raise ValueError(f"{spec.name}: unsupported DMA stride or K range")
    result = []
    first = 0
    while first < oh:
        for rows in range(min(oh - first, 1023), 0, -1):
            try:
                _regions(spec, first, rows, weight_bytes, len(result) % 2)
            except ValueError:
                continue
            break
        else:
            raise ValueError(f"{spec.name}: no aligned Phase 17 macro tile fits into 1 MiB")
        result.append((first, rows))
        first += rows
    return tuple(result)


def needs_zero_row(spec):
    _, ow = spec.output_hw
    return (((min(16, ow) - 1) * spec.stride + spec.kernel) * spec.padded_channels
            if ow > 16 and spec.pad else 0)


def _copy_crop(stages, label, spec, src, crop, zero, first, rows, x, width):
    """Materialize an explicit halo while preserving global stride phase."""
    x0 = x * spec.stride - spec.pad
    xspan = (width - 1) * spec.stride + spec.kernel
    y0 = first * spec.stride - spec.pad
    yspan = (rows - 1) * spec.stride + spec.kernel
    ci = spec.padded_channels
    pitch = xspan * ci
    x_start = max(0, x0)
    left = x_start - x0
    middle = min(spec.width, x0 + xspan) - x_start
    right = xspan - left - middle
    for row in range(yspan):
        y = y0 + row
        where = crop + row * pitch
        prefix = f"{label}/gather{row}"
        if not 0 <= y < spec.height:
            stages.append(_ew(prefix, EW_COPY, zero, where, length=pitch))
            continue
        if left:
            stages.append(_ew(prefix + "/left", EW_COPY, zero, where, length=left * ci))
        start = src + (y - max(0, y0)) * spec.input_pitch + x_start * ci
        stages.append(_ew(prefix + "/valid", EW_COPY, start, where + left * ci, length=middle * ci))
        if right:
            stages.append(_ew(prefix + "/right", EW_COPY, zero, where + (left + middle) * ci, length=right * ci))
    return yspan, xspan


def _chain(stages, label, spec, *, count, size, a, b, c, descriptors, shape, roi,
           mul, shift, lut, pad, initial_lut, setup, split_a):
    bias, offsets = descriptors
    flags = (1 << 9 | int(spec.kernel == 3) << 10 | int(spec.stride == 2) << 11 |
             pad << 12 | 1 << 13 | 1 << 14)
    if split_a:
        flags = 0
    table = channel_configuration(count, mul, shift, lut if initial_lut else None) if setup else ()
    pos = 0
    for index, (depth, weight) in enumerate(zip(chain_depths(spec.depth), offsets, strict=True)):
        final = index + 1 == len(offsets)
        mode = flags | (CONTINUE_K if index else 1 << 1)
        mode |= ((1 << 2 | 1 << 15 | int(lut is not None) << 16 | 1 << 17) if final else DEFER_K)
        stages.append(NativeStage(f"{label}/chain{index}", "NPU", (table if not index else ()) + (
            (0x60, size | count << 16), (0x64, depth | 128 << 16), (0x68, split_a[index] if split_a else a),
            (0x6C, b + weight), (0x70, c), (0x74, b + bias), (0x78, mode),
            (0x7C, 0 if split_a else shape), (0x80, 0 if split_a else roi), (0x84, 0 if split_a else pos), (Z15_CTRL, Z15_NPU_START))))
        pos += depth


def _join(stages, label, start, auxiliary, widths, h, w):
    count = h * w
    step = 0
    while len(widths) > 1:
        positions = [0, *accumulate(widths)]
        result = []
        for i in range(0, len(widths), 2):
            offset = positions[i] * count
            a = start + offset
            c = auxiliary + offset
            n = widths[i]
            stage = f"{label}/join{step}_{i}"
            if i + 1 == len(widths):
                stages.append(_ew(stage, EW_COPY, a, c, length=n * count))
            else:
                other = widths[i + 1]
                stages.append(_ew(stage, EW_CONCAT, a, c, b=a + n * count,
                                  h=h, w=w, ca=n, cb=other))
                n += other
            result.append(n)
        widths = result
        start, auxiliary = auxiliary, start
        step += 1
    return start


def spatial_tiles(spec, *, input_ddr, output_ddr, weights_ddr, weight_bytes, blocks,
                  zero_ddr, mul, shift, lut):
    """Pack and retain geometrically legal NPU chains within independently owned halves."""
    _, ow = spec.output_hw
    channels = panels(spec)
    result = []
    for first, rows in row_plan(spec, weight_bytes):
        zone = len(result) % 2
        scratch, weights, src, crop, zero, a, b, out, split = _regions(spec, first, rows, weight_bytes, zone)
        y, h, local_first, _ = _extents(spec, first, rows)
        loads = [Transfer("load", input_ddr + y * spec.input_pitch, src.offset, src.capacity),
                 Transfer("load", weights_ddr, weights.offset, weights.capacity)]
        if zero:
            loads.append(Transfer("load", zero_ddr, zero.offset, zero.capacity))
        stages = []
        if split:
            chunks = chain_depths(spec.depth)
            a0, a1 = split.offset, split.offset + rows * ow * chunks[0]
            stages.append(_ew(f"{spec.name}/rows{first}/split_k", EW_SPLIT, src.offset, a0, d=a1,
                              h=rows, w=ow, ca=chunks[0], cb=chunks[1]))
        for x in range(0, ow, 16):
            w = min(16, ow - x)
            label = f"{spec.name}/rows{first}_{first+rows}/width{x}_{x+w}"
            if crop:
                ph, pw = _copy_crop(stages, label, spec, src.offset, crop.offset,
                                    zero.offset if zero else 0, first, rows, x, w)
                base, pad = crop.offset, 0
            else:
                ph, pw, base, pad = h, spec.width, src.offset, spec.pad
            shape = ph | pw << 9 | spec.input_channels << 18
            roi_rows = 16 // w
            channel = 0
            for panel, n in enumerate(channels):
                for row in range(0, rows, roi_rows):
                    m = min(roi_rows, rows - row)
                    split_a = (a0 + row * ow * chunks[0], a1 + row * ow * chunks[1]) if split else ()
                    _chain(stages, f"{label}/oc{channel}/y{row}", spec, count=n, size=m*w,
                           a=base, b=weights.offset, c=a.offset + (channel*rows + row*n)*w,
                           descriptors=blocks[panel], shape=shape, roi=(local_first+row) | m<<16,
                           mul=mul[channel:channel+n], shift=shift[channel:channel+n], lut=lut, pad=pad,
                           initial_lut=not result and not x and not panel and not row, setup=not row, split_a=split_a)
                channel += n
            joined = _join(stages, label, a.offset, b.offset if b else 0, channels, rows, w)
            if out:
                for row in range(rows):
                    stages.append(_ew(f"{label}/scatter{row}", EW_COPY, joined + row*w*spec.output_channels,
                                      out.offset + row*spec.output_pitch + x*spec.output_channels,
                                      length=w*spec.output_channels))
        output = out.offset if out else joined
        stores = (Transfer("store", output_ddr + first*spec.output_pitch, output,
                           align_dma(rows*spec.output_pitch)),)
        assert scratch.used <= HALF_BYTES
        result.append(Tile(zone, tuple(loads), tuple(stages), stores, scratch.used))
    return tuple(result)
