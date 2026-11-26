"""Lower static attention matrices with frozen Z15 native dot and vector ops."""

from math import gcd, prod

from .c6_gpu import PAGE_BYTES, build_gpu_stage
from .c6_memory import ALIGNMENT, Scratch, align_dma
from .c6_tensor import NativeStage, Tile, Transfer, channel_configuration
from .npu import Z15_CTRL, Z15_NPU_START, Z15_VPU_START


def _transpose(name, source, destination, rows, cols):
    return NativeStage(name, "VPU", (
        (0xA0, source), (0xA8, destination), (0xB4, rows),
        (0xB8, cols | rows << 16), (0xBC, 9), (Z15_CTRL, Z15_VPU_START),
    ))


def lower_matmul(name, left, right, output, left_shape, right_shape, mul, shift, *,
                 unsigned_rhs=False, raw_output=False, rhs_pretransposed=False):
    """Map one batch of dense row-major byte matrices with one output scale."""
    m, k = left_shape[-2:]
    rk, n = right_shape[-2:][::-1] if rhs_pretransposed else right_shape[-2:]
    if (list(left_shape[:-2]) != list(right_shape[:-2]) or k != rk or
            any(not 4 <= dim <= 511 or dim % 4 for dim in (m, n, k))):
        raise ValueError(f"{name}: unsupported native attention matrix shapes")
    if rhs_pretransposed and not unsigned_rhs:
        raise ValueError(f"{name}: only unsigned PV accepts pre-transposed B")
    if raw_output and unsigned_rhs:
        raise ValueError(f"{name}: raw QK output requires signed operands")
    batch = prod(left_shape[:-2])
    if not batch or any(at < 0 or at % ALIGNMENT for at in (left, right, output)):
        raise ValueError(f"{name}: invalid matrix placement")
    scratch = Scratch(0)
    a = scratch.reserve(batch * m * k)
    b = scratch.reserve(batch * k * n)
    output_width = 4 if raw_output else 1
    c = scratch.reserve(batch * m * n * output_width)
    operand = scratch.reserve(k * n) if not rhs_pretransposed else None
    if unsigned_rhs:
        result_transposed = scratch.reserve(n * m)
    # NPU consumes unsigned bytes only on A. For V * P^T evaluate P * V^T
    # and transpose the output while keeping the same integer reduction.
    native_m, native_n = (n, m) if unsigned_rhs else (m, n)
    config = () if raw_output else channel_configuration(native_n, [mul] * native_n, [shift] * native_n)
    tile_k = 128
    partial = scratch.reserve(4 * native_m * native_n).offset if not raw_output and k > tile_k else 0
    stages = []
    for head in range(batch):
        label = f"{name}.{head}"
        sa = a.offset + head * m * k
        sb = b.offset + head * k * n
        sc = c.offset + head * m * n * output_width
        if unsigned_rhs:
            if operand is not None:
                stages.append(_transpose(f"{label}.prob-transpose", sb, operand.offset, k, n))
                sb = operand.offset
            sa, sb, sc = sb, sa, result_transposed.offset
        else:
            stages.append(_transpose(f"{label}.pack", sb, operand.offset, k, n))
            sb = operand.offset
        # Packed B is column-major. Native tile-K is a power of two.
        flags = 0 if raw_output else int(unsigned_rhs) | 1 << 2 | 1 << 15 | 1 << 17
        stages.append(NativeStage(f"{label}.dot", "NPU", config + (
            (0x60, native_m | native_n << 16), (0x64, k | tile_k << 16),
            (0x68, sa), (0x6C, sb), (0x70, sc), (0x74, 0),
            (0x78, flags), (0x7C, 0), (0x80, 0), (0x84, partial),
            (Z15_CTRL, Z15_NPU_START),
        )))
        if unsigned_rhs:
            stages.append(_transpose(f"{label}.unpack", sc, c.offset + head * m * n, n, m))
    loads = (Transfer("load", left, a.offset, a.capacity), Transfer("load", right, b.offset, b.capacity))
    stores = (Transfer("store", output, c.offset, c.capacity),)
    return (Tile(0, loads, tuple(stages), stores, scratch.used),)


def _q23_kernel(columns, output, lut):
    """One RV32EM lane computes the max, denominator and outputs for one row."""
    def emit(p):
        p.li(6, columns)
        p.arithmetic(6, 5, 6, f7=1)
        p.addi(8, 6, columns)
        p.addi(7, 6, 0)
        p.li(9, -128)
        p.label("max_loop")
        p.load(10, 7, "int8")
        p.branch(5, 9, 10, "max_next")
        p.addi(9, 10, 0)
        p.label("max_next")
        p.addi(7, 7, 1)
        p.branch(6, 7, 8, "max_loop")

        p.li(11, 0)
        p.li(12, lut)
        p.addi(7, 6, 0)
        p.label("sum_loop")
        p.load(10, 7, "int8")
        p.arithmetic(10, 9, 10, f7=0x20)
        p.addi(10, 10, 2, f3=1)
        p.arithmetic(10, 10, 12)
        p.load(13, 10, "int32")
        p.arithmetic(11, 11, 13)
        p.addi(7, 7, 1)
        p.branch(6, 7, 8, "sum_loop")

        p.addi(7, 6, 0)
        p.li(5, output)
        p.arithmetic(6, 6, 5)
        p.label("normalize")
        p.load(10, 7, "int8")
        p.arithmetic(10, 9, 10, f7=0x20)
        p.addi(10, 10, 2, f3=1)
        p.arithmetic(10, 10, 12)
        p.load(13, 10, "int32")
        p.addi(13, 13, 8, f3=1)
        p.arithmetic(14, 13, 11, f3=5, f7=1)  # DIVU
        p.arithmetic(15, 13, 11, f3=7, f7=1)  # REMU
        p.addi(15, 15, 1, f3=1)
        p.branch(6, 15, 11, "clip")
        p.branch(1, 15, 11, "round_up")
        p.addi(10, 14, 1, f3=7)
        p.branch(0, 10, 0, "clip")
        p.label("round_up")
        p.addi(14, 14, 1)
        p.label("clip")
        p.li(10, 255)
        p.branch(7, 10, 14, "write")
        p.addi(14, 10, 0)
        p.label("write")
        p.store(14, 6, "uint8")
        p.addi(6, 6, 1)
        p.addi(7, 7, 1)
        p.branch(6, 7, 8, "normalize")
    return emit


def lower_q23_softmax(name, source, output, lookup, shape):
    """Use a DDR-backed Q23 lookup table and page-local softmax rows."""
    columns, rows = shape[-1], prod(shape[:-1])
    if not 1 <= columns <= 128 or not rows or any(at < 0 or at % ALIGNMENT for at in (source, output, lookup)):
        raise ValueError(f"{name}: invalid Q23 softmax extent")
    # End nonfinal tiles on a 32-byte boundary, with independent GPU pages.
    granularity = ALIGNMENT // gcd(ALIGNMENT, columns)
    max_rows = (PAGE_BYTES - 1024) // (2 * columns)
    tile_rows = max_rows // granularity * granularity
    if not tile_rows:
        raise ValueError(f"{name}: a full DMA-aligned softmax row group does not fit")
    tiles = []
    for first in range(0, rows, tile_rows):
        group = min(rows - first, tile_rows)
        nbytes = align_dma(group * columns)
        zone = len(tiles) % 2
        scratch = Scratch(zone)
        page = scratch.reserve(PAGE_BYTES)
        a, c, table = page.offset, page.offset + nbytes, page.offset + 2 * nbytes
        stage = build_gpu_stage(f"{name}.{first}", page.offset // PAGE_BYTES, group,
                                _q23_kernel(columns, c - a, table - a))
        loads = (Transfer("load", source + first * columns, a, nbytes),
                 Transfer("load", lookup, table, 1024))
        stores = (Transfer("store", output + first * columns, c, nbytes),)
        tiles.append(Tile(zone, loads, (stage,), stores, scratch.used))
    return tuple(tiles)


def lower_vpu_softmax(name, source, output, shape, shift):
    """Widen INT8 scores on the vector lanes, then run frozen EW Softmax."""
    columns, rows = shape[-1], prod(shape[:-1])
    if (not 4 <= columns <= 128 or columns % 4 or not rows or not 0 <= shift <= 31
            or any(at < 0 or at % ALIGNMENT for at in (source, output))):
        raise ValueError(f"{name}: invalid native softmax extent")
    granularity = ALIGNMENT // gcd(ALIGNMENT, columns)
    tile_rows = PAGE_BYTES // (5 * columns) // granularity * granularity
    if not tile_rows:
        raise ValueError(f"{name}: native softmax score rows do not fit one GPU page")
    tiles = []
    for first in range(0, rows, tile_rows):
        group = min(rows - first, tile_rows)
        size = group * columns
        wide = align_dma(size)
        if wide + 4 * size > PAGE_BYTES:
            raise ValueError(f"{name}: widened score scratch exceeds GPU page")
        zone = len(tiles) % 2
        scratch = Scratch(zone)
        page = scratch.reserve(PAGE_BYTES)
        probability = scratch.reserve(size)

        def widen(p):
            p.load(7, 5, "int8")
            p.addi(6, 5, 2, f3=1)
            p.li(8, wide)
            p.arithmetic(6, 6, 8)
            p.store(7, 6, "int32")

        label = f"{name}.{first}"
        gpu = build_gpu_stage(label + ".widen", page.offset // PAGE_BYTES, size, widen)
        vpu = NativeStage(label, "VPU", (
            (0xA0, page.offset + wide), (0xA8, probability.offset), (0xB4, group | columns << 16),
            (0xB8, 0), (0xBC, 12 | shift << 24), (Z15_CTRL, Z15_VPU_START),
        ))
        loads = (Transfer("load", source + first * columns, page.offset, align_dma(size)),)
        stores = (Transfer("store", output + first * columns, probability.offset, probability.capacity),)
        tiles.append(Tile(zone, loads, (gpu, vpu), stores, scratch.used))
    return tuple(tiles)


def lower_vpu_softmax_words(name, source, output, shape, shift):
    """Consume signed INT32 QK accumulators directly with native EW Softmax."""
    columns, rows = shape[-1], prod(shape[:-1])
    if (not rows or not 4 <= columns <= 128 or columns % 4 or not 0 <= shift <= 31
            or any(at < 0 or at % ALIGNMENT for at in (source, output))):
        raise ValueError(f"{name}: invalid raw native softmax extent")
    granularity = ALIGNMENT // gcd(ALIGNMENT, columns)
    tile_rows = 511 // granularity * granularity
    tiles = []
    for first in range(0, rows, tile_rows):
        group = min(rows - first, tile_rows)
        size = group * columns
        zone = len(tiles) % 2
        scratch = Scratch(zone)
        scores = scratch.reserve(4 * size)
        probability = scratch.reserve(size)
        stage = NativeStage(f"{name}.{first}", "VPU", (
            (0xA0, scores.offset), (0xA8, probability.offset), (0xB4, group | columns << 16),
            (0xB8, 0), (0xBC, 12 | shift << 24), (Z15_CTRL, Z15_VPU_START),
        ))
        loads = (Transfer("load", source + 4 * first * columns, scores.offset, scores.capacity),)
        stores = (Transfer("store", output + first * columns, probability.offset, probability.capacity),)
        tiles.append(Tile(zone, loads, (stage,), stores, scratch.used))
    return tuple(tiles)
