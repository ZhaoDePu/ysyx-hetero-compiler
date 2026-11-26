"""Packed log-normalizer lowering for Q23 attention on the frozen GPU."""

from dataclasses import dataclass
from math import ceil, exp, gcd, isfinite, log, prod
from struct import pack

import numpy as np

from .c6_gpu import PAGE_BYTES, build_gpu_stage
from .c6_memory import ALIGNMENT, Scratch, align_dma
from .c6_tensor import Tile, Transfer

_DENOMINATOR_SHIFT = 15
_BASE = 1 << 23


@dataclass(frozen=True)
class LogSoftmaxTables:
    payload: bytes
    thresholds: tuple
    probabilities: bytes
    probability_offset: int
    index_shift: int


def log_softmax_tables(weights, scale, steps, columns):
    """Build shared Q23 exp, uint16 denominator and uint8 score tables."""
    weights = tuple(int(w) for w in weights)
    if (len(weights) != 256 or weights[0] != _BASE or not 0 < scale or not isfinite(scale)
            or min(weights) < 0 or max(weights) > _BASE or steps not in (2, 4, 8)
            or not 4 <= columns <= 128 or columns % 4):
        raise ValueError("invalid packed log-softmax specification")
    intervals = ceil(steps * log(columns) / scale)
    scores = ceil(steps * log(512) / scale)
    if intervals > PAGE_BYTES // 2 or scores > PAGE_BYTES:
        raise ValueError("softmax tables exceed the frozen data page")
    divisor = 1 << _DENOMINATOR_SHIFT
    thresholds = tuple(ceil((_BASE / divisor) * exp((i + .5) * scale / steps))
                       for i in range(intervals))
    if any(th > 65535 for th in thresholds):
        raise ValueError("packed log-softmax threshold overflows uint16")
    probabilities = bytes(min(255, round(256 * exp(-i * scale / steps))) for i in range(scores))
    raw_thresholds = pack(f"<{intervals}H", *thresholds)
    offset = 1024 + align_dma(len(raw_thresholds))
    result = pack("<256I", *weights) + raw_thresholds
    result += bytes(offset - len(result)) + probabilities
    result += bytes(align_dma(len(result)) - len(result))
    return LogSoftmaxTables(result, thresholds, probabilities, offset, steps.bit_length() - 1)


def log_softmax(scores, axis, weights, scale, steps):
    """Reference the same integer denominator bins as the frozen kernel."""
    table = log_softmax_tables(weights, scale, steps, scores.shape[axis])
    delta = np.max(scores, axis=axis, keepdims=True).astype(np.int16) - scores
    totals = np.asarray(weights, dtype=np.int64)[delta].sum(axis=axis, keepdims=True)
    bins = np.searchsorted(table.thresholds, totals >> _DENOMINATOR_SHIFT, side="right")
    lookup = np.frombuffer(table.probabilities + b"\0", dtype=np.uint8)
    return lookup[np.minimum((delta.astype(np.int32) << table.index_shift) + bins, len(lookup) - 1)]


def _signed_byte(p, dst, word, lane):
    if lane != 3:
        p.addi(dst, word, 24 - 8 * lane, f3=1)
        word = dst
    p.addi(dst, word, 24 | 0x400, f3=5)


def _log_kernel(columns, table):
    """Compute one row per lane with word transfers and one shared warp."""
    def emit(p):
        p.li(6, columns)
        p.arithmetic(6, 5, 6, f7=1)
        p.li(15, len(table.payload))
        p.arithmetic(6, 6, 15)
        p.addi(8, 6, columns)
        p.addi(7, 6, 0)
        p.li(9, -128)
        p.label("max_word")
        p.load(10, 7, "int32")
        for lane in range(4):
            _signed_byte(p, 13, 10, lane)
            p.branch(5, 9, 13, f"max_next{lane}")
            p.addi(9, 13, 0)
            p.label(f"max_next{lane}")
        p.addi(7, 7, 4)
        p.branch(6, 7, 8, "max_word")

        p.li(11, 0)
        p.addi(7, 6, 0)
        p.label("sum_word")
        p.load(10, 7, "int32")
        for lane in range(4):
            _signed_byte(p, 13, 10, lane)
            p.arithmetic(13, 9, 13, f7=0x20)
            p.addi(13, 13, 2, f3=1)
            p.load(13, 13, "int32")
            p.arithmetic(11, 11, 13)
        p.addi(7, 7, 4)
        p.branch(6, 7, 8, "sum_word")

        p.addi(11, 11, _DENOMINATOR_SHIFT, f3=5)
        p.li(13, 0)
        p.li(14, len(table.thresholds))
        p.label("find_bin")
        p.branch(7, 13, 14, "bin_found")
        p.arithmetic(15, 13, 14)
        p.addi(15, 15, 1, f3=5)
        p.addi(10, 15, 1, f3=1)
        p.emit(1024 << 20 | 10 << 15 | 5 << 12 | 10 << 7 | 0x03)  # LHU x10, 1024(x10).
        p.branch(6, 11, 10, "lower_bin")
        p.addi(13, 15, 1)
        p.jump("find_bin")
        p.label("lower_bin")
        p.addi(14, 15, 0)
        p.jump("find_bin")

        p.label("bin_found")
        p.addi(7, 6, 0)
        p.li(12, table.probability_offset)
        p.li(14, len(table.probabilities))
        p.label("normalize_word")
        p.load(5, 7, "int32")
        p.li(10, 0)
        for lane in range(4):
            _signed_byte(p, 11, 5, lane)
            p.arithmetic(11, 9, 11, f7=0x20)
            p.addi(11, 11, table.index_shift, f3=1)
            p.arithmetic(11, 11, 13)
            p.branch(7, 11, 14, f"store_next{lane}")
            p.arithmetic(11, 11, 12)
            p.load(15, 11, "uint8")
            if lane:
                p.addi(15, 15, lane * 8, f3=1)
            p.arithmetic(10, 10, 15, f3=6)
            p.label(f"store_next{lane}")
        p.store(10, 7, "int32")
        p.addi(7, 7, 4)
        p.branch(6, 7, 8, "normalize_word")
    return emit


def lower_log_softmax(name, source, output, lookup, shape, table):
    """Emit aligned row tiles, overwriting each input word after the sum pass."""
    columns, rows = shape[-1], prod(shape[:-1])
    if (not rows or not 4 <= columns <= 128 or columns % 4
            or any(at < 0 or at % ALIGNMENT for at in (source, output, lookup))):
        raise ValueError(f"{name}: invalid packed log-softmax extent")
    granularity = ALIGNMENT // gcd(ALIGNMENT, columns)
    tile_rows = (PAGE_BYTES - len(table.payload)) // columns // granularity * granularity
    if not tile_rows:
        raise ValueError(f"{name}: aligned Softmax rows do not fit the frozen data page")
    result = []
    for first in range(0, rows, tile_rows):
        group = min(rows - first, tile_rows)
        nbytes = align_dma(group * columns)
        zone = len(result) % 2
        scratch = Scratch(zone)
        page = scratch.reserve(PAGE_BYTES)
        data = page.offset + len(table.payload)
        stage = build_gpu_stage(f"{name}.{first}", page.offset // PAGE_BYTES, group, _log_kernel(columns, table))
        loads = (Transfer("load", source + first * columns, data, nbytes),
                 Transfer("load", lookup, page.offset, len(table.payload)))
        stores = (Transfer("store", output + first * columns, data, nbytes),)
        result.append(Tile(zone, loads, (stage,), stores, scratch.used))
    return tuple(result)
