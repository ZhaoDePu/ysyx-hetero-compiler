"""Byte reindexing for logical ONNX layouts over the frozen Balanced tile ABI."""

from dataclasses import dataclass
from itertools import groupby
from math import gcd, prod
import struct

import numpy as np

from .c6_gpu import PAGE_BYTES, build_gpu_stage
from .c6_memory import ALIGNMENT, HALF_BYTES, align_dma
from .c6_vector import _TileBuilder, _ew, _row_windows, _validate_operand, vector_tiles

EW_TRANSPOSE = 9


def _axes(shape, layout):
    if layout == "row-major":
        return tuple(range(len(shape)))
    if layout == "NHWC" and len(shape) == 4:
        return (0, 2, 3, 1)
    raise ValueError(f"unsupported physical layout {layout!r} for rank {len(shape)}")


def _matrix_step(shape, permutation):
    """Recognize a batched rotation of two contiguous axis groups."""
    axes = tuple(i for i, n in enumerate(shape) if n != 1)
    permutation = tuple(i for i in permutation if shape[i] != 1)
    if permutation == axes:
        return ()
    prefix = next(i for i, (a, b) in enumerate(zip(axes, permutation, strict=True)) if a != b)
    leading, perm = axes[:prefix], permutation[prefix:]
    split = axes.index(perm[0])
    if perm != axes[split:] + axes[prefix:split]:
        return None
    rows = prod(shape[i] for i in axes[prefix:split])
    columns = prod(shape[i] for i in axes[split:])
    if max(rows, columns) > 1023:
        return None
    return ((prod(shape[i] for i in leading), rows, columns),)


def _transpose_stage(name, tile, a, c, batches, rows, cols):
    size = rows * cols
    for i in range(batches):
        offset = i * size
        tile.stages.append(_ew(f"{name}/matrix_{i}", EW_TRANSPOSE, a + offset, c + offset,
                               h=rows, ca=cols, cb=rows))


@dataclass(frozen=True)
class _NativeLayout:
    name: str
    elements: int
    matrices: tuple
    constant_data: bytes = b""

    def tiles(self, input_ddr, output_ddr, constant_ddr=None):
        for address in (input_ddr, output_ddr):
            _validate_operand(address, (self.elements,))
        if not self.matrices:
            return vector_tiles(self.name, "Copy", [(self.elements,)], [(self.elements,)],
                                [input_ddr], [output_ddr])
        if len(self.matrices) == 1:
            batches, rows, cols = self.matrices[0]
            record = rows * cols
            tiles = []
            for first, count in _row_windows(batches, (record,), batches,
                                              lambda _, n: 2 * align_dma(n * record)):
                tile = _TileBuilder(len(tiles))
                size = count * record
                a = tile.load(input_ddr + first * record, size)
                c = tile.scratch.reserve(size).offset
                _transpose_stage(f"{self.name}/batches_{first}_{first + count}", tile,
                                 a, c, count, rows, cols)
                tile.store(output_ddr + first * record, c, size)
                tiles.append(tile.end())
            return tuple(tiles)
        tile = _TileBuilder(0)
        a = tile.load(input_ddr, self.elements)
        for step, (batches, rows, columns) in enumerate(self.matrices):
            c = tile.scratch.reserve(self.elements).offset
            _transpose_stage(f"{self.name}/permute_{step}", tile, a, c, batches, rows, columns)
            a = c
        tile.store(output_ddr, a, self.elements)
        return (tile.end(),)


@dataclass(frozen=True)
class _GatherPiece:
    first: int
    count: int
    index: int
    ranges: tuple


def _gather_pieces(mapping):
    """Pack source footprints into 32-byte DMA runs for bounded GPU windows."""
    blob = bytearray()
    pieces = []
    first = 0
    while first < len(mapping):
        blocks = set()
        stop = first
        while stop < min(first + 512, len(mapping)):
            next_stop = min(stop + ALIGNMENT, len(mapping))
            expanded = blocks | {int(i) // ALIGNMENT for i in mapping[stop:next_stop]}
            count = next_stop - first
            if len(expanded) * ALIGNMENT + align_dma(4 * count) + align_dma(count) > PAGE_BYTES:
                break
            blocks, stop = expanded, next_stop
        count = stop - first
        if not count:
            raise ValueError("one aligned output chunk exceeds the GPU data-page capacity")
        blocks = sorted(blocks)
        # Group runs of atomically loadable, consecutive source blocks.
        runs = []
        for _, members in groupby(enumerate(blocks), key=lambda p: p[1] - p[0]):
            run = [blk for _, blk in members]
            runs.append((ALIGNMENT * run[0], ALIGNMENT * len(run)))
        destination_block = {src: i * ALIGNMENT for i, src in enumerate(blocks)}
        offsets = [destination_block[int(pos) // ALIGNMENT] + int(pos) % ALIGNMENT
                   for pos in mapping[first:stop]]
        pieces.append(_GatherPiece(first, count, len(blob), tuple(runs)))
        blob.extend(struct.pack(f"<{count}I", *offsets))
        blob.extend(bytes(align_dma(4 * count) - 4 * count))
        first = stop
    return tuple(pieces), bytes(blob)


def _copy_kernel(index, output):
    def kernel(p):
        p.address(6, 5, index, "int32")
        p.load(8, 6, "int32")
        p.load(7, 8, "uint8")
        p.address(6, 5, output, "uint8")
        p.store(7, 6, "uint8")
    return kernel


@dataclass(frozen=True)
class _GatherLayout:
    name: str
    elements: int
    pieces: tuple
    constant_data: bytes

    def tiles(self, input_ddr, output_ddr, constant_ddr=None):
        for address, size in ((input_ddr, self.elements), (output_ddr, self.elements),
                              (constant_ddr, len(self.constant_data))):
            _validate_operand(address, (size,))
        tiles = []
        for piece in self.pieces:
            tile = _TileBuilder(len(tiles))
            for start, count in piece.ranges:
                tile.load(input_ddr + start, count)
            index = tile.load(constant_ddr + piece.index, 4 * piece.count)
            output = tile.scratch.reserve(piece.count).offset
            base = tile.zone * HALF_BYTES
            stage = build_gpu_stage(f"{self.name}/bytes_{piece.first}_{piece.first + piece.count}",
                                    base // PAGE_BYTES, piece.count,
                                    _copy_kernel(index - base, output - base))
            tile.stages.append(stage)
            tile.store(output_ddr + piece.first, output, piece.count)
            tiles.append(tile.end())
        return tuple(tiles)


def plan_layout(name, op, input_shape, output_shape, *, input_layout="row-major",
                output_layout="row-major", perm=None):
    """Convert one static byte reshape/transpose from logical shape to physical lanes."""
    source, target = tuple(input_shape), tuple(output_shape)
    for shape in (source, target):
        _validate_operand(0, shape)
    count = prod(source)
    if prod(target) != count:
        raise ValueError(f"{name}: layout operator output shape changes the element count")
    src_axes, dst_axes = _axes(source, input_layout), _axes(target, output_layout)
    physical_source = tuple(source[i] for i in src_axes)
    to_logical = tuple(src_axes.index(i) for i in range(len(source)))
    if op == "Transpose":
        order = tuple(range(len(source) - 1, -1, -1)) if perm is None else tuple(perm)
        if tuple(sorted(order)) != tuple(range(len(source))) or target != tuple(source[i] for i in order):
            raise ValueError(f"{name}: transpose permutation disagrees with output shape")
        combined = tuple(src_axes.index(order[i]) for i in dst_axes)
        steps = [_matrix_step(physical_source, combined)]
    elif op == "Reshape":
        if source == target and src_axes == dst_axes:
            return _NativeLayout(name, count, ())
        steps = [_matrix_step(physical_source, to_logical), _matrix_step(target, dst_axes)]
    else:
        raise ValueError(f"{name}: unsupported layout operation {op}")
    if all(step is not None for step in steps):
        matrices = tuple(matrix for step in steps for matrix in step)
        if not matrices:
            return _NativeLayout(name, count, ())
        if len(matrices) == 1:
            batches, rows, columns = matrices[0]
            record = rows * columns
            chunk = min(batches, ALIGNMENT // gcd(record, ALIGNMENT))
            if 2 * align_dma(chunk * record) <= HALF_BYTES:
                return _NativeLayout(name, count, matrices)
        elif align_dma(count) * (1 + len(matrices)) <= HALF_BYTES:
            return _NativeLayout(name, count, matrices)
    data = np.arange(count, dtype=np.int32).reshape(physical_source).transpose(to_logical)
    logical = data.transpose(order) if op == "Transpose" else data.reshape(target)
    mapping = logical.transpose(dst_axes).reshape(-1)
    pieces, blob = _gather_pieces(mapping)
    return _GatherLayout(name, count, pieces, blob)
