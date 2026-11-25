"""Independent scalar evaluation for layout transformations and reductions."""

from decimal import Decimal, localcontext
from functools import lru_cache
import itertools
import math
import struct


def _coordinates(shape):
    return itertools.product(*(range(dimension) for dimension in shape))


def _offset(index, shape):
    offset = 0
    for position, size in zip(index, shape, strict=True):
        offset = offset * size + position
    return offset


@lru_cache(maxsize=32)
def _exponents_q23(numerator, denominator):
    # An independent high-precision oracle for the constant table. The
    # emitted kernel compares integer weight and remainder magnitudes.
    with localcontext() as context:
        context.prec = 60
        step = Decimal(-numerator) / denominator
        return tuple(round((step*i).exp() * (1 << 23)) for i in range(256))


def _softmax(data, shape, attrs):
    def f32(value):
        return struct.unpack("<f", struct.pack("<f", value))[0]

    dim = attrs["axis"]
    si = attrs["in_scale"][0] / attrs["in_scale"][1]
    so = attrs["out_scale"][0] / attrs["out_scale"][1]
    point = attrs["zero_point"]
    causal = attrs.get("causal_offset")
    table = _exponents_q23(*attrs["in_scale"]) if attrs.get("algorithm") == "q23" else None
    quantum = attrs["out_scale"][1]
    result = [point] * len(data)
    for before in _coordinates(shape[:dim]):
        active = shape[dim] if causal is None else min(shape[dim], before[-1] + causal + 1)
        for after in _coordinates(shape[dim+1:]):
            indexes = [_offset(before + (i,) + after, shape) for i in range(active)]
            highest = max(data[i] for i in indexes)
            changes = [highest - data[i] for i in indexes]
            if table is None:
                weights = [f32(math.exp(-si*d)) for d in changes]
                total = sum(weights)
                for i, weight in zip(indexes, weights, strict=True):
                    result[i] += round(f32(f32(weight / total) / so))
            else:
                weights = [table[d] for d in changes]
                total = sum(weights)
                for i, weight in zip(indexes, weights, strict=True):
                    q, rem = divmod(weight * quantum, total)
                    result[i] += q + (rem*2 > total or (rem*2 == total and q % 2))
    return result


def evaluate_layout(op, args, shapes, dest, attrs):
    data, src = args[0], shapes[0]
    if op in ("Reshape", "Cast"):
        return data[:]
    if op == "Transpose":
        inverse = [attrs["perm"].index(i) for i in range(len(src))]
        return [data[_offset([out[j] for j in inverse], src)] for out in _coordinates(dest)]
    if op == "Slice":
        return [data[_offset([a + b*c for a, b, c in zip(attrs["starts"], out, attrs["steps"], strict=True)], src)]
                for out in _coordinates(dest)]
    if op == "Resize":
        return [data[_offset([i*a//b for i, a, b in zip(out, src, dest, strict=True)], src)]
                for out in _coordinates(dest)]
    if op == "Expand":
        return [data[_offset([i if d != 1 else 0 for i, d in zip(
            out[len(dest)-len(src):], src, strict=True)], src)] for out in _coordinates(dest)]
    if op == "Concat":
        axis = attrs["axis"]
        prefix = math.prod(dest[:axis])
        step = math.prod(dest[axis+1:])
        result = []
        for i in range(prefix):
            for chunk, shape in zip(args, shapes, strict=True):
                block = shape[axis] * step
                result.extend(chunk[i*block:(i+1)*block])
        return result
    if op == "MaxPool":
        _, height, width, _ = src
        kh, kw = attrs["kernel_shape"]
        sy, sx = attrs["strides"]
        py, px = attrs["pads"][:2]
        return [
            max(data[_offset((b, iy, ix, c), src)]
                for iy in range(max(0, y*sy-py), min(height, y*sy-py+kh))
                for ix in range(max(0, x*sx-px), min(width, x*sx-px+kw)))
            for b, y, x, c in _coordinates(dest)
        ]
    if op == "TopKIndices":
        return sorted(range(len(data)), key=lambda i: (-data[i], i))[:attrs["k"]]
    if op == "Gather":
        width = math.prod(src[1:])
        result = []
        for idx in args[1]:
            if -src[0] <= idx < src[0]:
                idx %= src[0]
                result.extend(data[idx*width:(idx+1)*width])
            else:
                result.extend([0]*width)
        return result
    if op == "ArgMax":
        axis = attrs["axis"]
        candidates = range(src[axis])
        if attrs["select_last_index"]:
            candidates = reversed(candidates)
        candidates = tuple(candidates)
        reduced = src[:axis] + src[axis+1:]
        result = []
        for out in _coordinates(reduced):
            result.append(max(candidates, key=lambda i: data[_offset(out[:axis] + (i,) + out[axis:], src)]))
        return result
    if op == "Lookup":
        return [attrs["table"][value & 255] for value in data]
    if op == "QSoftmax":
        return _softmax(data, src, attrs)
    if op == "Sigmoid":
        return [round(16/(1 + math.exp(-value/16))) for value in data]
    raise ValueError(f"no layout evaluator for {op}")
