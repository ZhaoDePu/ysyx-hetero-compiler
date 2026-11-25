"""Independent scalar evaluation for layout transformations and reductions."""

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
        def f32(value):
            return struct.unpack("<f", struct.pack("<f", value))[0]

        axis = attrs["axis"]
        si = attrs["in_scale"][0] / attrs["in_scale"][1]
        so = attrs["out_scale"][0] / attrs["out_scale"][1]
        out = [0] * len(data)
        for before in _coordinates(src[:axis]):
            for after in _coordinates(src[axis+1:]):
                indexes = [_offset(before + (j,) + after, src) for j in range(src[axis])]
                highest = max(data[j] for j in indexes)
                weights = [f32(math.exp(si*(data[j]-highest))) for j in indexes]
                scale = sum(weights)
                for pos, val in zip(indexes, weights, strict=True):
                    out[pos] = round(f32(f32(val/scale)/so)) + attrs["zero_point"]
        return out
    if op == "Sigmoid":
        return [round(16/(1 + math.exp(-value/16))) for value in data]
    raise ValueError(f"no layout evaluator for {op}")
