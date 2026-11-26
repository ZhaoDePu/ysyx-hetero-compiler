"""Portable C6 quantized graph and interpreter for the frozen Z15 arithmetic.

Runtime arrays have ONNX logical shape. Images use NCHW logically and NHWC
storage. The layout conversion is explicit in the graph's tensor metadata.
Dense integer products are executed through BLAS only in an exact mantissa
envelope. The native epilogue then runs on integers without float rounding.
"""
import json
import math
from pathlib import Path

import numpy as np


def multiplier_shifts(ratio):
    ratio = np.asarray(ratio, dtype=np.float64)
    if np.any(~np.isfinite(ratio)) or np.any(ratio < 0) or np.any(ratio > 65535):
        raise ValueError("requant ratio is outside the frozen u16/u5 format")
    safe = np.where(ratio > 0, ratio, 1)
    shift = np.clip(np.floor(np.log2(65535. / safe)), 0, 31).astype(np.int64)
    mult = np.clip(np.rint(np.ldexp(safe, shift)), 0, 65535).astype(np.int64)
    return np.where(ratio == 0, 0, mult), shift


def requant(accum, multiplier, shift):
    product = np.asarray(accum, dtype=np.int64) * np.asarray(multiplier, dtype=np.int64)
    return np.clip(product >> np.asarray(shift, dtype=np.int64), -128, 127).astype(np.int8)


def lut_apply(x, table):
    return np.asarray(table, dtype=np.int8)[x.astype(np.int8, copy=False).view(np.uint8)]


def ew_mul(x, y, mul, shift):
    product = x.astype(np.int64) * y.astype(np.int64) * mul
    if shift:
        product += 1 << (shift - 1)
    return np.clip(product >> shift, -128, 127).astype(np.int8)


def _matmul_int(a, b, bias=None):
    """Exact integer dot using mantissa proof, followed by RTL int32 wrapping."""
    # Every multiply and partial sum takes absolute value <= this bound.
    # BLAS computes exact integers regardless of dot/reduction order.
    bound = int(np.max(np.sum(np.abs(b.astype(np.int64)), axis=-2))) * int(np.max(np.abs(a.astype(np.int16))))
    if bias is not None:
        bound += int(np.max(np.abs(bias.astype(np.int64))))
    if bound >= 1 << 53:
        raise ValueError("integer dot exceeds the exact binary64 envelope")
    dtype = np.float32 if bound < 1 << 24 else np.float64
    product = a.astype(dtype) @ b.astype(dtype)
    if bias is not None:
        product += bias
    return product.astype(np.int64).astype(np.int32)


def conv_acc(x, w, bias, attrs):
    """Accumulate logical NCHW from HWIO signed-byte weights."""
    sy, sx = attrs["strides"]
    pt, pl, pb, pr = attrs["pads"]
    dy, dx = attrs["dilations"]
    groups = attrs["groups"]
    kh, kw, channels, outputs = w.shape
    if x.shape[1] != groups * channels or outputs % groups:
        raise ValueError("Conv shapes disagree with groups")
    x = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)))
    eh, ew = dy * (kh - 1) + 1, dx * (kw - 1) + 1
    fields = np.lib.stride_tricks.sliding_window_view(x, (eh, ew), axis=(2, 3))
    fields = fields[:, :, ::sy, ::sx, ::dy, ::dx]
    batches, _, out_h, out_w, _, _ = fields.shape
    chunks = []
    each = outputs // groups
    for group in range(groups):
        channel = slice(group * channels, (group + 1) * channels)
        out = slice(group * each, (group + 1) * each)
        patch = fields[:, channel].transpose(0, 2, 3, 4, 5, 1).reshape(-1, kh * kw * channels)
        matrix = np.ascontiguousarray(w[..., out]).reshape(-1, each)
        chunks.append(_matmul_int(patch, matrix, bias[out]))
    return np.concatenate(chunks, axis=-1).reshape(batches, out_h, out_w, outputs).transpose(0, 3, 1, 2)


def native_softmax(scores, axis, shift):
    """Match the Z15 unsigned probabilities and floor normalization."""
    scores = scores.astype(np.int32)
    delta = np.max(scores, axis=axis, keepdims=True) - scores
    weights = 256 >> np.minimum(delta >> shift, 31)
    denominator = np.sum(weights, axis=axis, keepdims=True)
    return ((255 * weights) // denominator).astype(np.uint8)


def _resize_nearest(value, out_shape, attrs, args):
    if attrs.get("mode", "nearest") != "nearest":
        raise ValueError("C6 only implements nearest Resize")
    transform = attrs.get("coordinate_transformation_mode", "half_pixel")
    nearest = attrs.get("nearest_mode", "round_prefer_floor")
    scales = args[2].ravel() if len(args) > 2 and args[2] is not None and args[2].size else None
    axes = attrs.get("axes", list(range(value.ndim)))
    for i, axis in enumerate(axes):
        source, target = value.shape[axis], out_shape[axis]
        if source == target:
            continue
        scale = float(scales[i]) if scales is not None else target / source
        pos = np.arange(target, dtype=np.float64)
        if transform == "asymmetric":
            pos /= scale
        elif transform == "align_corners":
            pos *= (source - 1) / max(target - 1, 1)
        elif transform in {"half_pixel", "pytorch_half_pixel"}:
            pos = (pos + 0.5) / scale - 0.5
            if transform == "pytorch_half_pixel" and target == 1:
                pos[:] = 0
        else:
            raise ValueError(f"unsupported resize coordinates {transform}")
        if nearest == "floor":
            index = np.floor(pos)
        elif nearest == "ceil":
            index = np.ceil(pos)
        elif nearest == "round_prefer_floor":
            index = np.ceil(pos - 0.5)
        elif nearest == "round_prefer_ceil":
            index = np.floor(pos + 0.5)
        else:
            raise ValueError(f"unsupported nearest rounding {nearest}")
        value = np.take(value, np.clip(index, 0, source - 1).astype(np.int64), axis=axis)
    return value


def _maxpool(value, attrs):
    kh, kw = attrs["kernel_shape"]
    sy, sx = attrs.get("strides", [1, 1])
    pt, pl, pb, pr = attrs.get("pads", [0, 0, 0, 0])
    if attrs.get("ceil_mode") or attrs.get("dilations", [1, 1]) != [1, 1]:
        raise ValueError("unsupported MaxPool ceil_mode/dilation")
    pad = np.iinfo(value.dtype).min if np.issubdtype(value.dtype, np.integer) else -np.inf
    source = np.pad(value, ((0, 0), (0, 0), (pt, pb), (pl, pr)), constant_values=pad)
    windows = np.lib.stride_tricks.sliding_window_view(source, (kh, kw), axis=(2, 3))
    return windows[:, :, ::sy, ::sx].max(axis=(-2, -1))


def _reshape(value, shape, allowzero):
    shape = [int(x) for x in shape]
    if not allowzero:
        shape = [source if n == 0 else n for source, n in zip(value.shape, shape)] + shape[value.ndim:]
    return value.reshape(shape)


def _ordinary(op, args, attrs, shape):
    a, *rest = args
    if op == "Identity":
        return a
    if op == "Transpose":
        return a.transpose(attrs.get("perm"))
    if op == "Reshape":
        return _reshape(a, rest[0].ravel(), attrs.get("allowzero"))
    if op == "Slice":
        slices = [slice(None)] * a.ndim
        starts, ends = rest[0].ravel(), rest[1].ravel()
        axes = rest[2].ravel() if len(rest) > 2 and rest[2] is not None else range(len(starts))
        steps = rest[3].ravel() if len(rest) > 3 and rest[3] is not None else [1] * len(starts)
        for start, end, axis, step in zip(starts, ends, axes, steps):
            slices[int(axis)] = slice(int(start), int(end), int(step))
        return a[tuple(slices)]
    if op == "Split":
        axis = attrs.get("axis", 0)
        sizes = rest[0].ravel() if rest and rest[0] is not None else attrs.get("split")
        if sizes is None:
            return tuple(np.array_split(a, attrs["num_outputs"], axis=axis))
        return tuple(np.split(a, np.cumsum(sizes)[:-1], axis=axis))
    if op == "Concat":
        return np.concatenate(args, axis=attrs["axis"])
    if op == "Resize":
        return _resize_nearest(a, shape, attrs, args)
    if op == "MaxPool":
        return _maxpool(a, attrs)
    if op == "Gather":
        return np.take(a, rest[0], axis=attrs.get("axis", 0))
    if op == "GatherElements":
        return np.take_along_axis(a, rest[0], axis=attrs.get("axis", 0))
    if op == "GatherND":
        if attrs.get("batch_dims", 0):
            raise ValueError("batched GatherND is unsupported")
        return a[tuple(np.moveaxis(rest[0], -1, 0))]
    if op == "TopK":
        axis = attrs.get("axis", -1)
        k = int(rest[0].item())
        keys = a.astype(np.float64) if np.issubdtype(a.dtype, np.integer) else a
        order = np.argsort(-keys if attrs.get("largest", 1) else keys, axis=axis, kind="stable")
        indices = np.take(order, np.arange(k), axis=axis)
        return np.take_along_axis(a, indices, axis=axis), indices.astype(np.int64)
    if op == "Unsqueeze":
        axes = rest[0].ravel() if rest else attrs["axes"]
        return np.expand_dims(a, tuple(axes))
    if op == "Squeeze":
        axes = rest[0].ravel() if rest else attrs.get("axes")
        return np.squeeze(a, axis=None if axes is None else tuple(axes))
    if op == "Flatten":
        return a.reshape(shape)
    if op == "Expand":
        return np.broadcast_to(a, shape)
    if op == "Cast":
        return a.astype({1: np.float32, 2: np.uint8, 3: np.int8, 6: np.int32, 7: np.int64, 9: np.bool_}[attrs["to"]])
    if op == "Shape":
        return np.array(a.shape[slice(attrs.get("start"), attrs.get("end"))], dtype=np.int64)
    if op in {"ReduceMax", "ReduceMean", "ReduceSum"}:
        axes = rest[0].ravel() if rest and rest[0] is not None else attrs.get("axes")
        if axes is not None and not len(axes) and attrs.get("noop_with_empty_axes"):
            return a
        func = {"ReduceMax": np.max, "ReduceMean": np.mean, "ReduceSum": np.sum}[op]
        return func(a, axis=None if axes is None or not len(axes) else tuple(axes), keepdims=bool(attrs.get("keepdims", 1)))
    if op == "ArgMax":
        axis = attrs.get("axis", 0)
        idx = np.argmax(a, axis=axis)
        if attrs.get("select_last_index"):
            idx = a.shape[axis] - 1 - np.argmax(np.flip(a, axis=axis), axis=axis)
        return np.expand_dims(idx, axis) if attrs.get("keepdims", 1) else idx
    if op == "Clip":
        lo, hi = (rest + [None, None])[:2]
        return np.clip(a, lo, hi)
    if op == "Sigmoid":
        return 1. / (1. + np.exp(-np.clip(a, -80, 80)))
    if op == "Softmax":
        axis = attrs.get("axis", -1)
        v = np.exp(a - np.max(a, axis=axis, keepdims=True))
        return v / np.sum(v, axis=axis, keepdims=True)
    if op == "Relu":
        return np.maximum(a, 0)
    if op == "Neg":
        return -a
    if op == "Exp":
        return np.exp(a)
    if op == "Sqrt":
        return np.sqrt(a)
    if op == "Where":
        return np.where(a, *rest)
    if op in {"Add", "Sub", "Mul", "Div", "Pow", "Max", "Min", "Greater", "Equal", "Mod"}:
        fun = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Div": np.divide, "Pow": np.power,
               "Max": np.maximum, "Min": np.minimum, "Greater": np.greater, "Equal": np.equal, "Mod": np.mod}[op]
        if op == "Div" and np.issubdtype(a.dtype, np.integer):
            return np.trunc(fun(a, rest[0])).astype(a.dtype)
        value = a
        for b in rest:
            value = fun(value, b)
        return value
    raise ValueError(f"C6 reference does not implement {op}")


class QuantGraph:
    def __init__(self, root):
        root = Path(root)
        self.graph = json.loads((root / "quant.json").read_text(encoding="utf-8"))
        self.tensors = self.graph["tensors"]
        with np.load(root / "weights.npz", allow_pickle=False) as store:
            self.constants = dict(store.items())

    def _float(self, name, val):
        scale = self.tensors[name].get("scale")
        return val.astype(np.float32) * np.float32(scale) if scale is not None else val

    def infer(self, image, *, capture=None):
        """Evaluate one logical normalized [1,3,H,W] RGB input."""
        name = self.graph["inputs"][0]
        if list(image.shape) != self.tensors[name]["shape"]:
            raise ValueError("calibrated model was compiled for a different input image shape")
        values = self.constants.copy()
        values[name] = np.clip(np.rint(image / self.tensors[name]["scale"]), -128, 127).astype(np.int8)
        uses = dict(self.graph["use_counts"])
        for node in self.graph["nodes"]:
            op, attrs = node["op"], node["attrs"]
            ins, outs = node["inputs"], node["outputs"]
            args = [values[n] if n else None for n in ins]
            shape = self.tensors[outs[0]]["shape"]
            if op == "Conv":
                acc = conv_acc(args[0], self.constants[attrs["weight"]], self.constants[attrs["bias"]], attrs)
                m, s = (np.array(attrs[k]).reshape(1, -1, 1, 1) for k in ("mul", "shift"))
                q = requant(acc, m, s)
                y = lut_apply(q, attrs["lut"]) if "lut" in attrs else q
            elif op == "MatMul":
                acc = _matmul_int(*args)
                y = acc if attrs.get("raw_output") else requant(acc, attrs["mul"], attrs["shift"])
            elif op in {"Lookup", "RequantLUT"}:
                y = lut_apply(args[0], attrs["lut"])
            elif op == "EWAdd":
                y = np.clip(args[0].astype(np.int16) + args[1].astype(np.int16), -128, 127).astype(np.int8)
            elif op == "EWMul":
                y = ew_mul(*args, attrs["mul"], attrs["shift"])
            elif op == "VpuSoftmax":
                y = native_softmax(args[0], attrs["axis"], attrs["shift"])
            elif op == "C7LogSoftmax":
                from .c7_softmax import log_softmax
                y = log_softmax(args[0], attrs["axis"], attrs["lut"], attrs["scale"], attrs["steps"])
            elif op == "Q23Softmax":
                axis = attrs["axis"]
                delta = np.max(args[0], axis=axis, keepdims=True).astype(np.int16) - args[0]
                weight = np.asarray(attrs["lut"], dtype=np.int64)[delta]
                total = np.sum(weight, axis=axis, keepdims=True)
                q, r = np.divmod(256 * weight, total)
                y = np.minimum(255, q + ((2*r > total) | ((2*r == total) & (q & 1) != 0))).astype(np.uint8)
            elif node["device"] == "PS":
                a = [self._float(n, v) if n else None for n, v in zip(ins, args)]
                y = _ordinary(op, a, attrs, shape)
            else:
                y = _ordinary(op, args, attrs, shape)
            if not isinstance(y, tuple):
                y = (y,)
            for n, v in zip(outs, y, strict=True):
                if self.tensors[n]["shape"] != list(v.shape):
                    raise ValueError(f"{node['id']}: expected {n} shape {self.tensors[n]['shape']}, got {v.shape}")
                values[n] = v
                if capture is not None:
                    capture(node, n, v)
            for n in ins:
                if n:
                    uses[n] -= 1
                    if uses[n] == 0 and n not in self.graph["outputs"]:
                        values.pop(n, None)
        return {n: self._float(n, values[n]) for n in self.graph["outputs"]}
