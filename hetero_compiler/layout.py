"""Shape and attribute contracts for static integer layout operations."""

from math import prod

ARITIES = {
    "Transpose": (1,), "Reshape": (1,), "Slice": (1,), "Resize": (1,),
    "Concat": range(1, 33), "MaxPool": (1,), "ArgMax": (1,), "Sigmoid": (1,),
    "Cast": (1,), "Expand": (1,), "Lookup": (1,), "QSoftmax": (1,),
}

ATTRIBUTES = {
    "Transpose": {"perm"}, "Reshape": {"shape"}, "Slice": {"starts", "ends", "steps"},
    "Resize": {"shape"}, "Concat": {"axis"}, "MaxPool": {"kernel_shape", "strides", "pads"},
    "ArgMax": {"axis", "keepdims", "select_last_index"}, "Sigmoid": set(),
    "Cast": set(), "Expand": {"shape"}, "Lookup": {"table"},
    "QSoftmax": {"axis", "in_scale", "out_scale", "zero_point", "algorithm", "causal_offset"},
}


def axis(value, rank):
    if type(value) is not int or not -rank <= value < rank:
        raise ValueError(f"axis {value!r} is outside rank {rank}")
    return value % rank


def _integer_list(attrs, name, count, minimum=0, maximum=1 << 28):
    values = attrs.get(name)
    if not (isinstance(values, list) and len(values) == count and
            all(type(v) is int and minimum <= v <= maximum for v in values)):
        raise ValueError(f"{name} must contain {count} integers in [{minimum}, {maximum}]")
    return values


def broadcast_shape(left, right):
    rank = max(len(left), len(right))
    a = [1] * (rank - len(left)) + left
    b = [1] * (rank - len(right)) + right
    if any(x != y and 1 not in (x, y) for x, y in zip(a, b, strict=True)):
        raise ValueError("operands have incompatible broadcast dimensions")
    return [max(x, y) for x, y in zip(a, b, strict=True)]


def infer(op, inputs, attrs, output_dtype):
    base = inputs[0]
    shape, dtype = base["shape"], base["dtype"]
    rank = len(shape)
    if op == "ArgMax":
        if dtype not in ("int8", "uint8", "int32") or output_dtype not in (None, "int64"):
            raise ValueError("ArgMax requires an 8-bit or 32-bit integer input and int64 output")
        idx = attrs["axis"] = axis(attrs.get("axis", 0), rank)
        for flag, default in (("keepdims", True), ("select_last_index", False)):
            if type(attrs.setdefault(flag, default)) is not bool:
                raise ValueError(f"{flag} must be boolean")
        out = shape[:]
        if attrs["keepdims"]:
            out[idx] = 1
        else:
            del out[idx]
        if not out:
            raise ValueError("rank-0 ArgMax results are unsupported")
        return out, "int64"
    if op == "Cast":
        if output_dtype is None:
            raise ValueError("Cast requires an explicit integer dtype")
        return shape[:], output_dtype
    if op in ("Lookup", "QSoftmax"):
        if dtype not in ("int8", "uint8") or output_dtype not in ("int8", "uint8"):
            raise ValueError(f"{op} requires byte input and output")
        lo, hi = (-128, 127) if output_dtype == "int8" else (0, 255)
        if op == "Lookup":
            _integer_list(attrs, "table", 256, lo, hi)
        else:
            attrs["axis"] = axis(attrs.get("axis", -1), rank)
            for name in ("in_scale", "out_scale"):
                _integer_list(attrs, name, 2, 1, 1 << 31)
            point = attrs.setdefault("zero_point", 0)
            if type(point) is not int or not lo <= point <= hi:
                raise ValueError("QSoftmax zero_point does not fit output dtype")
            algorithm = attrs.get("algorithm", "float32")
            if algorithm not in ("float32", "q23"):
                raise ValueError("QSoftmax unknown arithmetic contract")
            if algorithm == "q23":
                numerator, denominator = attrs["out_scale"]
                if (shape[attrs["axis"]] > 255 or numerator != 1 or denominator > 1 << 16
                        or denominator & (denominator - 1)):
                    raise ValueError("Q23 Softmax requires at most 255 keys and a reciprocal power-of-two output scale")
            if "causal_offset" in attrs and (
                    rank < 2 or attrs["axis"] != rank - 1 or type(attrs["causal_offset"]) is not int
                    or not 0 <= attrs["causal_offset"] < shape[-1]):
                raise ValueError("causal QSoftmax requires a valid query/key prefix on the final axis")
        return shape[:], output_dtype
    if op == "Sigmoid":
        if dtype != "int8" or output_dtype not in (None, "int8", "uint8"):
            raise ValueError("Q4 Sigmoid requires int8 input and byte output")
        return shape[:], output_dtype or "int8"
    if output_dtype not in (None, dtype):
        raise ValueError(f"{op} retains its input dtype")
    if op == "Transpose":
        perm = attrs.setdefault("perm", list(reversed(range(rank))))
        if sorted(_integer_list(attrs, "perm", rank, 0, rank - 1)) != list(range(rank)):
            raise ValueError("perm is not a permutation")
        return [shape[i] for i in perm], dtype
    if op in ("Reshape", "Resize", "Expand"):
        dest = attrs.get("shape")
        if not isinstance(dest, list) or not 1 <= len(dest) <= 4:
            raise ValueError("shape requires 1..4 positive static dimensions")
        dest = _integer_list(attrs, "shape", len(dest), 1)
        if op == "Reshape" and prod(shape) != prod(dest):
            raise ValueError("Reshape changes the tensor element count")
        if op == "Resize" and rank != len(dest):
            raise ValueError("Resize changes tensor rank")
        if op == "Expand" and broadcast_shape(shape, dest) != dest:
            raise ValueError("Expand changes an existing non-unit dimension")
        return dest[:], dtype
    if op == "Slice":
        starts = _integer_list(attrs, "starts", rank)
        ends = _integer_list(attrs, "ends", rank)
        steps = _integer_list(attrs, "steps", rank, 1)
        if any(start >= stop or stop > dim for start, stop, dim in zip(starts, ends, shape, strict=True)):
            raise ValueError("Slice must have nonempty in-bounds intervals")
        return [(end - start + step - 1) // step
                for start, end, step in zip(starts, ends, steps, strict=True)], dtype
    if op == "Concat":
        idx = attrs["axis"] = axis(attrs.get("axis"), rank)
        if any(t["dtype"] != dtype or len(t["shape"]) != rank or
               t["shape"][:idx] != shape[:idx] or t["shape"][idx+1:] != shape[idx+1:]
               for t in inputs):
            raise ValueError("Concat inputs disagree in dtype, rank or a non-axis dimension")
        dest = shape[:]
        dest[idx] = sum(t["shape"][idx] for t in inputs)
        return dest, dtype
    if op == "MaxPool":
        if rank != 4 or dtype not in ("int8", "uint8"):
            raise ValueError("MaxPool requires NHWC int8 or uint8 tensors")
        attrs.setdefault("strides", [1, 1])
        attrs.setdefault("pads", [0, 0, 0, 0])
        kh, kw = _integer_list(attrs, "kernel_shape", 2, 1, 256)
        sy, sx = _integer_list(attrs, "strides", 2, 1, 256)
        top, left, bottom, right = _integer_list(attrs, "pads", 4, 0, 255)
        if top >= kh or bottom >= kh or left >= kw or right >= kw:
            raise ValueError("MaxPool padding must be smaller than the window")
        oh = (shape[1] + top + bottom - kh) // sy + 1
        ow = (shape[2] + left + right - kw) // sx + 1
        if min(oh, ow) <= 0:
            raise ValueError("MaxPool window has no output")
        return [shape[0], oh, ow, shape[3]], dtype
    raise ValueError(f"no shape rule for {op}")
