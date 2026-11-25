"""Decode a static integer ONNX subset using only the Python standard library.

Wire schema: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3
"""

import itertools
import math
import struct
from collections import defaultdict
from fractions import Fraction

from .ir import DTYPES, GraphError, _attributes, _infer
from .layout import axis, broadcast_shape
from .onnx_qdq import QuantizedIslands

TYPES = {1: "float32", 2: "uint8", 3: "int8", 6: "int32", 7: "int64"}
WIDTH = {"float32": ("f", 4), "uint8": ("B", 1), "int8": ("b", 1),
         "int32": ("i", 4), "int64": ("q", 8)}
BYTES = {"int8", "uint8"}


def _varint(data, pos):
    value = 0
    for i in range(10):
        if pos == len(data):
            raise GraphError("truncated ONNX varint")
        byte = data[pos]
        pos += 1
        if i == 9 and byte > 1:
            raise GraphError("ONNX varint overflow")
        value |= (byte & 127) << (7 * i)
        if byte < 128:
            return value, pos
    raise GraphError("ONNX varint overflow")


def _message(data):
    if not isinstance(data, (bytes, memoryview)):
        raise GraphError("invalid ONNX submessage")
    data = memoryview(data)
    pos, fields = 0, defaultdict(list)
    while pos < len(data):
        key, pos = _varint(data, pos)
        tag, wire = key >> 3, key & 7
        if not 0 < tag < (1 << 29):
            raise GraphError("invalid ONNX field number")
        if wire == 0:
            value, pos = _varint(data, pos)
        elif wire in (1, 2, 5):
            if wire == 2:
                length, pos = _varint(data, pos)
            else:
                length = 8 if wire == 1 else 4
            if length > len(data) - pos:
                raise GraphError("truncated ONNX field")
            value = data[pos:pos + length]
            pos += length
        else:
            raise GraphError(f"unsupported protobuf wire type {wire}")
        fields[tag].append((wire, value))
    return fields


def _one(msg, tag, wire, default=None):
    values = _all(msg, tag, wire)
    if len(values) > 1:
        raise GraphError(f"duplicate ONNX field {tag}")
    return values[0] if values else default


def _all(msg, tag, wire):
    if any(w != wire for w, _ in msg.get(tag, [])):
        raise GraphError(f"wrong ONNX wire type for field {tag}")
    return [v for _, v in msg.get(tag, [])]


def _text(value):
    if value is None:
        return ""
    try:
        return bytes(value).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GraphError("invalid ONNX UTF-8") from exc


def _repeated_ints(msg, tag):
    for wire, value in msg.get(tag, []):
        if wire == 0:
            yield value
        elif wire == 2:
            pos = 0
            while pos < len(value):
                x, pos = _varint(value, pos)
                yield x
        else:
            raise GraphError("invalid ONNX integer list")


def _signed(value, bits=64):
    value &= (1 << bits) - 1
    return value if value < 1 << (bits-1) else value - (1 << bits)


def _value_info(payload):
    v = _message(payload)
    name = _text(_one(v, 1, 2))
    typ = _message(_one(v, 2, 2))
    t = _message(_one(typ, 1, 2))
    shape = _message(_one(t, 2, 2))
    dims = []
    for entry in _all(shape, 1, 2):
        d = _message(entry)
        value = _one(d, 1, 0, 0)
        if 2 in d or value <= 0:
            raise GraphError(f"{name}: ONNX dynamic dimensions are unsupported")
        dims.append(value)
    dtype = TYPES.get(_one(t, 1, 0))
    if dtype is None:
        raise GraphError("unsupported ONNX tensor dtype")
    return name, {"shape": dims, "dtype": dtype}


def _floats(msg, tag):
    for wire, data in msg.get(tag, []):
        if wire not in (2, 5) or len(data) % 4:
            raise GraphError("invalid ONNX float list")
        yield from (value for value, in struct.iter_unpack("<f", data))


def _initializer(payload):
    t = _message(payload)
    name = _text(_one(t, 8, 2))
    dtype = TYPES.get(_one(t, 2, 0))
    if dtype is None or 3 in t or 13 in t or _one(t, 14, 0, 0):
        raise GraphError(f"{name}: unsupported ONNX initializer type, segmented or external")
    dims = list(_repeated_ints(t, 1))
    if len(dims) > 4 or any(d > (1 << 28) for d in dims):
        raise GraphError(f"{name}: excessive initializer dimensions")
    fmt, width = WIDTH[dtype]
    fields = {"float32": 4, "int64": 7}
    stored_field = fields.get(dtype, 5)
    raw = _one(t, 9, 2)
    if (set(t) & {4, 5, 6, 7, 10, 11}) - {stored_field}:
        raise GraphError(f"{name}: initializer data type mismatch")
    if raw is not None:
        if stored_field in t or len(raw) != math.prod(dims) * width:
            raise GraphError(f"{name}: invalid initializer raw_data")
        data = [v for v, in struct.iter_unpack("<" + fmt, raw)]
    elif dtype == "float32":
        data = list(_floats(t, 4))
    else:
        data = [_signed(v, 64 if dtype == "int64" else 32) for v in _repeated_ints(t, stored_field)]
    if len(data) != math.prod(dims):
        raise GraphError(f"{name}: initializer value count differs from shape")
    if dtype in DTYPES:
        _, lo, hi = DTYPES[dtype]
        if any(not lo <= v <= hi for v in data):
            raise GraphError(f"{name}: initializer data outside {dtype}")
    return name, {"shape": dims, "dtype": dtype, "data": data, "kind": "constant"}


def _node_attrs(node):
    attrs = {}
    for payload in _all(node, 5, 2):
        msg = _message(payload)
        name = _text(_one(msg, 1, 2))
        typ = _one(msg, 20, 0)
        if not name or name in attrs:
            raise GraphError("missing or duplicate ONNX attribute name")
        if typ == 1:
            value = struct.unpack("<f", _one(msg, 2, 5))[0]
        elif typ == 2:
            value = _signed(_one(msg, 3, 0, 0))
        elif typ == 3:
            value = _text(_one(msg, 4, 2))
        elif typ == 6:
            value = list(_floats(msg, 7))
        elif typ == 7:
            value = [_signed(v) for v in _repeated_ints(msg, 8)]
        else:
            raise GraphError(f"attribute {name!r} uses unsupported ONNX type {typ}")
        attrs[name] = value
    return attrs


def _check_args(args, minimum, maximum=None):
    if maximum is None:
        maximum = minimum
    if not minimum <= len(args) <= maximum or not all(args[:minimum]):
        raise GraphError(f"expected {minimum}..{maximum} inputs")


def _check_attrs(attrs, *supported):
    extra = sorted(attrs.keys() - set(supported))
    if extra:
        raise GraphError(f"unsupported attributes {extra}")


def _geom(attrs, shape, kernel):
    if attrs.get("group", 1) != 1:
        raise GraphError("group must be 1")
    if attrs.get("dilations", [1, 1]) != [1, 1]:
        raise GraphError("dilations must be [1, 1]")
    if len(kernel) != 2 or kernel != attrs.get("kernel_shape", kernel):
        raise GraphError("kernel_shape disagrees with the filter")
    strides = attrs.get("strides", [1, 1])
    pads = attrs.get("pads", [0, 0, 0, 0])
    if not (isinstance(strides, list) and len(strides) == 2 and
            all(type(s) is int and s > 0 for s in strides)):
        raise GraphError("strides must be two positive integers")
    if not (isinstance(pads, list) and len(pads) == 4 and
            all(type(p) is int and p >= 0 for p in pads)):
        raise GraphError("pads must contain four nonnegative integers")
    auto = attrs.get("auto_pad", "NOTSET")
    if auto != "NOTSET" and "pads" in attrs:
        raise GraphError("auto_pad and pads are mutually exclusive")
    if auto in ("VALID", "NOTSET"):
        return strides, pads
    if auto not in ("SAME_UPPER", "SAME_LOWER"):
        raise GraphError(f"unsupported auto_pad {auto!r}")
    total = [max(0, ((d+s-1)//s-1)*s + k-d) for d, k, s in zip(shape, kernel, strides, strict=True)]
    before = [(p + (auto == "SAME_LOWER")) // 2 for p in total]
    return strides, before + [p-q for p, q in zip(total, before, strict=True)]


def _reshape_dims(source, requested, allowzero):
    if allowzero not in (0, 1):
        raise GraphError("allowzero must be 0 or 1")
    dest = []
    for i, dim in enumerate(requested):
        if dim == 0 and not allowzero:
            if i >= len(source):
                raise GraphError("Reshape zero cannot copy a missing dimension")
            dim = source[i]
        dest.append(dim)
    if not 1 <= len(dest) <= 4 or dest.count(-1) > 1 or any(d == 0 or d < -1 for d in dest):
        raise GraphError("Reshape must have nonempty, positive dimensions or one inferred -1")
    if -1 in dest:
        ratio, remainder = divmod(math.prod(source), math.prod(d for d in dest if d != -1))
        if not ratio or remainder:
            raise GraphError("Reshape -1 is inconsistent with the element count")
        dest[dest.index(-1)] = ratio
    return dest


class _Import:
    def __init__(self, declarations, initializers, defined, reserved):
        self.declarations = declarations
        self.specs = {name: declarations[name] for name in defined}
        self.initializers = initializers
        self.reserved = reserved
        self.nodes = []
        self.tag = 0

    def name(self, purpose):
        while True:
            name = f"@onnx/{self.tag}/{purpose}"
            self.tag += 1
            if name not in self.reserved:
                self.reserved.add(name)
                return name

    def spec(self, name):
        if name not in self.specs:
            raise GraphError(f"unbound ONNX operand {name!r}")
        return self.specs[name]

    def static(self, name, allowed):
        spec = self.spec(name)
        if name not in self.initializers or spec.get("kind") != "constant" or spec["dtype"] not in allowed:
            raise GraphError(f"{name}: expected static {', '.join(sorted(allowed))} initializer")
        return spec["data"]

    def scalar(self, name, allowed):
        value = self.static(name, allowed)
        if self.spec(name)["shape"] not in ([], [1]) or len(value) != 1:
            raise GraphError(f"{name}: expected scalar quantization parameter")
        return value[0]

    def zpoint(self, value, name):
        if not name:
            return 0
        return self.scalar(name, {self.spec(value)["dtype"]})

    def scale(self, name):
        scale = self.scalar(name, {"float32"})
        if not math.isfinite(scale) or scale <= 0:
            raise GraphError(f"{name}: scale must be finite and positive")
        return Fraction(scale)

    def emit(self, op, ins, out, attrs=None, ident=None):
        if out in self.specs:
            raise GraphError(f"{out}: multiply defined ONNX value")
        specs = [self.spec(i) for i in ins]
        if any(s["dtype"] not in DTYPES or not s["shape"] for s in specs):
            raise GraphError(f"{op}: requires nonscalar integer tensor operands")
        ident = ident or self.name("node")
        attrs = _attributes(op, attrs or {}, ident)
        declared = self.declarations.get(out)
        shape, dtype = _infer(op, specs, attrs, False, attrs.get("dtype"))
        if declared and (declared["shape"] != shape or declared["dtype"] != dtype):
            raise GraphError(f"{out}: inferred {dtype}{shape} differs from declared "
                             f"{declared['dtype']}{declared['shape']}")
        self.nodes.append({"id": ident, "op": op, "inputs": ins, "output": out, "attrs": attrs})
        self.specs[out] = {"shape": shape, "dtype": dtype}
        return out

    def temp(self, op, ins, attrs=None):
        return self.emit(op, ins, self.name(op.lower()), attrs)

    def constant(self, shape, dtype, data, purpose):
        name = self.name(purpose)
        self.specs[name] = {"shape": shape, "dtype": dtype, "kind": "constant", "data": data}
        self.initializers.add(name)
        return name

    def transpose(self, source, perm, output=None, ident=None):
        # Fold only immutable weights. A graph input with a default initializer
        # remains runtime-patchable and must execute the transpose at runtime.
        if output is None and source in self.initializers and self.spec(source)["kind"] == "constant":
            tensor = self.spec(source)
            shape = tensor["shape"]
            arrange = [shape[i] for i in perm]
            strides = [math.prod(shape[i+1:]) for i in perm]
            values = [tensor["data"][sum(i*s for i, s in zip(idx, strides, strict=True))]
                      for idx in itertools.product(*(range(x) for x in arrange))]
            return self.constant(arrange, tensor["dtype"], values, "transposed")
        return self.emit("Transpose", [source], output or self.name("transpose"), {"perm": perm}, ident)

    def signed_weights(self, name, zp):
        if self.spec(name)["dtype"] == "int8":
            return name, zp
        values = self.static(name, {"uint8"})
        return self.constant(self.spec(name)["shape"], "int8", [v-128 for v in values],
                             "signed_weight"), zp-128

    def bound(self, name, zero):
        tensor = self.spec(name)
        if name in self.initializers and tensor["kind"] == "constant":
            return max((abs(x-zero) for x in tensor["data"]), default=0)
        _, lo, hi = DTYPES[tensor["dtype"]]
        return max(abs(lo-zero), abs(hi-zero))

    def dq_attrs(self, ratio, bound, dtype, zero=0, source_zero=0):
        multiplier, divisor = ratio.numerator, ratio.denominator
        if (not 0 < multiplier < (1 << 31) or not 1 <= divisor <= (1 << 31) or
                (bound * multiplier + abs(zero) >= (1 << 31))):
            raise GraphError("exact quantization ratio or worst-case product exceeds the int32 lowering envelope")
        options = {"dtype": dtype, "multiplier": multiplier, "zero_point": zero,
                   "input_zero_point": source_zero}
        if divisor != 1:
            options["rounding"] = "nearest_even"
            if divisor & (divisor - 1):
                options["divisor"] = divisor
            else:
                options["shift"] = divisor.bit_length() - 1
        return options

    def matrix(self, op, inputs, output, attrs, ident):
        conv = op in ("Conv", "ConvInteger", "QLinearConv")
        qlinear = op.startswith("QLinear")
        if qlinear:
            _check_args(inputs, 8, 9 if conv else 8)
            a, sa, za, b, sb, zb, sy, zy, *bias = inputs
        elif op == "Conv":
            _check_args(inputs, 2, 3)
            a, b, *bias = inputs
            za = zb = ""
        else:
            _check_args(inputs, 2, 4)
            a, b, *zeros = inputs
            za, zb = (zeros + ["", ""])[:2]
            bias = []
        for operand in (a, b):
            if self.spec(operand)["dtype"] not in BYTES:
                raise GraphError(f"{operand}: integer matrix operands must be int8 or uint8")
        az, bz = self.zpoint(a, za), self.zpoint(b, zb)
        b, bz = self.signed_weights(b, bz)
        shape, weights = self.spec(a)["shape"], self.spec(b)["shape"]
        target_attrs = {"dtype": "int32"}
        if az:
            target_attrs["a_zero_point"] = az
        if bz:
            target_attrs["b_zero_point"] = bz
        if conv:
            _check_attrs(attrs, "auto_pad", "dilations", "group", "kernel_shape", "pads", "strides")
            if len(shape) != 4 or len(weights) != 4:
                raise GraphError("Conv uses NCHW activations and OIHW weights")
            syx, pads = _geom(attrs, shape[2:], weights[2:])
            target_attrs.update(strides=syx, pads=pads)
            k = math.prod(weights[1:])
            a = self.transpose(a, [0, 2, 3, 1])
            b = self.transpose(b, [2, 3, 1, 0])
        else:
            _check_attrs(attrs)
            if len(shape) != 2 or len(weights) != 2:
                raise GraphError("MatMul requires rank-2 tensors")
            k = shape[1]
        bound = k * self.bound(a, az) * self.bound(b, bz)
        if bias:
            bound += self.bound(bias[0], 0)
        destination = self.name("accumulator") if conv or qlinear else output
        self.emit("Conv" if conv else "Gemm", [a, b] + bias, destination, target_attrs,
                  None if conv or qlinear else ident)
        if qlinear:
            yp = self.scalar(zy, BYTES)
            ytype = self.spec(zy)["dtype"]
            ratio = self.scale(sa) * self.scale(sb) / self.scale(sy)
            qattrs = self.dq_attrs(ratio, bound, ytype, yp)
            dest = self.name("requantized") if conv else output
            self.emit("Requant", [destination], dest, qattrs, None if conv else ident)
            destination = dest
        if conv:
            self.transpose(destination, [0, 3, 1, 2], output, ident)

    def broadcast(self, a, b):
        a_shape, b_shape = self.spec(a)["shape"], self.spec(b)["shape"]
        out = broadcast_shape(a_shape, b_shape)
        if self.spec(a)["dtype"] != self.spec(b)["dtype"]:
            raise GraphError("ONNX elementwise inputs must have the same dtype")
        if a_shape != out:
            a = self.temp("Expand", [a], {"shape": out})
        if b_shape != out:
            b = self.temp("Expand", [b], {"shape": out})
        return a, b

    def qadd(self, inputs, out, ident):
        _check_args(inputs, 8)
        a, sa, az, b, sb, bz, sy, yz = inputs
        az, bz = self.zpoint(a, az), self.zpoint(b, bz)
        yp = self.scalar(yz, BYTES)
        yt = self.spec(yz)["dtype"]
        ra, rb = self.scale(sa)/self.scale(sy), self.scale(sb)/self.scale(sy)
        a, b = self.broadcast(a, b)
        if ra == rb == 1 and az == bz == yp == 0:
            return self.emit("Add", [a, b], out, {"dtype": yt}, ident)
        divisor = math.lcm(ra.denominator, rb.denominator)
        ma, mb = int(ra*divisor), int(rb*divisor)
        bounda, boundb = self.bound(a, az), self.bound(b, bz)
        a = self.temp("Requant", [a], self.dq_attrs(Fraction(ma), bounda, "int32", source_zero=az))
        b = self.temp("Requant", [b], self.dq_attrs(Fraction(mb), boundb, "int32", source_zero=bz))
        summed = self.temp("Add", [a, b], {"dtype": "int32"})
        return self.emit("Requant", [summed], out,
                         self.dq_attrs(Fraction(1, divisor), ma*bounda + mb*boundb, yt, yp), ident)

    def section(self, source, starts, ends, axes, steps):
        shape = self.spec(source)["shape"]
        if not (len(starts) == len(ends) == len(axes) == len(steps)):
            raise GraphError("Slice initializer lengths disagree")
        lower, upper, stride = [0]*len(shape), shape[:], [1]*len(shape)
        axes = [axis(i, len(shape)) for i in axes]
        if len(axes) != len(set(axes)):
            raise GraphError("Slice axes are repeated")
        for start, stop, i, step in zip(starts, ends, axes, steps, strict=True):
            if step <= 0:
                raise GraphError("Slice steps must be positive")
            lower[i], upper[i], stride[i] = slice(start, stop, step).indices(shape[i])
        return {"starts": lower, "ends": upper, "steps": stride}

    def split(self, inputs, outs, attrs, ident):
        _check_args(inputs, 1, 2)
        _check_attrs(attrs, "axis", "num_outputs", "split")
        source = inputs[0]
        shape = self.spec(source)["shape"]
        split_axis = axis(attrs.get("axis", 0), len(shape))
        size = shape[split_axis]
        if len(inputs) > 1 and inputs[1]:
            if "split" in attrs or "num_outputs" in attrs:
                raise GraphError("Split uses either static sizes or num_outputs")
            widths = self.static(inputs[1], {"int64"})
        elif "split" in attrs:
            widths = attrs["split"]
        else:
            count = attrs.get("num_outputs", len(outs))
            if count != len(outs):
                raise GraphError("Split num_outputs does not match output count")
            width = (size + count - 1)//count
            widths = [width] * (count - 1) + [size - width*(count - 1)]
        if len(widths) != len(outs) or min(widths) <= 0 or sum(widths) != size:
            raise GraphError("Split must partition the selected axis into nonempty outputs")
        offset = 0
        for i, (out, width) in enumerate(zip(outs, widths, strict=True)):
            params = self.section(source, [offset], [offset+width], [split_axis], [1])
            self.emit("Slice", [source], out, params, ident if i == 0 else None)
            offset += width

    def resize(self, op, inputs, out, attrs, ident):
        _check_args(inputs, 1, 2 if op == "Upsample" else 4)
        shape = self.spec(inputs[0])["shape"]
        if op == "Upsample":
            _check_attrs(attrs, "mode", "scales")
            if attrs.get("mode", "nearest") != "nearest":
                raise GraphError("Upsample mode must be nearest")
            scales = self.static(inputs[1], {"float32"}) if len(inputs) > 1 and inputs[1] else attrs.get("scales", [])
            sizes, axes = [], list(range(len(shape)))
        else:
            _check_attrs(attrs, "mode", "coordinate_transformation_mode", "nearest_mode",
                         "axes", "antialias", "keep_aspect_ratio_policy", "exclude_outside")
            if (attrs.get("mode", "nearest") != "nearest" or
                    attrs.get("coordinate_transformation_mode", "half_pixel") != "asymmetric" or
                    attrs.get("nearest_mode", "round_prefer_floor") != "floor" or
                    attrs.get("antialias", 0) or attrs.get("exclude_outside", 0) or
                    attrs.get("keep_aspect_ratio_policy", "stretch") != "stretch"):
                raise GraphError("Resize requires nearest/asymmetric/floor with no antialias")
            roi, scale_input, size_input = (inputs[1:] + ["", "", ""])[:3]
            if roi and self.static(roi, {"float32"}):
                raise GraphError("Resize ROI is unsupported")
            scales = self.static(scale_input, {"float32"}) if scale_input else []
            sizes = self.static(size_input, {"int64"}) if size_input else []
            axes = [axis(v, len(shape)) for v in attrs.get("axes", list(range(len(shape))))]
        if len(axes) != len(set(axes)) or bool(scales) == bool(sizes):
            raise GraphError("Resize requires distinct axes and exactly one of sizes or scales")
        if (len(sizes or scales) != len(axes) or any(not math.isfinite(x) or x <= 0 for x in scales)
                or any(type(x) is not int or x <= 0 for x in sizes)):
            raise GraphError("Resize sizes/scales must be positive and match axes")
        if scales:
            sizes = [math.floor(shape[i] * Fraction(scale)) for i, scale in zip(axes, scales, strict=True)]
            for i, size, scale in zip(axes, sizes, scales, strict=True):
                if Fraction(scale) != Fraction(size, shape[i]):
                    raise GraphError("Resize with explicit scales must map exactly to integral extents")
        dest = shape[:]
        for i, size in zip(axes, sizes, strict=True):
            dest[i] = size
        self.emit("Resize", [inputs[0]], out, {"shape": dest}, ident)

    def operator(self, op, inputs, outs, attrs, domain, ident, opset):
        custom = domain == "com.ysyx"
        if domain not in ("", "ai.onnx", "com.ysyx") and not (domain == "com.microsoft" and op == "QLinearAdd"):
            raise GraphError(f"unsupported domain {domain!r}")
        if any(not v or v in self.specs for v in outs) or len(outs) != len(set(outs)):
            raise GraphError("empty or multiply defined ONNX output")
        if op == "Split":
            if not outs:
                raise GraphError("Split requires outputs")
            return self.split(inputs, outs, attrs, ident)
        if len(outs) != 1:
            raise GraphError("operator requires one output")
        out = outs[0]
        if op in ("MatMulInteger", "ConvInteger", "QLinearConv", "QLinearMatMul") or (op == "Conv" and custom):
            return self.matrix(op, inputs, out, attrs, ident)
        if op == "QLinearAdd" and domain in ("com.microsoft", "com.ysyx"):
            _check_attrs(attrs)
            return self.qadd(inputs, out, ident)
        if op in ("Add", "Mul"):
            _check_args(inputs, 2)
            _check_attrs(attrs)
            a, b = self.broadcast(*inputs)
            return self.emit(op, [a, b], out, {"narrowing": "wrap"}, ident)
        if op in ("Relu", "Identity"):
            _check_args(inputs, 1)
            _check_attrs(attrs)
            if op == "Identity":
                return self.emit("Reshape", inputs, out, {"shape": self.spec(inputs[0])["shape"]}, ident)
            return self.emit(op, inputs, out, {"narrowing": "wrap"}, ident)
        if op == "Transpose":
            _check_args(inputs, 1)
            _check_attrs(attrs, "perm")
            return self.emit(op, inputs, out, attrs, ident)
        if op == "Cast":
            _check_args(inputs, 1)
            _check_attrs(attrs, "to")
            dtype = TYPES.get(attrs.get("to"))
            if dtype not in DTYPES:
                raise GraphError("only integer Cast destinations are supported")
            return self.emit(op, inputs, out, {"dtype": dtype, "narrowing": "wrap"}, ident)
        if op == "Reshape":
            _check_args(inputs, 2)
            _check_attrs(attrs, "allowzero")
            dims = _reshape_dims(self.spec(inputs[0])["shape"], self.static(inputs[1], {"int64"}),
                                 attrs.get("allowzero", 0))
            return self.emit(op, inputs[:1], out, {"shape": dims}, ident)
        if op == "Slice":
            _check_args(inputs, 3, 5)
            _check_attrs(attrs)
            args = [self.static(i, {"int64", "int32"}) if i else None for i in inputs[1:]]
            starts, ends, ax, steps = (args + [None]*4)[:4]
            count = len(starts)
            params = self.section(inputs[0], starts, ends, list(range(count)) if ax is None else ax,
                                  [1]*count if steps is None else steps)
            return self.emit(op, inputs[:1], out, params, ident)
        if op == "Concat":
            _check_args(inputs, 1, 32)
            _check_attrs(attrs, "axis")
            return self.emit(op, inputs, out, attrs, ident)
        if op in ("Upsample", "Resize"):
            if op == "Resize" and opset == 10:
                return self.resize("Upsample", inputs, out, attrs, ident)
            return self.resize(op, inputs, out, attrs, ident)
        if op == "MaxPool":
            _check_args(inputs, 1)
            _check_attrs(attrs, "kernel_shape", "auto_pad", "ceil_mode", "dilations", "pads",
                         "storage_order", "strides")
            if attrs.get("ceil_mode", 0) or attrs.get("storage_order", 0):
                raise GraphError("MaxPool requires floor output sizing and row-major storage")
            source = inputs[0]
            shape = self.spec(source)["shape"]
            kernel = attrs.get("kernel_shape", [])
            if len(shape) != 4:
                raise GraphError("MaxPool requires rank-4 NCHW input")
            strides, pads = _geom(attrs, shape[2:], kernel)
            src = self.transpose(source, [0, 2, 3, 1])
            pooled = self.temp(op, [src], {"kernel_shape": kernel, "strides": strides, "pads": pads})
            return self.transpose(pooled, [0, 3, 1, 2], out, ident)
        if op == "ArgMax":
            _check_args(inputs, 1)
            _check_attrs(attrs, "axis", "keepdims", "select_last_index")
            options = {"axis": attrs.get("axis", 0)}
            for flag, default in (("keepdims", 1), ("select_last_index", 0)):
                value = attrs.get(flag, default)
                if value not in (0, 1):
                    raise GraphError(f"{flag} must be 0 or 1")
                options[flag] = bool(value)
            return self.emit(op, inputs, out, options, ident)
        if custom and op in ("RMSNorm", "Softmax", "SiLU", "Gelu", "GELU", "Sigmoid", "Requant"):
            _check_args(inputs, 1)
            if op == "Softmax":
                _check_attrs(attrs, "axis", "causal")
                if axis(attrs.get("axis", -1), len(self.spec(inputs[0])["shape"])) != 1:
                    raise GraphError("fixed-point Softmax requires axis 1 of a matrix")
                causal = attrs.get("causal", 0)
                if causal not in (0, 1):
                    raise GraphError("causal must be 0 or 1")
                return self.emit("RowSoftmax", inputs, out, {"causal_mask": bool(causal)}, ident)
            if op == "Requant":
                _check_attrs(attrs, "shift", "zero_point", "multiplier", "input_zero_point",
                             "divisor", "rounding")
            else:
                _check_attrs(attrs)
            return self.emit("GELU" if op == "Gelu" else op, inputs, out, attrs, ident)
        if op in ("Conv", "Sigmoid", "SiLU", "Gelu", "Softmax", "RMSNormalization"):
            raise GraphError(f"floating {op} requires a supported static quantization pattern or com.ysyx integer form")
        raise GraphError(f"unsupported ONNX operator {op}")


def _commute_pointwise_through_transpose(importer, outputs):
    nodes = importer.nodes
    uses = defaultdict(int)
    for node in nodes:
        for source in node["inputs"]:
            uses[source] += 1
    producer = {}
    for i, node in enumerate(nodes):
        if node["op"] in ("Relu", "SiLU", "GELU", "Sigmoid", "Lookup"):
            source = node["inputs"][0]
            j = producer.get(source)
            if j is not None and source not in outputs and uses[source] == 1 and nodes[j]["op"] == "Transpose":
                transpose = nodes[j]
                value = importer.name("pointwise")
                source_spec = importer.spec(transpose["inputs"][0])
                importer.specs[value] = {"shape": source_spec["shape"], "dtype": importer.spec(node["output"])["dtype"]}
                nodes[j] = {**node, "inputs": transpose["inputs"], "output": value}
                nodes[i] = node = {**transpose, "inputs": [value], "output": node["output"]}
        producer[node["output"]] = i


def parse_model(data):
    model = _message(data)
    graphs = _all(model, 7, 2)
    if len(graphs) != 1 or 20 in model or 25 in model:
        raise GraphError("expected one static ONNX graph, without training or functions")
    opsets = {}
    for payload in _all(model, 8, 2):
        spec = _message(payload)
        domain = _text(_one(spec, 1, 2))
        domain = "" if domain == "ai.onnx" else domain
        if domain in opsets:
            raise GraphError(f"duplicate ONNX opset domain {domain!r}")
        opsets[domain] = _one(spec, 2, 0, 0)
    if not 7 <= opsets.get("", 0) <= 23:
        raise GraphError("supported main ONNX opsets are 7..23")
    graph = _message(graphs[0])
    if 15 in graph:
        raise GraphError("sparse ONNX initializers are unsupported")
    tensors, inputs, outputs = {}, set(), []
    for field in (11, 12, 13):
        for payload in _all(graph, field, 2):
            name, spec = _value_info(payload)
            if not name or name in tensors and tensors[name] != spec:
                raise GraphError(f"missing or conflicting ONNX ValueInfo {name}")
            if field in (11, 12) and (spec["dtype"] not in DTYPES or not spec["shape"]):
                raise GraphError(f"{name}: graph interface requires nonscalar integer tensors")
            tensors[name] = spec
            if field == 11:
                inputs.add(name)
            if field == 12:
                outputs.append(name)
    initializers = set()
    for payload in _all(graph, 5, 2):
        name, spec = _initializer(payload)
        if not name or name in initializers:
            raise GraphError(f"missing or duplicate ONNX initializer {name}")
        if name in tensors and any(tensors[name][k] != spec[k] for k in ("shape", "dtype")):
            raise GraphError(f"ONNX initializer differs from ValueInfo for {name}")
        initializers.add(name)
        tensors[name] = spec
    for name in inputs:
        tensors[name]["kind"] = "input"
    raw_nodes = [_message(p) for p in _all(graph, 1, 2)]
    reserved = set(tensors)
    for node in raw_nodes:
        for tag in (1, 2, 3):
            reserved.update(_text(v) for v in _all(node, tag, 2))
    importer = _Import(tensors, initializers, inputs | initializers, reserved)
    islands = QuantizedIslands(importer)
    for idx, node in enumerate(raw_nodes):
        ident = _text(_one(node, 3, 2)) or f"onnx_{idx}"
        op, domain = (_text(_one(node, tag, 2)) for tag in (4, 7))
        domain = "" if domain == "ai.onnx" else domain
        ins = [_text(i) for i in _all(node, 1, 2)]
        outs = [_text(i) for i in _all(node, 2, 2)]
        try:
            if domain not in opsets or (domain and opsets[domain] != 1) or 8 in node:
                raise GraphError("unsupported or missing domain opset, or node overload")
            for name in ins:
                if name and name not in islands.virtual:
                    importer.spec(name)
            attrs = _node_attrs(node)
            if not domain and islands.resolve(op, ins, outs, attrs, ident, opsets[""]):
                continue
            if any(name in islands.virtual for name in outs):
                raise GraphError("node redefines a float-island output")
            importer.operator(op, ins, outs, attrs, domain, ident, opsets[""])
        except (ValueError, TypeError, struct.error) as exc:
            raise GraphError(f"{ident} ({op}): {exc}") from exc
    if any(name not in importer.specs for name in outputs):
        raise GraphError("unbound ONNX graph output")
    islands.finish()
    _commute_pointwise_through_transpose(importer, outputs)
    used = set(outputs)
    for node in importer.nodes:
        used.update(node["inputs"])
        used.add(node["output"])
    return {"name": _text(_one(graph, 2, 2)),
            "tensors": {name: spec for name, spec in importer.specs.items() if name in used},
            "nodes": importer.nodes, "outputs": outputs}
