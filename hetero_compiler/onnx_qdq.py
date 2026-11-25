"""Recognize static float islands with quantized input and output boundaries."""

from dataclasses import dataclass
from fractions import Fraction
import math
import struct

from .ir import GraphError
from .layout import axis


@dataclass(frozen=True)
class Dequantized:
    tensor: str
    scale_name: str
    point_name: str
    scale: Fraction
    point: int


@dataclass(frozen=True)
class Operation:
    op: str
    inputs: tuple
    attrs: dict


def _power_of_two(value):
    n, d = value.numerator, value.denominator
    return not ((n & (n-1)) or (d & (d-1))) and (1 << 90) * n >= d and n <= (1 << 90)*d


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _activation(value, op, approximate):
    if op in ("Sigmoid", "SiLU"):
        # Separate stable branches prevent exp overflow for large inputs.
        if value >= 0:
            sigmoid = _f32(1 / (1 + math.exp(-value)))
        else:
            e = math.exp(value)
            sigmoid = _f32(e / (1 + e))
        return sigmoid if op == "Sigmoid" else _f32(value * sigmoid)
    if approximate == "tanh":
        if abs(value) >= 16:
            return value if value > 0 else 0
        return _f32(0.5 * value * (1 + math.tanh(math.sqrt(2/math.pi) *
                                                  (value + 0.044715*value**3))))
    return _f32(0.5 * value * (1 + math.erf(value / math.sqrt(2))))


class QuantizedIslands:
    def __init__(self, importer):
        self.importer = importer
        self.virtual = {}
        self.consumed = set()

    def _define(self, name, value):
        if not name or name in self.virtual or name in self.importer.specs:
            raise GraphError(f"{name!r}: empty or multiply defined ONNX value")
        decl = self.importer.declarations.get(name)
        if decl and decl["dtype"] != "float32":
            raise GraphError(f"{name}: float-island declaration must be float32")
        self.virtual[name] = value

    def resolve(self, op, inputs, outputs, attrs, ident, opset):
        if op not in ("DequantizeLinear", "QuantizeLinear") and not any(n in self.virtual for n in inputs):
            return False
        if len(outputs) != 1:
            raise GraphError("quantization island operator requires one output")
        out = outputs[0]
        ir = self.importer
        if op == "DequantizeLinear":
            if not 2 <= len(inputs) <= 3 or attrs.keys() - {"axis"}:
                raise GraphError("DequantizeLinear requires scalar static scale and optional zero point")
            val, scale_name = inputs[:2]
            point_name = inputs[2] if len(inputs) == 3 else ""
            dtype = ir.spec(val)["dtype"]
            if dtype not in ("int8", "uint8", "int32"):
                raise GraphError("unsupported DequantizeLinear input dtype")
            point = ir.zpoint(val, point_name)
            if dtype == "int32" and point:
                raise GraphError("int32 DequantizeLinear requires zero point 0")
            if not point_name:
                point_name = ir.constant([], dtype, [0], "implicit_zero")
            self._define(out, Dequantized(val, scale_name, point_name, ir.scale(scale_name), point))
            return True
        if op == "QuantizeLinear":
            if not 2 <= len(inputs) <= 3 or attrs.keys() - {"axis"} or inputs[0] not in self.virtual:
                raise GraphError("QuantizeLinear requires a supported dequantized island and scalar static scale")
            if out in self.virtual:
                raise GraphError("QuantizeLinear redefines a float-island name")
            val, scale_name = inputs[:2]
            point_name = inputs[2] if len(inputs) == 3 and inputs[2] else ir.constant([], "uint8", [0], "output_zero")
            point = ir.scalar(point_name, {"int8", "uint8"})
            value, scale, dtype = self.virtual[val], ir.scale(scale_name), ir.spec(point_name)["dtype"]
            self.quantize(value, scale, point, dtype, scale_name, point_name, out, ident)
            self.consumed.add(val)
            return True
        if op in ("Sigmoid", "Gelu", "SiLU", "Softmax"):
            if len(inputs) != 1 or not isinstance(self.virtual.get(inputs[0]), Dequantized):
                raise GraphError(f"{op} requires a direct scalar DequantizeLinear input")
            if op == "Gelu":
                if attrs.keys() - {"approximate"} or attrs.get("approximate", "none") not in ("none", "tanh"):
                    raise GraphError("unsupported Gelu approximate mode")
            elif op == "Softmax":
                if opset < 13 or attrs.keys() - {"axis"}:
                    raise GraphError("quantized Softmax requires opset 13+ and an optional axis")
            elif attrs:
                raise GraphError(f"unsupported {op} attributes")
            self._define(out, Operation(op, (self.virtual[inputs[0]],), attrs))
        elif op == "Mul":
            if len(inputs) != 2 or attrs or not all(n in self.virtual for n in inputs):
                raise GraphError("only Sigmoid(x)×x floating Mul is supported")
            left, right = [self.virtual[n] for n in inputs]
            plain, sigmoid = (left, right) if isinstance(left, Dequantized) else (right, left)
            if (not isinstance(plain, Dequantized) or not isinstance(sigmoid, Operation)
                    or sigmoid.op != "Sigmoid" or sigmoid.inputs[0] != plain):
                raise GraphError("only Sigmoid(x)×x floating Mul is supported")
            self._define(out, Operation("SiLU", (plain,), {}))
        elif op == "Conv":
            if (not 2 <= len(inputs) <= 3 or any(not isinstance(self.virtual.get(x), Dequantized)
                                                 for x in inputs[:2])):
                raise GraphError("Q/DQ Conv requires direct dequantized activations and weights")
            a, b = [self.virtual[x] for x in inputs[:2]]
            if ir.spec(a.tensor)["dtype"] not in ("int8", "uint8") or ir.spec(b.tensor)["dtype"] not in ("int8", "uint8"):
                raise GraphError("Q/DQ Conv requires byte activations and weights")
            bias_name = self.conv_bias(inputs, a.scale*b.scale)
            self._define(out, Operation(op, (a, b, bias_name), attrs))
        else:
            raise GraphError(f"unsupported floating operator {op} between quantized boundaries")
        self.consumed.update(name for name in inputs if name in self.virtual)
        return True

    def conv_bias(self, inputs, scale):
        ir = self.importer
        if len(inputs) < 3 or not inputs[2]:
            return None
        name = inputs[2]
        if name in self.virtual:
            bias = self.virtual[name]
            if (not isinstance(bias, Dequantized) or ir.spec(bias.tensor)["dtype"] != "int32"
                    or bias.scale != scale or bias.point):
                raise GraphError("Conv bias DequantizeLinear must use int32 and activation×weight scale")
            return bias.tensor
        floats = ir.static(name, {"float32"})
        if any(not math.isfinite(v) for v in floats):
            raise GraphError("Conv bias must be finite")
        values = [Fraction(v)/scale for v in floats]
        if any(v.denominator != 1 or not -(1 << 31) <= v < 1 << 31 for v in values):
            raise GraphError("Conv bias is not representable at activation×weight scale")
        return ir.constant(ir.spec(name)["shape"], "int32", [int(v) for v in values], "quantized_bias")

    def quantize(self, value, scale, point, dtype, scale_name, point_name, out, ident):
        ir = self.importer
        if isinstance(value, Dequantized):
            if not (_power_of_two(value.scale) and _power_of_two(scale)):
                raise GraphError("direct Q/DQ requires power-of-two float scales")
            bound = ir.bound(value.tensor, value.point)
            if bound >= 1 << 24:
                raise GraphError("direct Q/DQ exceeds exact float32 integer mantissa")
            attrs = ir.dq_attrs(value.scale/scale, bound, dtype, point, value.point)
            return ir.emit("Requant", [value.tensor], out, attrs, ident)
        source = value.inputs[0]
        if not (_power_of_two(source.scale) and _power_of_two(scale)):
            raise GraphError("exact Q/DQ lowering requires power-of-two float scales")
        if value.op == "Conv":
            a, b, bias = value.inputs
            product = a.scale*b.scale
            terms = math.prod(ir.spec(b.tensor)["shape"][1:])
            bound = terms*ir.bound(a.tensor, a.point)*ir.bound(b.tensor, b.point)
            if bias:
                bound += ir.bound(bias, 0)
            if not (_power_of_two(b.scale) and _power_of_two(product) and bound < 1 << 24):
                raise GraphError("Q/DQ Conv exceeds exact float32 dyadic mantissa or exponent envelope")
            ins = [a.tensor, a.scale_name, a.point_name, b.tensor, b.scale_name, b.point_name,
                   scale_name, point_name] + ([bias] if bias else [])
            return ir.matrix("QLinearConv", ins, out, value.attrs, ident)
        source_dtype = ir.spec(source.tensor)["dtype"]
        if source_dtype not in ("int8", "uint8"):
            raise GraphError("quantized activation requires byte input")
        if value.op == "Softmax":
            dim = axis(value.attrs.get("axis", -1), len(ir.spec(source.tensor)["shape"]))
            return ir.emit("QSoftmax", [source.tensor], out,
                           {"dtype": dtype, "axis": dim, "zero_point": point,
                            "in_scale": [source.scale.numerator, source.scale.denominator],
                            "out_scale": [scale.numerator, scale.denominator]}, ident)
        table = []
        lo, hi = (-128, 127) if dtype == "int8" else (0, 255)
        for index in range(256):
            code = index - 256 if source_dtype == "int8" and index >= 128 else index
            x = _f32(float((code-source.point)*source.scale))
            y = _activation(x, value.op, value.attrs.get("approximate", "none"))
            exact = round(Fraction(y)/scale) + point
            table.append(max(lo, min(hi, exact)))
        return ir.emit("Lookup", [source.tensor], out, {"dtype": dtype, "table": table}, ident)

    def finish(self):
        stranded = self.virtual.keys() - self.consumed
        if stranded:
            raise GraphError(f"floating value {min(stranded)!r} has no quantized consumer")
