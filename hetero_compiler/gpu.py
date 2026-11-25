"""Lower row-major integer tensors to YSYX GPU device images without a cross toolchain."""
from math import erf, exp, prod, sqrt
from pathlib import Path
from struct import pack
import json


_WIDTH = {"int8": 1, "uint8": 1, "int32": 4, "int64": 8}
_LAYOUT = {"Transpose", "Reshape", "Slice", "Resize", "Concat", "Expand"}
_ADVANCED = _LAYOUT | {"Cast", "Conv", "MaxPool", "ArgMax", "RowSoftmax", "RMSNorm", "QSoftmax"}
_LOAD = {"int8": 0, "uint8": 4, "int32": 2}
_VARIANTS = {"v6", "g3", "g4"}
_LEGACY_SILU_KNOTS = (0, 0, 1, 2, 5, 12, 31, 69, 128, 187, 225, 244, 251, 254, 255, 256, 256)
_EXIT = 0x00100073
_SCALAR_ENABLE = 0x5340000F


def supports(stage, graph, profile):
    """Check the mathematical contract; allocation and instruction limits are checked at emit time."""
    config = profile["gpu"]
    if config.get("variant") not in _VARIANTS:
        return False
    parts = stage.get("fused")
    if not parts:
        return _supports_node(stage, graph)
    if not (2 <= len(parts) <= 4 and parts[0]["op"] in {"Gemm", "Conv"} and
            parts[-1]["output"] == stage["output"] and
            all(_supports_node(node, graph) for node in parts)):
        return False
    phases = {"Add": 0, "Requant": 1, "Relu": 2, "SiLU": 2, "GELU": 2}
    produced = {node["output"] for node in parts}
    tail, last_phase = parts[0]["output"], -1
    external = set(parts[0]["inputs"])
    for node in parts[1:]:
        phase = phases.get(node["op"], -1)
        if (phase <= last_phase or node["inputs"][0] != tail or
                any(name in produced for name in node["inputs"][1:]) or
                tail in graph.get("outputs", [])):
            return False
        external.update(node["inputs"][1:])
        tail, last_phase = node["output"], phase
    if external != set(stage["inputs"]) or external & produced:
        return False
    # Reject fusion when the external operands alone cannot fit in one GPU page.
    memory = profile.get("sram", {})
    alignment = memory.get("alignment", 1)
    total = sum((prod(graph["tensors"][name]["shape"]) * _WIDTH[graph["tensors"][name]["dtype"]]
                 + alignment - 1) // alignment * alignment for name in {*external, tail})
    return total <= config["window_bytes"] - memory.get("metadata_reserve", 0)


def _supports_node(stage, graph):
    tensors = graph["tensors"]
    op = stage["op"]
    inputs = [tensors[name] for name in stage["inputs"]]
    out = tensors[stage["output"]]
    attrs = stage.get("attrs", {})
    shape = out["shape"]
    if any(t["dtype"] not in _WIDTH for t in [out, *inputs]):
        return False
    if attrs.get("narrowing", "saturate") not in {"wrap", "saturate"}:
        return False
    if op in _ADVANCED:
        return _supports_advanced(op, inputs, out, attrs)
    if any(t["dtype"] == "int64" for t in [out, *inputs]):
        return False
    if op in {"Relu", "Requant", "SiLU", "GELU", "Sigmoid", "GatedBranch", "Lookup"}:
        if len(inputs) != 1 or inputs[0]["shape"] != shape:
            return False
        if op == "Lookup":
            lo, hi = (-128, 127) if out["dtype"] == "int8" else (0, 255)
            table = attrs.get("table")
            return (inputs[0]["dtype"] in {"int8", "uint8"} and out["dtype"] in {"int8", "uint8"}
                    and isinstance(table, list) and len(table) == 256
                    and all(type(value) is int and lo <= value <= hi for value in table))
        if op == "Sigmoid":
            return inputs[0]["dtype"] == "int8" and out["dtype"] in {"int8", "uint8"}
        if op in {"SiLU", "GELU"}:
            mode = attrs.get("mode")
            return (inputs[0]["dtype"] == out["dtype"] == "int8"
                    and (mode is None or (op == "SiLU" and mode == "legacy_q4_knots")))
        if op == "GatedBranch":
            dtype = inputs[0]["dtype"]
            lo, hi = (-128, 127) if dtype == "int8" else (0, 255)
            threshold, rounds = attrs.get("threshold", 0), attrs.get("rounds", 64)
            return (dtype in {"int8", "uint8"} and dtype == out["dtype"] and
                    type(threshold) is int and lo <= threshold <= hi and
                    type(rounds) is int and 64 <= rounds <= 4096)
        if op == "Requant":
            shift, rounding = attrs.get("shift", 0), attrs.get("rounding", "floor")
            if type(shift) is not int or not 0 <= shift <= 31 or rounding not in {"floor", "nearest_even"}:
                return False
            divisor = attrs.get("divisor", 1 << shift)
            return (type(divisor) is int and 1 <= divisor <= 1 << 31 and
                    (rounding == "nearest_even" or divisor == 1 << shift) and
                    all(type(attrs.get(a, default)) is int and
                        -(1 << 31) <= attrs.get(a, default) < 1 << 31
                        for a, default in (("multiplier", 1), ("zero_point", 0), ("input_zero_point", 0))))
        return True
    if op in {"Add", "Mul"}:
        if len(inputs) != 2 or inputs[0]["shape"] != shape or not shape:
            return False
        if attrs.get("broadcast") == "last":
            return op == "Add" and inputs[1]["shape"] == [shape[-1]]
        return attrs.get("broadcast", "none") == "none" and inputs[1]["shape"] == shape
    if op == "Gemm":
        if (len(inputs) not in {2, 3} or len(shape) != 2 or
                len(inputs[0]["shape"]) != 2 or len(inputs[1]["shape"]) != 2):
            return False
        a, b = (inp["shape"] for inp in inputs[:2])
        transpose = attrs.get("transpose_b", False)
        if type(transpose) is not bool:
            return False
        contract, columns = b[::-1] if transpose else b
        return (a[1] == contract and shape == [a[0], columns] and _valid_zero_points(inputs, attrs) and
                (len(inputs) == 2 or inputs[2]["shape"] == [columns]))
    return False


def _valid_zero_points(inputs, attrs):
    for tensor, key in zip(inputs, ("a_zero_point", "b_zero_point")):
        zp = attrs.get(key, 0)
        dtype = tensor["dtype"]
        lo, hi = {"int8": (-128, 127), "uint8": (0, 255), "int32": (0, 0)}[dtype]
        if type(zp) is not int or not lo <= zp <= hi:
            return False
    return True


def _supports_advanced(op, inputs, out, attrs):
    src, dst = inputs[0]["shape"], out["shape"]
    dtypes = [tensor["dtype"] for tensor in inputs]
    dtype = out["dtype"]
    if op == "Cast":
        return (len(inputs) == 1 and src == dst and
                (dtypes[0] != "int64" or dtype == "int64" or attrs.get("narrowing") == "wrap"))
    if op in _LAYOUT:
        if not all(d == dtype for d in dtypes):
            return False
        if op == "Concat":
            axis = attrs.get("axis")
            if type(axis) is not int or not -len(dst) <= axis < len(dst):
                return False
            axis %= len(dst)
            return (all(len(s) == len(dst) and s[:axis] + s[axis+1:] == dst[:axis] + dst[axis+1:]
                        for s in [tensor["shape"] for tensor in inputs]) and
                    sum(tensor["shape"][axis] for tensor in inputs) == dst[axis])
        if len(inputs) != 1:
            return False
        if op == "Reshape":
            return prod(src) == prod(dst)
        if op == "Resize":
            return len(src) == len(dst)
        if op == "Expand":
            return (len(src) <= len(dst) and
                    all(a == 1 or a == b for a, b in zip(src, dst[len(dst)-len(src):])))
        if op == "Transpose":
            perm = attrs.get("perm")
            return (isinstance(perm, (tuple, list)) and sorted(perm) == list(range(len(src))) and
                    [src[i] for i in perm] == dst)
        starts, ends, steps = (attrs.get(k, ()) for k in ("starts", "ends", "steps"))
        return (len(starts) == len(ends) == len(steps) == len(src) == len(dst) and
                all(type(a) is int and type(b) is int and type(d) is int and 0 <= a < b <= dim
                    and d > 0 and n == (b - a + d - 1) // d
                    for a, b, d, dim, n in zip(starts, ends, steps, src, dst)))
    if op == "QSoftmax":
        axis = attrs.get("axis", len(src) - 1)
        return (len(inputs) == 1 and src == dst and dtypes[0] in {"int8", "uint8"}
                and dtype in {"int8", "uint8"} and attrs.get("algorithm") == "q23"
                and type(axis) is int and 0 <= axis < len(src) and src[axis] <= 255)
    if op in {"RMSNorm", "RowSoftmax"}:
        return (len(inputs) == 1 and src == dst and len(src) == 2 and dtypes[0] == "int8"
                and dtype == ("uint8" if op == "RowSoftmax" else "int8") and src[1] <= 4096)
    if op == "ArgMax":
        axis = attrs.get("axis", 0)
        if (len(inputs) != 1 or dtypes[0] == "int64" or dtype not in {"int32", "int64"}
                or type(axis) is not int or not -len(src) <= axis < len(src)):
            return False
        axis %= len(src)
        return dst == src[:axis] + ([1] if attrs.get("keepdims", True) else []) + src[axis+1:]
    if len(src) != 4 or len(dst) != 4 or dst[0] != src[0] or dtypes[0] == "int64" or dtype == "int64":
        return False
    if op == "Conv":
        if len(inputs) not in {2, 3} or len(inputs[1]["shape"]) != 4 or "int64" in dtypes:
            return False
        kh, kw, ci, co = inputs[1]["shape"]
        if ci != src[-1] or co != dst[-1] or (len(inputs) == 3 and inputs[2]["shape"] != [co]):
            return False
        if not _valid_zero_points(inputs, attrs):
            return False
    else:
        if (len(inputs) != 1 or dtypes[0] != dtype or src[-1] != dst[-1] or
                len(attrs.get("kernel_shape", [])) != 2):
            return False
        kh, kw = attrs["kernel_shape"]
    sy, sx = attrs.get("strides", [1, 1])
    top, left, bottom, right = attrs.get("pads", [0, 0, 0, 0])
    return (1 <= kh <= 7 and 1 <= kw <= 7 and 1 <= sx <= 64 and 1 <= sy <= 64 and
            0 <= min(top, left, bottom, right) <= max(top, left, bottom, right) <= 64 and
            dst[1:3] == [(src[1] + top + bottom - kh) // sy + 1, (src[2] + left + right - kw) // sx + 1])


class Program:
    """RV32EM fixed-width instruction builder with local branch fixups."""

    def __init__(self, scalar_register):
        self.words = []
        self.labels = {}
        self.fixups = []
        self.scalar_register = scalar_register
        self.namespace = ""
        self.pair_candidates = 0

    def emit(self, word):
        self.words.append(word)

    def label(self, name):
        self.labels[self.namespace + name] = 4 * len(self.words)

    def addi(self, dst, a, imm, f3=0):
        self.emit(((imm & 0xFFF) << 20) | (a << 15) | (f3 << 12) | (dst << 7) | 0x13)

    def li(self, dst, value):
        lo = (value + 2048) % 4096 - 2048
        if value != lo:
            self.emit(((value - lo) & 0xFFFFF000) | (dst << 7) | 0x37)
            if lo:
                self.addi(dst, dst, lo)
        else:
            self.addi(dst, 0, lo)

    def arithmetic(self, dst, a, b, f3=0, f7=0):
        self.emit((f7 << 25) | (b << 20) | (a << 15) | (f3 << 12) | (dst << 7) | 0x33)

    def branch(self, f3, a, b, label):
        self.fixups.append((len(self.words), self.namespace + label))
        self.emit((b << 20) | (a << 15) | (f3 << 12) | 0x63)

    def jump(self, label):
        self.fixups.append((len(self.words), self.namespace + label))
        self.emit(0x6F)

    def load(self, dst, addr, dtype, disp=0):
        self.emit((disp << 20) | (addr << 15) | (_LOAD[dtype] << 12) | (dst << 7) | 0x03)

    def store(self, src, addr, dtype, disp=0):
        access = 2 if dtype == "int32" else 0
        self.emit((disp >> 5) << 25 | (src << 20) | (addr << 15) | (access << 12) | (disp & 31) << 7 | 0x23)

    def address(self, dst, index, base, dtype, temp=9):
        self.li(temp, base)
        self.indexed(dst, index, temp, dtype)

    def indexed(self, dst, index, base_register, dtype):
        if dtype in {"int32", "int64"}:
            self.addi(dst, index, 2 if dtype == "int32" else 3, f3=1)
            index = dst
        self.arithmetic(dst, index, base_register)

    def independent_constant(self, dst, value):
        # Appears immediately after a vector load or ALU op that does not read dst.
        # The scalar constant has no incoming RAW dependency on that vector op.
        reg = self.scalar_register or dst
        if self.scalar_register:
            self.pair_candidates += 1
        self.li(reg, value)
        return reg

    def image(self):
        words = self.words.copy()
        for index, label in self.fixups:
            word = words[index]
            delta = self.labels[label] - 4 * index
            if word & 127 == 0x63:
                if not -4096 <= delta < 4096:
                    raise ValueError("GPU branch target outside signed 13-bit range")
                imm = delta & 0x1FFF
                encoded = (((imm >> 12) & 1) << 31) | (((imm >> 5) & 63) << 25)
                encoded |= (((imm >> 1) & 15) << 8) | (((imm >> 11) & 1) << 7)
            else:
                if not -(1 << 20) <= delta < (1 << 20):
                    raise ValueError("GPU jump target outside signed 21-bit range")
                imm = delta & 0x1FFFFF
                encoded = (((imm >> 20) & 1) << 31) | (((imm >> 1) & 1023) << 21)
                encoded |= (((imm >> 11) & 1) << 20) | (((imm >> 12) & 255) << 12)
            words[index] = word | encoded
        return words


def _clip(p, dtype, mode, materialized):
    if dtype in {"int32", "int64"}:
        return
    if mode == "wrap":
        if not materialized:
            if dtype == "int8":
                p.addi(7, 7, 24, f3=1)
                p.addi(7, 7, 24 | 0x400, f3=5)
            else:
                p.addi(7, 7, 255, f3=7)
        return
    low, high = (-128, 127) if dtype == "int8" else (0, 255)
    p.li(8, low)
    p.branch(5, 7, 8, "clip_low_done")
    p.addi(7, 8, 0)
    p.label("clip_low_done")
    p.li(8, high)
    p.branch(5, 8, 7, "clip_high_done")
    p.addi(7, 8, 0)
    p.label("clip_high_done")


def _activation(p, op, mode):
    values = []
    for x in range(-128, 128):
        if mode == "legacy_q4_knots":
            knot = ((x + 8) >> 4) + 8
            scaled = abs(x) * _LEGACY_SILU_KNOTS[knot] // 256
            value = -scaled if x < 0 else scaled
        else:
            q = x / 16
            if op == "Sigmoid":
                y = 1 / (1 + exp(-q))
            elif op == "SiLU":
                y = q / (1 + exp(-q))
            else:
                y = q * (1 + erf(q / sqrt(2))) / 2
            value = min(127, max(-128, round(16 * y)))
        values.append(value)
    _lookup(p, values, -128)


def _lookup(p, values, first_input, value_reg=7, compare_reg=8, label="activation"):
    runs = []
    for x, value in enumerate(values, first_input):
        if not runs or runs[-1][1] != value:
            runs.append((x, value))

    def tree(first, end):
        if end - first == 1:
            p.li(value_reg, runs[first][1])
            p.jump(f"{label}_end")
            return
        mid = (first + end) // 2
        upper = f"{label}_{first}_{mid}"
        p.li(compare_reg, runs[mid][0])
        p.branch(5, value_reg, compare_reg, upper)
        tree(first, mid)
        p.label(upper)
        tree(mid, end)

    tree(0, len(runs))
    p.label(f"{label}_end")


def _gemm(p, inputs, offsets, attrs):
    transpose = attrs.get("transpose_b", False)
    a, b = inputs[:2]
    da, db = a["dtype"], b["dtype"]
    contract, columns = b["shape"][::-1] if transpose else b["shape"]
    # x5 is the global output element. Quotient picks A's row, remainder B's column.
    p.li(14, columns)
    p.arithmetic(6, 5, 14, f3=5, f7=1)
    p.arithmetic(8, 5, 14, f3=7, f7=1)
    p.li(11, contract)
    p.arithmetic(6, 6, 11, f7=1)
    p.address(9, 6, offsets[0], da, temp=15)
    column_offset = 8
    if transpose:
        p.arithmetic(6, 8, 11, f7=1)
        column_offset = 6
    p.address(10, column_offset, offsets[1], db, temp=15)
    p.li(14, (1 if transpose else columns) * _WIDTH[db])
    if len(inputs) == 3:
        bias = inputs[2]["dtype"]
        p.address(15, 8, offsets[2], bias, temp=12)
        p.load(7, 15, bias)
    else:
        p.li(7, 0)
    p.label("dot")
    p.load(12, 9, da)
    p.load(13, 10, db)
    _center_pair(p, attrs, 12, 13)
    p.arithmetic(12, 12, 13, f7=1)
    p.arithmetic(7, 7, 12)
    p.addi(9, 9, _WIDTH[da])
    p.arithmetic(10, 10, 14)
    p.addi(11, 11, -1)
    p.branch(1, 11, 0, "dot")


def _gated(p, attrs):
    def rotate(dst, left):
        p.addi(13, dst, left, f3=1)
        p.addi(dst, dst, 32 - left, f3=5)
        p.arithmetic(dst, dst, 13, f3=6)

    p.addi(14, 7, 0)
    p.addi(9, 7, 255, f3=7)
    p.li(10, 0x01234567)
    p.arithmetic(7, 9, 10)
    p.li(10, 0x9E3779B9)
    p.arithmetic(8, 7, 10, f3=4)
    p.li(10, attrs.get("threshold", 0))
    p.li(11, 0)
    p.li(12, attrs.get("rounds", 64))
    p.branch(4, 14, 10, "gated_low_setup")

    p.li(10, 0xA5A5A5A5)
    p.addi(15, 9, 11, f3=1)
    p.label("gated_high")
    rotate(7, 5)
    p.arithmetic(14, 8, 9)
    p.arithmetic(14, 14, 11)
    p.arithmetic(7, 7, 14, f3=4)
    rotate(8, 29)
    p.arithmetic(14, 7, 10, f3=4)
    p.arithmetic(8, 8, 14)
    p.addi(14, 8, 7, f3=5)
    p.arithmetic(7, 7, 14)
    p.arithmetic(7, 7, 15, f3=4)
    p.addi(11, 11, 1)
    p.branch(1, 11, 12, "gated_high")
    p.jump("gated_done")

    p.label("gated_low_setup")
    p.li(10, 0x3C6EF372)
    p.addi(15, 9, 1, f3=5)
    p.label("gated_low")
    rotate(7, 25)
    p.arithmetic(14, 8, 9, f3=4)
    p.arithmetic(14, 14, 11, f3=4)
    p.arithmetic(7, 7, 14)
    rotate(8, 3)
    p.arithmetic(14, 7, 10)
    p.arithmetic(8, 8, 14, f3=4)
    p.addi(14, 8, 9, f3=1)
    p.arithmetic(7, 7, 14, f3=4)
    p.arithmetic(7, 7, 15)
    p.addi(11, 11, 1)
    p.branch(1, 11, 12, "gated_low")

    p.label("gated_done")
    p.arithmetic(7, 7, 8, f3=4)


def _elementwise(p, op, second, offset, attrs, dtype):
    # The previous IR result is in x7, either read from memory or held by this lane.
    if op in {"Add", "Mul"}:
        dtype = second["dtype"]
        base = p.independent_constant(10, offset)
        index = 5
        if attrs.get("broadcast") == "last":
            p.li(9, second["shape"][0])
            p.arithmetic(11, 5, 9, f3=7, f7=1)
            index = 11
        p.indexed(6, index, base, dtype)
        p.load(8, 6, dtype)
        p.arithmetic(7, 7, 8, f7=int(op == "Mul"))
    elif op == "Relu":
        p.branch(5, 7, 0, "relu_done")
        p.li(7, 0)
        p.label("relu_done")
    elif op == "Requant":
        _requant(p, attrs)
    elif op == "GatedBranch":
        _gated(p, attrs)
    elif op == "Lookup":
        table = attrs["table"]
        if dtype == "int8":
            _lookup(p, table[128:] + table[:128], -128)
        else:
            _lookup(p, table, 0)
    else:
        _activation(p, op, attrs.get("mode"))


def _center_pair(p, attrs, a, b):
    for reg, key in ((a, "a_zero_point"), (b, "b_zero_point")):
        zero = attrs.get(key, 0)
        if zero:
            p.addi(reg, reg, -zero)


def _constant_arith(p, dst, src, value, operation, temp):
    if operation in {"mul", "divu"} and value == 1:
        if dst != src:
            p.addi(dst, src, 0)
        return
    if operation == "remu" and value == 1:
        p.li(dst, 0)
        return
    f3 = {"mul": 0, "divu": 5, "remu": 7}[operation]
    p.li(temp, value)
    p.arithmetic(dst, src, temp, f3=f3, f7=1)


def _requant(p, attrs):
    multiplier = attrs.get("multiplier", 1)
    shift = attrs.get("shift", 0)
    divisor = attrs.get("divisor", 1 << shift)
    input_zero = attrs.get("input_zero_point", 0)
    if input_zero:
        p.li(9, input_zero)
        p.arithmetic(7, 7, 9, f7=0x20)
    if multiplier != 1:
        scale = p.independent_constant(10, multiplier)
        p.arithmetic(7, 7, scale, f7=1)
    if divisor > 1 and attrs.get("rounding", "floor") == "nearest_even":
        if divisor < (1 << 31):
            # RISC-V DIV rounds toward zero. Compare the unsigned remainder's
            # magnitude with its complement to avoid overflowing 2*remainder.
            p.li(9, divisor)
            p.arithmetic(8, 7, 9, f3=6, f7=1)
            p.arithmetic(10, 7, 9, f3=4, f7=1)
            p.addi(11, 7, 31 | 0x400, f3=5)
            p.branch(5, 8, 0, "rounded_abs")
            p.arithmetic(8, 0, 8, f7=0x20)
            p.label("rounded_abs")
            p.arithmetic(9, 9, 8, f7=0x20)
            p.branch(6, 8, 9, "rounded_done")
            p.branch(6, 9, 8, "rounded_up")
            p.addi(8, 10, 1, f3=7)
            p.branch(0, 8, 0, "rounded_done")
            p.label("rounded_up")
            p.addi(11, 11, 1, f3=6)
            p.arithmetic(10, 10, 11)
        else:
            # A divisor of 2**31 is not representable as signed RV32.
            p.addi(10, 7, 31 | 0x400, f3=5)
            p.addi(8, 7, 1, f3=1)
            p.li(9, 1 << 31)
            p.branch(6, 8, 9, "rounded_done")
            p.branch(6, 9, 8, "rounded_shift_up")
            p.branch(0, 10, 0, "rounded_done")
            p.label("rounded_shift_up")
            p.addi(10, 10, 1)
        p.label("rounded_done")
        p.addi(7, 10, 0)
    elif shift and "divisor" not in attrs:
        p.addi(7, 7, shift | 0x400, f3=5)
    zero = attrs.get("zero_point", 0)
    if zero:
        p.li(9, zero)
        p.arithmetic(7, 7, 9)


def _spatial(p, op, inputs, offsets, out, attrs):
    _, h, w, channels = inputs[0]["shape"]
    _, oh, ow, cout = out["shape"]
    da = inputs[0]["dtype"]
    conv = op == "Conv"
    kh, kw = inputs[1]["shape"][:2] if conv else attrs["kernel_shape"]
    sy, sx = attrs.get("strides", [1, 1])
    top, left = attrs.get("pads", [0, 0, 0, 0])[:2]
    # Keep output channel, start x/y, and base of input batch live across taps.
    _constant_arith(p, 8, 5, cout, "remu", 13)
    _constant_arith(p, 12, 5, cout, "divu", 13)
    _constant_arith(p, 9, 12, ow, "remu", 13)
    _constant_arith(p, 12, 12, ow, "divu", 13)
    _constant_arith(p, 10, 12, oh, "remu", 13)
    _constant_arith(p, 11, 12, oh, "divu", 13)
    _constant_arith(p, 11, 11, h * w * channels, "mul", 13)
    _constant_arith(p, 9, 9, sx, "mul", 13)
    _constant_arith(p, 10, 10, sy, "mul", 13)
    if conv and len(inputs) == 3:
        p.address(6, 8, offsets[2], inputs[2]["dtype"], temp=13)
        p.load(7, 6, inputs[2]["dtype"])
    else:
        p.li(7, 0 if conv else {"int8": -128, "uint8": 0, "int32": -(1 << 31)}[da])
    for ky in range(kh):
        for kx in range(kw):
            prefix = f"tap_{ky}_{kx}"
            end = prefix + "_end"
            p.addi(12, 10, ky - top)
            p.branch(4, 12, 0, end)
            p.li(13, h)
            p.branch(5, 12, 13, end)
            p.addi(13, 9, kx - left)
            p.branch(4, 13, 0, end)
            p.li(14, w)
            p.branch(5, 13, 14, end)
            p.arithmetic(12, 12, 14, f7=1)
            p.arithmetic(12, 12, 13)
            _constant_arith(p, 12, 12, channels, "mul", 13)
            p.arithmetic(12, 12, 11)
            if not conv:
                p.arithmetic(12, 12, 8)
            p.address(12, 12, offsets[0], da, temp=13)
            if conv:
                db = inputs[1]["dtype"]
                weight = offsets[1] + (ky * kw + kx) * channels * cout * _WIDTH[db]
                p.address(13, 8, weight, db, temp=14)
                p.li(14, channels)
                p.label(prefix + "_channels")
                p.load(15, 12, da)
                p.load(6, 13, db)
                _center_pair(p, attrs, 15, 6)
                p.arithmetic(15, 15, 6, f7=1)
                p.arithmetic(7, 7, 15)
                p.addi(12, 12, _WIDTH[da])
                p.li(15, cout * _WIDTH[db])
                p.arithmetic(13, 13, 15)
                p.addi(14, 14, -1)
                p.branch(1, 14, 0, prefix + "_channels")
            else:
                p.load(13, 12, da)
                p.branch(5, 7, 13, end)
                p.addi(7, 13, 0)
            p.label(end)
    if conv and attrs.get("shift", 0):
        p.addi(7, 7, attrs["shift"] | 0x400, f3=5)


def _layout_load(p, index, base, dtype):
    p.address(6, index, base, dtype)
    p.load(7, 6, "int32" if dtype == "int64" else dtype)
    if dtype == "int64":
        p.load(12, 6, "int32", 4)


def _layout(p, op, inputs, offsets, out, attrs):
    dtype, dst = out["dtype"], out["shape"]
    src = inputs[0]["shape"]
    if op == "Expand":
        src = [1] * (len(dst) - len(src)) + src
    if op == "Reshape":
        _layout_load(p, 5, offsets[0], dtype)
        return
    if op == "Concat":
        axis = attrs["axis"] % len(dst)
        inner = prod(dst[axis + 1:])
        _constant_arith(p, 8, 5, inner * dst[axis], "divu", 10)
        _constant_arith(p, 11, 5, inner, "divu", 10)
        _constant_arith(p, 11, 11, dst[axis], "remu", 10)
        _constant_arith(p, 14, 5, inner, "remu", 10)
        accumulated = 0
        for i, tensor in enumerate(inputs):
            width = tensor["shape"][axis]
            p.li(10, accumulated + width)
            if i + 1 < len(inputs):
                p.branch(7, 11, 10, f"concat_{i + 1}")
            _constant_arith(p, 13, 8, width * inner, "mul", 10)
            p.li(10, accumulated)
            p.arithmetic(15, 11, 10, f7=0x20)
            _constant_arith(p, 15, 15, inner, "mul", 10)
            p.arithmetic(13, 13, 15)
            p.arithmetic(13, 13, 14)
            _layout_load(p, 13, offsets[i], dtype)
            if i + 1 < len(inputs):
                p.jump("concat_done")
                p.label(f"concat_{i + 1}")
            accumulated += width
        p.label("concat_done")
        return
    p.li(8, 0)
    for axis, extent in enumerate(dst):
        if op == "Expand" and src[axis] == 1:
            continue
        _constant_arith(p, 11, 5, prod(dst[axis+1:]), "divu", 13)
        _constant_arith(p, 11, 11, extent, "remu", 13)
        source_axis = axis
        if op == "Transpose":
            source_axis = attrs["perm"][axis]
        elif op == "Slice":
            _constant_arith(p, 11, 11, attrs["steps"][axis], "mul", 13)
            p.li(13, attrs["starts"][axis])
            p.arithmetic(11, 11, 13)
        elif extent != src[axis]:
            _constant_arith(p, 11, 11, src[axis], "mul", 13)
            _constant_arith(p, 11, 11, extent, "divu", 13)
        _constant_arith(p, 11, 11, prod(src[source_axis+1:]), "mul", 13)
        p.arithmetic(8, 8, 11)
    _layout_load(p, 8, offsets[0], dtype)


def _argmax(p, tensor, offset, attrs):
    shape, dtype = tensor["shape"], tensor["dtype"]
    axis = attrs.get("axis", 0) % len(shape)
    inner = prod(shape[axis+1:])
    extent = shape[axis]
    _constant_arith(p, 8, 5, inner, "divu", 10)
    _constant_arith(p, 8, 8, extent * inner, "mul", 10)
    _constant_arith(p, 11, 5, inner, "remu", 10)
    p.arithmetic(8, 8, 11)
    p.address(8, 8, offset, dtype)
    p.load(15, 8, dtype)
    p.li(7, 0)
    p.li(13, 0)
    p.li(9, inner * _WIDTH[dtype])
    p.li(12, extent - 1)
    p.label("argmax_next")
    p.branch(0, 13, 12, "argmax_done")
    p.arithmetic(8, 8, 9)
    p.addi(13, 13, 1)
    p.load(14, 8, dtype)
    if attrs.get("select_last_index", False):
        p.branch(4, 14, 15, "argmax_next")
    else:
        p.branch(5, 15, 14, "argmax_next")
    p.addi(15, 14, 0)
    p.addi(7, 13, 0)
    p.jump("argmax_next")
    p.label("argmax_done")


def _rows(p, op, tensor, offset, attrs):
    cols = tensor["shape"][-1]
    _constant_arith(p, 8, 5, cols, "divu", 15)
    p.li(10, cols)
    if op == "RowSoftmax" and attrs.get("causal_mask"):
        p.addi(11, 8, 1)
        p.branch(7, 11, 10, "softmax_active")
        p.addi(10, 11, 0)
        p.label("softmax_active")
        _constant_arith(p, 11, 5, cols, "remu", 15)
        p.li(7, 0)
        p.branch(7, 11, 10, "row_done")
    _constant_arith(p, 8, 8, cols, "mul", 15)
    p.address(8, 8, offset, "int8")
    p.arithmetic(10, 8, 10)
    p.address(14, 5, offset, "int8")
    p.addi(9, 8, 0)
    if op == "RMSNorm":
        p.li(11, 0)
        p.label("rms_sum")
        p.load(12, 9, "int8")
        p.arithmetic(12, 12, 12, f7=1)
        p.arithmetic(11, 11, 12)
        p.addi(9, 9, 1)
        p.branch(1, 9, 10, "rms_sum")
        p.li(13, 0)
        for exponent in range(7):
            p.li(12, 2 * cols * 4 ** exponent)
            p.branch(6, 11, 12, "rms_scale")
            p.addi(13, 13, 1)
        p.label("rms_scale")
        p.load(7, 14, "int8")
        p.addi(7, 7, 4, f3=1)
        p.arithmetic(7, 7, 13, f3=5, f7=0x20)
        return
    p.li(11, -128)
    p.label("softmax_max")
    p.load(12, 9, "int8")
    p.branch(5, 11, 12, "softmax_max_next")
    p.addi(11, 12, 0)
    p.label("softmax_max_next")
    p.addi(9, 9, 1)
    p.branch(1, 9, 10, "softmax_max")
    p.addi(9, 8, 0)
    p.li(13, 0)
    p.label("softmax_sum")
    p.load(12, 9, "int8")
    p.arithmetic(12, 11, 12, f7=0x20)
    p.addi(12, 12, 8)
    p.addi(12, 12, 4, f3=5)
    for d in range(17):
        p.branch(0, 12, 0, f"softmax_knot_{d}")
        p.addi(12, 12, -1)
    p.li(12, 0)
    p.jump("softmax_weight")
    for d in range(17):
        p.label(f"softmax_knot_{d}")
        p.li(12, int(65536 * exp(-d) + 0.5))
        p.jump("softmax_weight")
    p.label("softmax_weight")
    p.branch(1, 9, 14, "softmax_sum_next")
    p.addi(7, 12, 0)
    p.label("softmax_sum_next")
    p.arithmetic(13, 13, 12)
    p.addi(9, 9, 1)
    p.branch(1, 9, 10, "softmax_sum")
    p.addi(7, 7, 8, f3=1)
    p.arithmetic(7, 7, 13, f3=5, f7=1)
    p.label("row_done")


def _qsoftmax(p, tensor, offset, attrs):
    """Integer Q23 exponent weights and RNE output on the RV32 GPU."""
    shape, dtype = tensor["shape"], tensor["dtype"]
    dim = attrs["axis"]
    width, step = shape[dim], prod(shape[dim+1:])
    span = width * step
    zero = attrs["zero_point"]
    causal = attrs.get("causal_offset")

    def advance(reg):
        if step <= 2047:
            p.addi(reg, reg, step)
        else:
            p.li(15, step)
            p.arithmetic(reg, reg, 15)

    # Convert the flat output ID to a strided softmax lane.
    _constant_arith(p, 8, 5, span, "divu", 15)
    p.li(10, width)
    if causal is not None:
        _constant_arith(p, 11, 8, shape[-2], "remu", 15)
        p.addi(11, 11, causal + 1)
        p.branch(7, 11, 10, "qsoft_active")
        p.addi(10, 11, 0)
        p.label("qsoft_active")
        _constant_arith(p, 11, 5, width, "remu", 15)
        p.branch(6, 11, 10, "qsoft_unmasked")
        p.li(7, zero)
        p.jump("qsoft_done")
        p.label("qsoft_unmasked")
    _constant_arith(p, 8, 8, span, "mul", 15)
    _constant_arith(p, 11, 5, step, "remu", 15)
    p.arithmetic(8, 8, 11)
    p.address(8, 8, offset, dtype, temp=15)
    p.address(14, 5, offset, dtype, temp=15)
    _constant_arith(p, 10, 10, step, "mul", 15)
    p.arithmetic(10, 8, 10)

    # Pass one finds the maximum among unmasked scores.
    p.addi(9, 8, 0)
    p.li(11, -128 if dtype == "int8" else 0)
    p.label("qsoft_max")
    p.load(12, 9, dtype)
    p.branch(5, 11, 12, "qsoft_max_continue")
    p.addi(11, 12, 0)
    p.label("qsoft_max_continue")
    advance(9)
    p.branch(1, 9, 10, "qsoft_max")

    # Pass two sums exponent weights. The delta is independent of the
    # input quantizer's zero point and is always in [0, 255].
    numerator, denominator = attrs["in_scale"]
    si = numerator / denominator
    weights = [round((1 << 23) * exp(-si*d)) for d in range(256)]
    p.addi(9, 8, 0)
    p.li(13, 0)
    p.label("qsoft_sum")
    p.load(12, 9, dtype)
    p.arithmetic(12, 11, 12, f7=0x20)
    _lookup(p, weights, 0, value_reg=12, compare_reg=15, label="qsoft_exponent")
    p.branch(1, 9, 14, "qsoft_add")
    p.addi(7, 12, 0)
    p.label("qsoft_add")
    p.arithmetic(13, 13, 12)
    advance(9)
    p.branch(0, 9, 10, "qsoft_normalize")
    p.jump("qsoft_sum")

    # Integer fractional long division avoids a wide numerator. The
    # denominator is at most 255 * 2**23 and doubled remainder fits u32.
    p.label("qsoft_normalize")
    quantum = attrs["out_scale"][1]
    p.branch(1, 7, 13, "qsoft_fraction")
    p.li(7, quantum)
    p.jump("qsoft_zero_point")
    p.label("qsoft_fraction")
    p.addi(8, 7, 0)
    p.li(7, 0)
    for bit in range(quantum.bit_length() - 1):
        skip = f"qsoft_bit_{bit}"
        p.addi(8, 8, 1, f3=1)
        p.addi(7, 7, 1, f3=1)
        p.branch(6, 8, 13, skip)
        p.arithmetic(8, 8, 13, f7=0x20)
        p.addi(7, 7, 1)
        p.label(skip)
    p.addi(8, 8, 1, f3=1)
    p.branch(6, 8, 13, "qsoft_zero_point")
    p.branch(6, 13, 8, "qsoft_round_up")
    p.addi(9, 7, 1, f3=7)
    p.branch(0, 9, 0, "qsoft_zero_point")
    p.label("qsoft_round_up")
    p.addi(7, 7, 1)
    p.label("qsoft_zero_point")
    if zero:
        p.addi(7, 7, zero)
    p.label("qsoft_done")


def _advanced(p, op, inputs, offsets, output, attrs):
    if op in _LAYOUT:
        _layout(p, op, inputs, offsets, output, attrs)
    elif op == "Cast":
        source, dest = inputs[0]["dtype"], output["dtype"]
        _layout_load(p, 5, offsets[0], source)
        if dest == "int64" and source != "int64":
            p.addi(12, 7, 31 | 0x400, f3=5)
    elif op in {"Conv", "MaxPool"}:
        _spatial(p, op, inputs, offsets, output, attrs)
    elif op == "ArgMax":
        _argmax(p, inputs[0], offsets[0], attrs)
    elif op == "QSoftmax":
        _qsoftmax(p, inputs[0], offsets[0], attrs)
    else:
        _rows(p, op, inputs[0], offsets[0], attrs)


def _elf(words, entry):
    """ELF32 little-endian ET_EXEC / EM_RISCV / EF_RISCV_RVE."""
    data_size = 4 * len(words)
    elf_header_size, program_header_size, section_header_size = 52, 32, 40
    text_offset = 4096
    shstrings = b"\0.text\0.shstrtab\0"
    string_offset = text_offset + data_size
    section_offset = (string_offset + len(shstrings) + 3) & ~3
    ident = b"\x7fELF\x01\x01\x01" + bytes(9)
    header = pack("<16sHHIIIIIHHHHHH", ident, 2, 243, 1, entry, elf_header_size,
                  section_offset, 8, elf_header_size, program_header_size, 1,
                  section_header_size, 3, 2)
    segment = pack("<8I", 1, text_offset, 0, 0, data_size, data_size, 5, text_offset)
    text = pack(f"<{len(words)}I", *words)
    pad = bytes(section_offset - string_offset - len(shstrings))
    sections = (bytes(section_header_size) +
                pack("<10I", 1, 1, 6, 0, text_offset, data_size, 0, 0, 4, 0) +
                pack("<10I", 7, 3, 0, 0, string_offset, len(shstrings), 0, 0, 1, 0))
    return header + segment + bytes(text_offset - len(header) - len(segment)) + text + shstrings + pad + sections


def _ranges(graph, plan, stage, window, variant):
    ranges = {}
    names = [*stage["inputs"], stage["output"]]
    for name in names:
        tensor = graph["tensors"][name]
        width = _WIDTH[tensor["dtype"]]
        size = width * prod(tensor["shape"])
        allocation = plan["allocations"][name]
        offset = allocation["offset"]
        if size <= 0 or size > allocation["size"] or offset < 0 or offset % width:
            raise ValueError(f"invalid GPU allocation for {name}")
        if offset // window != (offset + size - 1) // window:
            raise ValueError(f"GPU tensor {name} crosses a data page boundary")
        ranges[name] = {"offset": offset, "size": size}
    output = ranges[stage["output"]]
    for name in stage["inputs"]:
        operand = ranges[name]
        if max(output["offset"], operand["offset"]) < min(
                output["offset"] + output["size"], operand["offset"] + operand["size"]):
            raise ValueError(f"GPU output overlaps operand {name}")
    pages = {item["offset"] // window for item in ranges.values()}
    if len(pages) != 1:
        raise ValueError("GPU operands must share one data page")
    page = pages.pop()
    if page >= 512 or (variant == "v6" and page):
        raise ValueError("GPU data page is outside target's address window")
    return page, ranges


def _is_pairable(node):
    return (node["op"] in {"Add", "Mul"} or
            (node["op"] == "Requant" and node.get("attrs", {}).get("multiplier", 1) != 1))


def _stage_program(stage, graph, ranges, config, page):
    variant = config["variant"]
    lanes, warps = config["lanes"], config["warps"]
    slots = config.get("launch_warps", warps)
    if type(slots) is not int or not 1 <= slots <= warps:
        raise ValueError("GPU launch_warps must fit physical Warp count")
    window = config["window_bytes"]
    parts = stage.get("fused") or [stage]
    # The frozen G3 cluster faults with scalar mode kept live across the
    # divergent tap loop. Leave that kernel in the vector register namespace.
    paired = (config.get("issue_width", 1) > 1 and parts[0]["op"] != "Conv"
              and any(_is_pairable(node) for node in parts))
    public = (16 if variant == "v6" else 24) if paired else None
    p = Program(public)
    tensors = graph["tensors"]
    dst = tensors[stage["output"]]
    count = prod(dst["shape"])
    job_threads = 255 if variant == "v6" else lanes
    jobs = []
    for start in range(0, count, job_threads):
        warp = len(jobs) % slots
        threads = min(job_threads, count - start)
        entry = len(p.words) * 4
        jobs.append({"entry": entry, "threads": threads, "warp": warp, "page": page,
                     "first_element": start, "wave": len(jobs) // slots})
        if variant != "v6":
            jobs[-1]["gpu_job_reg"] = (warp << 24) | (threads << 16) | page
        p.li(15, start)
        p.jump("kernel")
    p.label("kernel")
    if paired:
        p.emit(_SCALAR_ENABLE)
    p.emit(0xCC3022F3)  # csrrs x5, logical GID CSR, x0
    p.arithmetic(5, 5, 15)
    offsets = {name: rec["offset"] % window for name, rec in ranges.items()}
    for i, node in enumerate(parts):
        p.namespace = f"part_{i}/"
        op, names = node["op"], node["inputs"]
        attrs = node.get("attrs", {})
        if op == "Gemm":
            _gemm(p, [tensors[name] for name in names], [offsets[name] for name in names], attrs)
        elif op in _ADVANCED:
            _advanced(p, op, [tensors[name] for name in names], [offsets[name] for name in names],
                      tensors[node["output"]], attrs)
        else:
            if i == 0:
                dtype = tensors[names[0]]["dtype"]
                p.address(6, 5, offsets[names[0]], dtype)
                p.load(7, 6, dtype)
            second = tensors[names[1]] if len(names) == 2 else None
            offset = offsets[names[1]] if second else None
            _elementwise(p, op, second, offset, attrs, tensors[names[0]]["dtype"])
        _clip(p, tensors[node["output"]]["dtype"], attrs.get("narrowing", "saturate"),
              materialized=i == len(parts) - 1)
    p.address(6, 5, ranges[stage["output"]]["offset"] % window, dst["dtype"])
    if dst["dtype"] == "int64":
        p.store(7, 6, "int32")
        p.store(12 if parts[0]["op"] in _LAYOUT | {"Cast"} else 0, 6, "int32", 4)
    else:
        p.store(7, 6, dst["dtype"])
    p.emit(_EXIT)
    words = p.image()
    if len(words) * 4 > config["code_bytes"]:
        raise ValueError(f"GPU stage {stage['id']} exceeds instruction RAM budget")
    return words, jobs, p.pair_candidates


def lower(graph, plan, profile, output: Path):
    """Write per-stage linked device ELF, word HEX and FPGA job metadata."""
    config = profile["gpu"]
    variant, lanes, warps, window = (config[k] for k in ("variant", "lanes", "warps", "window_bytes"))
    if variant not in _VARIANTS or window != 4096:
        raise ValueError("unknown GPU ABI or data page size")
    if variant == "v6" and (lanes, warps) != (8, 1):
        raise ValueError("v6 profile requires 8 lanes and one launch slot")
    if variant in {"g3", "g4"} and (lanes, warps) not in {(16, 4), (8, 8)}:
        raise ValueError("G3/G4 requires 16 threads / 4 warps or 8 threads / 8 warps")
    abi = "v6-launch"
    slots = config.get("launch_warps", warps)
    if variant != "v6":
        abi = "z15-mmio-v1" if slots <= 4 else "z15-mmio-8warp-projected"
    bundle = {"variant": variant, "lanes": lanes, "warps": warps,
              "launch_warps": slots, "window_bytes": window, "dispatch_abi": abi, "stages": []}
    artifacts = []
    for index, stage in enumerate(plan["stages"]):
        if stage["device"] != "GPU":
            continue
        if not supports(stage, graph, profile):
            raise ValueError(f"GPU does not support stage {stage['id']} ({stage['op']})")
        page, ranges = _ranges(graph, plan, stage, window, variant)
        words, jobs, pairs = _stage_program(stage, graph, ranges, config, page)
        base = f"gpu/{index:04d}"
        bundle["stages"].append({"id": stage["id"], "elf": base + ".elf", "hex": base + ".hex",
                                 "jobs": jobs, "tensor_ranges": ranges,
                                 "static_words": len(words), "scalar_pair_candidates": pairs})
        artifacts.append((base, words, jobs[0]["entry"]))
    output.mkdir(parents=True, exist_ok=True)
    for base, words, entry in artifacts:
        elf = output / (base + ".elf")
        elf.parent.mkdir(exist_ok=True)
        elf.write_bytes(_elf(words, entry))
        (output / (base + ".hex")).write_text("".join(f"{word:08x}\n" for word in words))
    (output / "gpu.json").write_text(json.dumps(bundle, indent=2) + "\n")
    return bundle
