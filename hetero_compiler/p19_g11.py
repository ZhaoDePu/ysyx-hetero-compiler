"""G11B RV32 register-window microkernel templates and offline instruction model."""

import math
import struct

from .p19 import ABI

MASK = 0xffffffff
Q = 1 << 16
CUSTOM = ABI["g11_opcode"]
RFORM = ABI["g11_rform"]


def rtype(opcode, rd, rs1, rs2, funct7, funct3):
    if not all(type(r) is int and 0 <= r < 32 for r in (rd, rs1, rs2)):
        raise ValueError("invalid GPR")
    return funct7 << 25 | rs2 << 20 | rs1 << 15 | funct3 << 12 | rd << 7 | opcode


def g11(name, rd, rs1=0, rs2=0):
    f7, f3 = RFORM[name]
    if f7 == 0 and (rd or any(r != 0 and r < 24 for r in (rs1, rs2))):
        raise ValueError("barrier instructions require shared source GPRs and x0 destination")
    return rtype(CUSTOM, rd, rs1, rs2, f7, f3)


class Program:
    def __init__(self):
        self.words = []

    def alu(self, rd, a, b, f3=0, f7=0):
        self.words.append(rtype(0x33, rd, a, b, f7, f3))

    def imm(self, rd, a, value, f3=0):
        self.words.append((value & 4095) << 20 | a << 15 | f3 << 12 | rd << 7 | 0x13)

    def custom(self, name, rd, a, b=0):
        self.words.append(g11(name, rd, a, b))

    def qmul(self, rd, a, b, temp):
        self.alu(temp, a, b, 1, 1)   # MULH
        self.alu(rd, a, b, 0, 1)     # MUL
        self.imm(temp, temp, 16, 1)
        self.imm(rd, rd, 16, 5)
        self.alu(rd, rd, temp, 6)    # OR

    def select(self, rd, yes, no, mask, temp):
        self.alu(temp, yes, no, 4)
        self.alu(temp, temp, mask, 7)
        self.alu(rd, no, temp, 4)


def kernel(name, lanes=8, k=1):
    if lanes not in (4, 8, 16):
        raise ValueError("supported complete warp windows have 4, 8 or 16 lanes")
    p = Program()
    if name == "rmsnorm-a8":
        # x5 Q16 input, x6 per-lane Q16 gamma. x10 normalized x13 packed A8.
        p.qmul(8, 5, 5, 15)
        p.custom("REDUCE_SUM", 8, 8)
        p.imm(8, 8, lanes.bit_length() - 1, 5)
        p.imm(8, 8, 1)
        p.custom("RSQRT", 9, 8)
        p.qmul(10, 5, 9, 15)
        p.qmul(10, 10, 6, 15)
        p.custom("REDUCE_AMAX", 11, 10)
        p.custom("RCP", 12, 11)
        p.imm(14, 0, 127)
        p.alu(12, 12, 14, 0, 1)
        p.custom("QUANT_I8", 13, 10, 12)
    elif name == "flashattention-pv":
        # x5..x7 old max/sum/PV scalar; x8,x9 score and one V column per lane.
        p.custom("REDUCE_MAX", 10, 8)
        p.alu(14, 5, 10, 2)
        p.alu(14, 0, 14, 0, 32)
        p.select(10, 10, 5, 14, 15)
        p.alu(11, 5, 10, 0, 32)
        p.alu(12, 8, 10, 0, 32)
        p.custom("EXP", 11, 11)
        p.custom("EXP", 12, 12)
        p.qmul(14, 12, 9, 15)
        p.custom("REDUCE_SUM", 13, 12)
        p.custom("REDUCE_SUM", 14, 14)
        p.qmul(6, 6, 11, 15)
        p.qmul(7, 7, 11, 15)
        p.alu(6, 6, 13)
        p.alu(7, 7, 14)
        p.imm(5, 10, 0)
        p.custom("RCP", 12, 6)
        p.qmul(13, 7, 12, 15)
    elif name == "topk":
        if not 1 <= k <= min(4, lanes):
            raise ValueError("Top-K window supports 1..4 retained entries")
        # x5 scores > INT32_MIN; x6 distinct nonnegative IDs. x16..x23 outputs.
        p.imm(15, 0, 1)
        p.imm(15, 15, 31, 1)
        for i in range(k):
            score, token = 16 + 2 * i, 17 + 2 * i
            p.custom("REDUCE_MAX", score, 5)
            p.alu(7, 5, score, 4)
            p.alu(7, 0, 7, 3)
            p.alu(7, 0, 7, 0, 32)
            p.alu(8, 0, 6, 0, 32)
            p.select(8, 15, 8, 7, 9)
            p.custom("REDUCE_MAX", token, 8)
            p.alu(token, 0, token, 0, 32)
            p.alu(7, 6, token, 4)
            p.alu(7, 0, 7, 3)
            p.imm(7, 7, -1)
            p.select(5, 15, 5, 7, 9)
    else:
        raise ValueError("unknown G11B template")
    return p.words


def encode_binary(words):
    return struct.pack(f"<{len(words)}I", *words)


def signed(value):
    value &= MASK
    return value if value < 1 << 31 else value - (1 << 32)


def sfu(name, x):
    # Offline ideal Q16 value. Hardware LUT error remains outside this model.
    if name == "EXP":
        return 0 if x < -4 * Q else round(math.exp(min(x, 0) / Q) * Q)
    if x <= 0:
        return 0
    value = Q * (Q / x if name == "RCP" else math.sqrt(Q / x))
    return min((1 << 31) - 1, round(value))


def emulate(words, inputs, lanes):
    """Interpret the supported register-window instructions for a full active warp."""
    reg = [[0] * lanes for _ in range(32)]
    for i, values in inputs.items():
        if not 0 < i < 24 or len(values) != lanes:
            raise ValueError("invalid private register input")
        reg[i] = [signed(v) for v in values]
    custom = {tuple(value): name for name, value in RFORM.items()}
    for word in words:
        kind, rd = word & 127, (word >> 7) & 31
        f3, a, b, f7 = (word >> 12) & 7, (word >> 15) & 31, (word >> 20) & 31, (word >> 25) & 127
        x, y = reg[a], reg[b]
        if kind == CUSTOM:
            operation = custom.get((f7, f3))
            if operation in ("REDUCE_SUM", "REDUCE_MAX", "REDUCE_AMAX"):
                value = {"REDUCE_SUM": sum(x), "REDUCE_MAX": max(x),
                         "REDUCE_AMAX": max(map(abs, x))}[operation]
                out = [value] * lanes
            elif operation in ("EXP", "RCP", "RSQRT"):
                out = [sfu(operation, value) for value in x]
            elif operation == "QUANT_I8":
                q = [min(127, max(-127, round((a * b) / (1 << 32)))) for a, b in zip(x, y)]
                packed = [sum((q[i + j] & 255) << (8 * j) for j in range(4))
                          for i in range(0, lanes, 4)]
                out = [packed[i // 4] for i in range(lanes)]
            else:
                raise ValueError("unsupported G11B reference instruction")
        elif kind == 0x13 and f3 == 0:
            imm = (word >> 20) - (4096 if word >> 31 else 0)
            out = [v + imm for v in x]
        elif kind == 0x13 and f3 in (1, 5):
            out = [v << b if f3 == 1 else (v & MASK) >> b for v in x]
        elif kind == 0x33:
            operations = {
                (0, 0): lambda x, y: x + y, (32, 0): lambda x, y: x - y,
                (0, 2): lambda x, y: x < y, (0, 3): lambda x, y: (x & MASK) < (y & MASK),
                (0, 4): lambda x, y: x ^ y, (0, 6): lambda x, y: x | y,
                (0, 7): lambda x, y: x & y, (1, 0): lambda x, y: x * y,
                (1, 1): lambda x, y: (x * y) >> 32,
            }
            try:
                out = list(map(operations[f7, f3], x, y))
            except KeyError as exc:
                raise ValueError("unknown RV32 ALU form") from exc
        else:
            raise ValueError("unknown microkernel word")
        if rd:
            reg[rd] = [signed(v) for v in out]
    return reg
