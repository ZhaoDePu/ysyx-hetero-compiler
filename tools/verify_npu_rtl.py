"""Exercise generated native queues using pinned RTL and independent integer goldens."""

from argparse import ArgumentParser
from pathlib import Path
import math
import struct
import subprocess

from hetero_compiler.npu import lower


def sat(x):
    return min(127, max(-128, x))


def tensor(shape, data=None, dtype="int8"):
    return {"shape": shape, "dtype": dtype, **({"data": data} if data is not None else {})}


def deterministic(length, seed):
    return [((i * i * 13 + i * 19 + seed * 7) % 57) - 28 for i in range(length)]


def gemm(a, b, bias, m, n, k, shift, *, relu=False):
    matrix = [
        sat((sum(a[r * k + x] * b[x * n + c] for x in range(k)) + bias[c]) >> shift)
        for r in range(m) for c in range(n)
    ]
    return [max(0, v) for v in matrix] if relu else matrix


def conv(a, b, bias, height, width, cin, cout, shift):
    values = []
    for h in range(height):
        for w in range(width):
            for n in range(cout):
                total = bias[n]
                for kh in range(3):
                    for kw in range(3):
                        row, col = h + kh - 1, w + kw - 1
                        if 0 <= row < height and 0 <= col < width:
                            for channel in range(cin):
                                total += a[(row * width + col) * cin + channel] * b[((kh * 3 + kw) * cin + channel) * cout + n]
                values.append(sat(total >> shift))
    return values


def attention(q, keys, values, m, s, d, n, shift, position, causal):
    scores, probabilities, output = [], [], []
    for r in range(m):
        row = [sum(q[r*d+x] * keys[j*d+x] for x in range(d)) for j in range(s)]
        scores.extend(row)
        stop = position + r + 1 if causal else s
        best = max(row[:stop])
        w = [256 >> ((best-v) >> shift) if j < stop else 0 for j, v in enumerate(row)]
        p = [255 * x // sum(w) for x in w]
        probabilities.extend(p)
        output.extend(sat(sum(p[j] * values[j*n+c] for j in range(s)) >> 8) for c in range(n))
    return scores, probabilities, output


class Case:
    def __init__(self, target, out):
        self.path = out
        self.out = out / "bundle"
        self.out.mkdir(parents=True, exist_ok=True)
        self.graph = {"tensors": {}}
        self.plan = {"stages": [], "allocations": {}}
        self.base = 0
        self.memory = bytearray([165]) * 4096
        self.expected = {}
        self.profile = (
            {"id": "asic_nangate45_p8", "npu": {"rows": 4, "cols": 4}, "sram": {"capacity": 4096}}
            if target == "p8" else
            {"id": "fpga_zu15eg_balanced", "npu": {"rows": 16, "cols": 16}, "vpu": {"enabled": True},
             "sram": {"capacity": 2097152}}
        )
        self.target = target

    def add_tensor(self, name, shape, offset, data=None, dtype="int8"):
        self.graph["tensors"][name] = tensor(shape, data, dtype)
        size = (len(data) if data is not None else math.prod(shape)) * (4 if dtype == "int32" else 1)
        self.allocate(name, offset, size)
        if data is not None:
            self.store(name, data, dtype)

    def allocate(self, name, offset, size):
        self.plan["allocations"][name] = {"offset": offset, "size": size}

    def store(self, name, data, dtype):
        offset = self.plan["allocations"][name]["offset"]
        payload = struct.pack(f"<{len(data)}i", *data) if dtype == "int32" else bytes(x & 255 for x in data)
        self.memory[offset:offset + len(payload)] = payload

    def stage(self, name, op, inputs, output, gold, *, fused=None, extra_expected=None, device="NPU", **attrs):
        stage = dict(id=name, op=op, inputs=inputs, output=output, attrs=attrs, device=device)
        if fused:
            stage["fused"] = fused
        self.plan["stages"].append(stage)
        self.expected[name] = [(output, gold, self.graph["tensors"][output]["dtype"]), *(extra_expected or ())]

    def compile_and_run(self, executable):
        native = lower(self.graph, self.plan, self.profile, self.out)
        for constant in native["constants"]:
            self.store(constant["tensor"], constant["data"], "uint8")
        if self.target == "p8":
            base = native["binary_base"]
            payload = (self.out / "npu.bin").read_bytes()
            self.memory[base:base + len(payload)] = payload
        initial = self.path / "startup.bin"
        initial.write_bytes(self.memory)
        lines = [str(initial) if self.target == "p8" else f"{initial} 0 {len(self.memory)}"]
        for cmd in native["commands"]:
            name = cmd["stage"]
            for tensor_name, values, dtype in self.expected[name]:
                self.store(tensor_name, values, dtype)
            expected = self.path / f"after-{name}.bin"
            expected.write_bytes(self.memory)
            if self.target == "p8":
                lines.append(f"{name} {cmd['a_base']} {cmd['descriptor']} {cmd['program_slots']} {expected}")
            else:
                lines.extend(f"write {reg} {value}" for reg, value in cmd["writes"])
                lines.append(f"check_npu {name} {expected} 0 {len(self.memory)}")
        control = self.path / "run.txt"
        control.write_text("\n".join(lines) + "\n", encoding="ascii")
        subprocess.run([str(executable), str(control)], check=True)


def p8_case(path):
    case = Case("p8", path)
    m, n, k = 5, 8, 32
    a, b, bias = deterministic(m*k, 3), deterministic(k*n, 5), [-400, 10000, -20000, 50, -500, 300, 420, -220]
    case.add_tensor("a", [m, k], 0, a)
    case.add_tensor("b", [k, n], 224, b)
    case.add_tensor("bias", [n], 560, bias, "int32")
    case.add_tensor("direct", [m, n], 640)
    case.add_tensor("tiled", [m, n], 704)
    case.add_tensor("residual", [m, n], 768)
    case.graph["tensors"]["wide"] = tensor([m, n], dtype="int32")
    case.graph["tensors"]["quant"] = tensor([m, n])
    direct = gemm(a, b, [0] * n, m, n, k, 6)
    case.stage("linear", "Gemm", ["a", "b"], "direct", direct, shift=6)
    tiled = gemm(a, b, bias, m, n, k, 5, relu=True)
    case.stage("tiled", "Gemm", ["a", "b", "bias"], "tiled", tiled, fused=[
        dict(op="Gemm", inputs=["a", "b", "bias"], output="wide", attrs={}),
        dict(op="Requant", inputs=["wide"], output="quant", attrs={"shift": 5}),
        dict(op="Relu", inputs=["quant"], output="tiled", attrs={}),
    ])
    case.stage("residual", "Add", ["direct", "tiled"], "residual", [sat(a+b) for a,b in zip(direct,tiled)])
    a_conv = deterministic(3*3*4, 8)
    b_conv = deterministic(3*3*4*4, 9)
    case.add_tensor("conv_a", [1, 3, 3, 4], 832, a_conv)
    case.add_tensor("conv_b", [3, 3, 4, 4], 960, b_conv)
    case.add_tensor("conv_c", [1, 3, 3, 4], 1152)
    case.stage("conv", "Conv", ["conv_a", "conv_b"], "conv_c",
               conv(a_conv, b_conv, [0]*4, 3, 3, 4, 4, 5), shift=5, pads=[1]*4)
    dyn_b = deterministic(4*8, 11)
    case.add_tensor("dynamic_b", [4, 8], 1216, dyn_b)
    # Model this binding as a runtime tensor, while seeding its input bytes.
    del case.graph["tensors"]["dynamic_b"]["data"]
    case.add_tensor("transpose_c", [m, 4], 1280)
    case.add_tensor("norm", [m, 4], 1344)
    case.add_tensor("softmax", [m, 4], 1408, dtype="uint8")
    case.add_tensor("causal", [m, 4], 1472, dtype="uint8")
    transposed = gemm(direct, [dyn_b[c * 8 + x] for x in range(8) for c in range(4)],
                      [0]*4, m, 4, n, 3)
    case.stage("transpose", "Gemm", ["direct", "dynamic_b"], "transpose_c", transposed, shift=3, transpose_b=True)
    normalized = []
    base_softmax, causal_softmax = [], []
    for i in range(m):
        row = transposed[4*i:4*i+4]
        s2 = sum(x*x for x in row)
        exponent = sum(s2 >= 8 * 4**j for j in range(7))
        normalized.extend(sat((16*x) >> exponent) for x in row)
        for count, expected in ((4, base_softmax), (min(4, i+1), causal_softmax)):
            maximum = max(row[:count])
            weights = [int(65536 * math.exp(-((maximum-x+8)//16)) + 0.5) for x in row[:count]]
            expected.extend(min(255, 256*w//sum(weights)) for w in weights)
            expected.extend([0]*(4-count))
    case.stage("rmsnorm", "RMSNorm", ["transpose_c"], "norm", normalized)
    case.stage("softmax", "RowSoftmax", ["transpose_c"], "softmax", base_softmax)
    case.stage("causal", "RowSoftmax", ["transpose_c"], "causal", causal_softmax, causal_mask=True)
    return case


def z15_case(path):
    case = Case("z15", path)
    m, n, k = 5, 4, 132
    a = deterministic(m*k, 3)
    b = deterministic(k*n, 5)
    bias = [-540, 220, -150, 90]
    case.add_tensor("a", [m, k], 0, a)
    case.add_tensor("b", [k, n], 672, b)
    case.allocate("b.__npu_colmajor", 1216, k*n)
    case.add_tensor("bias", [n], 1760, bias, "int32")
    case.allocate("gemm.__npu_partial", 1792, m*n*4)
    case.add_tensor("tiled", [m, n], 1888)
    partial = [bias[c] + sum(a[r*k+x]*b[x*n+c] for x in range(128)) for r in range(m) for c in range(n)]
    tiled = gemm(a, b, bias, m, n, k, 6)
    case.stage("gemm", "Gemm", ["a", "b", "bias"], "tiled", tiled, shift=6,
               extra_expected=[("gemm.__npu_partial", partial, "int32")])
    case.add_tensor("add_rhs", [m, n], 1920, deterministic(m*n, 9))
    case.add_tensor("sum", [m, n], 1952)
    case.add_tensor("relu", [m, n], 1984)
    sums = [sat(a+b) for a, b in zip(tiled, case.graph["tensors"]["add_rhs"]["data"])]
    case.stage("add", "Add", ["tiled", "add_rhs"], "sum", sums, device="VPU")
    case.stage("relu", "Relu", ["sum"], "relu", [max(0, v) for v in sums], device="VPU")
    rows, length, d, vdim = 3, 8, 4, 4
    q, key, value = deterministic(rows*d, 7), deterministic(length*d, 8), deterministic(length*vdim, 9)
    case.add_tensor("q", [rows, d], 2048, q)
    case.add_tensor("k", [length, d], 2080, key)
    case.add_tensor("v", [length, vdim], 2240, value)
    case.allocate("v.__npu_colmajor", 2112, length*vdim)
    case.allocate("attn.__npu_scores", 2272, rows*length*4)
    case.allocate("attn.__npu_probabilities", 2368, rows*length)
    case.add_tensor("attn_out", [rows, vdim], 2400)
    scores, prob, attn = attention(q, key, value, rows, length, d, vdim, 8, 3, True)
    case.stage("attn", "Attention", ["q", "k", "v"], "attn_out", attn, causal=True, exp_shift=8, position=3,
               extra_expected=[("attn.__npu_scores", scores, "int32"), ("attn.__npu_probabilities", prob, "int8")])
    a_conv, b_conv = deterministic(3*3*4, 14), deterministic(3*3*4*4, 16)
    case.add_tensor("conv_a", [1, 3, 3, 4], 2464, a_conv)
    case.add_tensor("conv_b", [3, 3, 4, 4], 2528, b_conv)
    case.allocate("conv_b.__npu_colmajor", 2688, 144)
    case.add_tensor("conv_c", [1, 3, 3, 4], 2848)
    case.stage("conv", "Conv", ["conv_a", "conv_b"], "conv_c", conv(a_conv, b_conv, [0]*4, 3, 3, 4, 4, 5),
               shift=5, pads=[1]*4)
    return case


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("target", choices=("p8", "z15"))
    parser.add_argument("executable", type=Path)
    parser.add_argument("scratch", type=Path)
    args = parser.parse_args()
    fixture = p8_case(args.scratch) if args.target == "p8" else z15_case(args.scratch)
    fixture.compile_and_run(args.executable)
