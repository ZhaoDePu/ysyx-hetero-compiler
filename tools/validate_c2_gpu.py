"""Replay C2 GPU kernels on frozen Z15 using independently calculated full-page goldens."""

from math import exp, prod
from pathlib import Path
import struct
import subprocess
import tempfile

from hetero_compiler import gpu
from hetero_compiler.profile import load_profile

ROOT = Path("/dev/shm/ysyx_compiler_c4a")
PAGE = 4096
FORMATS = {"int8": "b", "uint8": "B", "int32": "i", "int64": "q"}


def fixture(name, op, operands, result_shape, dtype, expected, **attrs):
    tensors = {f"x{i}": {"shape": dims, "dtype": mode, "data": values}
               for i, (dims, mode, values) in enumerate(operands)}
    inputs = list(tensors)
    tensors["out"] = {"shape": result_shape, "dtype": dtype}
    node = {"id": name, "op": op, "inputs": inputs, "output": "out",
            "attrs": attrs, "device": "GPU"}
    return name, {"tensors": tensors}, node, expected


def conv_fixture(fused):
    a = [(n * 19 % 23) - 11 for n in range(24)]
    b = [(n * 13 % 19) - 9 for n in range(54)]
    bias = [-22, 5, 6]
    golden = []
    for y in range(3):
        for x in range(4):
            for ch in range(3):
                acc = bias[ch]
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if not (0 <= y + dy < 3 and 0 <= x + dx < 4):
                            continue
                        for c in range(2):
                            ac = a[((y + dy) * 4 + x + dx) * 2 + c] + 2
                            bc = b[(((dy + 1) * 3 + dx + 1) * 2 + c) * 3 + ch] - 3
                            acc += ac * bc
                golden.append(acc)
    args = [([1, 3, 4, 2], "int8", a), ([3, 3, 2, 3], "int8", b), ([3], "int32", bias)]
    case = fixture("fused_conv" if fused else "conv", "Conv", args, [1, 3, 4, 3], "int32",
                   golden, a_zero_point=-2, b_zero_point=3, strides=[1, 1], pads=[1, 1, 1, 1])
    if not fused:
        return case
    name, graph, conv, _ = case
    graph["tensors"]["tmp"] = graph["tensors"]["out"].copy()
    graph["tensors"]["out"]["dtype"] = "int8"
    conv = {**conv, "output": "tmp"}
    requant = {"id": "r", "op": "Requant", "inputs": ["tmp"], "output": "out",
               "attrs": {"rounding": "nearest_even", "multiplier": 3, "divisor": 7}}
    node = {**conv, "output": "out", "fused": [conv, requant]}
    return name, graph, node, [max(-128, min(127, round(v * 3 / 7))) for v in golden]


def cases():
    yield conv_fixture(False)
    yield conv_fixture(True)
    yield fixture("gemm", "Gemm", [([1, 4], "uint8", [230, 231, 240, 250]),
                                   ([4, 2], "int8", [3, -2, 5, -1, 5, 3, 8, 3])],
                  [1, 2], "int32", [153, 27], a_zero_point=230, b_zero_point=2)
    signed = [-(1 << 31), -125, -93, -19, -11, -5, -3, -1, 0, 1, 3, 5, 11, 19,
              93, 125, (1 << 31) - 1]
    yield fixture("round_ties", "Requant", [([len(signed)], "int32", signed)],
                  [len(signed)], "int32", [round(n / 2) for n in signed],
                  shift=1, rounding="nearest_even")
    boundary = [-(1 << 31), -1073741825, -1073741824, -1, 0, 1, 1073741824,
                1073741825, (1 << 31) - 1]
    yield fixture("round_shift31", "Requant", [([len(boundary)], "int32", boundary)],
                  [len(boundary)], "int32", [round(n / (1 << 31)) for n in boundary],
                  shift=31, rounding="nearest_even")
    yield fixture("requant_center", "Requant", [([7], "uint8", [0, 19, 20, 21, 24, 45, 255])],
                  [7], "uint8", [min(255, max(0, round((v - 20) * -9 / 7) + 120))
                                 for v in [0, 19, 20, 21, 24, 45, 255]],
                  input_zero_point=20, zero_point=120, multiplier=-9, divisor=7, rounding="nearest_even")
    yield fixture("reshape_int64", "Reshape",
                  [([2, 2], "int64", [-(1 << 50), -1, (1 << 47) + 17, -614])],
                  [4], "int64", [-(1 << 50), -1, (1 << 47) + 17, -614], shape=[4])
    yield fixture("requant_wrap", "Requant", [([3], "int32", [1 << 30, 5, -5])],
                  [3], "int8", [-128, 5, -5], shift=1, multiplier=2, rounding="nearest_even")
    yield fixture("requant_divisor_override", "Requant", [([3], "int32", [1 << 30, 5, -5])],
                  [3], "int32", [1 << 30, 5, -5], shift=3, divisor=1, rounding="nearest_even")
    yield fixture("transpose", "Transpose", [([1, 2, 2, 3], "int8", list(range(12)))],
                  [1, 2, 3, 2], "int8", [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11], perm=[0, 2, 3, 1])
    yield fixture("slice", "Slice", [([2, 6], "int8", list(range(12)))],
                  [2, 2], "int8", [1, 3, 7, 9], starts=[0, 1], ends=[2, 5], steps=[1, 2])
    yield fixture("concat", "Concat", [([2, 2], "int8", [1, 2, 3, 4]), ([2, 1], "int8", [-1, -2])],
                  [2, 3], "int8", [1, 2, -1, 3, 4, -2], axis=1)
    yield fixture("resize", "Resize", [([1, 2, 2, 2], "int8", list(range(8)))],
                  [1, 4, 4, 2], "int8", [2 * ((r // 2) * 2 + c // 2) + k
                                        for r in range(4) for c in range(4) for k in range(2)])
    yield fixture("pool", "MaxPool", [([1, 2, 3, 1], "int8", [-14, -5, -3, -17, -1, -30])],
                  [1, 2, 3, 1], "int8", [-14, -5, -3, -14, -1, -1],
                  strides=[1, 1], pads=[1, 1, 0, 0], kernel_shape=[2, 2])
    yield fixture("argmax", "ArgMax", [([2, 3, 2], "int8", [7, 5, 7, 1, 3, 8, 0, 9, 4, 9, 4, 8])],
                  [2, 2], "int64", [1, 2, 2, 1], axis=1, keepdims=False, select_last_index=True)
    yield fixture("rmsnorm", "RMSNorm", [([2, 4], "int8", [0, 1, -1, 0, 1, -12, 8, -128])],
                  [2, 4], "int8", [0, 16, -16, 0, 0, -3, 2, -32])
    yield fixture("softmax", "RowSoftmax", [([2, 3], "int8", [-11, 5, 9, -16, 0, 13])],
                  [2, 3], "uint8", [255, 0, 0, 68, 187, 0], causal_mask=True)
    source = [-128, -56, -8, 0, 8, 32, 64, 127]
    yield fixture("sigmoid", "Sigmoid", [([len(source)], "int8", source)],
                  [len(source)], "int8", [round(16 / (1 + exp(-x / 16))) for x in source])

    yield fixture("expand", "Expand", [([2, 1], "int8", [27, -19])],
                  [3, 2, 2], "int8", [27, 27, -19, -19] * 3)
    source = [-31, -29, 0, 27, 63, 127]
    yield fixture("cast_extend", "Cast", [([len(source)], "int8", source)],
                  [len(source)], "int64", source, narrowing="wrap")
    source = [-128, -73, -1, 0, 1, 73, 127]
    table = [(i * 31 + 7) % 256 for i in range(256)]
    yield fixture("lookup_u8", "Lookup", [([len(source)], "int8", source)],
                  [len(source)], "uint8", [table[v & 255] for v in source], table=table)


def pack(mode, values):
    return struct.pack("<" + FORMATS[mode] * len(values), *values)


def validate(root, profile, example, page_index=47):
    name, graph, stage, golden = example
    page_offset = PAGE * page_index
    folder = root / name
    folder.mkdir()
    page = bytearray(PAGE)
    positions = {}
    output = None
    for i, key in enumerate([*stage["inputs"], stage["output"]]):
        spec = graph["tensors"][key]
        offset = i * 512
        if key == stage["output"]:
            output = offset
            content = pack(spec["dtype"], golden)
        else:
            content = pack(spec["dtype"], spec["data"])
            page[offset:offset + len(content)] = content
        if len(content) > 512 or prod(spec["shape"]) * struct.calcsize(FORMATS[spec["dtype"]]) != len(content):
            raise ValueError(f"invalid fixture size for {name}/{key}")
        positions[key] = {"offset": page_offset + offset, "size": len(content)}
    manifest = gpu.lower(graph, {"stages": [stage], "allocations": positions}, profile, folder)
    native = manifest["stages"][0]
    image, expected, control = [folder / path for path in ("input.bin", "golden.bin", "replay.txt")]
    image.write_bytes(page)
    page[output:output + len(content)] = content
    expected.write_bytes(page)
    commands = [f"{image} {page_offset} {PAGE}"]
    commands.extend(f"write {0x4000 + i * 4} {int(w, 16)}"
                    for i, w in enumerate((folder / native["hex"]).read_text().splitlines()))
    commands.extend(f"submit_gpu {j['entry']} {j['gpu_job_reg']}" for j in native["jobs"])
    commands.append(f"check_gpu {name} {expected} {page_offset} {PAGE}")
    control.write_text("\n".join(commands) + "\n")
    binary = ROOT / "npu/z15-build/run-npu-z15"
    subprocess.run(["nice", "-n", "10", "ionice", "-c2", "-n7", str(binary), str(control)],
                   cwd=folder, check=True, timeout=40)


if __name__ == "__main__":
    profile = load_profile("fpga_zu15eg_balanced")
    with tempfile.TemporaryDirectory(prefix="c2-gpu-z15-", dir=ROOT / "gpu") as path:
        for case in cases():
            validate(Path(path), profile, case)
