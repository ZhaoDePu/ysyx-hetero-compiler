"""Replay native vector registers against the frozen Step 2 Z15 MMIO shell."""

import argparse
import json
from pathlib import Path
import random
import struct
import subprocess
from math import prod

from hetero_compiler import npu
from hetero_compiler.profile import load_profile
from hetero_compiler.z15_vpu import attention, mul, rmsnorm, softmax

SCRATCH = Path("/dev/shm/ysyx_compiler_c4vpu")
RUNNER = SCRATCH / "npu/z15-build/run-npu-z15"
PROFILE = {**load_profile("fpga_zu15eg_balanced"), "vpu_fused_lowering": True}


def ir(dtype, shape, data=None):
    return {"kind": "input", "shape": shape, "dtype": dtype, **({"data": data} if data is not None else {})}


def replay(output, name, op, tensors, attrs, expected, expected_temps=None):
    path = output / name
    path.mkdir(exist_ok=True)
    stage = {"id": "native", "op": op, "device": "NPU" if op == "Attention" else "VPU",
             "inputs": list(tensors)[:-1], "output": "y", "attrs": attrs}
    graph = {"tensors": tensors}
    req = npu.requirements(stage, graph, PROFILE)
    order = npu.ordered_buffers(stage, graph, PROFILE) or list(tensors)
    space = bytearray()
    allocations = {}
    for key in order:
        t = tensors.get(key)
        size = req[key]["size"] if key in req else prod(t["shape"]) * npu.BYTES[t["dtype"]]
        data = t.get("data") if t else None
        if data is None:
            encoded = b""
        elif t["dtype"] == "int32":
            encoded = struct.pack(f"<{len(data)}i", *data)
        else:
            encoded = bytes(x & 255 for x in data)
        allocations[key] = {"offset": len(space), "size": size}
        space.extend(encoded + bytes([0xa7]) * (((size + 31) // 32) * 32 - len(encoded)))
    bundle = npu.lower(graph, {"stages": [stage], "allocations": allocations}, PROFILE, path)
    (path / "input.bin").write_bytes(space)
    content = [f"{path / 'input.bin'} 0 {len(space)}"]
    content += [f"write {reg} {word}" for reg, word in bundle["commands"][0]["writes"]]
    for key, values in {"y":expected, **(expected_temps or {})}.items():
        n = len(values)
        pad = (n + 31) // 32 * 32
        answer = path / (key + ".bin")
        answer.write_bytes(bytes(x & 255 for x in values) + bytes([0xa7]) * (pad - n))
        content.append(f"check_npu {key} {answer} {allocations[key]['offset']} {pad}")
    control = path / "control.txt"
    control.write_text("\n".join(content) + "\n")
    with (path / "rtl.log").open('w') as log:
        p = subprocess.run(["nice", "-n", "10", str(RUNNER), str(control)],
                           check=False, stdout=log, stderr=subprocess.STDOUT)
    lines = (path / "rtl.log").read_text().splitlines()
    exact = sum(line.startswith("Z15_NPU_RTL_STAGE_PASS ") for line in lines)
    if p.returncode or exact != 1 + len(expected_temps or {}):
        raise RuntimeError(f"{name} did not pass all exact checks. See {path / 'rtl.log'}")
    metrics, = (json.loads(line.split(" ", 1)[1]) for line in lines if line.startswith("Z15_METRICS "))
    print(f"{name} exact, {metrics['wall_npu_ticks']:,} wall ticks", flush=True)
    return {"case": name, "status": "byte_exact", "comparisons": exact, **metrics}



def cases():
    rng = random.Random(85)

    def draw(n, limit):
        return [rng.randrange(-limit, limit + 1) for _ in range(n)]

    scores = draw(4 * 48, 50)
    for fast in (False, True):
        mode = {"exp_shift": 3, "causal": True, "position": 44, "fast": fast}
        tensors = {"a": ir("int32", [4, 48], scores), "y": ir("uint8", [4, 48])}
        yield f"softmax-{int(fast)}", "Z15Softmax", tensors, mode, softmax(scores, 4, 48, **mode)

    for width in (4, 48, 64, 68, 128):
        a, gamma = draw(width * 4, 127), draw(width, 127)
        mode = {"gain": round(256 * width**0.5), "epsilon": 7, "shift": 15}
        tensors = {"a": ir("int8", [4, width], a), "gamma": ir("int8", [width], gamma),
                   "y": ir("int8", [4, width])}
        yield f"rms-{width}", "Z15RMSNorm", tensors, mode, rmsnorm(a, gamma, 4, width, **mode)

    a, b = draw(96, 127), draw(96, 127)
    binary = {"a": ir("int8", [4, 24], a), "b": ir("int8", [4, 24], b), "y": ir("int8", [4, 24])}
    mode = {"scale": 61, "shift": 12}
    yield "scaled-mul", "Z15Mul", binary, mode, mul(a, b, **mode)
    yield "ew-add", "Add", binary, {}, [max(-128, min(127, x + y)) for x, y in zip(a, b, strict=True)]
    lut = npu._lookup("SiLU", None)
    unary = {"a": ir("int8", [4, 24], a), "y": ir("int8", [4, 24])}
    yield "activation", "SiLU", unary, {}, [lut[x & 255] for x in a]
    arbitrary = [((i + 71) % 219) - 109 for i in range(256)]
    yield "lookup", "Lookup", unary, {"table": arbitrary}, [arbitrary[x & 255] for x in a]

    source = [19, -5, 80, 92, -122, 0, -38, 57]
    stride, rows, cols = 7, 2, 4
    size = rows + (cols - 1) * stride
    packed = [0xa7] * size
    for row in range(rows):
        for col in range(cols):
            packed[col * stride + row] = source[row * cols + col]
    native = {"a": ir("int8", [rows, cols], source), "y": ir("int8", [size])}
    yield "strided-transpose", "Z15Transpose", native, {"stride": stride}, packed

    q, k, v = draw(8, 8), draw(48 * 4, 8), draw(4 * 48, 16)
    scores = [sum(q[row * 4 + i] * k[col * 4 + i] for i in range(4))
              for row in range(2) for col in range(48)]
    qkv = {"q": ir("int8", [2, 4], q), "k": ir("int8", [48, 4], k),
           "v": ir("int8", [4, 48], v), "y": ir("int8", [2, 4])}
    for fast in (False, True):
        mode = {"causal": True, "position": 46, "exp_shift": 6, "packed_v": True, "fast": fast}
        answer = attention(q, k, v, [2, 4], [48, 4], [4, 48], mode)
        probs = softmax(scores, 2, 48, exp_shift=6, causal=True, position=46, fast=fast)
        temps = {"native.__npu_scores": struct.pack(f"<{len(scores)}i", *scores),
                 "native.__npu_probabilities": probs}
        yield f"attention-{int(fast)}", "Attention", qkv, mode, answer, temps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    output = args.out.resolve()
    if SCRATCH not in output.parents or SCRATCH.resolve() != SCRATCH:
        p.error("output must be under the C4-VPU RAM directory")
    output.mkdir()
    summary = [replay(output, *case) for case in cases()]
    (output / "native.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
