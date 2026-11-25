#!/usr/bin/env python3
"""Compare compiler-generated GPU kernel outputs with independently calculated data."""
import argparse
import json
from pathlib import Path
from struct import pack
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hetero_compiler import gpu
from verify_gpu import SCRATCH, simulate


def encode(dtype, values):
    if dtype == "int32":
        return pack(f"<{len(values)}i", *values)
    return bytes(v & 255 for v in values)


def saturated(values, low=-128, high=127):
    return [min(high, max(low, v)) for v in values]


def i32(v):
    return (v + (1 << 31)) % (1 << 32) - (1 << 31)


def cases():
    # Long using a partial Warp and exceeding the v6 limit of 255 threads.
    count = 273
    a = [(n * 53 + 7) % 256 - 128 for n in range(count)]
    b = [(n * 47 + 3) % 256 - 128 for n in range(count)]
    yield "Add", {}, [([count], "int8", a), ([count], "int8", b)], ([count], "int8",
            saturated([x + y for x, y in zip(a, b)]))
    yield "Mul", {"narrowing": "wrap"}, [([count], "int8", a), ([count], "int8", b)], (
        [count], "int8", [x * y for x, y in zip(a, b)])
    yield "Relu", {}, [([count], "int8", a)], ([count], "int8", [max(0, v) for v in a])
    # Intentional signed product overflow, sign-extending shifts and every clamp direction.
    raw = [13, -42, 32, -91, 8192, -8192, -2147483648, 2147483647, 13421773]
    yield "Requant", {"multiplier": -19, "shift": 3, "zero_point": 7}, [
        ([len(raw)], "int32", raw)], ([len(raw)], "uint8",
        saturated([(i32(v * -19) >> 3) + 7 for v in raw], 0, 255))
    yield "AddLast", {"broadcast": "last"}, [([2, 3], "int8", [-7, 42, 83, 7, -21, 22]),
        ([3], "int32", [100, -123, 33])], ([2, 3], "int8", [93, -81, 116, 107, -128, 55])
    yield "Gemm", {}, [([2, 3], "int8", [-5, 42, 19, -38, 12, -7]),
        ([3, 2], "int8", [5, 4, -1, 9, -7, -6]), ([2], "int32", [-50, -50])], (
        [2, 2], "int32", [-250, 194, -203, -52])
    # The tensor contains the same numeric weights as Gemm, physically [N,K].
    yield "GemmTranspose", {"transpose_b": True}, [
        ([2, 3], "int8", [-5, 42, 19, -38, 12, -7]),
        ([2, 3], "int8", [5, -1, -7, 4, 9, -6]), ([2], "int32", [-50, -50])], (
        [2, 2], "int32", [-250, 194, -203, -52])
    yield "GatedBranch", {"rounds": 64, "threshold": 13, "narrowing": "wrap"}, [
        ([8], "int8", [-128, -1, 0, 12, 13, 25, 63, 127])], ([8], "int8",
        [44, -6, 46, -78, -103, -26, 82, -78])
    yield "GatedBranchUnsigned", {"rounds": 65, "threshold": 128, "narrowing": "wrap"}, [
        ([6], "uint8", [0, 32, 127, 128, 139, 255])], ([6], "uint8",
        [241, 172, 232, 205, 55, 134])
    activation_input = [-128, -64, -32, -16, -8, 0, 8, 16, 32, 64, 127]
    yield "SiLU", {}, [([11], "int8", activation_input)], ([11], "int8",
        [0, -1, -4, -4, -3, 0, 5, 12, 28, 63, 127])
    values = [-128, -64, -32, -24, -16, -9, -8, -7, 0, 7, 8, 9, 16, 23, 24, 32, 64, 127]
    yield "LegacySiLU", {"mode": "legacy_q4_knots"}, [([18], "int8", values)], ([18], "int8",
        [0, -1, -3, -6, -4, -2, -4, -3, 0, 3, 5, 6, 11, 16, 21, 28, 62, 127])
    yield "GELU", {}, [([11], "int8", activation_input)], ([11], "int8",
        [0, 0, -1, -3, -2, 0, 6, 13, 31, 64, 127])


def run_case(profile, root, case):
    name, attrs, inputs, (out_shape, out_dtype, expected) = case
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    page = bytearray([0xB5] * 4096)
    tensors, allocations = {}, {}
    for i, (shape, dtype, values) in enumerate(inputs, 1):
        tensor = f"in{i}"
        pos = 32 if i == 1 else 2048 + (i - 2) * 512
        binary = encode(dtype, values)
        tensors[tensor] = {"shape": shape, "dtype": dtype}
        allocations[tensor] = {"offset": pos, "size": len(binary)}
        page[pos:pos + len(binary)] = binary
    tensors["out"] = {"shape": out_shape, "dtype": out_dtype}
    output_offset = 3008
    payload = encode(out_dtype, expected)
    allocations["out"] = {"offset": output_offset, "size": len(payload)}
    graph = {"tensors": tensors}
    plan = {"allocations": allocations, "stages": [{"id": name, "device": "GPU",
            "op": {"AddLast": "Add", "GemmTranspose": "Gemm",
                   "GatedBranchUnsigned": "GatedBranch", "LegacySiLU": "SiLU"}.get(name, name), "inputs": list(tensors)[:-1],
            "output": "out", "attrs": attrs}]}
    gpu.lower(graph, plan, profile, folder)
    initial, result = folder / "initial.bin", folder / "result.bin"
    initial.write_bytes(page)
    metrics = simulate(folder, name, initial, result)
    page[output_offset:output_offset + len(payload)] = payload
    actual = result.read_bytes()
    if actual != page:
        at = next(i for i, (got, want) in enumerate(zip(actual, page)) if got != want)
        raise RuntimeError(f"{name} byte at {at}: actual {actual[at]}, expected {page[at]}")
    return f"PASS {name} bytes_checked={len(page)} {metrics}"


def run_fused(profile, root, legacy=False):
    label = "SiLU" if legacy else "GELU"
    folder = root / f"fused-{label}"
    golden = [-3, 59, 3] if legacy else [-1, 61, 5]
    folder.mkdir(parents=True, exist_ok=True)
    fixtures = [
        ("x", [1, 2], "int8", 32, [1, 2]),
        ("w", [2, 3], "int8", 2048, [80, -60, 1, 70, -60, -2]),
        ("bias", [3], "int32", 2304, [9, -17, 8]),
        ("result", [1, 3], "int8", 3008, golden),
    ]
    tensors, allocations = {}, {}
    page = bytearray([0xB5] * 4096)
    for name, shape, dtype, offset, values in fixtures:
        binary = encode(dtype, values)
        tensors[name] = {"shape": shape, "dtype": dtype}
        allocations[name] = {"offset": offset, "size": len(binary)}
        if name != "result":
            page[offset:offset + len(binary)] = binary
    for name in ("linear", "biased", "scaled"):
        tensors[name] = {"shape": [1, 3], "dtype": "int8"}
    nodes = [
        {"id": "linear", "op": "Gemm", "inputs": ["x", "w"], "output": "linear",
         "attrs": {"narrowing": "wrap"}},
        {"id": "bias", "op": "Add", "inputs": ["linear", "bias"], "output": "biased",
         "attrs": {"broadcast": "last"}},
        {"id": "scale", "op": "Requant", "inputs": ["biased"], "output": "scaled",
         "attrs": {"multiplier": 2, "shift": 1, "zero_point": 2}},
        {"id": "activation", "op": label, "inputs": ["scaled"], "output": "result",
         "attrs": {"mode": "legacy_q4_knots"} if legacy else {}},
    ]
    stage = dict(nodes[0], device="GPU", inputs=["x", "w", "bias"], output="result", fused=nodes)
    graph = {"tensors": tensors}
    plan = {"allocations": allocations, "stages": [stage]}
    gpu.lower(graph, plan, profile, folder)
    initial, result = folder / "initial.bin", folder / "result.bin"
    initial.write_bytes(page)
    stats = simulate(folder, stage["id"], initial, result)
    page[3008:3011] = encode("int8", golden)
    if result.read_bytes() != page:
        raise RuntimeError(f"fused output differs: {list(result.read_bytes()[3008:3011])}")
    return f"PASS FusedGemmBiasRequant{label} bytes_checked={len(page)} {stats}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=["v6", "g4-16t4w", "g4-8t8w"])
    args = parser.parse_args()
    target = args.target
    lanes, warps = {"v6": (8, 1), "g4-16t4w": (16, 4), "g4-8t8w": (8, 8)}[target]
    config = {"variant": "v6" if target == "v6" else "g4", "lanes": lanes, "warps": warps,
              "window_bytes": 4096, "code_bytes": 16384, "issue_width": 2}
    print(json.dumps(config, sort_keys=True), flush=True)
    for fixture in cases():
        print(run_case({"gpu": config}, SCRATCH / "fixtures" / target, fixture), flush=True)
    print(run_fused({"gpu": config}, SCRATCH / "fixtures" / target), flush=True)
    print(run_fused({"gpu": config}, SCRATCH / "fixtures" / target, legacy=True), flush=True)


if __name__ == "__main__":
    main()
