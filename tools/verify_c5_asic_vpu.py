"""Verify C5 vector YHB on the pinned A3 SRAM host ports and C99 loader."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import struct
import subprocess

from examples.generate_quantized_onnx import FLOAT, INT8, INT32, INT64, UINT8, Model
from hetero_compiler import asic_ip, reference
from hetero_compiler.compiler import compile_bundle, tensor_bytes
from hetero_compiler.onnx import parse_model
from hetero_compiler.profile import load_profile
from hetero_compiler.yhb import inspect_yhb
from runtime.verify_z15_abi import corrupt_words


RAM = Path("/dev/shm/ysyx_compiler_c5/asic")
DEFAULT_RTL = RAM / "a3/obj/Vasic_soc_sram_top"
PIN = "6fe0028a7b7def51c1f2754a058c1984830ee6d0"


def fixture(*, mixed):
    """An ONNX RMSNorm→SwiGLU block with optional causal attention scores."""
    shape = [2, 8]
    x = [-128, 127, 0, -1, 1, 7, -7, 8, 0, 0, 0, 0, 15, -16, 16, -17]
    model = Model("asic_a3_qdq_vector")
    model.qparams()
    model.input("x", shape, INT8, x)
    model.tensor("gamma", [8], INT8, [16, 15, 12, 1, 7, 16, -16, 4])
    model.tensor("eps", [], FLOAT, [1/1024])
    model.tensor("axis", [1], INT64, [-1])
    model.tensor("out_scale", [], FLOAT, [1/32])
    model.tensor("out_zp", [], UINT8, [128])

    def op(name, operation, inputs, **attributes):
        return model.op(name, operation, inputs, name, **attributes)

    def dq(name, value, scale="q4", point="z"):
        return op(name, "DequantizeLinear", [value, scale, point] if point else [value, scale])

    def quant(name, value, scale="q4", point="z"):
        return op(name, "QuantizeLinear", [value, scale, point])

    f = dq("x_float", "x")
    gamma = dq("gamma_float", "gamma")
    power = op("power", "Mul", [f,f])
    mean = op("mean", "ReduceMean", [power, "axis"], keepdims=1)
    den = op("den", "Sqrt", [op("add_eps", "Add", [mean, "eps"])])
    norm = op("unit_norm", "Div", [f, den])
    normalized = quant("norm", op("gamma_norm", "Mul", [norm, gamma]))
    f_norm = dq("norm_float", normalized)
    swish = op("act", "Mul", [f_norm, op("sigmoid", "Sigmoid", [f_norm])])
    gate = quant("gate", swish)
    product = op("product", "Mul", [dq("gate_float", gate), f])
    hidden = quant("hidden", product)
    residual = quant("residual", op("residual_add", "Add", [dq("hidden_float", hidden), f_norm]))
    requant = quant("byte_requant", f_norm,"out_scale","out_zp")
    for result,dtype in ((normalized, INT8), (gate, INT8), (hidden, INT8),
                         (residual, INT8), (requant, UINT8)):
        model.output(result, shape, dtype)

    if mixed:
        model.input("scores", shape, INT8, x[::-1])
        model.tensor("mask", shape, FLOAT, [0.] * 7 + [-float("inf")] + [0.] * 8)
        model.tensor("prob_scale", [], FLOAT, [1/256])
        model.tensor("zero_u", [], UINT8, [0])
        logits = op("mask_add", "Add", [dq("score_float", "scores"), "mask"])
        probability = quant("probability", op("softmax", "Softmax", [logits], axis=-1),
                            "prob_scale", "zero_u")
        model.output(probability, shape, UINT8)
        model.tensor("wide", shape, INT32, [1000, -1000, 1, -1, 3, -3, 127,-128] * 2)
        wide = quant("wide_requant", dq("wide_float","wide","q4",None))
        model.output(wide, shape, INT8)
    return model.encode()


def run(binary, args, log):
    result = subprocess.run(["nice", "-n", "10", "ionice", "-c2", "-n7", str(binary), *map(str,args)],
                            capture_output=True, text=True, timeout=120)
    log.write_text(result.stdout + result.stderr)
    return result


def c99_negatives(blob, output, runner):
    directory = inspect_yhb(blob)["sections"]
    found = {}
    for i in range(directory["stages"]["count"]):
        entry = directory["stages"]["offset"] + i * 32
        engine, offset = struct.unpack_from("<II", blob, entry+8)
        start = directory["data"]["offset"] + offset
        if engine == 3:
            opcode = struct.unpack_from("<I", blob, start+4)[0]
            found.setdefault(opcode,start)
    probes = (
        ("requant_divisor", 3, {27: 0}),
        ("requant_shift", 3, {26: 32}),
        ("norm_rows", 11, {13: 1024,21:1024}),
        ("norm_gamma", 11, {17: 4}),
        ("norm_epsilon", 11, {26: 1 << 24}),
        ("norm_bias", 11, {9: 0}),
        ("softmax_position", 12, {26: 7}),
        ("softmax_shift", 12, {25: 32}),
        ("softmax_dtype", 12, {12: 1}),
        ("lookup_bounds", 13, {25: 65528}),
        ("lookup_same_address", 13, {25: struct.unpack_from("<I",blob,found[13]+4*6)[0]}),
        ("scaled_mul_scale", 14, {25: 65536}),
        ("scaled_mul_shift", 14, {26: 32}),
        ("reserved", 14, {31: 1}),
    )
    for name, opcode, changes in probes:
        bad = corrupt_words(blob, output, name, found[opcode], changes)
        result = run(runner, [bad, "--inspect"], output / (name+".log"))
        if result.returncode == 0 or "YHB CRC32 mismatch" in result.stderr:
            raise RuntimeError(f"C99 failed descriptor-specific rejection for {name}")
    return len(probes)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--runner", type=Path, default=DEFAULT_RTL)
    cli.add_argument("--c99", type=Path, default=RAM / "c99/ysyx_run")
    args = cli.parse_args()
    out = args.out.resolve()
    if RAM not in out.parents or RAM.resolve() != RAM:
        cli.error("choose a fresh directory in C5-only ASIC scratch")
    out.mkdir()
    rows = []
    for name, mixed, mode in (("native-lut",False,"exact"),("mixed-exact",True,"exact"),("mixed-fast",True,"fast")):
        case = out/name
        model = fixture(mixed=mixed)
        manifest = compile_bundle(parse_model(model), load_profile("asic_nangate45_ip"),
                                  case, vpu_attention=mode)
        (case/"model.onnx").write_bytes(model)
        graph = manifest["graph"]
        evaluated = reference.evaluate(graph)
        expected = {name: tensor_bytes(evaluated[name],graph["tensors"][name]["dtype"]) for name in graph["outputs"]}
        program = case/"program.yhb"
        blob = program.read_bytes()
        records, bindings = asic_ip.hardware_vector_records(blob, expected)
        if not mixed and asic_ip.replay(blob) != expected:
            raise RuntimeError("software target-5 replay differs from the fixed-point graph")
        control = case/"replay.ops"
        control.write_text("".join(" ".join(f"{item:x}" for item in row)+"\n" for row in records))
        rtl = run(args.runner,["--replay",control],case/"rtl.log")
        inspect = run(args.c99,[program,"--inspect"],case/"inspect.log")
        if rtl.returncode or "PASS workload" not in rtl.stdout or inspect.returncode:
            raise RuntimeError(f"{name}: see {case}/rtl.log and inspect.log")
        if mixed and [r["device"] for r in bindings].count("GPU") != 2:
            raise RuntimeError("expected stand-alone score Cast and wide Requant on frozen v6")
        negative = c99_negatives(blob,case,args.c99) if name == "mixed-fast" else 0
        windows = [line for line in rtl.stdout.splitlines() if line.startswith("WINDOW ")]
        row = {"case":name,"frozen_asic_revision":PIN,"status":"byte_exact",
               "bytes":sum(map(len,expected.values())),"yhb_sha256":sha256(blob).hexdigest(),
               "native_software_replay":not mixed,"c99_bad_descriptors_rejected":negative,
               "frozen_rtl_counts":windows,"host_bindings":bindings}
        rows.append(row)
        print(name, row["bytes"], windows, flush=True)
    (out/"result.json").write_text(json.dumps(rows,indent=2)+"\n")


if __name__=="__main__":
    main()
