"""Replay compiler output against the isolated Z15 or Phase 8 RTL runners."""

import argparse
import json
import os
import shlex
import struct
import subprocess
import sys
from pathlib import Path

from hetero_compiler.compiler import tensor_bytes
from hetero_compiler.reference import evaluate

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("/dev/shm/ysyx_compiler")
P8 = SCRATCH / "npu/p8-build/run-npu-p8"
Z15 = SCRATCH / "npu/z15-build/run-npu-z15"
LOW_PRIORITY = ["nice", "-n", "15", "ionice", "-c", "3"]


def _words(data):
    return [word for word, in struct.iter_unpack("<I", data)]


def _hex(path):
    return [int(line, 16) for line in path.read_text(encoding="ascii").splitlines()]


def _require(cond, msg):
    if not cond:
        raise ValueError(msg)


def _execute(command):
    subprocess.run(LOW_PRIORITY + [str(word) for word in command], cwd=ROOT, check=True,
                   timeout=180, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                                     "OMP_NUM_THREADS": "1", "TMPDIR": str(SCRATCH)})


def _stage_value(arena, graph, plan, values, stage):
    name = stage["output"]
    offset = plan["allocations"][name]["offset"]
    dtype = graph["tensors"][name]["dtype"]
    physical = tensor_bytes(values[name], dtype)
    arena[offset:offset + len(physical)] = physical
    return offset, len(physical)


def _check_backend_files(bundle, manifest, contents):
    npu_commands = manifest["backends"]["npu"]
    payload = (bundle / "npu.bin").read_bytes()
    words = _words(payload)
    _require(words == _hex(bundle / "npu.hex"), "NPU binary differs from HEX")
    if npu_commands["abi"] == "z15-mmio-v1":
        oracle = [word for c in npu_commands["commands"] for rw in c["writes"] for word in rw]
        _require(words == oracle, "NPU MMIO binary differs from manifest")
    else:
        base = npu_commands["binary_base"]
        _require(contents[base:base + len(payload)] == payload,
                 "initial arena differs from the NPU descriptor image")
    for stage in manifest["backends"]["gpu"]["stages"]:
        elf = (bundle / stage["elf"]).read_bytes()
        elf_offset, _, _, size = struct.unpack_from("<4I", elf, 56)
        _require(_words(elf[elf_offset:elf_offset + size]) == _hex(bundle / stage["hex"]),
                 f"{stage['id']}: GPU ELF segment differs from HEX")


def _case_path(path):
    # The C++ replay files use whitespace-delimited absolute paths.
    if any(c.isspace() for c in str(path)):
        raise ValueError("scratch path contains whitespace")
    return path


def _verify_z15(bundle, case, manifest, arena, values):
    graph, plan, products = (manifest[x] for x in ("graph", "plan", "backends"))
    initial = case / "initial-beat.bin"
    initial.write_bytes(bytes(32))
    setup = [f"{initial} 0 32"]
    for index, move in enumerate(manifest["dma"]):
        if move["direction"] != "load":
            continue
        lo, size = move["scratch_offset"], move["bytes"]
        region = case / f"upload-{index}.bin"
        region.write_bytes(arena[lo:lo + size])
        setup.append(f"upload {region} {lo} {size}")
    gpu = {s["id"]: s for s in products["gpu"]["stages"]}
    npu = {s["stage"]: s for s in products["npu"]["commands"]}
    for index, stage in enumerate(plan["stages"]):
        key = stage["id"]
        # Writes start from a clean sticky status and the previous launch has drained.
        setup.append("write 16 2")
        if key in gpu:
            program = gpu[key]
            words = _hex(bundle / program["hex"])
            setup += [f"write {0x4000+4*i} {w}" for i, w in enumerate(words)]
            for job in program["jobs"]:
                setup.append(f"submit_gpu {job['entry']} {job['gpu_job_reg']}")
            check = "check_gpu"
        else:
            setup += [f"write {addr} {value}" for addr, value in npu[key]["writes"]]
            check = "check_npu"
        off, size = _stage_value(arena, graph, plan, values, stage)
        lo, high = off // 32 * 32, (off + size + 31) // 32 * 32
        golden = case / f"expected-{index}.bin"
        golden.write_bytes(arena[lo:high])
        setup.append(f"{check} s{index} {golden} {lo} {high-lo}")
        print(f"replay stage=s{index} node={key!r} device={stage['device']} "
              f"output_offset={off} output_bytes={size}", flush=True)
    control = case / "z15.txt"
    control.write_text("\n".join(setup) + "\n", encoding="ascii")
    _execute([Z15, control])


def _verify_p8(bundle, case, manifest, arena, values):
    graph, plan, products = (manifest[x] for x in ("graph", "plan", "backends"))
    gpu = {s["id"]: s for s in products["gpu"]["stages"]}
    npu = {s["stage"]: s for s in products["npu"]["commands"]}
    source = case / "p8-start.bin"
    source.write_bytes(arena)
    for index, stage in enumerate(plan["stages"]):
        name = stage["id"]
        _stage_value(arena, graph, plan, values, stage)
        expected = case / f"expected-{index}.bin"
        result = case / f"actual-{index}.bin"
        expected.write_bytes(arena)
        if name in gpu:
            _execute([sys.executable, "-B", ROOT / "tools/verify_gpu.py", "--bundle", bundle,
                      "--stage", name, "--initial", source, "--result", result])
        else:
            op = npu[name]
            control = case / f"p8-{index}.txt"
            control.write_text(f"{source}\ns{index} {op['a_base']} {op['descriptor']} "
                               f"{op['program_slots']} {expected}\n", encoding="ascii")
            _execute([P8, control, result])
        _require(result.read_bytes() == expected.read_bytes(), f"{name}: full Phase 8 arena mismatch")
        source = result
        print(f"arena relay stage={name!r} device={stage['device']} compared_bytes=4096", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--case", required=True, type=Path, help="New directory inside compiler RAM")
    parser.add_argument("--inputs", type=Path, help="Runtime tensor values as a JSON mapping")
    args = parser.parse_args()
    bundle = args.bundle.resolve(strict=True)
    case = _case_path(args.case.resolve())
    _require(SCRATCH in case.parents, "verification scratch must be in the compiler RAM directory")
    case.mkdir()
    manifest = json.loads((bundle / "bundle.json").read_text())
    graph = manifest["graph"]
    profile = manifest["profile"]
    name = profile["name"]
    _require(name in ("fpga_zu15eg_balanced", "asic_nangate45_p8"), "target has no pinned replay binding")
    contents = (bundle / "initial.bin").read_bytes()
    _check_backend_files(bundle, manifest, contents)
    arena = bytearray(contents)
    runtime = json.loads(args.inputs.read_text()) if args.inputs else {}
    values = evaluate(graph, runtime)
    for tensor, data in runtime.items():
        binding = manifest["input_bindings"][tensor]
        encoded = tensor_bytes(data, binding["dtype"])
        offset, size = binding["offset"], binding["size"]
        _require(len(encoded) == size, f"{tensor}: incorrect runtime input size")
        arena[offset:offset + size] = encoded
    print("replay command=" + shlex.join(sys.argv) + " binding=" + profile["binding"], flush=True)
    if name == "fpga_zu15eg_balanced":
        _verify_z15(bundle, case, manifest, arena, values)
    else:
        _verify_p8(bundle, case, manifest, arena, values)
    print("BUNDLE_RTL_PASS " + json.dumps({name: values[name] for name in graph["outputs"]},
                                          ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
