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
SCRATCH = Path("/dev/shm/ysyx_compiler_c3")
P8 = SCRATCH / "npu/p8-build/run-npu-p8"
Z15 = SCRATCH / "npu/z15-build/run-npu-z15"
LOW_PRIORITY = ["nice", "-n", "10", "ionice", "-c2", "-n7"]


def _words(data):
    return [word for word, in struct.iter_unpack("<I", data)]


def _hex(path):
    return [int(line, 16) for line in path.read_text(encoding="ascii").splitlines()]


def _require(cond, msg):
    if not cond:
        raise ValueError(msg)


def _execute(command):
    subprocess.run(LOW_PRIORITY + [str(word) for word in command], cwd=ROOT, check=True,
                   timeout=int(os.environ.get("YSYX_REPLAY_TIMEOUT", "900")), env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
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


def _z15_loads(case, manifest, arena):
    setup = []
    for index, move in enumerate(manifest["dma"]):
        if move["direction"] != "load":
            continue
        lo, size = move["scratch_offset"], move["bytes"]
        _require(lo == move["ddr_offset"], "replay requires an identity DDR offset")
        region = case / f"upload-{index}.bin"
        region.write_bytes(arena[lo:lo + size])
        tail = f"{region} {lo} {size}"
        setup.append("upload " + tail if setup else tail)
    _require(bool(setup), "missing Z15 startup extents")
    return setup


def _z15_launch(bundle, key, gpu, npu):
    if key not in gpu:
        return [f"write {addr} {value}" for addr, value in npu[key]["writes"]]
    program = gpu[key]
    lines = [f"write {0x4000 + 4*i} {word}"
             for i, word in enumerate(_hex(bundle / program["hex"]))]
    return lines + [f"submit_gpu {job['entry']} {job['gpu_job_reg']}" for job in program["jobs"]]


def _verify_z15_plan(bundle, case, manifest, arena, values, setup, gpu, npu):
    graph, plan = manifest["graph"], manifest["plan"]
    stage_labels = {stage["id"]: f"s{i}" for i, stage in enumerate(plan["stages"])}
    launched, drained = set(), set()
    in_flight = {}

    def wait(key):
        _require(key in launched, f"wait references an unlaunched stage: {key}")
        if key in drained:
            return
        engine = "gpu" if key in gpu else "npu"
        _require(in_flight.get(engine) == key, f"stage {key} lost its engine completion")
        setup.append(f"wait_{engine} {stage_labels[key]}")
        drained.add(key)
        del in_flight[engine]

    for cmd in plan["commands"]:
        if cmd["type"] == "fence":
            wait(cmd["wait_for"])
            print(f"plan fence wait_for={cmd['wait_for']!r} zone={cmd['zone']}", flush=True)
            continue
        _require(cmd["type"] == "launch", "unknown Z15 schedule command")
        key = cmd["stage"]
        _require(key in stage_labels and key not in launched, "invalid or repeated Z15 launch")
        if "engine_predecessor" in cmd:
            wait(cmd["engine_predecessor"])
        engine = "gpu" if key in gpu else "npu"
        _require(engine not in in_flight, f"stage {key} omitted its engine predecessor")
        setup += _z15_launch(bundle, key, gpu, npu)
        in_flight[engine] = key
        launched.add(key)
        print(f"plan launch stage={stage_labels[key]} node={key!r} "
              f"device={cmd['device']} zone={cmd['zone']}", flush=True)
    _require(launched == stage_labels.keys(), "incomplete Z15 command schedule")
    for key in list(in_flight.values()):
        wait(key)
    for stage in plan["stages"]:
        _stage_value(arena, graph, plan, values, stage)
    for index, move in enumerate(manifest["dma"]):
        if move["direction"] != "store":
            continue
        lo, size = move["scratch_offset"], move["bytes"]
        _require(lo == move["ddr_offset"], "replay requires an identity DDR offset")
        expected = case / f"output-{index}.bin"
        expected.write_bytes(arena[lo:lo + size])
        setup.append(f"download o{index} {expected} {lo} {size}")


def _verify_z15(bundle, case, manifest, arena, values, execute_plan):
    graph, plan, products = (manifest[x] for x in ("graph", "plan", "backends"))
    setup = _z15_loads(case, manifest, arena)
    gpu = {s["id"]: s for s in products["gpu"]["stages"]}
    npu = {s["stage"]: s for s in products["npu"]["commands"]}
    if execute_plan:
        _verify_z15_plan(bundle, case, manifest, arena, values, setup, gpu, npu)
    else:
        for index, stage in enumerate(plan["stages"]):
            key = stage["id"]
            setup.append("write 16 2")
            setup.extend(_z15_launch(bundle, key, gpu, npu))
            check = "check_gpu" if key in gpu else "check_npu"
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


def _verify_z15_stream(bundle, case, manifest, values, ddr):
    img = case / "ddr-initial.bin"
    img.write_bytes(ddr)
    setup = ["- 0 0", f"ddr_put {img} 0 {len(ddr)}"]
    products = manifest["backends"]
    gpu = {s["id"]: s for s in products["gpu"]["stages"]}
    npu = {s["stage"]: s for s in products["npu"]["commands"]}
    stage_labels = {s["id"]: f"s{i}" for i, s in enumerate(manifest["plan"]["stages"])}
    started, drained = set(), set()
    for cmd in manifest["execution"]:
        kind = cmd["type"]
        if kind == "launch":
            name = cmd["stage"]
            _require(name not in started, f"repeated stage launch {name}")
            started.add(name)
            setup.extend(_z15_launch(bundle, name, gpu, npu))
        elif kind == "fence":
            name = cmd["wait_for"]
            _require(name in started, f"wait references unlaunched stage {name}")
            if name not in drained:
                engine = "gpu" if name in gpu else "npu"
                setup.append(f"wait_{engine} {stage_labels[name]}")
                drained.add(name)
        elif kind == "dma":
            action = "dma_in" if cmd["direction"] == "load" else "dma_out"
            setup.append(f"{action} {cmd['ddr_offset']} {cmd['scratch_offset']} {cmd['bytes']}")
        elif kind == "dma-fence":
            setup.append("dma_wait")
        else:
            raise ValueError(f"unrecognized streaming command {kind!r}")
    _require(started == drained == stage_labels.keys(), "incomplete streaming execution")
    for index, (name, spec) in enumerate(manifest["output_bindings"].items()):
        data = tensor_bytes(values[name], spec["dtype"])
        expected = case / f"output-{index}.bin"
        expected.write_bytes(data)
        setup.append(f"compare_ddr {expected} {spec['ddr_offset']} {len(data)}")
    control = case / "z15-stream.txt"
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
    parser.add_argument("--execute-plan", action="store_true",
                        help="Run the Z15 launch/fence plan and check the final DMA table")
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
    streaming = "execution" in manifest
    arena = bytearray((bundle / manifest["initial_ddr"]).read_bytes() if streaming else contents)
    runtime = json.loads(args.inputs.read_text()) if args.inputs else {}
    values = evaluate(graph, runtime)
    for tensor, data in runtime.items():
        binding = manifest["input_bindings"][tensor]
        encoded = tensor_bytes(data, binding["dtype"])
        offset, size = binding.get("ddr_offset", binding["offset"]), binding["size"]
        _require(len(encoded) == size, f"{tensor}: incorrect runtime input size")
        arena[offset:offset + size] = encoded
    print("replay command=" + shlex.join(sys.argv) + " binding=" + profile["binding"], flush=True)
    if name == "fpga_zu15eg_balanced":
        if streaming:
            _verify_z15_stream(bundle, case, manifest, values, arena)
        else:
            _verify_z15(bundle, case, manifest, arena, values, args.execute_plan)
    else:
        _require(not args.execute_plan, "scheduled MMIO replay is available on balanced Z15")
        _verify_p8(bundle, case, manifest, arena, values)
    print("BUNDLE_RTL_PASS " + json.dumps({name: values[name] for name in graph["outputs"]},
                                          ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
