"""Exercise YHB KV sessions on the frozen C99 Verilator bridge."""

import argparse
from copy import deepcopy
from pathlib import Path
import struct
import subprocess
import zlib

from hetero_compiler.gpu import lower
from hetero_compiler.profile import load_profile
from hetero_compiler.yhb import emit_yhb, inspect_yhb


RAM = Path("/dev/shm/ysyx_compiler_c5/runtime")


def make_fixture(folder):
    profile = load_profile("fpga_zu15eg_balanced")
    values = {"query": (2, 3, 4, 5), "resident": (11, 22, 33, 44)}
    inputs, outputs = ["query", "feed"], ["first", "last"]
    tensors = {name: dict(dtype="int8", kind="input" if name in inputs else "value", shape=[1, 4])
               for name in (*inputs, *outputs, "resident")}
    for name, data in values.items():
        tensors[name]["data"] = list(data)
    tensors["resident"]["kind"] = "constant"
    graph = {"name": "persistent_session_fixture", "inputs": inputs, "outputs": outputs,
             "tensors": tensors}
    offsets = {name: i*32 for i, name in enumerate(tensors)}
    allocations = {name: dict(offset=offsets[name], size=4, stored_size=32, zone=0)
                   for name in tensors}
    stages = [
        {"id": "prefill", "op": "Add", "inputs": ["query", "resident"], "output": "first",
         "device": "GPU", "zone": 0, "attrs": {}},
        {"id": "decode", "op": "Add", "inputs": ["feed", "resident"], "output": "last",
         "device": "GPU", "zone": 0, "attrs": {}},
    ]
    plan = {"allocations": allocations, "stages": stages}
    products = lower(graph, plan, profile, folder)
    bindings = {name: {"ddr_offset": offsets["first"] if name == "feed" else offsets[name]}
                for name in inputs + outputs}
    bindings["feed"]["needs_data"] = False
    image = bytearray(4096)
    for name, data in values.items():
        struct.pack_into("<4b", image, offsets[name], *data)
    (folder / "initial.bin").write_bytes(image)

    def dma(name, direction):
        return {"type": "dma", "direction": direction, "bytes": 32, "scratch_offset": offsets[name],
                "ddr_offset": bindings.get(name, {"ddr_offset": offsets[name]})["ddr_offset"]}

    wait = {"type": "dma-fence"}
    execution = [
        dma("query", "load"), wait, dma("resident", "load"), wait,
        {"type": "launch", "stage": "prefill"}, {"type": "fence", "wait_for": "prefill"},
        dma("first", "store"), wait, {"type": "yield", "step": 0},
        dma("feed", "load"), wait,
        {"type": "launch", "stage": "decode"}, {"type": "fence", "wait_for": "decode"},
        dma("last", "store"), wait, {"type": "yield", "step": 1},
    ]
    return {"graph": graph, "profile": profile, "plan": plan, "initial_image": "initial.bin",
            "input_bindings": {name: bindings[name] for name in inputs},
            "output_bindings": {name: bindings[name] for name in outputs}, "execution": execution,
            "backends": {"gpu": products, "npu": {"commands": []}}}


def verify(folder, runner):
    manifest = make_fixture(folder)
    program = emit_yhb(folder, manifest)
    blob = program.read_bytes()
    info = inspect_yhb(blob)

    def call(bundle, label, extra, failure=None):
        result = subprocess.run(["nice", "-n", "10", str(runner), str(bundle), *extra],
                                check=False, text=True, capture_output=True, timeout=90)
        (folder / (label + ".log")).write_text(result.stdout + result.stderr)
        if failure is None:
            if result.returncode:
                raise RuntimeError(result.stderr)
        elif result.returncode == 0 or failure not in result.stderr:
            raise RuntimeError(f"{label}: expected {failure!r}, got {result.stderr!r}")
        return result.stdout

    def compare(name, data):
        if (folder / name).read_bytes() != bytes(data):
            raise RuntimeError(f"session output {name} differs")

    call(program, "whole", ["--backend", "verilator", "--output", f"first={folder / 'first.bin'}",
                            "--output", f"last={folder / 'last.bin'}", "--dump-stats"])
    compare("first.bin", [13, 25, 37, 49])
    compare("last.bin", [24, 47, 70, 93])

    (folder / "override.bin").write_bytes(struct.pack("<4b", 1, 0, -1, -2))
    for label, source, expect in (
        ("feedback", "feedback\tfirst\tfeed", [24, 47, 70, 93]),
        ("override", f"input\tfeed\t{folder / 'override.bin'}", [12, 22, 32, 42]),
    ):
        script = folder / (label + ".tsv")
        script.write_text(f"step\t0\noutput\tfirst\t{folder / 'pause.bin'}\n{source}\n"
                          f"step\t1\noutput\tlast\t{folder / 'resume.bin'}\n")
        trace = call(program, label, ["--backend", "verilator", "--dump-stats", "--kv-script", str(script)])
        compare("pause.bin", [13, 25, 37, 49])
        compare("resume.bin", expect)
        steps = [line for line in trace.splitlines() if line.startswith("KV_STEP ")]
        if len(steps) != 2 or "dma_bytes=96" not in steps[0] or "dma_bytes=64" not in steps[1]:
            raise RuntimeError("incomplete session evidence or repeated residency transfer")

    for action, expect_error in (
        (f"output\tlast\t{folder / 'future.bin'}", "KV script tensor symbol or file is invalid"),
        ("feedback\tlast\tfeed", "KV feedback read failed"),
    ):
        label = f"future-{action.split()[0]}"
        script = folder / (label + ".tsv")
        script.write_text(f"step\t0\n{action}\nstep\t1\n")
        call(program, label, ["--backend", "verilator", "--kv-script", str(script)], expect_error)

    broken = folder / "bad-crc.yhb"
    broken.write_bytes(blob[:-1] + bytes([blob[-1] ^ 1]))
    call(broken, "bad-crc", ["--inspect"], "YHB CRC32 mismatch")

    # Mutate the final prefill WAIT to a same-offset marker and repair the
    # checksum, so the decoder has to detect the undrained DMA itself.
    raw = bytearray(blob)
    command_offset = info["sections"]["commands"]["offset"]
    struct.pack_into("<II", raw, command_offset + 7 * 16, 5, 0)
    struct.pack_into("<I", raw, 16, 0)
    struct.pack_into("<I", raw, 16, zlib.crc32(raw))
    broken = folder / "undrained.yhb"
    broken.write_bytes(raw)
    call(broken, "undrained", ["--inspect"], "undrained YHB KV checkpoint")

    invalid = deepcopy(manifest)
    invalid["execution"][-1]["step"] = 3
    try:
        emit_yhb(folder, invalid)
    except ValueError as exc:
        if "consecutive" not in str(exc):
            raise
    else:
        raise RuntimeError("nonconsecutive marker was accepted")
    print("KV session, device feedback, host feedback/override, CRC and fence rejection: PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runner", type=Path, default=RAM / "verilator/ysyx_run")
    args = parser.parse_args()
    if RAM not in args.out.resolve().parents:
        parser.error(f"output must be a new child directory of {RAM}")
    args.out.mkdir()
    verify(args.out, args.runner)


if __name__ == "__main__":
    main()
