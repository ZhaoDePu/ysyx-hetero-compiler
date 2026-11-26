"""Export standalone C99 sources and sampled YHB/IO under the Z15 board kit."""

import argparse
import json
from pathlib import Path
import shutil
import struct

from hetero_compiler.compiler import tensor_bytes
from hetero_compiler.reference import evaluate
from hetero_compiler.yhb import inspect_yhb


FORMATS = {"int8": "b", "uint8": "B", "int16": "h", "uint16": "H", "int32": "i", "int64": "q"}
SOURCES = ("hal_linux_uio.c", "ysyx_rt.c", "ysyx_rt.h", "ysyx_run.c", "z15_regs.h", "c10_session_core.h")


def export(kit, specs):
    source = Path(__file__).resolve().parents[1] / "runtime"
    (kit / "source").mkdir(exist_ok=True)
    for name in SOURCES:
        shutil.copyfile(source / name, kit / "source" / name)
    (kit / "cases").mkdir(exist_ok=True)
    for record in specs:
        case, bundle, trial, samples = record
        if not case or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for c in case):
            raise ValueError(f"invalid sample case {case!r}")
        bundle = Path(bundle)
        manifest = json.loads((bundle / "bundle.json").read_bytes())
        graph = manifest["graph"]
        program = bundle / "program.yhb"
        if inspect_yhb(program.read_bytes())["target"] != 1:
            raise ValueError("deployment kit requires a Balanced YHB binding")
        inputs = {}
        for name, binding in manifest["input_bindings"].items():
            if trial != "-":
                if Path(name).name != name or name in (".", ".."):
                    raise ValueError(f"trial input symbol {name!r} is not a filename")
                binary = (Path(trial) / f"{name}.input.bin").read_bytes()
                if len(binary) != binding["size"]:
                    raise ValueError(f"{name}: input size disagrees with the YHB symbol")
                inputs[name] = [x for x, in struct.iter_unpack("<" + FORMATS[binding["dtype"]], binary)]
            else:
                inputs[name] = graph["tensors"][name]["data"]
        expected = evaluate(graph, inputs)
        folder = kit / "cases" / case
        folder.mkdir()
        shutil.copyfile(program, folder / "program.yhb")
        bindings = []
        for role, names, values in (("input", graph["inputs"], inputs), ("output", graph["outputs"], expected)):
            for index, name in enumerate(names):
                if any(x in name for x in ("\n", "\r", "\t", "=")):
                    raise ValueError(f"unsupported CLI symbol {name!r}")
                dtype = graph["tensors"][name]["dtype"]
                file = f"{role}-{index}.bin"
                (folder / file).write_bytes(tensor_bytes(values[name], dtype))
                bindings.append(f"{role}\t{name}\t{file}\n")
        (folder / "io.tsv").write_text("".join(bindings), encoding="utf-8")
        if int(samples) <= 0:
            raise ValueError("item count must be positive")
        (folder / "samples.txt").write_text(str(int(samples)) + "\n", encoding="ascii")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", nargs=4, action="append", required=True,
                        metavar=("NAME", "BUNDLE", "TRIAL_OR_DASH", "ITEMS"))
    args = parser.parse_args()
    export(args.out, args.case)
