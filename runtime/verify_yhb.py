"""Drive the compiled C99 runtime with several independent runtime input vectors."""
import argparse
import json
from pathlib import Path
import random
import struct
import subprocess

from hetero_compiler.reference import evaluate
from hetero_compiler.yhb import emit_yhb, inspect_yhb

ROOT = Path("/dev/shm/ysyx_compiler_c2/runtime")
DEFAULT_RUNNER = ROOT / "verilator/ysyx_run"
PRIORITY = ["nice", "-n", "15", "ionice", "-c", "3"]
FIELD = {"int8": "b", "uint8": "B", "int16": "h", "uint16": "H", "int32": "i", "int64": "q"}


def _tensor(values, dtype):
    return struct.pack(f"<{len(values)}{FIELD[dtype]}", *values)


def _random_inputs(manifest, index):
    rng = random.Random(20261125 + index)
    result = {}
    for name, spec in manifest["input_bindings"].items():
        low, high = (0, 63) if spec["dtype"].startswith("uint") else (-32, 31)
        elements = spec["size"] // struct.calcsize("<" + FIELD[spec["dtype"]])
        result[name] = [rng.randint(low, high) for _ in range(elements)]
    return result


def verify(bundle, scratch, runner, vectors):
    bundle, scratch = Path(bundle).resolve(), Path(scratch).resolve()
    if ROOT.resolve() != ROOT or ROOT not in scratch.parents:
        raise ValueError("runtime verification scratch must be inside the C2 runtime RAM directory")
    if vectors < 1:
        raise ValueError("vector count must be positive")
    scratch.mkdir()
    manifest = json.loads((bundle / "bundle.json").read_bytes())
    graph = manifest["graph"]
    program = bundle / "program.yhb"
    if not program.exists():
        raise ValueError("bundle does not contain program.yhb")
    binary = program.read_bytes()
    metadata = inspect_yhb(binary)
    if metadata["target"] != 1:
        raise ValueError("numeric RTL replay requires the frozen Balanced binding")
    broken = scratch / "bad-crc.yhb"
    broken.write_bytes(binary[:-1] + bytes((binary[-1] ^ 128,)))
    bad = subprocess.run(PRIORITY + [str(runner), str(broken), "--inspect"],
                         check=False, capture_output=True, text=True)
    if bad.returncode == 0 or "CRC32" not in bad.stderr:
        raise ValueError("C loader admitted a corrupted program")
    results = []
    for i in range(vectors):
        case = scratch / f"trial-{i}"
        case.mkdir()
        supplied = _random_inputs(manifest, i)
        expected = evaluate(graph, supplied)
        (case / "input-values.json").write_text(json.dumps(supplied, indent=2) + "\n")
        command = PRIORITY + [str(runner), str(program), "--backend", "verilator", "--dump-stats"]
        for index, (name, values) in enumerate(supplied.items()):
            source = case / f"input-{index}.bin"
            source.write_bytes(_tensor(values, manifest["input_bindings"][name]["dtype"]))
            command += ["--input", f"{name}={source}"]
        outputs = {}
        for index, name in enumerate(graph["outputs"]):
            path = case / f"output-{index}.bin"
            command += ["--output", f"{name}={path}"]
            outputs[name] = path
        done = subprocess.run(command, capture_output=True, text=True)
        (case / "runtime.log").write_text(done.stdout + done.stderr)
        if done.returncode:
            raise RuntimeError(f"runtime trial {i} failed:\n{done.stdout}{done.stderr}")
        for name, path in outputs.items():
            desired = _tensor(expected[name], graph["tensors"][name]["dtype"])
            if path.read_bytes() != desired:
                (case / "expected.bin").write_bytes(desired)
                raise ValueError(f"runtime trial {i}: output {name!r} differs from Python golden")
        summary, = (line for line in done.stdout.splitlines() if line.startswith("STATS "))
        stats = {}
        for token in summary.split()[1:]:
            key, value = token.split("=", 1)
            stats[key] = int(value) if value.isdecimal() else value
        comparison = dict(trial=i, bytes=sum(p.stat().st_size for p in outputs.values()), **stats)
        results.append(comparison)
        print("YHB_RUNTIME_PASS " + json.dumps(comparison, sort_keys=True))
    result = {"program_bytes": len(binary), "trials": results}
    (scratch / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--bundle", required=True, type=Path)
    cli.add_argument("--scratch", required=True, type=Path)
    cli.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    cli.add_argument("--vectors", type=int, default=2)
    cli.add_argument("--pack", action="store_true", help="pack a scratch bundle lacking program.yhb")
    args = cli.parse_args()
    if args.pack:
        target = args.bundle.resolve()
        if ROOT not in target.parents:
            raise ValueError("runtime --pack requires a bundle inside C2 runtime RAM")
        emit_yhb(target, json.loads((target / "bundle.json").read_bytes()))
    verify(args.bundle, args.scratch, args.runner, args.vectors)


if __name__ == "__main__":
    main()
