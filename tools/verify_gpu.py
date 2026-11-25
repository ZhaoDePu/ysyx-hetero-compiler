#!/usr/bin/env python3
"""Execute one compiler-emitted stage on a pinned, out-of-tree GPU RTL model."""
import argparse
import json
from pathlib import Path
from struct import unpack_from, pack
import subprocess
import tempfile


SCRATCH = Path("/dev/shm/ysyx_compiler_c3/gpu")
BUILD = Path(__file__).with_name("build_gpu_rtl.sh")
PRIORITY = ["nice", "-n", "10", "ionice", "-c2", "-n7"]


def program_from_bundle(bundle, stage, code_limit):
    elf = (bundle / stage["elf"]).read_bytes()
    if len(elf) < 84 or elf[:7] != b"\x7fELF\x01\x01\x01":
        raise ValueError("invalid GPU ELF32 LSB header")
    kind, machine, version, entry, phoff, _, flags, _, phsize, phcount = unpack_from("<HHIIIIIHHH", elf, 16)
    if (kind, machine, version, phoff, flags, phsize, phcount) != (2, 243, 1, 52, 8, 32, 1):
        raise ValueError("invalid GPU executable ABI")
    ptype, offset, address, physical, size, memory, permissions, _ = unpack_from("<8I", elf, phoff)
    if (ptype, address, physical, permissions) != (1, 0, 0, 5) or size != memory:
        raise ValueError("GPU executable requires a single read/execute segment at address zero")
    if not size or size % 4 or size > code_limit or len(elf) < offset + size:
        raise ValueError("invalid GPU executable segment size")
    if entry != stage["jobs"][0]["entry"]:
        raise ValueError("GPU ELF entry does not match its first job")
    code = elf[offset:offset + size]
    words = [int(line, 16) for line in (bundle / stage["hex"]).read_text().splitlines()]
    if code != pack(f"<{len(words)}I", *words):
        raise ValueError("GPU ELF and word HEX differ")
    return code


def simulate(bundle, stage_id, initial, result, scratch=SCRATCH):
    if scratch != SCRATCH or scratch.resolve() != SCRATCH:
        raise ValueError(f"GPU simulator scratch must be {SCRATCH}")
    manifest = json.loads((bundle / "gpu.json").read_text())
    matches = [stage for stage in manifest["stages"] if stage["id"] == stage_id]
    if len(matches) != 1:
        raise ValueError(f"expected one GPU stage {stage_id!r}")
    stage = matches[0]
    variant = manifest["variant"]
    if variant == "v6":
        target = variant
    elif variant in {"g3", "g4"}:
        target = f"g4-{manifest['lanes']}t{manifest['warps']}w"
    else:
        raise ValueError("unknown simulation target")
    if initial.stat().st_size != manifest["window_bytes"]:
        raise ValueError("initial image must contain exactly the GPU page bytes")
    code = program_from_bundle(bundle, stage, 262144 if variant == "v6" else 16384)
    executable = scratch / target / "sim"
    if (not executable.is_file() or executable.stat().st_mtime_ns <
            max(BUILD.stat().st_mtime_ns, BUILD.with_name("gpu_rtl.cpp").stat().st_mtime_ns)):
        subprocess.run([*PRIORITY, "bash", str(BUILD), target], check=True)
    with tempfile.TemporaryDirectory(prefix="run-", dir=scratch) as temp:
        temp = Path(temp)
        image, commands = temp / "code.bin", temp / "jobs.txt"
        image.write_bytes(code)
        commands.write_text("".join(f"{j['entry']} {j['threads']} {j['warp']}\n" for j in stage["jobs"]))
        proc = subprocess.run([*PRIORITY, str(executable), str(image), str(initial.resolve()),
                               str(commands), str(result.resolve())], cwd=temp,
                              check=True, text=True, stdout=subprocess.PIPE)
    return proc.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--initial", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--scratch", type=Path, default=SCRATCH)
    args = parser.parse_args()
    print(simulate(args.bundle, args.stage, args.initial, args.result, args.scratch))


if __name__ == "__main__":
    main()
