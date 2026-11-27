"""Run the 15 legacy YHB cases and the 13 pinned O4 cases in C14 scratch."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from runtime import verify_z15_abi
from tools import verify_bundle, verify_gpu


# The resume script applies nice/ionice to this process. Children inherit both.
PRIORITY = []


def memory_available():
    host = dict(line.split()[:2] for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(host["MemAvailable:"]) * 1024
    group = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
                 if line.startswith("0::"))
    root = Path("/sys/fs/cgroup")
    current = root / group.lstrip("/")
    for path in [current, *current.parents]:
        if path != root and root not in path.parents:
            break
        limit = path / "memory.max"
        if limit.is_file() and (value := limit.read_text().strip()) != "max":
            available = min(available, int(value) - int((path / "memory.current").read_text()))
    return available


def guard(_root=None):
    while (value := memory_available()) < (8 << 30):
        print(f"memory wait legacy effective_available_B={value}", flush=True)
        time.sleep(60)


def yhb(root, native, simulator):
    verify_z15_abi.RAM = root
    verify_z15_abi.PRIORITY = PRIORITY
    run = verify_z15_abi.run

    def checked(runner, arguments, log):
        if runner == simulator and "--inspect" not in arguments:
            guard()
        return run(runner, arguments, log)

    verify_z15_abi.run = checked
    kit = verify_z15_abi.KIT
    cases = sorted(p.name for p in (kit / "cases").iterdir() if p.is_dir())
    if len(cases) != 15:
        raise ValueError("expected exactly 15 legacy YHB cases")
    verify_z15_abi.verify(kit, root / "legacy-yhb", native, simulator, cases)


def o4(root, models):
    verify_bundle.SCRATCH = root
    verify_bundle.P8 = root / "npu/p8-build/run-npu-p8"
    verify_bundle.LOW_PRIORITY = PRIORITY
    verify_gpu.SCRATCH = root / "gpu"
    verify_gpu.PRIORITY = PRIORITY
    execute = verify_bundle._execute

    def replay(command):
        guard()
        if len(command) > 3 and str(command[2]).endswith("/verify_gpu.py"):
            params = dict(zip(command[3::2], command[4::2], strict=True))
            print(verify_gpu.simulate(Path(params["--bundle"]), params["--stage"],
                                      Path(params["--initial"]), Path(params["--result"]),
                                      verify_gpu.SCRATCH))
        else:
            execute(command)

    verify_bundle._execute = replay
    tests = sorted(models.glob("*.json"))
    if len(tests) != 13:
        raise ValueError("expected exactly 13 pinned O4 cases")
    scratch = root / "o4"
    scratch.mkdir()
    failures = []
    for model in tests:
        bundle = scratch / model.stem
        subprocess.run([*PRIORITY, sys.executable, "-B", "-m", "hetero_compiler",
                        "--graph", model, "--profile", "asic_nangate45_p8", "--out", bundle,
                        "--vpu-lowering", "off"], check=True, env={**os.environ, "TMPDIR": str(root)})
        args = sys.argv
        sys.argv = ["verify_bundle", "--bundle", str(bundle), "--case", str(scratch / (model.stem + "-replay"))]
        try:
            verify_bundle.main()
        except (ValueError, subprocess.CalledProcessError) as error:
            failures.append(model.stem)
            print(f"O4 {model.stem} FAIL {error}", flush=True)
        else:
            print(f"O4 {model.stem} PASS", flush=True)
        finally:
            sys.argv = args
    if failures:
        raise ValueError(f"O4 failures: {failures}")
    print("O4_PASS 13/13")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scratch", type=Path)
    p.add_argument("parameters", type=Path, nargs="+")
    p.add_argument("--o4", action="store_true")
    args = p.parse_args()
    if args.scratch.resolve() != Path("/dev/shm/ysyx_compiler_c14v2"):
        p.error("use the task's private scratch directory")
    if args.o4:
        o4(args.scratch, *args.parameters)
    else:
        yhb(args.scratch, *args.parameters)


if __name__ == "__main__":
    main()
