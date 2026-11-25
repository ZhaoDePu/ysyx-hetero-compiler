"""Exercise valid and malformed YHB1 descriptors on the Step 2 C99 runtime."""

import argparse
import json
from pathlib import Path
import re
import struct
import subprocess
import zlib

from hetero_compiler.profile import load_profile
from hetero_compiler.yhb import emit_yhb, inspect_yhb


RAM = Path("/dev/shm/ysyx_compiler_c4vpu")
PRIORITY = ["nice", "-n", "10", "ionice", "-c2", "-n7"]
KIT = Path(__file__).resolve().parents[1] / "deploy/z15_board_kit"
HEX_OR_DEC = r"(?:0x[0-9a-fA-F]+|[0-9]+)"


def run(runner, arguments, log):
    result = subprocess.run(PRIORITY + [str(runner), *map(str, arguments)],
                            text=True, capture_output=True, timeout=900)
    log.write_text(result.stdout + result.stderr)
    return result


def replay(runner, case, folder):
    folder.mkdir()
    command = [case / "program.yhb", "--backend", "verilator", "--dump-stats"]
    checkpoint = case / "kv.tsv"
    if checkpoint.exists():
        command += ["--kv-script", checkpoint]
    outputs = []
    for row in (case / "io.tsv").read_text().splitlines():
        direction, symbol, filename = row.split("\t")
        if direction == "input":
            command += ["--input", f"{symbol}={case / filename}"]
        elif direction == "output":
            command += ["--output", f"{symbol}={folder / filename}"]
            outputs.append(filename)
        else:
            raise ValueError(f"invalid binding direction {direction}")
    result = run(runner, command, folder / "runtime.log")
    if result.returncode:
        raise ValueError(f"{case.name}: C99 RTL replay failed; see {folder / 'runtime.log'}")
    for filename in outputs:
        if (folder / filename).read_bytes() != (case / filename).read_bytes():
            raise ValueError(f"{case.name}: non-identical tensor {filename}")
    line, = [l for l in result.stdout.splitlines() if l.startswith("STATS ")]
    fields = {key: int(val, 0) for key, val in re.findall(r"(\w+)=(" + HEX_OR_DEC + ")", line)}
    if fields.get("abi_error") != 0 or fields.get("invalid_dma_descriptors") != 0:
        raise ValueError(f"{case.name}: missing or nonzero Step 2 ABI error evidence")
    return {"case": case.name, "status": "byte_exact", "outputs": len(outputs),
            "bytes": sum((folder / name).stat().st_size for name in outputs), **fields}


def corrupt_words(blob, folder, label, start, changes):
    broken = bytearray(blob)
    for word, value in changes.items():
        struct.pack_into("<I", broken, start + word * 4, value)
    struct.pack_into("<I", broken, 16, 0)
    struct.pack_into("<I", broken, 16, zlib.crc32(broken))
    fixture = folder / (label + ".yhb")
    fixture.write_bytes(broken)
    return fixture


def bad_descriptors(blob, folder):
    directory = inspect_yhb(blob)["sections"]
    start = directory["dma"]["offset"]
    if not directory["dma"]["count"]:
        raise ValueError("missing DMA in negative test fixture")
    tests = {"zero_length": {3: 0}, "unaligned_ddr": {1: 1}, "unaligned_scratch": {2: 1},
             "unaligned_length": {3: 31}, "pool_end": {1: 0x1fffffe0, 3: 64},
             "ddr_wrap": {1: 0xffffffe0, 3: 64}, "half_cross": {2: 0xfffe0, 3: 64},
             "bad_direction": {0: 3}, "scratch_end": {2: 0x1fffe0, 3: 64},
             "bad_flags": {4: 4}}
    for label, edits in tests.items():
        yield corrupt_words(blob, folder, label, start, edits)


def table_case(case):
    case.mkdir()
    values = bytes(range(256))
    table = bytes((5 * i + 19) & 255 for i in range(256))
    (case / "data.bin").write_bytes(values)
    (case / "expected.bin").write_bytes(table)
    (case / "io.tsv").write_text("input\tx\tdata.bin\noutput\ty\texpected.bin\n")
    writes = [(0xc4, 12), (0xc8, 1), (0x10, 128)]
    for i, value in enumerate(table):
        writes += [(0xcc, i | value << 8), (0x10, 256)]
    writes += [(0xa0, 0), (0xa8, 256), (0xb0, 256), (0xbc, 11), (0x10, 64)]
    allocs = {"x": {"offset": 0, "size": 256}, "y": {"offset": 256, "size": 256}}
    transfers = [{"type": "dma", "direction": direction, "bytes": 256,
                  "ddr_offset": offset, "scratch_offset": offset}
                 for direction, offset in (("load", 0), ("store", 256))]
    execution = [transfers[0], {"type": "dma-fence"}, {"type": "launch", "stage": "lookup"},
                 {"type": "fence", "wait_for": "lookup"}, transfers[1], {"type": "dma-fence"}]
    return emit_yhb(case, {
        "profile": load_profile("fpga_zu15eg_balanced"),
        "graph": {"inputs": ["x"], "outputs": ["y"],
                  "tensors": {name: {"shape": [256], "dtype": "uint8"} for name in allocs}},
        "plan": {"stages": [{"id": "lookup", "device": "VPU", "zone": 0}], "allocations": allocs},
        "backends": {"gpu": {"stages": []}, "npu": {"commands": [{"stage": "lookup", "writes": writes}]}},
        "input_bindings": {"x": {}}, "output_bindings": {"y": {}},
        "initial_image": "data.bin", "execution": execution,
    })


def bad_table_commands(blob, folder):
    directory = inspect_yhb(blob)["sections"]
    stage, data = directory["stages"]["offset"], directory["data"]["offset"]
    payload, words = struct.unpack_from("<II", blob, stage + 12)
    start = data + payload
    cases = {"stale_scale_index": {0: 0xc0}, "bad_scale_payload": {3: 0x200000},
             "stale_activation": {6: 0xc0}, "bad_activation_payload": {7: 0x10000},
             "combine_actions": {words - 1: 72}, "inplace_lookup": {words - 7: 0}}
    for label, fields in cases.items():
        yield corrupt_words(blob, folder, label, start, fields)


def bad_transposes(blob, folder):
    directory = inspect_yhb(blob)["sections"]
    stages = directory["stages"]
    for idx in range(stages["count"]):
        stage = struct.unpack_from("<8I", blob, stages["offset"] + 32 * idx)
        _, _, device, payload, words, _, _, zone = stage
        if device != 3:
            continue
        start = directory["data"]["offset"] + payload
        program = struct.unpack_from(f"<{words}I", blob, start)
        positions = {reg: i for i, reg in enumerate(program[::2])}
        values = dict(zip(program[::2], program[1::2], strict=True))
        if values.get(0x10) != 64 or values.get(0xbc, 0) & 15 != 9:
            continue
        cases = {"transpose_zero_rows": (0xb4, values[0xb4] & ~1023),
                 "transpose_short_stride": (0xb8, values[0xb8] & 1023),
                 "transpose_cross_half": (0xa8, ((zone + 1) << 20) - 1),
                 "transpose_overlap": (0xa8, values[0xa0])}
        for label, (register, value) in cases.items():
            yield corrupt_words(blob, folder, label, start, {2 * positions[register] + 1: value})
        break


def verify(kit, scratch, linux_runner, sim_runner, case_names):
    if RAM.resolve() != RAM or RAM not in scratch.resolve().parents:
        raise ValueError("qualification must use C4-VPU scratch")
    scratch.mkdir()
    cases = [kit / "cases" / name for name in case_names]
    modes = {"native": linux_runner, "verilator": sim_runner}
    fixtures = list(bad_descriptors((kit / "cases/00-mixed/program.yhb").read_bytes(), scratch))
    rejected, crc, measurements, bad_native = [], [], [], []
    lookup = table_case(scratch / "lookup")
    native_fixtures = list(bad_table_commands(lookup.read_bytes(), scratch))
    for mode, runner in modes.items():
        for fixture in fixtures:
            result = run(runner, [fixture, "--inspect"], scratch / f"{fixture.stem}-{mode}.log")
            if result.returncode == 0 or "invalid YHB DMA extent" not in result.stderr:
                raise ValueError(f"{mode} accepted malformed DMA {fixture.name}")
            rejected.append({"backend": mode, "fixture": fixture.stem, "status": "rejected"})
        for fixture in native_fixtures:
            result = run(runner, [fixture, "--inspect"], scratch / f"{fixture.stem}-{mode}.log")
            if result.returncode == 0 or "invalid YHB native stage" not in result.stderr:
                raise ValueError(f"{mode} accepted malformed control command {fixture.name}")
            bad_native.append({"backend": mode, "fixture": fixture.stem, "status": "rejected"})
    table_measurement = replay(sim_runner, lookup.parent, scratch / "lookup-rtl")
    print("ABI_TABLE_PASS " + json.dumps(table_measurement, sort_keys=True), flush=True)
    for case in cases:
        folder = scratch / case.name
        row = replay(sim_runner, case, folder)
        measurements.append(row)
        print("ABI_RTL_PASS " + json.dumps(row, sort_keys=True), flush=True)
        raw = (case / "program.yhb").read_bytes()
        broken = folder / "bad-crc.yhb"
        broken.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        for fixture in bad_transposes(raw, folder):
            for mode, runner in modes.items():
                result = run(runner, [fixture, "--inspect"], folder / f"{fixture.stem}-{mode}.log")
                if result.returncode == 0 or "invalid YHB native stage" not in result.stderr:
                    raise ValueError(f"{mode} accepted malformed transpose {fixture.name}")
                bad_native.append({"backend": mode, "fixture": fixture.stem, "status": "rejected"})
        for mode, runner in modes.items():
            good = run(runner, [case / "program.yhb", "--inspect"], folder / f"valid-{mode}.log")
            bad = run(runner, [broken, "--inspect"], folder / f"crc-{mode}.log")
            if good.returncode or bad.returncode == 0 or "YHB CRC32 mismatch" not in bad.stderr:
                raise ValueError(f"{mode} unexpected CRC result for {case.name}")
            crc.append({"backend": mode, "case": case.name, "status": "rejected"})
    (scratch / "result.json").write_text(json.dumps(
        {"rtl_replays": measurements, "abi_table": table_measurement,
         "malformed_dma": rejected, "malformed_native": bad_native, "crc": crc}, indent=2) + "\n")
    print(f"STEP2_ABI_PASS rtl_cases={len(measurements)} crc={len(crc)} "
          f"invalid_descriptors_rejected={len(rejected)} invalid_native_rejected={len(bad_native)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, default=KIT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--native", type=Path, default=RAM / "runtime/linux/ysyx_run")
    parser.add_argument("--sim", type=Path, default=RAM / "runtime/verilator/ysyx_run")
    parser.add_argument("cases", nargs="*")
    args = parser.parse_args()
    names = args.cases or sorted(p.name for p in (args.kit / "cases").iterdir() if p.is_dir())
    verify(args.kit, args.out, args.native, args.sim, names)


if __name__ == "__main__":
    main()
