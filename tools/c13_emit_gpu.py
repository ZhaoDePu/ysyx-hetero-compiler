"""Build a freestanding C13 SIMT image for the frozen Z15 instruction window."""

import argparse
import os
from pathlib import Path
import struct
import subprocess
import tempfile


ENTRIES = {
    "rms_reduce": 1,
    "rms_scale": 4,
    "swiglu": 4,
    "kv_append": 1,
    "attention_scores": 1,
    "attention_mix": 1,
}
DEFAULT_PREFIX = "/home/zdp/ysyx-migration/tools/riscv-toolchain/bin/riscv64-linux-gnu-"


def bootstrap():
    lines = [".section .text.boot", ".balign 4"]
    for kernel, warps in ENTRIES.items():
        for warp in range(warps):
            label = f"boot_{kernel}_{warp}"
            lines.extend((
                f".global {label}",
                f"{label}:",
                ".word 0xcc302573",  # csrr a0, 0xcc3 (SIMT lane ID)
                f"addi a0, a0, {warp * 16}",
                f"jal ra, c13_{kernel}",
                "ebreak",
            ))
    return "\n".join(lines) + "\n"


def emit(prefix, source, scratch):
    with tempfile.TemporaryDirectory(dir=scratch) as folder:
        root = Path(folder)
        (root / "boot.S").write_text(bootstrap(), encoding="ascii")
        linker = """SECTIONS {
            . = 0;
            .text : { *(.text.boot) *(.text .text.*) }
            /DISCARD/ : { *(.comment) *(.riscv.attributes) *(.note*) }
        }\n"""
        (root / "image.ld").write_text(linker, encoding="ascii")
        built = root / "gpu.elf"
        environ = os.environ.copy()
        libs = (Path(prefix).parent.parent / "host-lib").resolve()
        if libs.is_dir():
            previous = environ.get("LD_LIBRARY_PATH")
            environ["LD_LIBRARY_PATH"] = os.pathsep.join([str(libs), previous] if previous else [str(libs)])
        subprocess.run([
            prefix + "gcc", "-march=rv32em", "-mabi=ilp32e", "-O2", "-g0",
            "-ffreestanding", "-fno-strict-aliasing", "-fno-pic", "-fno-inline-functions-called-once",
            "-fno-unwind-tables", "-fno-asynchronous-unwind-tables", "-fno-stack-protector",
            "-Wall", "-Wextra", "-Werror", "-nostdlib", "-static", "-Wl,--no-relax",
            "-Wl,--build-id=none", "-Wl,-T," + str(root / "image.ld"),
            str(root / "boot.S"), str(source), "-o", str(built),
        ], env=environ, check=True)
        sections = subprocess.check_output([prefix + "objdump", "-h", str(built)], text=True, env=environ)
        for line in sections.splitlines():
            fields = line.split()
            if len(fields) >= 7 and fields[0].isdigit() and fields[1] != ".text" and int(fields[2], 16):
                raise ValueError(f"GPU ELF contains unexpected data section {fields[1]}")
        binary = root / "gpu.bin"
        subprocess.run([prefix + "objcopy", "-j", ".text", "-O", "binary", str(built), str(binary)],
                       check=True, env=environ)
        data = binary.read_bytes()
        if not data or len(data) & 3 or len(data) > 0x2800:
            raise ValueError("Program exceeds the C13 instruction slot")
        syms = subprocess.check_output([prefix + "nm", "--defined-only", str(built)], text=True, env=environ)
        addresses = {name: int(addr, 16) for addr, _, name in (row.split() for row in syms.splitlines())}
        words = [word for (word,) in struct.iter_unpack("<I", data)]
        print("/* Generated from the freestanding C13 GPU kernels. */")
        print("static const uint32_t c13_gpu_words[] = {")
        for i in range(0, len(words), 8):
            print("    " + ", ".join(f"0x{x:08x}u" for x in words[i:i + 8]) + ",")
        print("};")
        for name, warps in ENTRIES.items():
            items = ", ".join(str(addresses[f"boot_{name}_{warp}"]) for warp in range(warps))
            print(f"static const uint32_t c13_{name}_pc[] = {{{items}}};")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default=os.getenv("VALENCE_RISCV_PREFIX", DEFAULT_PREFIX))
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    emit(args.prefix, args.source, args.scratch)
