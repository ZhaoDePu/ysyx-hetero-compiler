"""Replay a checked P17 command stream against the C99 functional HAL."""

import argparse
from ctypes import CDLL, CFUNCTYPE, POINTER, Structure, byref, c_char_p, c_int, c_size_t, c_uint32, c_void_p, create_string_buffer
from pathlib import Path


WRITE = CFUNCTYPE(c_int, c_void_p, c_uint32, c_uint32)
COPY = CFUNCTYPE(c_int, c_void_p, c_uint32, c_void_p, c_size_t)
CLOSE = CFUNCTYPE(None, c_void_p)


class Hal(Structure):
    _fields_ = [("ctx", c_void_p), ("read", c_void_p), ("write", WRITE), ("ddr_write", COPY),
                ("ddr_read", COPY), ("sync", c_void_p), ("observe", c_void_p), ("close", CLOSE)]


def replay(library, path):
    lib = CDLL(str(library))
    open_ref = lib.ysyx_hal_reference_p17_open
    open_ref.argtypes = [POINTER(Hal), POINTER(c_char_p)]
    hal, error = Hal(), c_char_p()
    if open_ref(byref(hal), byref(error)):
        raise ValueError(error.value)
    heads = 0
    total = 0
    moves = 0
    try:
        def write(reg, value):
            if hal.write(hal.ctx, reg, value):
                raise ValueError(f"invalid MMIO {(reg, value)} while working on line {index}")

        with path.open() as control:
            if next(control).split() != ["-", "0", "0"]:
                raise ValueError("unexpected replay seed")
            for index, line in enumerate(control, start=2):
                op, *args = line.split()
                if op == "write":
                    write(*map(int, args))
                elif op == "submit_gpu":
                    pc, job = map(int, args)
                    write(0x40, pc)
                    write(0x44, job)
                    write(0x10, 16)
                elif op in {"dma_in", "dma_out"}:
                    ddr, sp, size = map(int, args)
                    for reg, value in ((0x20, 0x40000000), (0x24, ddr), (0x28, sp), (0x2c, size),
                                       (0x30, 2 | (op == "dma_out")), (0x10, 4)):
                        write(reg, value)
                    moves += 1
                elif op in {"ddr_put", "compare_ddr"}:
                    data = Path(args[0]).read_bytes()
                    where, size = map(int, args[1:])
                    if len(data) != size:
                        raise ValueError(f"wrong data extent on line {index}")
                    if op == "ddr_put":
                        if hal.ddr_write(hal.ctx, where, data, size):
                            raise ValueError(f"invalid DDR seed on line {index}")
                    else:
                        output = create_string_buffer(size)
                        if hal.ddr_read(hal.ctx, where, output, size) or output.raw != data:
                            raise ValueError(f"DDR output differs from golden on line {index}")
                        heads += 1
                        total += size
                        print(f"{Path(args[0]).name} {size} B exact")
                elif op not in {"dma_wait", "wait_gpu", "wait_npu"}:
                    raise ValueError(f"unknown replay operation {op}")
    finally:
        hal.close(hal.ctx)
    if (heads, total) != (6, 176400):
        raise ValueError(f"incomplete comparison {(heads, total)}")
    print(f"CPU_FUNCTIONAL_PASS heads={heads} bytes={total} descriptors={moves}")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("library", type=Path)
    cli.add_argument("replay", type=Path)
    args = cli.parse_args()
    replay(args.library, args.replay)


if __name__ == "__main__":
    main()
