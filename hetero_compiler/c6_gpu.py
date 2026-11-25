"""Bounded RV32EM microkernels sharing the Balanced GPU data-page contract."""

from dataclasses import dataclass

from .gpu import Program

PAGE_BYTES = 4096


@dataclass(frozen=True)
class GpuStage:
    id: str
    words: tuple
    jobs: tuple

    @property
    def device(self):
        return "GPU"


def build_gpu_stage(name, page, elements, kernel):
    """Emit one warp entry per 16 elements, with x5 as the linear lane index."""
    if not 0 <= page < 512 or not 1 <= elements <= PAGE_BYTES:
        raise ValueError(f"{name}: invalid GPU page or element count")
    p = Program(None)
    jobs = []
    for first in range(0, elements, 16):
        warp = len(jobs) % 4
        threads = min(elements - first, 16)
        jobs.append({"entry": len(p.words) * 4, "threads": threads, "warp": warp,
                     "gpu_job_reg": page | threads << 16 | warp << 24})
        p.li(15, first)
        p.jump("kernel")
    p.label("kernel")
    p.emit(0xCC3022F3)  # Read logical GID into x5.
    p.arithmetic(5, 5, 15)
    kernel(p)
    p.emit(0x00100073)  # Completion trap.
    words = tuple(p.image())
    if len(words) * 4 > 16384:
        raise ValueError(f"{name}: GPU kernel exceeds the 16 KiB program window")
    return GpuStage(name, words, tuple(jobs))


def emit_requant(p):
    """Given x7=acc and x8=mul|(shift<<16), leave saturated int8 in x7.

    This calculates the exact signed 48-bit product in two 16-bit limbs,
    including for values whose product overflows signed RV32.
    """
    p.addi(10, 8, 16, f3=1)
    p.addi(10, 10, 16, f3=5)
    p.addi(11, 8, 16, f3=5)
    p.addi(12, 7, 16, f3=1)
    p.addi(12, 12, 16, f3=5)
    p.addi(13, 7, 16 | 0x400, f3=5)
    p.arithmetic(12, 12, 10, f7=1)
    p.arithmetic(13, 13, 10, f7=1)
    p.addi(14, 12, 16, f3=5)
    p.arithmetic(13, 13, 14)  # high, signed
    p.li(9, 16)
    p.branch(4, 11, 9, "small_shift")
    p.addi(11, 11, -16)
    p.arithmetic(7, 13, 11, f3=5, f7=0x20)
    p.jump("clip")

    p.label("small_shift")
    p.arithmetic(9, 9, 11, f7=0x20)
    p.li(8, 127)
    p.arithmetic(8, 8, 9, f3=5, f7=0x20)
    p.branch(4, 8, 13, "upper")
    p.li(8, -128)
    p.arithmetic(8, 8, 9, f3=5, f7=0x20)
    p.branch(4, 13, 8, "lower")
    p.arithmetic(13, 13, 9, f3=1)
    p.addi(12, 12, 16, f3=1)
    p.addi(12, 12, 16, f3=5)
    p.arithmetic(12, 12, 11, f3=5)
    p.arithmetic(7, 13, 12)

    p.label("clip")
    p.li(8, -128)
    p.branch(4, 7, 8, "lower")
    p.li(8, 127)
    p.branch(4, 8, 7, "upper")
    p.jump("clipped")
    p.label("lower")
    p.li(7, -128)
    p.jump("clipped")
    p.label("upper")
    p.li(7, 127)
    p.label("clipped")
