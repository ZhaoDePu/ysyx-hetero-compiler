"""Check C9 GPU scratch bounds and signed DOT4 in the exact C99 reference."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from hetero_compiler.c6_gpu import DATA_WINDOW_BYTES
from hetero_compiler.c6_memory import align_dma
from hetero_compiler.c9_gpu_conv import dot_slices, tile_shape
from hetero_compiler.gpu import Program


ROOT = Path(__file__).resolve().parents[1]


class C9GPUConvTests(unittest.TestCase):
    def test_large_depth_stays_inside_owned_gpu_window(self):
        count, channels = tile_shape(16, 128, 8192, 512, 9, True, False)
        occupied = (8192 + 512 + align_dma(count * 9 * 4) + align_dma(count * 4608)
                    + align_dma(channels * 4608) + align_dma(channels * 8)
                    + align_dma(count * channels))
        self.assertLessEqual(occupied, DATA_WINDOW_BYTES)
        self.assertEqual(channels % 32, 0)
        self.assertLess(channels, 128)

    def test_long_dots_keep_contiguous_output_inside_the_poll_budget(self):
        self.assertEqual(list(dot_slices(4096, 576)), [(0, 2048), (2048, 4096)])
        self.assertEqual(list(dot_slices(128, 4608)), [(0, 128)])
        self.assertEqual(list(dot_slices(9, 4608)), [(0, 9)])

    def test_single_spatial_output_can_split_an_unaligned_channel_tail(self):
        _, channels = tile_shape(1, 232, 1280, 1280, 1, False, False)
        self.assertLess(channels, 232)
        self.assertEqual(channels % 32, 0)
        self.assertEqual(channels + align_dma(232 - channels), align_dma(232))

    @unittest.skipUnless(shutil.which("cc"), "C99 compiler required")
    def test_reference_dot4_signed_and_destinations(self):
        p = Program(None)
        p.li(2, 0x01807F80)  # -128, 127, -128, 1
        p.li(3, 0x807F807F)  # 127, -128, 127, -128
        p.emit(3 << 20 | 2 << 15 | 1 << 12 | 4 << 7 | 0x2B)
        p.store(4, 0, "int32")
        p.emit(2 << 20 | 2 << 15 | 1 << 12 | 4 << 7 | 0x2B)
        p.store(4, 0, "int32", 4)
        p.emit(0x00100073)
        instructions = ", ".join(f"0x{word:08x}u" for word in p.image())
        code = f"""
#include "hal_reference_c6.h"
#include "z15_regs.h"

int main(void)
{{
    static const uint32_t code[] = {{{instructions}}};
    static const uint8_t expected[] = {{0x00, 0x41, 0xff, 0xff, 0x02, 0xbf, 0, 0}};
    ysyx_hal_t hal = {{0}};
    if (ysyx_hal_reference_c6_open(&hal, 0))
        return 1;
    for (unsigned i = 0; i < sizeof(code) / sizeof(*code); ++i)
        if (hal.mmio_write32(hal.ctx, Z15_PROGRAM + 4 * i, code[i]))
            return 2;
    if (hal.mmio_write32(hal.ctx, Z15_GPU_JOB, 1u << 16) ||
        hal.mmio_write32(hal.ctx, Z15_CTRL, Z15_START_GPU))
        return 3;
    if (hal.mmio_write32(hal.ctx, Z15_DMA_BASE, Z15_POOL_BASE) ||
        hal.mmio_write32(hal.ctx, Z15_DMA_BYTES, 32) ||
        hal.mmio_write32(hal.ctx, Z15_DMA_FLAGS, 1) ||
        hal.mmio_write32(hal.ctx, Z15_CTRL, Z15_START_DMA))
        return 4;
    uint8_t actual[8];
    if (hal.ddr_read(hal.ctx, 0, actual, sizeof(actual)))
        return 5;
    hal.close(hal.ctx);
    for (unsigned i = 0; i < sizeof(actual); ++i)
        if (actual[i] != expected[i])
            return 6;
    return 0;
}}
"""
        with tempfile.TemporaryDirectory(dir=os.getenv("TMPDIR")) as work:
            path = Path(work)
            source, binary = path / "dot4.c", path / "dot4"
            source.write_text(code)
            subprocess.run(["cc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "runtime"), str(source),
                            str(ROOT / "runtime/hal_reference_c6.c"), "-o", str(binary)],
                           check=True)
            subprocess.run([binary], check=True)
