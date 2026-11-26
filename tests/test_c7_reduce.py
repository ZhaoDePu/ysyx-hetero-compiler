"""Checks for the Z15 compiler's packed bias epilogue."""
import unittest

from hetero_compiler.c6_tensor import Conv
from hetero_compiler import c7_reduce_conv


class NativeEpilogueTests(unittest.TestCase):
    def test_plan_preserves_whole_scale_periods_and_initializes_each_half(self):
        conv = Conv("large", 10, 10, 256, 16, 3, 1, 1)
        tiles = c7_reduce_conv.native_combine_tiles(
            conv, output_ddr=0x50000, partials_ddr=(0, 0x10000, 0x20000, 0x30000, 0x40000),
            zero_ddr=0x60000, mul=list(range(16)), shift=[24] * 16,
        )
        self.assertEqual([tile.zone for tile in tiles], [i % 2 for i in range(len(tiles))])
        self.assertEqual(sum(tile.stores[0].size for tile in tiles), 1600)
        self.assertEqual([len(tile.loads) for tile in tiles[:3]], [6, 6, 5])
        self.assertEqual([stage.device for stage in tiles[0].stages], ["GPU", "NPU"])
        self.assertEqual(len(tiles[0].stages[0].jobs), 192 // 32)
        cmds = tiles[0].stages[1].writes
        n = dict(cmds)[0x60] >> 16
        self.assertEqual(n, 192)
        self.assertEqual([data & 65535 for addr, data in cmds if addr == 0xc8], list(range(16)) * 12)
        self.assertFalse(any(addr == 0xc8 for addr, _ in tiles[1].stages[1].writes))

    def test_four_word_kernel_keeps_all_warps_active(self):
        conv = Conv("large", 10, 10, 128, 128, 3, 1, 1)
        tiles = c7_reduce_conv.native_combine_tiles(
            conv, output_ddr=0x30000, partials_ddr=(0, 0x10000, 0x20000),
            zero_ddr=0x40000, mul=[1000] * 128, shift=[24] * 128,
        )
        self.assertEqual([job["warp"] for job in tiles[0].stages[0].jobs], [0, 1, 2, 3])

    def test_native_core_falls_back_for_channel_period_above_page_capacity(self):
        conv = Conv("wide", 10, 10, 256, 256, 3, 1, 1)
        self.assertEqual(c7_reduce_conv.native_combine_width(conv), 0)
