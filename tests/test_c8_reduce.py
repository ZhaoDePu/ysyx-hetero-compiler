"""Checks for bank-skewed and SRAM-resident wide Conv reductions."""

import unittest

from hetero_compiler import c7_reduce_conv
from hetero_compiler.c6_gpu import build_gpu_stage
from hetero_compiler.c6_memory import HALF_BYTES
from hetero_compiler.c6_tensor import Conv


class WideReduceTests(unittest.TestCase):
    def test_skew_selects_distinct_banks_and_leaves_narrow_tail_intact(self):
        stage = build_gpu_stage("skew", 0, 68, lambda _p: None, warp_skew=True)
        starts = [stage.words[job["entry"] // 4] >> 20 for job in stage.jobs]
        self.assertEqual(starts, [0, 17, 34, 51, 64])
        self.assertEqual([job["threads"] for job in stage.jobs], [16, 16, 16, 16, 4])
        for start in range(0, 16, 4):
            for lane in range(start, start + 4):
                self.assertEqual({(lane ^ base) % 4 for base in starts[:4]}, {0, 1, 2, 3})

    def test_wide_reduce_keeps_raw_planes_in_one_sram_half(self):
        spec = Conv("large", 40, 40, 128, 128, 3, 2, 1)
        tiles = c7_reduce_conv.scratch_combine_tiles(
            spec, input_ddr=0x10000, output_ddr=0x50000,
            weights_ddr=(0x80000, 0x90000, 0xa0000), bias_ddr=0xb0000,
            zero_ddr=0xc0000, mul=[17] * 128, shift=[15] * 128,
        )
        self.assertEqual(sum(t.stores[0].size for t in tiles), 20 * 20 * 128)
        self.assertEqual([t.zone for t in tiles], [i % 2 for i in range(len(tiles))])
        for tile in tiles:
            self.assertEqual(len(tile.stores), 1)
            self.assertLessEqual(tile.scratch_bytes, HALF_BYTES)
            raw = [s for s in tile.stages if s.id.rpartition("/k")[2].isdigit()]
            outputs = [dict(stage.writes)[0x70] for stage in raw]
            stage, = [s for s in tile.stages if s.device == "GPU"]
            page = (stage.jobs[0]["gpu_job_reg"] & 511) * 4096
            self.assertEqual(page, outputs[0])
            self.assertGreater(outputs[-1] - page, 4096)
            self.assertLess(outputs[-1] - page, 240 * 1024)
            self.assertEqual([job["warp"] for job in stage.jobs], [0, 1, 2, 3])
            self.assertEqual(dict(tile.stages[-1].writes)[0x74] // HALF_BYTES, tile.zone)


if __name__ == "__main__":
    unittest.main()
