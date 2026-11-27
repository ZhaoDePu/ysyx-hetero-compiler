import unittest

import numpy as np

from hetero_compiler.c14_spatial import pack_spatial_weights, row_plan, spatial_tiles
from hetero_compiler.c6_memory import HALF_BYTES
from hetero_compiler.c6_tensor import Conv


class SpatialTest(unittest.TestCase):
    def test_wide_strided_halos_and_partial_oc(self):
        conv = Conv("wide", 40, 40, 64, 40, 3, 2, 1)
        weights = np.zeros((3, 3, 64, 40), dtype=np.int8)
        bias = np.arange(40, dtype=np.int32)
        packed, blocks = pack_spatial_weights(conv, weights, bias)
        self.assertEqual(row_plan(conv, len(packed)), ((0, 20),))
        tile, = spatial_tiles(conv, input_ddr=0x100000, output_ddr=0x200000,
                             weights_ddr=0x300000, weight_bytes=len(packed), blocks=blocks,
                             zero_ddr=0x400000, mul=[1] * 40, shift=[0] * 40, lut=None)
        self.assertLessEqual(tile.scratch_bytes, HALF_BYTES)
        for dma in tile.loads + tile.stores:
            self.assertEqual((dma.ddr_offset | dma.scratch_offset | dma.size) & 31, 0)
        chains = [dict(stage.writes) for stage in tile.stages if "/chain" in stage.id]
        self.assertTrue(all(1 <= (r[0x60] & 511) <= 16 and 1 <= r[0x60] >> 16 <= 16 for r in chains))
        self.assertEqual({(r[0x7C] >> 9) & 511 for r in chains}, {9, 33})
        final = [r for r in chains if r[0x78] & (1 << 17)]
        self.assertEqual(sum((r[0x60] & 511) * (r[0x60] >> 16) for r in final), 20 * 20 * 40)
        self.assertEqual(tile.stores[0].size, 20 * 20 * 40)

    def test_512_input_channels_use_split_gemm(self):
        spec = Conv("512cin", 10, 10, 512, 256, 1, 1, 0)
        from hetero_compiler.c14_spatial import split_1x1

        self.assertTrue(split_1x1(spec))
        w = np.zeros((1, 1, 512, 256), dtype=np.int8)
        packed, blocks = pack_spatial_weights(spec, w, np.zeros(256, dtype=np.int32))
        tile, = spatial_tiles(spec, input_ddr=0, output_ddr=0x100000, weights_ddr=0x200000,
                               weight_bytes=len(packed), blocks=blocks, zero_ddr=0,
                               mul=[1]*256, shift=[0]*256, lut=None)
        self.assertEqual(dict(tile.stages[0].writes)[0xB8], 508 | 4 << 16)
        self.assertEqual({dict(stage.writes)[0x84] for stage in tile.stages if "/chain" in stage.id}, {0})


if __name__ == "__main__":
    unittest.main()
