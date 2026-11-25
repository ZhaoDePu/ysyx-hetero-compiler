"""Resource and boundary checks for C6 native DDR tiles."""

import unittest

from hetero_compiler.c6_memory import Lifetime, Scratch, reserve_ddr
from hetero_compiler.c6_reduce_conv import combine_tiles, raw_conv_tiles, reduction_widths
from hetero_compiler.c6_tensor import Conv, conv_tiles
from hetero_compiler.yhb import POOL_BYTES


class C6CapacityTests(unittest.TestCase):
    def test_static_reuse_waits_for_final_consumer(self):
        slots, peak = reserve_ddr([Lifetime("input", 53, 0, 2), Lifetime("consumer_out", 63, 2, 3),
                                   Lifetime("next", 60, 3, 5)], base=4096)
        self.assertEqual(peak, 4224)
        self.assertEqual(slots["input"].offset, 4096)
        self.assertEqual(slots["consumer_out"].offset, 4160)
        self.assertEqual(slots["next"].offset, slots["input"].offset)

    def test_ddr_and_scratch_guards(self):
        with self.assertRaisesRegex(ValueError, "capacity"):
            reserve_ddr([Lifetime("over", 33, 0, 0)], base=POOL_BYTES - 32)
        with self.assertRaisesRegex(ValueError, "ownership half"):
            Scratch(1).reserve((1 << 20) + 1)
        with self.assertRaisesRegex(ValueError, "ownership half"):
            Scratch(2)

    def test_conv_tail_aligns_dma_without_changing_public_size(self):
        stem = Conv("head", 10, 10, 16, 4, 1, 1, 0)
        tiles = conv_tiles(stem, input_ddr=0, output_ddr=16384, weights_ddr=32768,
                           bias_ddr=49152, mul=[1] * 4, shift=[0] * 4)
        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[-1].stores[0].size, 416)
        self.assertEqual(dict(tiles[-1].stages[0].writes)[0x60] & 511, 100)

    def test_rgb_stem_uses_halo_rows_and_caches_constants_per_half(self):
        stem = Conv("stem", 320, 320, 3, 16, 3, 2, 1)
        tiles = conv_tiles(stem, input_ddr=0, output_ddr=1 << 20, weights_ddr=2 << 20,
                           bias_ddr=3 << 20, mul=[1] * 16, shift=[0] * 16)
        self.assertEqual([tile.zone for tile in tiles[:4]], [0, 1, 0, 1])
        self.assertEqual([len(tile.loads) for tile in tiles[:4]], [3, 3, 1, 1])
        self.assertEqual(sum(move.size for tile in tiles for move in tile.stores), 409600)
        self.assertEqual(tiles[0].loads[0].size, 6 * 320 * 4)
        self.assertEqual(tiles[1].loads[0].ddr_offset, 4 * 320 * 4)
        self.assertEqual(dict(tiles[1].stages[0].writes)[0x80], 1 | 3 << 16)

    def test_raw_reduction_and_gpu_folding_use_full_tensor_geometry(self):
        deep = Conv("deep", 10, 10, 256, 16, 3, 1, 1)
        self.assertEqual(reduction_widths(deep), (56, 56, 56, 56, 32))
        partials = [1 << 20 | idx * 16384 for idx in range(5)]
        raw = raw_conv_tiles(deep, input_ddr=0, bias_ddr=32768,
                             weights_ddr=[65536 + idx * 16384 for idx in range(5)],
                             partials_ddr=partials)
        self.assertEqual(len(raw), 1)
        self.assertEqual([stage.device for stage in raw[0].stages].count("NPU"), 5)
        self.assertEqual([stage.device for stage in raw[0].stages].count("VPU"), 4)
        self.assertEqual([move.size for move in raw[0].stores], [6400] * 5)
        folded = combine_tiles(deep, output_ddr=2 << 20, partials_ddr=partials, params_ddr=65536)
        self.assertTrue(all(tile.scratch_bytes <= 4096 for tile in folded))
        self.assertEqual(sum(tile.stores[0].size for tile in folded), 1600)


if __name__ == "__main__":
    unittest.main()
