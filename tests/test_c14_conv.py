import unittest

import numpy as np

from hetero_compiler.c14_conv import CONTINUE_K, DEFER_K, can_chain_conv, chain_tiles, pack_chain_weights
from hetero_compiler.c6_memory import HALF_BYTES
from hetero_compiler.c6_tensor import Conv, pack_weights


class Phase17ConvTest(unittest.TestCase):
    def test_in_place_chain_preserves_k_order_and_writes_only_the_final_tile(self):
        spec = Conv("conv", 10, 10, 256, 16, 3, 1, 1)
        self.assertTrue(can_chain_conv(spec))
        self.assertFalse(can_chain_conv(Conv("wide", 10, 20, 256, 16, 3, 1, 1)))
        self.assertFalse(can_chain_conv(Conv("many_channels", 10, 10, 256, 32, 3, 1, 1)))
        rng = np.random.default_rng(7)
        weights = rng.integers(-127, 128, (3, 3, 256, 16), dtype=np.int8)
        groups = pack_chain_weights(spec, weights)
        tiles = chain_tiles(spec, input_ddr=0x100000, output_ddr=0x200000,
                            weights_ddr=[0x300000 + i * 0x10000 for i in range(len(groups))],
                            bias_ddr=0x400000, mul=[1] * 16, shift=[10] * 16,
                            lut=list(range(128)) + list(range(-128, 0)))
        tile, = tiles
        self.assertEqual(tile.zone, 0)
        self.assertEqual(len(tile.stores), 1)
        self.assertEqual(tile.stores[0].size, 1600)
        self.assertEqual(tile.loads[0].size, 25600)
        self.assertLessEqual(tile.scratch_bytes, HALF_BYTES)
        self.assertEqual(len(tile.stages), 50)
        all_regs = [dict(stage.writes) for stage in tile.stages]
        self.assertEqual(len({r[0x68] for r in all_regs}), 1)
        for row in range(10):
            regs = all_regs[row * 5:(row + 1) * 5]
            self.assertEqual({r[0x60] for r in regs}, {10 | 16 << 16})
            self.assertEqual({r[0x70] for r in regs}, {tile.stores[0].scratch_offset + 160 * row})
            self.assertEqual({r[0x80] for r in regs}, {row | 1 << 16})
            self.assertEqual([r[0x78] & (CONTINUE_K | DEFER_K) for r in regs],
                             [DEFER_K] + [DEFER_K | CONTINUE_K] * 3 + [CONTINUE_K])
            self.assertEqual([bool(r[0x78] & 2) for r in regs], [True] + [False] * 4)
            self.assertEqual([bool(r[0x78] & (1 << 17)) for r in regs], [False] * 4 + [True])
        group_regs = all_regs[:5]
        patch = rng.integers(-127, 128, spec.depth, dtype=np.int32)
        bias = rng.integers(-100000, 100000, spec.output_channels, dtype=np.int32)
        accum = bias.copy()
        k = 0
        for packed, regs in zip(groups, group_regs, strict=True):
            depth = regs[0x64] & 511
            self.assertEqual(regs[0x84], k)
            w = np.frombuffer(packed, dtype=np.int8).reshape(16, depth)
            accum += w @ patch[k:k + depth]
            k += depth
        self.assertEqual(k, spec.depth)
        expected = np.frombuffer(pack_weights(spec, weights), dtype=np.int8).reshape(16, -1) @ patch + bias
        np.testing.assert_array_equal(accum, expected)


if __name__ == "__main__":
    unittest.main()
