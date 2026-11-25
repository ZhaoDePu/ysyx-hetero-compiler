"""Static layout contracts for the frozen Z15 VPU/GPU."""
import unittest

from hetero_compiler.c6_layout import plan_layout
from hetero_compiler.c6_memory import HALF_BYTES


class LayoutPlan(unittest.TestCase):
    def test_batched_attention_swap_uses_native_vpu(self):
        plan = plan_layout("attention", "Transpose", (1, 2, 32, 100), (1, 2, 100, 32),
                           perm=(0, 1, 3, 2))
        self.assertEqual(plan.constant_data, b"")
        tile, = plan.tiles(0, 0x10000)
        self.assertEqual([s.device for s in tile.stages], ["VPU", "VPU"])
        first, second = [dict(s.writes) for s in tile.stages]
        self.assertEqual(first[0xB4], 32)
        self.assertEqual(first[0xB8], (32 << 16) | 100)
        self.assertEqual(second[0xA0] - first[0xA0], 3200)

    def test_nhwc_reshape_keeps_onnx_element_order(self):
        plan = plan_layout("qkv", "Reshape", (1, 256, 10, 10), (1, 2, 128, 100),
                           input_layout="NHWC")
        tile, = plan.tiles(0, 0x10000)
        stage, = tile.stages
        self.assertEqual(dict(stage.writes)[0xB4], 100)
        self.assertEqual(dict(stage.writes)[0xB8], (100 << 16) | 256)

    def test_nonmatrix_permutation_uses_aligned_gpu_gather(self):
        plan = plan_layout("strided", "Transpose", (2, 5, 7, 9), (7, 2, 9, 5),
                           perm=(2, 0, 3, 1))
        tiles = plan.tiles(0, 0x10000, 0x20000)
        self.assertTrue(plan.constant_data)
        self.assertGreater(len(tiles), 1)
        self.assertEqual([t.zone for t in tiles], [i % 2 for i in range(len(tiles))])
        self.assertTrue(all(t.scratch_bytes <= 4096 and t.stages[0].device == "GPU" for t in tiles))
        for tile in tiles:
            for move in (*tile.loads, *tile.stores):
                self.assertEqual(move.ddr_offset % 32, 0)
                self.assertEqual(move.size % 32, 0)
                self.assertEqual((move.scratch_offset + move.size - 1) // HALF_BYTES, tile.zone)

    def test_inconsistent_shape_or_unsupported_layout_rejected(self):
        with self.assertRaisesRegex(ValueError, "output shape"):
            plan_layout("bad", "Transpose", (2, 3, 4), (4, 2, 3), perm=(0, 2, 1))
        with self.assertRaisesRegex(ValueError, "layout"):
            plan_layout("bad", "Reshape", (2, 3, 4), (6, 4), input_layout="NHWC")


if __name__ == "__main__":
    unittest.main()
