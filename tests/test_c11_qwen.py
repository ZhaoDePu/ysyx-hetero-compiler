import unittest

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "C11 exporter uses NumPy")
class TestQwenPack(unittest.TestCase):
    def test_signed_pair_order_and_scale(self):
        from tools.c11_qwen_export import quantize_grouped

        pair = np.array([-7, 7, -1, 0], dtype=np.float32) / 32
        weights = np.tile(pair, (2, 64))
        packed, scales = quantize_grouped(weights)
        np.testing.assert_array_equal(packed[:, 0::2], 0x79)
        np.testing.assert_array_equal(packed[:, 1::2], 0x0F)
        np.testing.assert_array_equal(scales, 524288)

    def test_exact_zero_group_and_invalid_extent(self):
        from tools.c11_qwen_export import quantize_grouped

        packed, scales = quantize_grouped(np.zeros((1, 128), dtype=np.float32))
        np.testing.assert_array_equal(packed, 0)
        np.testing.assert_array_equal(scales, 1)
        with self.assertRaises(ValueError):
            quantize_grouped(np.zeros((1, 129), dtype=np.float32))
        with self.assertRaises(ValueError):
            quantize_grouped(np.full((1, 128), np.nan, dtype=np.float32))
