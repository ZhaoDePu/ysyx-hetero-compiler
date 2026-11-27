import unittest

import numpy as np

from tools.c12_smoothquant import group_quantize, pack4, smooth_scale, unpack4


class SmoothQuantTest(unittest.TestCase):
    def test_signed_nibbles(self):
        signed = np.array([[-8, 7, 1, -1, 0, -2]], dtype=np.int8)
        packed = pack4(signed)
        np.testing.assert_array_equal(packed, [[0x78, 0xf1, 0xe0]])
        np.testing.assert_array_equal(unpack4(packed), signed)
        with self.assertRaises(ValueError):
            pack4(np.array([0, 8]))
        with self.assertRaises(ValueError):
            pack4(np.array([0]))

    def test_folded_scale_cancels_for_shared_norm(self):
        x = np.array([[0.125, 1, 16]], dtype=np.float32)
        w = np.array([[1, 0.125, 2], [2, 0.25, -1]], dtype=np.float32)
        scale = smooth_scale(np.abs(x).max(axis=0), np.abs(w).max(axis=0), 0.6)
        np.testing.assert_allclose((x / scale) @ (w * scale).T, x @ w.T, rtol=2e-7)
        self.assertTrue(np.all(np.isfinite(smooth_scale([0], [0], 0.75))))

    def test_groups_and_clipping(self):
        w = np.tile(np.linspace(-1, 1, 64, dtype=np.float32), (2, 1))
        for group in (64, 128):
            both = np.concatenate((w, w / 3), axis=1)
            for bits in (4, 8):
                q, scales = group_quantize(both, group, bits)
                self.assertEqual(q.shape, (2, 128 // group, group))
                self.assertEqual(scales.shape, (2, 128 // group))
                self.assertGreater(int(scales.min()), 0)
                bound = (1 << (bits - 1)) - 1
                self.assertLessEqual(int(np.abs(q).max()), bound)
                zero, zero_scales = group_quantize(np.zeros_like(both), group, bits)
                np.testing.assert_array_equal(zero, 0)
                np.testing.assert_array_equal(zero_scales, 1)


if __name__ == "__main__":
    unittest.main()
