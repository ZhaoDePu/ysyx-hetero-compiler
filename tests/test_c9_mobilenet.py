import unittest

import numpy as np

from tools.c9_mobilenet import fixed_conv, calibrate_multiplier, resize_codes


class MobileNetV2PreparationTests(unittest.TestCase):
    def test_fixed_depthwise_keeps_corresponding_channels(self):
        image = np.array([[[[1, 2], [3, 4]], [[10, 20], [30, 40]]]], dtype=np.int8)
        weights = np.array([[[[2]]], [[[-1]]]], dtype=np.int8)
        result = fixed_conv(image, weights, np.array([1, -2], dtype=np.int32), 1, 0, 2)
        np.testing.assert_array_equal(result, np.array([[[[3, 5], [7, 9]], [[-12, -22], [-32, -42]]]]))

    def test_stem_rgb_is_zero_padded_and_interleaved(self):
        planes = np.arange(12, dtype=np.int8).reshape(1, 3, 2, 2)
        packed = resize_codes(planes)
        self.assertEqual(packed.shape, (1, 2, 2, 4))
        self.assertEqual(packed[0, 1, 0].tolist(), [2, 6, 10, 0])

    def test_hardware_multiplier_preserves_q8_dyadic_ratio(self):
        factors, shifts = calibrate_multiplier(np.array([0.25, 0.125], dtype=np.float64))
        np.testing.assert_array_equal(np.ldexp(factors.astype(np.float64), -shifts), [0.25, 0.125])


class MobileNetV2A0LoweringTests(unittest.TestCase):
    @staticmethod
    def _context(tensors):
        from hetero_compiler.c6_compile import _Preparation

        return _Preparation(1, tensors, {}, {}, {})

    @staticmethod
    def _tensor(shape):
        return {"dtype": "int8", "layout": "NHWC", "physical_shape": shape}

    def test_logit_concat_uses_only_gpu_with_padded_final_dma(self):
        from hetero_compiler.c6_memory import Allocation
        from hetero_compiler.c9_gpu_misc import lower_gpu_partition

        names = ["left", "right"]
        ctx = self._context({"left": self._tensor([1, 1, 1, 768]),
                             "right": self._tensor([1, 1, 1, 232]),
                             "logits": self._tensor([1, 1, 1, 1000])})
        schedule = lower_gpu_partition({"id": "concat", "op": "Concat", "inputs": names,
                                        "outputs": ["logits"], "attrs": {"axis": 1}}, ctx)
        layers = schedule({"left": Allocation(0, 768), "right": Allocation(768, 232),
                           "logits": Allocation(1024, 1000)})
        tile, = layers[0]
        self.assertEqual([stage.device for stage in tile.stages], ["GPU", "GPU"])
        self.assertEqual([dma.size for dma in tile.loads], [768, 256])
        self.assertEqual(tile.stores[0].size, 1024)

    def test_requant_rejects_nonspatial_or_mismatched_shape(self):
        from hetero_compiler.c9_gpu_misc import lower_gpu_vector

        ctx = self._context({"input": self._tensor([1]), "output": self._tensor([1, 2, 2, 2])})
        with self.assertRaisesRegex(ValueError, "batch-one INT8 NHWC"):
            lower_gpu_vector({"id": "add", "op": "RequantLUT", "inputs": ["input"],
                              "outputs": ["output"], "attrs": {}}, ctx)
