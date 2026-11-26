"""Focused regressions for C9 ResNet quantization and ONNX array export."""

import unittest

import numpy as np

from hetero_compiler.onnx import _initializer
from tools.c9_resnet import QDQ, fixed_multiplier, requant_lut


class ResNetExportTests(unittest.TestCase):
    def test_requant_lut_indexes_signed_codes_before_relu(self):
        table = requant_lut(0.5, 0.25, relu=True)
        self.assertEqual((table[8], table[127], table[128], table[255]), (16, 127, 0, 0))

    def test_weight_requant_renders_exact_dyadic_scale(self):
        ratio = np.asarray([1 / 16, 1 / 4, 1 / 2, 2], dtype=np.float64)
        mul, shift = fixed_multiplier(ratio)
        np.testing.assert_array_equal(np.ldexp(mul, -np.asarray(shift)), ratio)
        self.assertTrue(all(0 <= value <= 65535 for value in mul))
        self.assertTrue(all(0 <= value <= 31 for value in shift))

    def test_array_writer_handles_scalar_and_vector_initializers(self):
        model = QDQ("small")
        model.array("scale", np.asarray(0.25, dtype=np.float32))
        model.array("codes", np.asarray([-128, 127], dtype=np.int8))
        scale = _initializer(model.tensors["scale"])
        codes = _initializer(model.tensors["codes"])
        self.assertEqual((scale[0], scale[1]["shape"], scale[1]["data"]), ("scale", [], [0.25]))
        self.assertEqual((codes[0], codes[1]["shape"], codes[1]["data"]), ("codes", [2], [-128, 127]))
