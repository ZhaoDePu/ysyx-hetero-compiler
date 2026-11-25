"""Bit-exact tests for arithmetic shared by the C6 PTQ and the Z15 backend."""
import unittest

import numpy as np

from hetero_compiler.c6_quant import _matmul_int, _ordinary, conv_acc, ew_mul, lut_apply, requant
from tools.c6_ptq import compare


class C6QuantTests(unittest.TestCase):
    def test_native_requant_floor_and_saturation(self):
        acc = np.array([[-1, 1000], [-129, 45]], dtype=np.int32)
        np.testing.assert_array_equal(requant(acc, [3, 7], [1, 1]), [[-2, 127], [-128, 127]])

    def test_lut_uses_twos_complement_byte_address(self):
        table = np.arange(-128, 128, dtype=np.int8)
        np.testing.assert_array_equal(lut_apply(np.array([-128, -1, 0, 127], dtype=np.int8), table),
                                      [0, 127, -128, -1])

    def test_elementwise_mul_ties_toward_positive_infinity(self):
        lhs = np.array([-3, -1, 1, 3], dtype=np.int8)
        np.testing.assert_array_equal(ew_mul(lhs, np.ones_like(lhs), 1, 1), [-1, 0, 1, 2])

    def test_unsigned_attention_operand_can_move_to_a_port(self):
        v = np.array([[-5, 8, 2], [7, -2, 4]], dtype=np.int8)
        p = np.array([[0, 220], [255, 8], [9, 129]], dtype=np.uint8)
        direct = v.astype(np.int32) @ p.astype(np.int32)
        np.testing.assert_array_equal(_matmul_int(v, p), direct)
        np.testing.assert_array_equal(_matmul_int(p.T, v.T).T, direct)

    def test_expand_uses_certified_onnx_output_shape(self):
        x = np.arange(3).reshape(1, 3, 1)
        y = _ordinary("Expand", [x, np.array([1, 1, 2])], {}, [1, 3, 2])
        np.testing.assert_array_equal(y, [[[0, 0], [1, 1], [2, 2]]])

    def test_confidence_gate_counts_filtered_and_wrong_class_as_failures(self):
        ref = [{"class_id": 3, "score": .40, "xyxy": [0, 0, 1, 1]},
               {"class_id": 2, "score": .8, "xyxy": [2, 2, 3, 3]}]
        candidate = [{"class_id": 3, "score": .399, "xyxy": [0, 0, 1, 1]},
                     {"class_id": 4, "score": .9, "xyxy": [2, 2, 3, 3]}]
        expected = compare(ref, candidate, .4)
        self.assertEqual(expected["class_agreement"], 0)
        self.assertEqual(expected["unmatched_fp32"], 1)
        self.assertEqual(expected["mean_iou"], 1)

    def test_grouped_conv_asymmetric_stride_and_padding(self):
        x = np.arange(24, dtype=np.int8).reshape(1, 2, 3, 4)
        w = np.arange(-4, 4, dtype=np.int8).reshape(2, 2, 1, 2)
        bias = np.array([3, -2], dtype=np.int32)
        attrs = {"strides":[2, 1], "pads":[1, 0, 0, 1], "groups":2, "dilations":[1, 1]}
        actual = conv_acc(x, w, bias, attrs)
        gold = np.zeros((1, 2, 2, 4), np.int32)
        src = np.pad(x, ((0,0),(0,0),(1,0),(0,1)))
        for c in range(2):
            for i in range(2):
                for j in range(4):
                    gold[0,c,i,j] = (src[0,c,2*i:2*i+2,j:j+2] * w[:,:,0,c]).sum() + bias[c]
        np.testing.assert_equal(actual, gold)


if __name__ == "__main__":
    unittest.main()
