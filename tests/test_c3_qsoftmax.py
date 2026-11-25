import unittest

from examples.generate_quantized_onnx import FLOAT, INT8, UINT8, Model
from hetero_compiler import gpu
from hetero_compiler.ir import GraphError, normalize
from hetero_compiler.onnx import parse_model
from hetero_compiler.profile import load_profile
from hetero_compiler.reference import evaluate
from tools.run_c3_qsoftmax import build_softmax


class C3QuantizedSoftmax(unittest.TestCase):
    def test_standard_qdq_uses_integer_rv32_backend(self):
        graph = normalize(parse_model(build_softmax([2, 2], [0, 0, 16, -16])))
        node, = graph["nodes"]
        self.assertEqual(node["op"], "QSoftmax")
        self.assertEqual(node["attrs"]["algorithm"], "q23")
        self.assertTrue(gpu.supports(node, graph, load_profile("fpga_zu15eg_balanced")))
        self.assertEqual(evaluate(graph)["probability"], [128, 128, 225, 31])

    def test_broadcast_causal_add_masks_before_max_and_sum(self):
        floor = float("-inf")
        graph = normalize(parse_model(build_softmax(
            [2, 2, 2], [0, 127, 16, -16, 2, 100, 0, 0],
            mask=([2, 2], [0, floor, 0, 0]))))
        node, = graph["nodes"]
        self.assertEqual(node["attrs"]["causal_offset"], 0)
        self.assertEqual(evaluate(graph)["probability"], [255, 0, 225, 31, 255, 0, 128, 128])

    def test_finite_mask_requires_provable_float_underflow(self):
        graph = normalize(parse_model(build_softmax(
            [2, 2], [0, 127, 16, -16], mask=([2, 2], [0, -1.0e9, 0, 0]))))
        self.assertEqual(evaluate(graph)["probability"], [255, 0, 225, 31])
        with self.assertRaisesRegex(GraphError, "underflow"):
            parse_model(build_softmax([2, 2], [0]*4, mask=([2, 2], [0, -4.0, 0, 0])))

    def test_non_causal_constant_mask_rejected(self):
        with self.assertRaisesRegex(GraphError, "causal"):
            parse_model(build_softmax([2, 2], [0] * 4, mask=([2, 2], [float("-inf"), 0, 0, 0])))

    def test_nonlast_axis_and_ties_to_even(self):
        graph = normalize(parse_model(build_softmax(
            [2, 3], [0] * 6, axis=0, scale=1)))
        self.assertEqual(evaluate(graph)["probability"], [0] * 6)

    def test_standard_dq_sum_and_quantize(self):
        model = Model("quantized_add")
        model.input("lhs", [4], INT8, [4, 2, 1, -1])
        model.input("rhs", [4], INT8, [0, 4, 1, -1])
        for name, scale in (("sl", 1/4), ("sr", 1/8), ("sy", 1/2)):
            model.tensor(name, [], FLOAT, [scale])
        model.tensor("z", [], INT8, [0])
        model.op("dq1", "DequantizeLinear", ["lhs", "sl", "z"], "l")
        model.op("dq2", "DequantizeLinear", ["rhs", "sr", "z"], "r")
        model.op("sum", "Add", ["l", "r"], "s")
        model.op("q", "QuantizeLinear", ["s", "sy", "z"], "y")
        model.output("y", [4], INT8)
        graph = normalize(parse_model(model.encode()))
        self.assertEqual(evaluate(graph)["y"], [2, 2, 1, -1])
