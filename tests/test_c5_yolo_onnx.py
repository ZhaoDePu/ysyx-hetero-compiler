"""Standards-only detection tail IR and reference tests."""

import unittest

from examples.generate_quantized_onnx import INT8, INT64, Model
from hetero_compiler.ir import GraphError, normalize
from hetero_compiler.onnx import parse_model
from hetero_compiler.reference import evaluate


def fixture():
    model = Model("detection_tail")
    scores = model.input("scores", [5], INT8, [-128, 4, -7, 4, 127])
    rows = model.input("boxes", [5, 2], INT8, list(range(10)))
    k = model.tensor("k", [1], INT64, [4])
    model.op("select", "TopK", [scores, k], ["selected_scores", "indices"], axis=0)
    model.op("fetch", "Gather", [rows, "indices"], "result", axis=0)
    model.output("selected_scores", [4], INT8)
    model.output("indices", [4], INT64)
    model.output("result", [4, 2], INT8)
    return model.encode()


class DetectionTailTests(unittest.TestCase):
    def test_topk_descending_uses_lower_index_for_ties(self):
        graph = normalize(parse_model(fixture()))
        result = evaluate(graph)
        self.assertEqual(result["indices"], [4, 1, 3, 2])
        self.assertEqual(result["selected_scores"], [127, 4, 4, -7])
        self.assertEqual(result["result"], [8, 9, 2, 3, 6, 7, 4, 5])

    def test_gather_negative_indices_and_defensive_zero(self):
        model = Model("negative")
        data = model.input("data", [4], INT8, [5, 6, 7, 8])
        indices = model.input("idx", [4], INT64, [-4, -1, -5, 1 << 32])
        model.op("gather", "Gather", [data, indices], "out", axis=0)
        model.output("out", [4], INT8)
        self.assertEqual(evaluate(normalize(parse_model(model.encode())))["out"], [5, 8, 0, 0])

    def test_topk_rejects_static_count_past_input(self):
        model = Model("invalid_k")
        scores = model.input("scores", [2], INT8, [1, 2])
        k = model.tensor("k", [1], INT64, [3])
        model.op("select", "TopK", [scores, k], ["selected_scores", "idx"])
        model.output("selected_scores", [3], INT8)
        model.output("idx", [3], INT64)
        with self.assertRaises(GraphError):
            parse_model(model.encode())
