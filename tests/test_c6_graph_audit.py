"""Check logical lifetime overlap, index sizes, and grouped Conv classification."""

import unittest

from examples.generate_quantized_onnx import FLOAT, INT64, Model, field, value_info
from hetero_compiler import onnx as wire
from hetero_compiler.ir import GraphError
from tools.audit_c6_yolo import audit_model


def with_value_info(model, declarations):
    model = wire._message(model.encode())
    graph = bytes(wire._one(model, 7, 2))
    graph += b"".join(field(13, value_info(name, dims, kind))
                      for name, dims, kind in declarations)
    return field(7, graph) + b"".join(field(8, payload) for payload in wire._all(model, 8, 2))


class GraphAudit(unittest.TestCase):
    def test_branch_lifetimes_include_the_output_at_the_consuming_node(self):
        shape = [1, 64, 128, 128]
        model = Model("depthwise_branch")
        model.inputs.append(value_info("x", shape, FLOAT))
        weight = model.tensor("w", [64, 1, 3, 3], FLOAT, [0] * 576)
        model.op("/model.2/conv", "Conv", ["x", weight], "a", group=64, pads=[1] * 4)
        model.op("/model.2/skip", "Identity", ["x"], "b")
        model.op("/model.2/merge", "Add", ["a", "b"], "y")
        model.output("y", shape, FLOAT)
        audit = audit_model(with_value_info(model, [("a", shape, FLOAT), ("b", shape, FLOAT)]))
        self.assertEqual(audit["peak_onnx_bytes"], 12 * 1024**2)
        self.assertEqual(audit["peak_int8_bytes"], 3 * 1024**2)
        self.assertEqual(audit["initializer_onnx_bytes"], 2304)
        self.assertEqual(audit["tensors"]["x"]["last_consumer"], 1)
        self.assertEqual(audit["tensors"]["y"]["last_consumer"], 3)
        self.assertEqual(audit["nodes"][0]["conv"]["family"], "depthwise")
        self.assertEqual(audit["nodes"][0]["macs"], 9_437_184)
        self.assertEqual(audit["tensors"]["a"]["int8_limit"], "within-half")

    def test_native_integer_indices_are_not_projected_to_one_byte(self):
        model = Model("argmax")
        model.inputs.append(value_info("x", [4], FLOAT))
        model.op("choose", "ArgMax", ["x"], "idx", keepdims=0)
        model.output("idx", [], INT64)
        audit = audit_model(model.encode())
        self.assertEqual(audit["tensors"]["idx"]["int8_bytes"], 8)
        self.assertEqual(audit["peak_int8_bytes"], 12)

    def test_missing_intermediate_has_no_invented_size(self):
        model = Model("incomplete")
        model.inputs.append(value_info("x", [4], FLOAT))
        model.op("first", "Identity", ["x"], "hidden")
        model.op("second", "Identity", ["hidden"], "y")
        model.output("y", [4], FLOAT)
        with self.assertRaisesRegex(GraphError, "hidden.*shape"):
            audit_model(model.encode())


if __name__ == "__main__":
    unittest.main()
