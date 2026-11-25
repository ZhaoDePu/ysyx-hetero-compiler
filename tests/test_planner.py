import unittest
from unittest.mock import patch

from hetero_compiler.ir import GraphError, normalize
from hetero_compiler.planner import make_plan
from hetero_compiler.profile import load_profile
from hetero_compiler.reference import evaluate


def matmul_graph(fork=False):
    return normalize({
        "tensors": {"x": {"shape": [1, 4], "dtype": "int8", "kind": "input",
                          "data": [-8, 1, 4, 7]},
                    "w": {"shape": [4, 4], "dtype": "int8",
                          "data": [int(i == j) for i in range(4) for j in range(4)]}},
        "nodes": [{"id": "dense", "op": "Gemm", "inputs": ["x", "w"], "output": "a"},
                  {"id": "scale", "op": "Requant", "inputs": ["a"], "output": "b",
                   "attrs": {"shift": 1}},
                  {"id": "act", "op": "Relu", "inputs": ["b"], "output": "c"},
                  {"id": "sum", "op": "Add", "inputs": ["c", "x"], "output": "y"}],
        "outputs": ["y", "a"] if fork else ["y"],
    })


class Stub:
    def __init__(self, ops):
        self.ops = ops

    def supports(self, stage, graph, profile):
        return stage["op"] in self.ops


class PlanTests(unittest.TestCase):
    def make(self, graph, target="asic_nangate45_p8"):
        with patch("hetero_compiler.planner._backend", side_effect=[
                Stub({"Gemm"}), Stub({"Requant", "Relu", "Add", "Mul"})]):
            return make_plan(graph, load_profile(target))

    def test_fusion_and_cross_device_residency(self):
        g = matmul_graph()
        p = self.make(g)
        self.assertEqual([(s["id"], s["device"]) for s in p["stages"]],
                         [("dense", "NPU"), ("sum", "GPU")])
        self.assertEqual([x["op"] for x in p["stages"][0]["fused"]],
                         ["Gemm", "Requant", "Relu"])
        self.assertNotIn("a", p["allocations"])
        self.assertNotIn("b", p["allocations"])
        self.assertEqual([c["type"] for c in p["commands"]], ["launch", "fence", "launch"])
        self.assertEqual(evaluate(g)["y"], [-8, 1, 6, 10])
        self.assertNotEqual(p["allocations"]["x"]["offset"], p["allocations"]["y"]["offset"])

    def test_public_intermediate_blocks_fusion_and_lifetimes_are_inclusive(self):
        p = self.make(matmul_graph(True))
        self.assertEqual(len(p["stages"]), 4)
        self.assertEqual(p["allocations"]["a"]["last"], 4)
        self.assertNotEqual(p["allocations"]["b"]["offset"], p["allocations"]["c"]["offset"])
        self.assertLess(p["peak_live_bytes"], sum(x["stored_size"] for x in p["allocations"].values()))

    def test_allocation_respects_gpu_address_window(self):
        g = normalize({"tensors": {"x": {"shape": [4096], "dtype": "uint8"}},
                       "nodes": [{"id": "relu", "op": "Relu", "inputs": ["x"], "output": "y"}],
                       "outputs": ["y"]})
        with self.assertRaisesRegex(GraphError, "GPU.*window"):
            self.make(g, "fpga_zu15eg_balanced")


if __name__ == "__main__":
    unittest.main()
