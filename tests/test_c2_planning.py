import unittest

from hetero_compiler.ir import normalize
from hetero_compiler.planner import make_plan
from hetero_compiler.streaming import make_stream_plan
from hetero_compiler.profile import load_profile


TARGET = load_profile("fpga_zu15eg_balanced")


class C2Planning(unittest.TestCase):
    def test_gpu_only_and_serial_policy(self):
        graph = normalize({
            "tensors": {"x": {"shape": [1, 4], "dtype": "int8"},
                        "w": {"shape": [4, 4], "dtype": "int8", "data": [1] * 16}},
            "nodes": [{"id": "g", "op": "Gemm", "inputs": ["x", "w"], "output": "y"}],
            "outputs": ["y"],
        })
        auto = make_plan(graph, TARGET)
        pure = make_plan(graph, TARGET, device_policy="gpu-only", overlap=False, reuse=False)
        self.assertEqual(auto["stages"][0]["device"], "NPU")
        self.assertEqual(pure["stages"][0]["device"], "GPU")

    def test_two_gpu_components_get_distinct_physical_pages(self):
        graph = normalize({
            "tensors": {name: {"shape": [1536], "dtype": "int8"} for name in ("a", "b")},
            "nodes": [{"id": "aa", "op": "Relu", "inputs": ["a"], "output": "x"},
                      {"id": "bb", "op": "Relu", "inputs": ["b"], "output": "y"}],
            "outputs": ["x", "y"],
        })
        plan = make_plan(graph, TARGET, device_policy="gpu-only", overlap=False)
        pages = []
        for stage in plan["stages"]:
            stage_pages = {plan["allocations"][name]["offset"] // 4096 for name in
                           (*stage["inputs"], stage["output"])}
            self.assertEqual(len(stage_pages), 1)
            pages.extend(stage_pages)
        self.assertEqual(len(set(pages)), 2)
        self.assertTrue(any(page > 0 for page in pages))


    def test_two_half_ddr_prefetch_is_gated_by_ablation(self):
        graph = normalize({
            "tensors": {"a": {"shape": [4096], "dtype": "int8"}},
            "nodes": [{"id": "r", "op": "Relu", "inputs": ["a"], "output": "y"}],
            "outputs": ["y"],
        })

        def order(overlap):
            plan, _, execution = make_stream_plan(
                graph, TARGET, fuse=True, reuse=True, overlap=overlap, device_policy="gpu-only")
            self.assertEqual([s["zone"] for s in plan["stages"]], [0, 1])
            self.assertEqual([s["tile_elements"] for s in plan["stages"]], [2048, 2048])
            return [c["type"] if c["type"] != "dma" else c["direction"] for c in execution]

        self.assertEqual(order(True)[:6], ["load", "dma-fence", "launch", "load", "fence", "store"])
        self.assertEqual(order(False)[:6], ["load", "dma-fence", "launch", "fence", "store", "load"])


if __name__ == "__main__":
    unittest.main()
