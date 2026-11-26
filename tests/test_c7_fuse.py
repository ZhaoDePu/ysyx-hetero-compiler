import unittest

from hetero_compiler import c6_compile


class ConvRequantFusionTests(unittest.TestCase):
    def test_composes_signed_lookup_into_conv_without_mutation(self):
        first = [i - 128 for i in range(256)]
        second = [127 - i for i in range(256)]
        graph = {
            "outputs": ["y"],
            "nodes": [
                {"op": "Conv", "inputs": ["x"], "outputs": ["middle"],
                 "id": "conv", "attrs": {"lut": first}},
                {"op": "RequantLUT", "inputs": ["middle"], "outputs": ["y"],
                 "id": "requant", "attrs": {"lut": second}},
            ],
        }
        fuse = getattr(c6_compile, "fuse_conv_edge_luts", None)
        self.assertIsNotNone(fuse, "the C7 fusion must be available")
        result = fuse(graph)
        child, = result["nodes"]
        self.assertEqual(child["outputs"], ["y"])
        self.assertEqual(child["attrs"]["lut"], [second[value & 255] for value in first])
        self.assertEqual(graph["nodes"][0]["outputs"], ["middle"])

    def test_preserves_live_branch_and_public_output(self):
        graph = {
            "outputs": ["middle", "y"],
            "nodes": [
                {"op": "Conv", "inputs": ["x"], "outputs": ["middle"], "id": "conv", "attrs": {}},
                {"op": "RequantLUT", "inputs": ["middle"], "outputs": ["y"], "id": "requant",
                 "attrs": {"lut": [0] * 256}},
            ],
        }
        self.assertEqual(c6_compile.fuse_conv_edge_luts(graph), graph)
        graph["outputs"] = ["y", "z"]
        graph["nodes"].append({"op": "Identity", "inputs": ["middle"], "outputs": ["z"], "attrs": {}})
        self.assertEqual(c6_compile.fuse_conv_edge_luts(graph), graph)
