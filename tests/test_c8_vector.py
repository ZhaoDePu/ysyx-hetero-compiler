import unittest

import numpy as np

from hetero_compiler.c8_vector import fuse_vector_partitions


def tensor(channels, side=2):
    return {"shape": [1, channels, side, side], "physical_shape": [1, side, side, channels],
            "layout": "NHWC", "dtype": "int8", "scale": 0.25}


def node(name, op, inputs, outputs, **attrs):
    return {"id": name, "op": op, "inputs": inputs, "outputs": outputs,
            "device": "VPU", "attrs": attrs}


class C8VectorFusion(unittest.TestCase):
    def setUp(self):
        table = [max(-128, min(127, int(value) + 8))
                 for value in np.arange(256, dtype=np.uint8).view(np.int8)]
        self.graph = {
            "inputs": ["data", "other"], "outputs": ["out"],
            "tensors": {"data": tensor(8), "other": tensor(4), "out": tensor(12),
                        **{name: tensor(4) for name in ("x", "y", "qx", "qy")}},
            "nodes": [
                node("split", "Split", ["data"], ["x", "y"], axis=1),
                node("qa", "RequantLUT", ["x"], ["qx"], lut=table),
                node("qb", "RequantLUT", ["y"], ["qy"], lut=table),
                node("join", "Concat", ["qx", "qy", "other"], ["out"], axis=1),
            ],
        }

    def test_commutes_common_lut_through_split_and_keeps_channel_order(self):
        fused = fuse_vector_partitions(self.graph)
        split, lut, join = fused["nodes"]
        self.assertEqual(split, self.graph["nodes"][0])
        self.assertEqual(lut["inputs"], ["data"])
        self.assertEqual(join["inputs"], [lut["outputs"][0], "other"])
        left, right, other = np.arange(48, dtype=np.int8).reshape(3, 2, 2, 4)
        table = np.asarray(lut["attrs"]["lut"], dtype=np.int8)
        old = np.concatenate((table[left.view(np.uint8)], table[right.view(np.uint8)], other), -1)
        new = np.concatenate((table[np.concatenate((left, right), -1).view(np.uint8)], other), -1)
        np.testing.assert_array_equal(old, new)

    def test_keeps_noncontiguous_or_public_split_branches(self):
        join = self.graph["nodes"][-1]
        join["inputs"] = ["qx", "other", "qy"]
        self.assertEqual(fuse_vector_partitions(self.graph), self.graph)
        join["inputs"] = ["qx", "qy", "other"]
        self.graph["outputs"].append("qx")
        self.assertEqual(fuse_vector_partitions(self.graph), self.graph)

    def test_moves_private_byte_lut_before_nearest_upsample(self):
        table = list(range(256))
        graph = {"inputs": ["in"], "outputs": ["out"],
                 "tensors": {"in": tensor(4), "ups": tensor(4, 4), "out": tensor(4, 4)},
                 "nodes": [node("resize", "Resize", ["in"], ["ups"],
                                mode="nearest", coordinate_transformation_mode="asymmetric"),
                           node("requant", "RequantLUT", ["ups"], ["out"], lut=table)]}
        lut, ups = fuse_vector_partitions(graph)["nodes"]
        self.assertEqual(lut["inputs"], ["in"])
        self.assertEqual(ups["inputs"], lut["outputs"])
        self.assertEqual(ups["outputs"], ["out"])

    def test_uses_unsigned_attention_probabilities_without_inverse_transposes(self):
        probability = {"shape": [1, 1, 12, 8], "physical_shape": [1, 1, 12, 8],
                       "layout": "row-major", "dtype": "uint8", "scale": 1.0 / 255}
        graph = {"inputs": ["p", "v"], "outputs": ["out"],
                 "tensors": {"p": probability}, "nodes": [
                     node("transpose", "Transpose", ["p"], ["pt"], perm=[0, 1, 3, 2]),
                     node("pv", "MatMul", ["v", "pt"], ["out"], mul=1, shift=12),
                 ]}
        pv, = fuse_vector_partitions(graph)["nodes"]
        self.assertEqual(pv["inputs"], ["v", "p"])
        self.assertTrue(pv["attrs"]["rhs_pretransposed"])

    def test_native_pv_drops_the_rhs_pack_when_already_transposed(self):
        from hetero_compiler.c6_attention import lower_matmul
        tiles = lower_matmul("pv", 0, 128, 256, (1, 4, 8), (1, 12, 8), 1, 12,
                             unsigned_rhs=True, rhs_pretransposed=True)
        self.assertEqual([stage.id for stage in tiles[0].stages], ["pv.0.dot", "pv.0.unpack"])
        registers = dict(tiles[0].stages[0].writes)
        self.assertEqual(registers[0x68], tiles[0].loads[1].scratch_offset)
        self.assertEqual(registers[0x6C], tiles[0].loads[0].scratch_offset)
