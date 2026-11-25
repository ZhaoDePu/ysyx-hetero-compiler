import tempfile
import unittest
from pathlib import Path

from hetero_compiler import gpu, npu

PROFILE = {
    "gpu": {"variant": "g3", "lanes": 16, "warps": 4, "window_bytes": 4096,
            "code_bytes": 16384, "issue_width": 2},
    "npu": {"variant": "z15", "rows": 16, "cols": 16},
    "sram": {"capacity": 2097152, "alignment": 32},
}


def t(shape, dtype="int8"):
    return {"shape": shape, "dtype": dtype}


def op(op, operands, out, **attrs):
    graph = {"tensors": {f"x{i}": spec for i, spec in enumerate(operands)}}
    graph["tensors"]["out"] = out
    stage = {"id": "test", "op": op, "device": "GPU", "inputs": list(graph["tensors"])[:-1],
             "output": "out", "attrs": attrs}
    return graph, stage


class C2GPUTests(unittest.TestCase):
    def test_layout_and_reductions(self):
        cases = [
            op("Transpose", [t([1, 4, 2, 3])], t([1, 2, 3, 4]), perm=[0, 2, 3, 1]),
            op("Reshape", [t([2, 3])], t([6]), shape=[6]),
            op("Slice", [t([2, 6])], t([2, 2]), starts=[0, 1], ends=[2, 5], steps=[1, 2]),
            op("Resize", [t([1, 2, 3, 4])], t([1, 4, 6, 4]), shape=[1, 4, 6, 4]),
            op("Concat", [t([2, 2]), t([2, 1])], t([2, 3]), axis=1),
            op("MaxPool", [t([1, 3, 4, 2])], t([1, 2, 3, 2]), kernel_shape=[2, 2],
               strides=[1, 1], pads=[0, 0, 0, 0]),
            op("RMSNorm", [t([2, 4])], t([2, 4])),
            op("RowSoftmax", [t([2, 2])], t([2, 2], "uint8"), causal_mask=True),
            op("ArgMax", [t([1, 2, 3, 4])], t([1, 2, 3, 1], "int64"), axis=3,
               keepdims=True, select_last_index=False),
        ]
        for graph, stage in cases:
            with self.subTest(op=stage["op"]):
                self.assertTrue(gpu.supports(stage, graph, PROFILE))

    def test_conv_with_centering_and_nonzero_page(self):
        graph, stage = op("Conv", [t([1, 2, 2, 4]), t([1, 1, 4, 4])],
                          t([1, 2, 2, 4], "int32"), a_zero_point=-2,
                          b_zero_point=3, strides=[1, 1], pads=[0, 0, 0, 0])
        graph["tensors"]["x1"]["kind"] = "constant"
        graph["tensors"]["x1"]["data"] = [0] * 16
        plan = {"stages": [stage], "allocations": {
            name: {"offset": 8192 + 128*i, "size": 64}
            for i, name in enumerate(graph["tensors"])}}
        self.assertTrue(gpu.supports(stage, graph, PROFILE))
        self.assertFalse(npu.supports({**stage, "device": "NPU"}, graph, PROFILE))
        with tempfile.TemporaryDirectory() as output:
            manifest = gpu.lower(graph, plan, PROFILE, Path(output))
        self.assertEqual(manifest["stages"][0]["jobs"][0]["page"], 2)

    def test_npu_cannot_fuse_dyadic_rounding(self):
        graph, primary = op("Gemm", [t([2, 4]), t([4, 4])], t([2, 4], "int32"))
        graph["tensors"]["x1"]["data"] = [1] * 16
        graph["tensors"]["narrow"] = t([2, 4])
        stage = {**primary, "device": "NPU", "output": "narrow", "fused": [
            primary,
            {"id": "round", "op": "Requant", "inputs": ["out"], "output": "narrow",
             "attrs": {"rounding": "nearest_even", "shift": 1}},
        ]}
        self.assertFalse(npu.supports(stage, graph, PROFILE))
        stage["fused"][1]["attrs"]["shift"] = 0
        self.assertTrue(npu.supports(stage, graph, PROFILE))


    def test_nearest_even_uses_the_same_signed_int32_arithmetic_as_rv32(self):
        from hetero_compiler.ir import normalize
        from hetero_compiler.reference import evaluate
        graph = normalize({
            "tensors": {"x": {"shape": [3], "dtype": "int32", "data": [1 << 30, 5, -5]}},
            "nodes": [
                {"id": "wrap", "op": "Requant", "inputs": ["x"], "output": "y",
                 "attrs": {"rounding": "nearest_even", "shift": 1, "multiplier": 2}},
                {"id": "divisor", "op": "Requant", "inputs": ["x"], "output": "z",
                 "attrs": {"rounding": "nearest_even", "shift": 3, "divisor": 1, "dtype": "int32"}},
            ],
            "outputs": ["y", "z"],
        })
        result = evaluate(graph)
        self.assertEqual(result["y"], [-128, 5, -5])
        self.assertEqual(result["z"], [1 << 30, 5, -5])


if __name__ == "__main__":
    unittest.main()
