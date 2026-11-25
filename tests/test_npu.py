import pathlib
import struct
import tempfile
import unittest

from hetero_compiler.npu import lower, ordered_buffers, requirements, supports


P8 = {"id": "asic_nangate45_p8", "npu": {"variant": "phase8"}, "sram": {"capacity": 4096}}
Z15 = {"id": "fpga_zu15eg_balanced", "npu": {"variant": "phase12c", "rows": 16, "cols": 16},
       "vpu": {"enabled": True}, "sram": {"capacity": 2097152}}


def fixture(bias=False):
    tensor = lambda shape, dtype, **kw: dict(shape=shape, dtype=dtype, **kw)
    tensors = {
        "a": tensor([2, 4], "int8"),
        "b": tensor([4, 4], "int8", data=list(range(16))),
        "c": tensor([2, 4], "int8"),
    }
    stage = dict(id="gemm", op="Gemm", inputs=["a", "b"], output="c", attrs={"shift": 2}, device="NPU")
    allocations = {"a": {"offset": 0, "size": 8}, "b": {"offset": 64, "size": 16},
                   "c": {"offset": 128, "size": 8}}
    if bias:
        tensors["bias"] = tensor([4], "int32", data=[-1000, 77, 250, -51])
        stage["inputs"].append("bias")
        allocations["bias"] = {"offset": 160, "size": 16}
    return dict(tensors=tensors), dict(stages=[stage], allocations=allocations, arena_end=3072)


class NpuTests(unittest.TestCase):
    def test_phase8_direct(self):
        graph, plan = fixture()
        self.assertTrue(supports(plan["stages"][0], graph, P8))
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp)
            result = lower(graph, plan, P8, path)
            cmd = result["commands"][0]
            self.assertEqual((cmd["a_base"], cmd["descriptor"]), (0, 3072))
            self.assertEqual(struct.unpack("<6I", (path / "npu.bin").read_bytes()),
                             (0x4E320100, 64, 128, 0xFFFFFFFF, 0x040402, 2))

    def test_requant_controls_phase8_narrowing_after_int32_gemm(self):
        graph, plan = fixture()
        graph["tensors"]["acc"] = {"shape": [2, 4], "dtype": "int32"}
        stage = plan["stages"][0]
        stage["fused"] = [
            dict(op="Gemm", inputs=["a", "b"], output="acc", attrs={"narrowing": "wrap"}),
            dict(op="Requant", inputs=["acc"], output="c", attrs={"shift": 3, "narrowing": "saturate"}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            result = lower(graph, plan, P8, pathlib.Path(tmp))
        self.assertEqual(result["records"][0]["words"][5], 3)

    def test_phase8_signed_wide(self):
        graph, plan = fixture(bias=True)
        self.assertTrue(supports(plan["stages"][0], graph, P8))
        with tempfile.TemporaryDirectory() as tmp:
            result = lower(graph, plan, P8, pathlib.Path(tmp))
            records = result["records"]
            self.assertEqual(records[0]["words"][:2], [0x4E340106, 1])
            self.assertEqual(records[1]["words"], [0x4E37010B, 64, 128, 0x02040402, 0x40004, 4, 0x40000, 160, 7, 64, 128, 0x04020000])
            self.assertEqual(records[2]["words"][:2], [11, 0])

    def test_z15_static_pack_and_wide_bias(self):
        graph, plan = fixture(bias=True)
        stage = plan["stages"][0]
        self.assertTrue(supports(stage, graph, Z15))
        self.assertEqual(requirements(stage, graph, Z15)["b.__npu_colmajor"]["size"], 16)
        plan["allocations"]["b.__npu_colmajor"] = {"offset": 192, "size": 16}
        with tempfile.TemporaryDirectory() as tmp:
            result = lower(graph, plan, Z15, pathlib.Path(tmp))
        self.assertEqual(result["constants"][0], {"tensor": "b.__npu_colmajor", "offset": 192,
                                                 "data": [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]})
        regs = dict(result["commands"][0]["writes"])
        self.assertEqual(regs[0x6C], 192)
        self.assertEqual(regs[0x74], 160)
        self.assertEqual(regs[0x78], (1 << 17) | (2 << 4) | (1 << 2) | (1 << 1))

    def test_z15_causal_attention_with_dynamic_qk_and_packed_v(self):
        graph = {"tensors": {name: {"shape": [4, 4], "dtype": "int8"} for name in ("q", "k", "v", "out")}}
        graph["tensors"]["v"]["data"] = list(range(16))
        stage = dict(id="attn", op="Attention", inputs=["q", "k", "v"], output="out", device="NPU",
                     attrs={"causal": True, "exp_shift": 4, "position": 0})
        self.assertTrue(supports(stage, graph, Z15))
        order = ordered_buffers(stage, graph, Z15)
        self.assertEqual(order, ["q", "k", "v.__npu_colmajor", "attn.__npu_scores", "attn.__npu_probabilities", "out"])
        alloc = dict(zip(order, ({"offset": i * 64, "size": 64} for i in range(6))))
        req = requirements(stage, graph, Z15)
        graph["tensors"]["v"]["kind"] = "input"
        self.assertFalse(supports(stage, graph, Z15))
        graph["tensors"]["v"]["kind"] = "constant"
        self.assertEqual(req["v.__npu_colmajor"]["replaces"], "v")
        self.assertEqual(req["attn.__npu_scores"]["size"], 64)
        self.assertEqual(req["attn.__npu_probabilities"]["size"], 16)
        with tempfile.TemporaryDirectory() as tmp:
            result = lower(graph, {"stages": [stage], "allocations": alloc}, Z15, pathlib.Path(tmp))
        regs = dict(result["commands"][0]["writes"])
        self.assertEqual(regs[0x88], (4 << 20) | (4 << 8) | 6)
        self.assertEqual(regs[0x10], 32)

    def test_z15_requires_static_weight_and_partition(self):
        graph, plan = fixture()
        stage = plan["stages"][0]
        del graph["tensors"]["b"]["data"]
        self.assertFalse(supports(stage, graph, Z15))
        graph["tensors"]["b"]["data"] = list(range(16))
        graph["tensors"]["b"]["kind"] = "input"
        self.assertFalse(supports(stage, graph, Z15))
        graph["tensors"]["b"]["kind"] = "constant"
        plan["allocations"]["b.__npu_colmajor"] = {"offset": 1048576, "size": 16}
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "ownership half"):
            lower(graph, plan, Z15, pathlib.Path(tmp))


if __name__ == "__main__":
    unittest.main()
