"""Native ABI descriptor checks for the frozen Z15 shared VPU."""

import json
import tempfile
import unittest
from pathlib import Path

from hetero_compiler import npu
from hetero_compiler.profile import load_profile
from hetero_compiler.kv_schedule import compile_kv_bundle
from hetero_compiler.z15_vpu import mul, rmsnorm, softmax


class NativeVPTest(unittest.TestCase):
    def setUp(self):
        self.profile = load_profile("fpga_zu15eg_balanced")

    def command(self, op, inputs, output, attrs):
        names = list(inputs)
        tensors = {name: {"shape": shape, "dtype": dtype, "kind": "value"}
                   for name, (shape, dtype) in {**inputs, "y": output}.items()}
        stage = {"id": "stage", "op": op, "device": "VPU", "inputs": names, "output": "y", "attrs": attrs}
        graph = {"tensors": tensors}
        self.assertTrue(npu.supports(stage, graph, self.profile))
        order = npu.ordered_buffers(stage, graph, self.profile) or tensors
        offsets = {name: {"offset": index * 128, "size": 128} for index, name in enumerate(order)}
        with tempfile.TemporaryDirectory() as folder:
            commands = npu.lower(graph, {"allocations": offsets, "stages": [stage]},
                                 self.profile, Path(folder))["commands"]
        return dict(commands[0]["writes"])

    def test_softmax_descriptor_uses_shape_and_causal_position(self):
        regs = self.command("Z15Softmax", {"x": ([2, 4], "int32")}, ([2, 4], "uint8"),
                            {"exp_shift": 3, "causal": True, "position": 1, "fast": True})
        self.assertEqual(regs[0x10], 1 << 6)
        self.assertEqual(regs[0xB4], 2 | 4 << 16)
        self.assertEqual(regs[0xB8], 1)
        self.assertEqual(regs[0xBC], 12 | 3 << 8 | 3 << 24)

    def test_rmsnorm_commits_gain_epsilon_and_gamma(self):
        regs = self.command("Z15RMSNorm", {"x": ([2, 4], "int8"), "gamma": ([4], "int8")},
                            ([2, 4], "int8"), {"gain": 512, "epsilon": 3, "shift": 15})
        self.assertEqual(regs[0xA4], 128)
        self.assertEqual(regs[0xB4], 2)
        self.assertEqual(regs[0xB8], 4)
        self.assertEqual(regs[0xBC], 7 | 512 << 8 | 15 << 24)
        self.assertEqual(regs[0xC0], 3)

    def test_attention_accepts_runtime_packed_values_and_selects_fast_reciprocal(self):
        regs = self.command("Attention", {"q": ([2, 4], "int8"), "k": ([4, 4], "int8"),
                                         "v": ([4, 4], "int8")}, ([2, 4], "int8"),
                            {"exp_shift": 3, "causal": True, "position": 2,
                             "packed_v": True, "fast": True})
        self.assertEqual(regs[0x10], 1 << 5)
        self.assertEqual(regs[0x88], 7 | 2 << 8 | 3 << 20)
        self.assertEqual([regs[reg] for reg in (0x68, 0x6C, 0x8C, 0x90, 0x94, 0x70)],
                         [0, 128, 256, 384, 512, 640])
        self.assertEqual(regs[0x98], 2)

    def test_scaled_multiply_does_not_truncate_scale(self):
        regs = self.command("Z15Mul", {"a": ([2, 4], "int8"), "b": ([2, 4], "int8")},
                            ([2, 4], "int8"), {"scale": 513, "shift": 14})
        self.assertEqual(regs[0xBC], 6 | 513 << 8 | 14 << 24)
        self.assertEqual(regs[0xB0], 8)


class FrozenVectorArithmeticTest(unittest.TestCase):
    def test_softmax_division_strategies_agree_with_causal_floor(self):
        scores = [8, 0, 127, 127, 0, 0, 0, 1000]
        for fast in (False, True):
            with self.subTest(fast=fast):
                result = softmax(scores, 2, 4, exp_shift=3, causal=True, position=1, fast=fast)
                self.assertEqual(result, [170, 85, 0, 0, 85, 85, 85, 0])

    def test_rmsnorm_preserves_negative_floor_and_saturation(self):
        a = [0, 0, 2, -2, 2, 2, 2, 2]
        self.assertEqual(rmsnorm(a, [16] * 4, 2, 4, gain=512, epsilon=0, shift=15),
                         [0, 0, 22, -23, 16, 16, 16, 16])
        self.assertEqual(rmsnorm(a, [127] * 4, 2, 4, gain=512, epsilon=0, shift=15)[:4],
                         [0, 0, 127, -128])

    def test_multiply_rounds_half_up_for_both_signs(self):
        self.assertEqual(mul([-3, -1, 1, 3], [1, 1, 1, 1], scale=1, shift=1), [-1, 0, 1, 2])


class NativeKVScheduleTest(unittest.TestCase):
    def test_transposed_cache_uses_fixed_pitch_and_appends_only_new_rows(self):
        model = json.loads(Path("examples/c4a_kv/causal_kv32.json").read_bytes())
        model["decode_steps"] = 5
        with tempfile.TemporaryDirectory() as tmp:
            bundle = compile_kv_bundle(model, Path(tmp) / "native",
                                       vpu_lowering="on", vpu_attention="fast")
        plan = bundle["plan"]
        self.assertEqual(plan["native_attention_pitch"], 40)
        self.assertEqual(bundle["output_bindings"]["v_cache"]["shape"], [37, 4])
        native = [stage for stage in plan["stages"] if stage["op"] == "Attention"]
        self.assertEqual([stage["attrs"]["position"] for stage in native],
                         [0, 8, 16, 24, 32, 33, 34, 35, 36])
        self.assertTrue(all(stage["attrs"]["fast"] for stage in native))
        for step in range(1, 6):
            append = next(stage for stage in plan["stages"] if stage["id"] == f"d{step}-pack-new-values")
            self.assertEqual(append["device"], "VPU")
            self.assertEqual(append["attrs"]["stride"], 40)
            self.assertEqual(plan["allocations"][append["inputs"][0]]["size"], 4)
            self.assertEqual(plan["allocations"][append["output"]]["byte_offset"], 31 + step)
            self.assertEqual(plan["allocations"][append["output"]]["view_of"], "value-pack-buffer")


if __name__ == "__main__":
    unittest.main()
