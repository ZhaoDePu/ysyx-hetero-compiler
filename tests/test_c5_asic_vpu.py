"""Phase A3 VPU bindings for new target-5 descriptors."""

import struct
from pathlib import Path
import tempfile
import unittest

from hetero_compiler import asic_ip, reference
from hetero_compiler.compiler import compile_bundle, tensor_bytes
from hetero_compiler.profile import load_profile


def fixture():
    shape = [2, 8]
    x = [-128, 127, 0, -1, 1, 7, -7, 8, 0, 0, 0, 0, 15, -16, 16, -17]
    bias_gain = [16, 15, 12, 1, 7, 16, -16, 4]
    scores = x[::-1]
    return {
        "name": "a3_vpu_modes",
        "tensors": {
            "x": {"shape": shape, "dtype": "int8", "kind": "input", "data": x},
            "scores": {"shape": shape, "dtype": "int8", "kind": "input", "data": scores},
            "g": {"shape": [8], "dtype": "int8", "kind": "constant", "data": bias_gain},
        },
        "nodes": [
            {"id": "norm", "op": "QRMSNorm", "inputs": ["x", "g"], "output": "n",
             "attrs": {"gain": 91, "epsilon": 35, "shift": 12}},
            {"id": "act", "op": "SiLU", "inputs": ["n"], "output": "s"},
            {"id": "gate", "op": "Mul", "inputs": ["s", "x"], "output": "gated",
             "attrs": {"dtype": "int32"}},
            {"id": "scale", "op": "Requant", "inputs": ["gated"], "output": "y",
             "attrs": {"dtype": "int8", "shift": 4}},
            {"id": "requant", "op": "Requant", "inputs": ["n"], "output": "r",
             "attrs": {"dtype": "uint8", "shift": 2, "multiplier": 9, "zero_point": 6,
                       "input_zero_point": -2, "rounding": "nearest_even"}},
            {"id": "softmax", "op": "QSoftmax", "inputs": ["scores"], "output": "p",
             "attrs": {"dtype": "uint8", "axis": 1, "causal_offset": 6,
                       "in_scale": [1, 16], "out_scale": [1, 256], "algorithm": "q23"}},
        ],
        "outputs": ["n", "s", "y", "r", "p"],
    }


class A3VectorTest(unittest.TestCase):
    def test_native_ports_and_software_match_each_output(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle = Path(folder) / "bundle"
            m = compile_bundle(fixture(), load_profile("asic_nangate45_ip"), bundle, vpu_attention="fast")
            payload = (bundle / "program.yhb").read_bytes()
            graph = m["graph"]
            expected = {name: tensor_bytes(reference.evaluate(graph)[name], graph["tensors"][name]["dtype"])
                        for name in graph["outputs"]}
            records, replays = asic_ip.hardware_vector_records(payload, expected)
            self.assertEqual([r["op"] for r in replays if r["device"] == "VPU"], [7, 11, 6, 11, 12])
            self.assertEqual(sum(entry[0] == 5 and entry[1] == 256 for entry in records), 2)
            self.assertEqual(replays[-1]["scale"], 3)
            self.assertEqual(sum(entry[3].bit_count() // 8 for entry in records if entry[0] == 4), 80)

    def test_native_softmax_extreme_signed_scores_and_causality(self):
        image = bytearray(asic_ip.CAPACITY)
        struct.pack_into("<8i", image, 0, -(1 << 31), 1 << 30, 3, -(1 << 30),
                        10, 10, 10, (1 << 31) - 1)
        cmd = [asic_ip.MAGIC, 12, 12, 2, 0, 2, 0, asic_ip.NONE, 40, asic_ip.NONE,
               3, 0, 2, 2, 4, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        asic_ip.execute_descriptor(image, cmd, "VPU")
        self.assertEqual(image[40:48], bytes([0, 255, 0, 0, 85, 85, 85, 0]))
        malformed = cmd.copy()
        malformed[26] = 3
        with self.assertRaises(ValueError):
            asic_ip.execute_descriptor(image, malformed, "VPU")
