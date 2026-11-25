import struct
import tempfile
import unittest
from pathlib import Path

from hetero_compiler.asic_ip import execute_descriptor, replay, WORDS
from hetero_compiler.compiler import compile_bundle
from hetero_compiler.profile import load_profile
from hetero_compiler.ir import normalize
from hetero_compiler.reference import evaluate
from hetero_compiler.yhb import inspect_yhb

from tools.verify_asic_c3_target import fixture


class AsicIPV3Tests(unittest.TestCase):
    def test_native_high_sram_and_shared_vpu_roundtrip(self):
        raw, inputs = fixture()
        with tempfile.TemporaryDirectory() as out:
            folder = Path(out) / "ip"
            native = load_profile("asic_nangate45_ip")
            plan = compile_bundle(raw, native, folder)
            bundle = (folder / "program.yhb").read_bytes()
            self.assertEqual(inspect_yhb(bundle)["target"], 5)
            self.assertEqual((folder / "initial.bin").stat().st_size, 65536)
            self.assertEqual(plan["plan"]["high_water_bytes"], 64512)
            records = plan["backends"]["npu"]["commands"]
            self.assertEqual([r["device"] for r in records], ["VPU"] * 6)
            self.assertEqual(len((folder / "npu.bin").read_bytes()), 6 * WORDS * 4)
            golden = evaluate(normalize(raw), inputs)
            packed = {name: struct.pack(f"<{len(values)}b", *values) for name, values in inputs.items()}
            self.assertEqual(replay(bundle, packed), {"probs": bytes(golden["probs"])})
            self.assertEqual(replay(bundle), {"probs": bytes(evaluate(normalize(raw))["probs"])})
            invalid = records[0]["words"][:]
            invalid[8] = 65528
            with self.assertRaisesRegex(ValueError, "extent"):
                execute_descriptor(bytearray(65536), invalid, "VPU")

    def test_explicit_compat_preserves_phase8_payload(self):
        raw = {"tensors": {n: {"shape": [1, 4], "dtype": "int8", "data": [1, 2, 3, 4]}
                           for n in ("a", "b")},
               "nodes": [{"id": "add", "op": "Add", "inputs": ["a", "b"], "output": "y"}],
               "outputs": ["y"]}
        with tempfile.TemporaryDirectory() as out:
            folder = Path(out) / "compat"
            plan = compile_bundle(raw, load_profile("asic_nangate45_ip_compat4k"), folder)
            self.assertEqual(inspect_yhb((folder / "program.yhb").read_bytes())["target"], 4)
            self.assertEqual(plan["backends"]["npu"]["abi"], "p8-ext-program")
            self.assertLessEqual(plan["plan"]["arena_end"], 4096)
