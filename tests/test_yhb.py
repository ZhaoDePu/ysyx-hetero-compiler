"""Container-level acceptance and CRC rejection tests."""
from pathlib import Path
import struct
import tempfile
import unittest

from hetero_compiler.yhb import emit_yhb, inspect_yhb


class YhbTests(unittest.TestCase):
    def test_four_target_ids_and_crc(self):
        for profile_id, profile in enumerate(("fpga_zu15eg_balanced", "fpga_zu15eg_full",
                                              "asic_nangate45_p8", "asic_nangate45_ip_compat4k"), 1):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "initial.bin").write_bytes(bytes(64))
                (root / "gpu.hex").write_text("00000013\n", encoding="ascii")
                manifest = {
                    "profile": {"name": profile, "sram": {"capacity": 4096}},
                    "graph": {"inputs": ["a"], "outputs": ["b"], "tensors": {
                        "a": {"dtype": "int8", "shape": [32], "data": [0] * 32},
                        "b": {"dtype": "int64", "shape": [4]}}},
                    "input_bindings": {"a": {"offset": 0, "size": 32,
                                             "dtype": "int8", "shape": [32]}},
                    "plan": {"allocations": {"b": {"offset": 32, "size": 32}},
                             "stages": [{"id": "launch", "device": "GPU", "output": "b"}],
                             "commands": [{"type": "launch", "stage": "launch"}]},
                    "initial_image": "initial.bin", "backends": {
                        "npu": {"commands": []}, "gpu": {"stages": [{
                            "id": "launch", "hex": "gpu.hex", "jobs": [
                                {"entry": 0, "threads": 1, "warp": 0,
                                 "gpu_job_reg": 1 << 16}]}]}},
                    "dma": [] if profile_id >= 3 else [
                        {"direction": "load", "ddr_offset": 0, "scratch_offset": 0, "bytes": 32},
                        {"direction": "store", "ddr_offset": 32, "scratch_offset": 32, "bytes": 32}],
                }
                path = emit_yhb(root, manifest)
                self.assertEqual(path.name, "program.yhb")
                code = path.read_bytes()
                info = inspect_yhb(code)
                self.assertEqual(info["target"], profile_id)
                self.assertEqual(info["sections"]["symbols"]["count"], 2)
                self.assertEqual(struct.unpack_from("<I", code, 12)[0], len(code))
                with self.assertRaisesRegex(ValueError, "CRC32"):
                    inspect_yhb(code[:-1] + bytes([code[-1] ^ 128]))

    def test_rejects_truncated_header(self):
        with self.assertRaisesRegex(ValueError, "header"):
            inspect_yhb(b"YHB1")
