import json
from pathlib import Path
import struct
import tempfile
import unittest

from hetero_compiler.gpu import lower, supports


class GpuLoweringTest(unittest.TestCase):
    def setUp(self):
        self.profile = {"gpu": {"variant": "g4", "lanes": 8, "warps": 8,
                                "window_bytes": 4096, "code_bytes": 16384, "issue_width": 2}}
        self.graph = {"tensors": {n: {"shape": [33], "dtype": "int8"} for n in ["a", "b", "c"]}}
        self.stage = {"id": "sum", "op": "Add", "inputs": ["a", "b"], "output": "c",
                      "attrs": {}, "device": "GPU"}
        self.plan = {"stages": [self.stage], "allocations": {
            n: {"offset": 4096 + 64 * i, "size": 33} for i, n in enumerate(["a", "b", "c"])}}

    def test_native_image_and_fpga_doorbells(self):
        with tempfile.TemporaryDirectory() as path:
            bundle = lower(self.graph, self.plan, self.profile, Path(path))
            stage = bundle["stages"][0]
            self.assertEqual(stage["id"], "sum")
            self.assertEqual([(j["threads"], j["warp"], j["page"], j["gpu_job_reg"])
                              for j in stage["jobs"]], [
                (8, 0, 1, 0x00080001), (8, 1, 1, 0x01080001), (8, 2, 1, 0x02080001),
                (8, 3, 1, 0x03080001), (1, 4, 1, 0x04010001)])
            elf = (Path(path) / stage["elf"]).read_bytes()
            self.assertEqual(elf[:7], b"\x7fELF\x01\x01\x01")
            self.assertEqual(struct.unpack_from("<H", elf, 18)[0], 243)
            self.assertEqual(struct.unpack_from("<I", elf, 36)[0], 8)
            self.assertEqual(struct.unpack_from("<I", elf, 24)[0], stage["jobs"][0]["entry"])
            _, off, _, _, length, _, _, _ = struct.unpack_from("<8I", elf, 52)
            words = [int(w, 16) for w in (Path(path) / stage["hex"]).read_text().splitlines()]
            self.assertEqual(elf[off:off + length], struct.pack(f"<{len(words)}I", *words))
            for start, job in zip((0, 8, 16, 24, 32), stage["jobs"]):
                self.assertEqual(words[job["entry"] // 4] >> 20, start)
            self.assertEqual(json.loads((Path(path) / "gpu.json").read_text()), bundle)

    def test_cross_page_operand_rejected(self):
        self.plan["allocations"]["c"]["offset"] = 8192
        with tempfile.TemporaryDirectory() as path, self.assertRaisesRegex(ValueError, "page"):
            lower(self.graph, self.plan, self.profile, Path(path))

    def test_unsupported_shape_and_signed_lut_type_rejected(self):
        self.graph["tensors"]["b"]["shape"] = [32]
        self.assertFalse(supports(self.stage, self.graph, self.profile))
        self.stage["op"], self.stage["inputs"] = "GELU", ["a"]
        self.graph["tensors"]["c"]["dtype"] = "uint8"
        self.assertFalse(supports(self.stage, self.graph, self.profile))
