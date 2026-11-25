import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from hetero_compiler.compiler import compile_bundle
from hetero_compiler.profile import load_profile


GRAPH = {
    "name": "mixed",
    "tensors": {
        "x": {"shape": [1, 4], "dtype": "int8", "kind": "input", "data": [-8, 4, 12, 22]},
        "w": {"shape": [4, 4], "dtype": "int8",
              "data": [int(i == j) for i in range(4) for j in range(4)]},
    },
    "nodes": [
        {"id": "gemm", "op": "Gemm", "inputs": ["x", "w"], "output": "a"},
        {"id": "quant", "op": "Requant", "inputs": ["a"], "output": "z",
         "attrs": {"shift": 1}},
        {"id": "add", "op": "Add", "inputs": ["z", "x"], "output": "y",
         "attrs": {"narrowing": "wrap"}},
    ],
    "outputs": ["y"],
}


class BundleTests(unittest.TestCase):
    def test_bundle_real_formats_and_resident_dependency(self):
        with TemporaryDirectory(dir=Path.cwd()) as tmp:
            output = Path(tmp) / "program"
            p = load_profile("fpga_zu15eg_balanced")
            manifest = compile_bundle(GRAPH, p, output)
            self.assertEqual(manifest["plan"]["stages"][0]["device"], "NPU")
            self.assertEqual(manifest["plan"]["stages"][-1]["device"], "GPU")
            self.assertEqual([c["type"] for c in manifest["plan"]["commands"]],
                             ["launch", "fence", "launch"])
            self.assertEqual(len((output / "initial.bin").read_bytes()), 2 << 20)
            self.assertTrue((output / "npu.bin").stat().st_size)
            self.assertEqual(manifest["program"], "program.yhb")
            self.assertEqual((output / "program.yhb").read_bytes()[:4], b"YHB1")
            self.assertTrue((output / manifest["backends"]["gpu"]["stages"][0]["elf"])
                            .read_bytes().startswith(b"\x7fELF"))
            self.assertEqual({n for cmd in manifest["dma"] for n in cmd["tensors"]},
                             {"x", "w.__npu_colmajor", "y"})
            self.assertEqual(json.loads((output / "bundle.json").read_text()), manifest)
            with self.assertRaises(FileExistsError):
                compile_bundle(GRAPH, p, output)
