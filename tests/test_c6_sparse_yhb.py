"""Validate sparse DDR initialization with an unchanged YHB1 container."""

from pathlib import Path
import struct
import tempfile
import unittest

from hetero_compiler.yhb import emit_yhb, inspect_yhb


def move(direction, ddr, scratch=0, size=32):
    return {"type": "dma", "direction": direction, "ddr_offset": ddr,
            "scratch_offset": scratch, "bytes": size}


class SparseYhbTests(unittest.TestCase):
    def manifest(self):
        return {
            "profile": {"name": "fpga_zu15eg_balanced", "sram": {"capacity": 2 << 20}},
            "plan": {"stages": [], "allocations": {}},
            "graph": {"inputs": ["image"], "outputs": [],
                      "tensors": {"image": {"dtype": "int8", "shape": [32]}}},
            "backends": {"gpu": {"stages": []}, "npu": {"commands": []}},
            "input_bindings": {"image": {"ddr_offset": 32, "size": 32, "needs_data": True}},
            "initial_ddr_extents": [
                {"ddr_offset": 0, "file": "weights.bin", "file_offset": 1, "bytes": 32}
            ],
        }

    def test_adjacent_external_input_and_completed_spill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "weights.bin").write_bytes(b"!" + bytes(range(32)))
            manifest = self.manifest()
            fence = {"type": "dma-fence"}
            spill = 400 << 20
            manifest["execution"] = [
                move("load", 0, size=64), fence, move("store", spill, size=64), fence,
                move("load", spill, 1 << 20, size=64), fence,
            ]
            blob = emit_yhb(root, manifest).read_bytes()
            sections = inspect_yhb(blob)["sections"]
            self.assertLess(len(blob), 512)
            self.assertEqual(sections["initial"]["count"], 1)
            ddr, length, offset, flags = struct.unpack_from("<4I", blob, sections["initial"]["offset"])
            self.assertEqual((ddr, length, flags), (0, 32, 0))
            start = sections["data"]["offset"] + offset
            self.assertEqual(blob[start:start + length], bytes(range(32)))

    def test_unfenced_spill_and_input_padding_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "weights.bin").write_bytes(bytes(33))
            manifest = self.manifest()
            spill = 400 << 20
            for sequence in [
                [move("store", spill), move("load", spill)],
                [move("load", 32, size=64)],
            ]:
                with self.subTest(sequence=sequence):
                    manifest["execution"] = sequence
                    with self.assertRaisesRegex(ValueError, "uninitialized DDR"):
                        emit_yhb(root, manifest)
