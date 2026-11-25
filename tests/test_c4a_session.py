"""YHB checkpoint encoding and malformed schedule checks."""

from copy import deepcopy
from pathlib import Path
import struct
import tempfile
import unittest

from hetero_compiler.yhb import emit_yhb, inspect_yhb
from tools.verify_c4a_session import make_fixture


class CheckpointTests(unittest.TestCase):
    def test_yhb_checkpoints_preserve_the_command_stride(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            fixture = make_fixture(folder)
            binary = emit_yhb(folder, fixture).read_bytes()
            commands = inspect_yhb(binary)["sections"]["commands"]
            rows = list(struct.iter_unpack("<4I", binary[commands["offset"]:]))
            self.assertEqual([entry for entry in rows if entry[0] == 5],
                             [(5, 0, 0, 0), (5, 1, 0, 0)])
            self.assertEqual(rows[-1], (5, 1, 0, 0))

    def test_yield_rejects_skipped_id_and_uncheckpointed_tail(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            fixture = make_fixture(folder)
            skipped = deepcopy(fixture)
            skipped["execution"][-1]["step"] += 1
            with self.assertRaisesRegex(ValueError, "consecutive"):
                emit_yhb(folder, skipped)
            fixture["execution"].pop()
            with self.assertRaisesRegex(ValueError, "end at a checkpoint"):
                emit_yhb(folder, fixture)


if __name__ == "__main__":
    unittest.main()
