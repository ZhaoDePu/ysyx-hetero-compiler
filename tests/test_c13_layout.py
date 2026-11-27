from pathlib import Path
import re
import unittest

from tools.c13_emit_gpu import ENTRIES, bootstrap


class TestC13GpuLayout(unittest.TestCase):
    def values(self):
        header = Path(__file__).resolve().parents[1] / "runtime/c13_qwen_layout.h"
        return {key: int(value, 0) for key, value in
                re.findall(r"\b(C13_[A-Z0-9_]+)\s*=\s*(0x[0-9a-f]+|\d+)", header.read_text())}

    def test_full_context_tiled_scratch_stays_below_frozen_lane_stacks(self):
        v = self.values()
        self.assertGreaterEqual(v["C13_CACHE"], v["C13_EXP"] + v["C13_EXP_COUNT"] * 4)
        self.assertLessEqual(v["C13_CACHE"] + v["C13_TILE"] * v["C13_KV_SLOT_BYTES"],
                             v["C13_SCORES"])
        self.assertLessEqual(v["C13_SCORES"] + v["C13_QUERIES"] * v["C13_DEPTH"] * 4, 0x3c000)
        self.assertLessEqual(v["C13_SUMS"] + v["C13_QUERIES"] * v["C13_HEAD"] * 8, v["C13_EXP"])
        self.assertLessEqual(v["C13_OUT"] + v["C13_INTER"] * 4, v["C13_SUMS"])
        self.assertGreaterEqual(v["C13_EXP"], 0x11000 + 496 * 4)
        self.assertEqual(v["C13_DEPTH"] % v["C13_TILE"], 0)
        self.assertEqual(v["C13_KV_SLOT_BYTES"] % 32, 0)
        self.assertEqual(v["C13_KV_BYTES"], 4 * v["C13_KV_HEAD_BYTES"])

    def test_bootstrap_keeps_hardware_assigned_stack_and_links_per_warp(self):
        program = bootstrap()
        self.assertNotRegex(program, r"(?m)^\s*\w+\s+sp\s*,")
        for name, count in ENTRIES.items():
            for warp in range(count):
                self.assertIn(f"boot_{name}_{warp}:", program)
                self.assertIn(f"jal ra, c13_{name}", program)
        self.assertEqual(program.count("ebreak"), sum(ENTRIES.values()))
