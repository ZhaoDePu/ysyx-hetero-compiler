import copy
import struct
import unittest

from hetero_compiler.p19 import BANK, DDR, Rectangle, audit, fold_1d, pipeline, reference, swizzled_address
from hetero_compiler.p19_g11 import Q, emulate, encode_binary, g11, kernel, sfu


class TmaTest(unittest.TestCase):
    def test_fold(self):
        rows = [(64 + i * 256, i * 128, 96, False, False) for i in range(16)]
        rect, = fold_1d(rows)
        self.assertEqual(rect.moves(), rows)
        self.assertEqual(rect.rows, 16)
        self.assertEqual(rect.commands()[0], 0x1000018040000040)
        self.assertEqual(rect.commands()[-1], 0x3000019c00000002)

    def test_fences(self):
        for fields in ((DDR - 32, 0, 64, 1, 64, 64), (0, BANK - 32, 64, 1, 64, 64),
                       (0, 0, 32, 2, 1, 32), (0, 0, 32, 0, 32, 32)):
            with self.assertRaises(ValueError):
                Rectangle(*fields).validate()

    def test_swizzle(self):
        for aligned in range(0, 1024, 32):
            seq = [swizzled_address(aligned + i) for i in range(32)]
            self.assertEqual(sorted(seq), list(range(aligned, aligned + 32)))
            self.assertEqual(list(map(swizzled_address, seq)), list(range(aligned, aligned + 32)))

    def test_reference_pipeline(self):
        plan = pipeline(8, 16, 64, 256, 128)
        result, words = reference(plan)
        self.assertEqual(result["stored_tiles"], 8)
        self.assertEqual(plan["summary"]["dma_1d"] // plan["summary"]["tma_2d"], 16)
        self.assertEqual(plan["summary"]["triple_overlap_slots"], 6)
        self.assertEqual(plan["summary"]["slots"], 20)
        self.assertEqual(len(words), plan["summary"]["phase19_dma_barrier_command_estimate"])

    def test_deadlock_and_race(self):
        plan = pipeline(8, 16, 64, 256, 128)
        bad = copy.deepcopy(plan["events"])
        bad[0]["wait"] = [0]
        with self.assertRaisesRegex(ValueError, "unsatisfied barrier"):
            audit(bad)
        bad = copy.deepcopy(plan["events"])
        first_compute = next(op for op in bad if op["op"] == "compute")
        first_compute["banks"][0] = 1
        with self.assertRaisesRegex(ValueError, "bank or engine conflict"):
            audit(bad)


class G11Test(unittest.TestCase):
    def test_encoding(self):
        exp = g11("EXP", 10, 11)
        self.assertEqual(exp, 0x0405855B)
        self.assertEqual(encode_binary([exp]), b"\x5b\x85\x05\x04")
        with self.assertRaises(ValueError):
            g11("BARRIER_WAIT", 1, 25)
        with self.assertRaises(ValueError):
            g11("BARRIER_WAIT", 0, 2)

    def test_topk(self):
        scores = [-8, 3, 200, 200, 0, 9, 1, -50]
        tokens = [7, 3, 5, 0, 2, 6, 4, 1]
        expected = sorted(zip(scores, tokens), key=lambda row: (-row[0], row[1]))[:4]
        words = kernel("topk", k=4)
        self.assertEqual(list(struct.unpack("<" + "I" * len(words), encode_binary(words))), words)
        state = emulate(words, {5: scores, 6: tokens}, 8)
        for i, (score, token) in enumerate(expected):
            self.assertEqual(state[16 + 2 * i], [score] * 8)
            self.assertEqual(state[17 + 2 * i], [token] * 8)

    def test_rmsnorm(self):
        x = [-41236, -9233, 0, 8194, 18235, 19568, -6425, 13860]
        gamma = [Q] * len(x)
        scale = sfu("RSQRT", (sum(v * v // Q for v in x) // len(x)) + 1)
        y = [v * scale // Q for v in x]
        coeff = sfu("RCP", max(map(abs, y))) * 127
        i8 = [max(-127, min(127, round(v * coeff / (Q * Q)))) for v in y]
        state = emulate(kernel("rmsnorm-a8"), {5: x, 6: gamma}, 8)
        self.assertEqual(state[10], y)
        self.assertEqual([((state[13][i] >> ((i % 4) * 8)) & 255) for i in range(8)],
                         [v & 255 for v in i8])

    def test_online_attention(self):
        m, count, pv = -5 * Q, 0, 0
        words = kernel("flashattention-pv")
        for step in range(3):
            scores = [i * Q // 3 + step * Q // 2 for i in range(8)]
            values = [i * Q // 8 for i in range(8)]
            state = emulate(words, {5: [m] * 8, 6: [count] * 8, 7: [pv] * 8, 8: scores, 9: values}, 8)
            maximum = max(m, *scores)
            decay = sfu("EXP", m - maximum)
            weights = [sfu("EXP", x - maximum) for x in scores]
            count = count * decay // Q + sum(weights)
            pv = pv * decay // Q + sum(w * v // Q for w, v in zip(weights, values))
            m = maximum
            self.assertEqual(state[5:8], [[m] * 8, [count] * 8, [pv] * 8])
            self.assertEqual(state[13], [pv * sfu("RCP", count) // Q] * 8)
