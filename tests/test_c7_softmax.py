"""Precision and data-page tests for the C7 packed-log Softmax kernel."""

import unittest

import numpy as np


class C7SoftmaxTests(unittest.TestCase):
    def test_packed_probability_model(self):
        from hetero_compiler.c7_softmax import log_softmax, log_softmax_tables

        score = np.arange(256, dtype=np.int64)
        scale = 0.09412467573195944
        lut = np.rint((1 << 23) * np.exp(-scale * score)).astype(int).tolist()
        rng = np.random.default_rng(701)
        values = rng.integers(-128, 128, (2, 16, 100), dtype=np.int8)
        values[0, 0] = -128
        values[0, 0, 0] = 127
        values[0, 1] = 0
        actual = log_softmax(values, -1, lut, scale, 8)
        delta = values.max(-1, keepdims=True).astype(np.int16) - values
        weights = np.asarray(lut)[delta]
        expected = np.rint(weights * 256 / weights.sum(-1, keepdims=True)).clip(0, 255).astype(np.uint8)
        self.assertLessEqual(np.abs(actual.astype(np.int16) - expected).max(), 2)
        self.assertEqual(actual[0, 0, 0], 255)
        self.assertEqual(actual[0, 1, 0], 3)
        tables = log_softmax_tables(lut, scale, 8, 100)
        self.assertLessEqual(len(tables.payload) + 16 * 100, 4096)

    def test_tiles_keep_full_warp_and_ownership(self):
        from hetero_compiler.c7_softmax import log_softmax_tables, lower_log_softmax

        scale = 0.09412467573195944
        weights = np.rint((1 << 23) * np.exp(-scale * np.arange(256))).astype(int)
        tables = log_softmax_tables(weights, scale, 8, 100)
        tiles = lower_log_softmax("softmax", 0, 3200, 6400, (2, 16, 100), tables)
        self.assertEqual([tile.zone for tile in tiles], [0, 1])
        self.assertEqual([tile.stages[0].jobs[0]["threads"] for tile in tiles], [16, 16])
        for tile in tiles:
            self.assertEqual(tile.loads[0].scratch_offset, tile.stores[0].scratch_offset)
            self.assertEqual(tile.loads[0].size, 1600)
            self.assertEqual(tile.scratch_bytes, 4096)

    def test_native_softmax_consumes_raw_npu_score_words(self):
        from hetero_compiler.c6_attention import lower_matmul, lower_vpu_softmax_words

        dot, = lower_matmul("scores", 0, 640, 1280, (2, 16, 16), (2, 16, 16),
                            1, 0, raw_output=True)
        self.assertEqual(dot.stores[0].size, 2 * 16 * 16 * 4)
        for stage in dot.stages:
            if stage.device == "NPU":
                cfg = dict(stage.writes)
                self.assertEqual(cfg[0x78], 0)
        normalizer, = lower_vpu_softmax_words("attention", 1280, 4096, (2, 16, 16), 3)
        self.assertEqual(normalizer.loads[0].size, dot.stores[0].size)
        self.assertEqual(normalizer.stores[0].size, 2 * 16 * 16)
        self.assertEqual([stage.device for stage in normalizer.stages], ["VPU"])
