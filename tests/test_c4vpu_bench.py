import math
from pathlib import Path
import unittest

import numpy as np

from examples.c4vpu.transformer import encode, evaluate_fp32
from tools.run_c4vpu_eval import accuracy_rows


class C4VPUBench(unittest.TestCase):
    def test_checked_in_fixture_is_reproducible(self):
        exported = Path("examples/c4vpu/TransformerLayer_INT8_C4VPU.onnx")
        self.assertEqual(exported.read_bytes(), encode())
        reference = evaluate_fp32()
        for i in range(4):
            prefix = f"layer{i:02d}_"
            logits = reference[prefix+"vocab_out"].reshape(4, 4)
            self.assertEqual(logits.dtype, np.float32)
            self.assertTrue(np.isfinite(logits).all())
            self.assertEqual(reference[prefix+"predict_out"].tolist(), logits.argmax(axis=1).tolist())

    def test_error_columns_use_dequantized_values_and_token_agreement(self):
        off = {"logits": [16, 32], "case_predict_out": [0, 1]}
        on = {"logits": [0, 32], "case_predict_out": [0, 0]}
        fp32 = {"logits": [1, 2], "case_predict_out": [0, 0]}
        rows = accuracy_rows("layer", "A3", off, on, fp32)
        summaries = {r["pair"]: r for r in rows if r["scope"] == "logits_and_token_summary"}
        self.assertEqual(summaries["vpu_vs_gpu"]["max_abs_err"], 1)
        self.assertEqual(summaries["vpu_vs_gpu"]["mae"], 0.5)
        self.assertAlmostEqual(summaries["vpu_vs_gpu"]["cosine_similarity"], 2/math.sqrt(5))
        self.assertEqual(summaries["vpu_vs_gpu"]["top1_agreement"], 0.5)
        self.assertEqual(summaries["gpu_vs_fp32"]["max_abs_err"], 0)
        self.assertEqual(summaries["vpu_vs_fp32"]["top1_agreement"], 1)


if __name__ == "__main__":
    unittest.main()
