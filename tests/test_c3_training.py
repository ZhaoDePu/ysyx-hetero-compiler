"""Reproduce training/export and verify the independent FP32 ONNX path."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
MODELS = Path(__file__).resolve().parent.parent / "examples/c3_training/models"


@unittest.skipUnless(HAS_NUMPY, "offline training fixtures require NumPy")
class TrainedQuantizedModels(unittest.TestCase):
    def test_reproducible_weights_and_exported_fp32_accuracy(self):
        import hashlib
        import numpy as np
        from examples.c3_training.export import export_models
        from examples.c3_training.fp32_onnx import Fp32Onnx
        from examples.c3_training.train import (accuracy, make_data, train, transformer_forward,
                                                vision_forward)

        with tempfile.TemporaryDirectory() as tmp:
            for task, fwd in (("vision", vision_forward), ("transformer", transformer_forward)):
                with self.subTest(task=task):
                    params, history = train(task)
                    cal, _ = make_data(task, "calibration")
                    path = Path(tmp)
                    m = export_models(task, params, fwd(cal, params)[1], path)
                    for mode in ("fp32", "int8"):
                        fname = m["models"][mode]
                        self.assertEqual(hashlib.sha256((path / fname).read_bytes()).hexdigest(),
                                         hashlib.sha256((MODELS / fname).read_bytes()).hexdigest())
                    self.assertGreater(history[-1]["validation_top1"], history[0]["validation_top1"])
                    x, y = make_data(task, "test")
                    runner = Fp32Onnx((path / m["models"]["fp32"]).read_bytes())
                    name, = runner.inputs
                    native = [
                        runner.evaluate(**{name: s.transpose(2, 0, 1)[None] if task == "vision" else s})
                        ["logits"] for s in x[:32]
                    ]
                    batch = np.concatenate(native) if task == "vision" else np.stack(native)
                    self.assertGreaterEqual(accuracy(batch, y[:32], task), .95)
                    np.testing.assert_allclose(batch, fwd(x[:32], params)[0], rtol=1e-5, atol=1e-5)

    def test_exported_attention_masks_future_positions(self):
        import numpy as np
        from examples.c3_training.fp32_onnx import Fp32Onnx
        from examples.c3_training.train import make_data
        x, _ = make_data("transformer", "test")
        sample = x[0].copy()
        model = Fp32Onnx((MODELS / "transformer_fp32.onnx").read_bytes())
        original = model.evaluate(tokens=sample)["logits"]
        sample[3, :4] = np.roll(sample[3, :4], 1)
        changed = model.evaluate(tokens=sample)["logits"]
        np.testing.assert_array_equal(original[:3], changed[:3])
