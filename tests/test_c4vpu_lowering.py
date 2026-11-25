from pathlib import Path
import tempfile
import unittest

from hetero_compiler.compiler import compile_bundle
from hetero_compiler.onnx import parse_model
from hetero_compiler.profile import load_profile


class VPULoweringTests(unittest.TestCase):
    def test_quantized_attention_selects_native_only_in_optimized_mode(self):
        graph = parse_model(Path("examples/c4a_streams/LongPrefix_Stream_C4A.onnx").read_bytes())
        profile = load_profile("fpga_zu15eg_balanced")
        profile["pipeline_scheduler"] = "bank-aware"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = compile_bundle(graph, profile, root / "off", pipeline_tiles=True, vpu_lowering="off")
            new = compile_bundle(graph, profile, root / "on", pipeline_tiles=True, vpu_lowering="on")
        self.assertEqual(set(old["output_bindings"]), set(new["output_bindings"]))
        self.assertTrue(any(s["op"] == "QSoftmax" and s["device"] == "GPU" for s in old["plan"]["stages"]))
        self.assertEqual(sum(s["op"] == "Attention" for s in new["plan"]["stages"]), 24)
        self.assertEqual(sum(s["op"] == "QSoftmax" for s in new["plan"]["stages"]), 0)

    def test_vision_is_unchanged_when_switching_vpu_lowering(self):
        graph = parse_model(Path("examples/c4a_streams/EdgeConv_Stream_C4A.onnx").read_bytes())
        profile = load_profile("fpga_zu15eg_balanced")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            compile_bundle(graph, profile, root / "off", pipeline_tiles=True, vpu_lowering="off")
            compile_bundle(graph, profile, root / "on", pipeline_tiles=True, vpu_lowering="on")
            self.assertEqual((root / "on/program.yhb").read_bytes(), (root / "off/program.yhb").read_bytes())

    def test_published_probability_preserves_a_materialized_output(self):
        graph = parse_model(Path("examples/c4a_streams/LongPrefix_Stream_C4A.onnx").read_bytes())
        probability = next(n["output"] for n in graph["nodes"] if n["op"] == "QSoftmax")
        graph["outputs"].append(probability)
        with tempfile.TemporaryDirectory() as folder:
            manifest = compile_bundle(graph, load_profile("fpga_zu15eg_balanced"),
                                      Path(folder) / "bundle", pipeline_tiles=True, vpu_attention="fast")
        self.assertIn(probability, manifest["output_bindings"])
        self.assertEqual(sum(s["op"] == "Attention" for s in manifest["plan"]["stages"]), 23)
        self.assertEqual(sum(s["op"] == "Z15Softmax" and s["attrs"]["fast"]
                             for s in manifest["plan"]["stages"]), 1)
