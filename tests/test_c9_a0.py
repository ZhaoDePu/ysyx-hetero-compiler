"""Check that A0 aggregation exposes coverage and measures full selected outputs."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from tools.c9_a0 import METRICS, cases, record


class A0AggregationTests(unittest.TestCase):
    def setUp(self):
        def desc(channels):
            return {"dtype": "int8", "layout": "NHWC", "shape": [1, channels, 1, 1],
                    "physical_shape": [1, 1, 1, channels]}

        nodes = [
            {"id": name, "op": "Split", "inputs": [name], "outputs": [name + "x", name + "y"],
             "attrs": {"axis": 1}} for name in ("first", "second")
        ] + [{"id": "last", "op": "RequantLUT", "inputs": ["firstx"], "outputs": ["last"],
              "attrs": {"lut": [0] * 256}}]
        tensors = {"first": desc(12), "second": desc(12), "last": desc(4)}
        tensors.update({"firstx": desc(4), "firsty": desc(8), "secondx": desc(4), "secondy": desc(8)})
        self.source = SimpleNamespace(graph={"nodes": nodes}, tensors=tensors, constants={})

    def test_equal_geometry_matches_all_outputs_and_keeps_multiplicity(self):
        partition, = [entry for entry in cases(self.source)
                      if entry["signature"]["operator"] == "Split"]
        self.assertEqual(partition["members"], ["first", "second"])
        self.assertEqual(partition["output_elements"], 12)

    def test_partial_coverage_does_not_publish_a_whole_graph_aggregate(self):
        plan = cases(self.source)
        with TemporaryDirectory() as tmp:
            args = SimpleNamespace(out=Path(tmp), npu_mhz=125, gpu_mhz=100)
            measured = {plan[0]["key"]: {key: 2 for key in
                        (*METRICS, "sram_peak_bytes", "ddr_peak_bytes")}}
            record(args, self.source, plan, measured, [])
            observed = json.loads((args.out / "aggregation.json").read_text())
        self.assertFalse(observed["coverage_complete"])
        self.assertIsNone(observed["whole_graph_aggregate"])
        self.assertEqual(observed["unique_measured"], 1)
        self.assertEqual(observed["unique_total"], 2)
