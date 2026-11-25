"""Detector dataset, learned fixture and no-NMS evaluation contracts."""

import unittest
from pathlib import Path

import numpy as np

from examples.c5_yolo.export import encode
from examples.c5_yolo.report import decode_fp32,decode_rtl,ledger
from examples.c5_yolo.scenes import held_out
from examples.c5_yolo.train import forward,load_parameters
from hetero_compiler.ir import normalize
from hetero_compiler.onnx import parse_model
from hetero_compiler.reference import evaluate


FIXTURE=Path(__file__).resolve().parents[1]/"examples/c5_yolo"


class TrainedDetectionTests(unittest.TestCase):
    def test_deployable_onnx_reproduces_and_detects_heldout_objects(self):
        cases = held_out()
        images = [x for x,_ in cases.values()]
        params=load_parameters(FIXTURE/"parameters.json")
        single=encode(params,images[:1])
        self.assertEqual(single,(FIXTURE/"YOLO26_Tiny_Style_INT8.onnx").read_bytes())
        raw=normalize(parse_model(encode(params,images,independent=True)))
        fixed=evaluate(raw)
        float_rows=forward(np.stack(images),params)[0]
        for n,(name,(_,truth)) in enumerate(cases.items()):
            actual=decode_rtl(fixed,f"im{n:02d}_")
            stats=ledger(name,actual,decode_fp32(float_rows[n]),truth)
            self.assertEqual(stats["detected"],len(truth))
            self.assertEqual(stats["gt_class_correct"],len(truth))
            self.assertGreater(stats["mean_iou_vs_fp32"],.75)

    def test_dense_holdout_contains_overlapping_objects(self):
        from examples.c5_yolo.report import iou
        _,truth=held_out()["dense-overlap"]
        self.assertGreaterEqual(len(truth),4)
        self.assertTrue(any(iou(a["box"],b["box"])>.25 for i,a in enumerate(truth) for b in truth[i+1:]))
