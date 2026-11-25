import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hetero_compiler.ir import GraphError
from hetero_compiler.reference import evaluate
from hetero_compiler.tinylm import PITCH, VOCAB, _TileLane, compile_tinylm, read_onnx
from tools.run_c5_tinylm import accuracy, fixed_reference

ASSETS = Path(__file__).resolve().parents[1] / "examples/c5_tinylm"
ONNX = ASSETS / "TinyLM-Chat.onnx"


class TinyLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = read_onnx(ONNX)
        cls.training = json.loads((ASSETS/"training.json").read_text())
        cls.prompts = cls.training["prompts"]

    def test_causal_reference_is_history_dependent_and_future_independent(self):
        graph = self.model[0]
        prompt = self.prompts[0]
        separate_history = self.prompts[1][:3] + [prompt[-1]]
        base = evaluate(graph,{"tokens":prompt+[0]*(PITCH-len(prompt))})["lm.out"]
        later = evaluate(graph,{"tokens":prompt+[7]*(PITCH-len(prompt))})["lm.out"]
        before = evaluate(graph,{"tokens":separate_history+[0]*(PITCH-len(prompt))})["lm.out"]
        self.assertEqual(base[:4*VOCAB], later[:4*VOCAB])
        self.assertNotEqual(base[3*VOCAB:4*VOCAB],before[3*VOCAB:4*VOCAB])

    def test_mode_specific_golden_text(self):
        for native in (False,True):
            expected = fixed_reference(self.model[0],self.prompts,native)
            with np.load(ASSETS/"weights.fp32.npz", allow_pickle=False) as weights:
                rows = accuracy(expected,self.prompts,weights,self.training["vocab"])
            self.assertEqual(sum(r["top1"] for r in rows),27)
            self.assertGreater(min(r["logit_cosine_mean"] for r in rows),.98)

    def test_native_attention_rejects_incompatible_imported_causal_mask(self):
        import copy

        graph,nodes,digest = copy.deepcopy(self.model)
        nodes["l0.h0.p"]["attrs"]["causal_offset"] = 1
        with self.assertRaisesRegex(GraphError,"causal last-axis"):
            _TileLane((graph,nodes,digest),0,self.prompts[0],"A3",True,True).make()

    def test_device_modes_and_checkpoint_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a0 = compile_tinylm(ONNX,root/"a0-off",self.prompts[:1],phase="A0",vpu=False)
            compile_tinylm(ONNX,root/"a0-on",self.prompts[:1],phase="A0",vpu=True)
            a2 = compile_tinylm(ONNX,root/"a2-off",self.prompts[:1],phase="A2",vpu=False)
            a3 = compile_tinylm(ONNX,root/"a3-on",self.prompts[:2],phase="A3",vpu=True)
            self.assertEqual((root/"a0-off/program.yhb").read_bytes(),(root/"a0-on/program.yhb").read_bytes())
        self.assertEqual(a0["estimates"]["device_stage_counts"]["NPU"],0)
        self.assertEqual(a2["estimates"]["device_stage_counts"]["VPU"],0)
        self.assertEqual(a2["estimates"]["native_attention_commands"],0)
        self.assertEqual(a3["estimates"]["native_attention_commands"],96)
        self.assertEqual(a3["input_bindings"]["prompt_0"]["dtype"],"int64")
        self.assertEqual([c["step"] for c in a3["execution"] if c["type"]=="yield"],list(range(9)))
        row = [s for s in a3["plan"]["stages"] if s["op"]=="Gather" and "row4/" in s["id"]]
        self.assertTrue(all(s["inputs"][1].endswith("/feedback") for s in row))


if __name__ == "__main__":
    unittest.main()
