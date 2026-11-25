import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from hetero_compiler.ir import GraphError
from hetero_compiler.kv_reference import evaluate_kv
from hetero_compiler.kv_schedule import compile_kv_bundle, normalize_model
from hetero_compiler.yhb import inspect_yhb


class KVScheduleTest(unittest.TestCase):
    def setUp(self):
        self.model = json.loads(Path("examples/c4a_kv/causal_kv32.json").read_bytes())

    def test_unbroken_prefix_and_disjoint_storage(self):
        with TemporaryDirectory() as scratch:
            root = Path(scratch)
            cache = compile_kv_bundle(self.model, root/"cached")
            baseline = compile_kv_bundle(self.model, root/"baseline", mode="recompute")
            self.assertEqual(inspect_yhb((root/"cached/program.yhb").read_bytes())["target"], 1)
        plan = cache["plan"]
        self.assertEqual([t["key_prefix_rows"] for t in plan["kv_tiles"]], [8,16,24,32,33,34,35,36])
        self.assertEqual([t["first_written_kv_row"] for t in plan["kv_tiles"]], [0,8,16,24,32,33,34,35])
        self.assertEqual([t["bank"] for t in plan["kv_tiles"]], [0,1,0,1,0,1,0,1])
        self.assertEqual(cache["estimates"]["projected_kv_rows"],72)
        self.assertEqual(baseline["estimates"]["projected_kv_rows"],340)
        self.assertEqual(plan["resident_kv"]["k-state"]["size"],144)
        banks = plan["transient_banks"]
        self.assertLessEqual(banks[0]["offset"]+banks[0]["size"],banks[1]["offset"])
        for rec in plan["resident_kv"].values():
            self.assertLessEqual(rec["offset"]+rec["stored_size"],banks[0]["offset"])
        for step in range(1,5):
            self.assertEqual(plan["allocations"][f"d{step}-append-k"]["size"],4)
            self.assertEqual(plan["allocations"][f"d{step}-append-k"]["byte_offset"],(31+step)*4)

    def test_feedback_uses_observed_ddr_slot_and_emits_distinct_outputs(self):
        with TemporaryDirectory() as scratch:
            manifest = compile_kv_bundle(self.model,Path(scratch)/"program")
        code = manifest["execution"]
        feedback = [c for c in code if c.get("binding")=="token_feedback"]
        self.assertEqual([c["direction"] for c in feedback],["store","load"]*4+["store"])
        self.assertEqual({c["ddr_offset"] for c in feedback},
                         {manifest["input_bindings"]["token_feedback"]["offset"]})
        self.assertEqual([c["step"] for c in code if c["type"]=="yield"],[0,1,2,3,4])
        out = list(manifest["output_bindings"].values())
        self.assertEqual(len({x["ddr_offset"] for x in out}),12)
        self.assertEqual(sum(x["size"] for x in out),720)

    def test_numpy_reference_actual_token_feedback_and_change(self):
        old = evaluate_kv(self.model)
        changed = evaluate_kv(self.model,step_overrides={2:3})
        self.assertEqual([old[f"decode_{i}_token"][0] for i in range(1,5)],[0,2,1,0])
        self.assertEqual(old["prefill_logits"],changed["prefill_logits"])
        self.assertEqual(old["decode_1_logits"],changed["decode_1_logits"])
        self.assertNotEqual(old["decode_2_logits"],changed["decode_2_logits"])
        self.assertEqual(old["k_cache"][:132],changed["k_cache"][:132])
        self.assertNotEqual(old["k_cache"][132:],changed["k_cache"][132:])

    def test_rejects_invalid_shape_and_quantized_weights(self):
        with self.assertRaisesRegex(GraphError,"geometry"):
            normalize_model({**self.model,"prefill_rows":31})
        with self.assertRaisesRegex(GraphError,"signed bytes"):
            normalize_model({**self.model,"prompt":[5]})
        with self.assertRaisesRegex(GraphError,"saturate"):
            normalize_model({**self.model,"token_gain":[127]*4})


if __name__=="__main__":
    unittest.main()
