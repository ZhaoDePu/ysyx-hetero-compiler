import unittest

from examples.c3_streams.generate import transformer, vision
from hetero_compiler.compiler import tensor_bytes
from hetero_compiler.ir import normalize
from hetero_compiler.onnx import parse_model
from hetero_compiler.pipeline import _batch_tiles, _independent, make_pipeline_plan
from hetero_compiler.profile import load_profile
from hetero_compiler.reference import evaluate


class PipelineTests(unittest.TestCase):
    def test_leading_batch_vision_preserves_outputs(self):
        graph = normalize(parse_model(vision(3)))
        whole = evaluate(graph)
        pieces = _batch_tiles(graph, 32)
        self.assertEqual(len(pieces), 3)
        for name in graph["outputs"]:
            dtype = graph["tensors"][name]["dtype"]
            gathered = b"".join(tensor_bytes(evaluate(piece)[name], dtype) for piece, _ in pieces)
            self.assertEqual(gathered, tensor_bytes(whole[name], dtype))

    def test_attention_contexts_remain_intact(self):
        graph = normalize(parse_model(transformer(3)))
        whole = evaluate(graph)
        parts = _independent(graph)
        self.assertEqual(len(parts), 3)
        for part, spans in parts:
            self.assertEqual(spans, {})
            result = evaluate(part)
            self.assertEqual({name: whole[name] for name in part["outputs"]},
                             {name: result[name] for name in part["outputs"]})
            self.assertTrue(any(n.get("attrs", {}).get("causal_mask") for n in part["nodes"]))

    def test_public_passthrough_input_is_preloaded_before_writeback(self):
        graph = normalize({
            "tensors": {n: {"shape": [2, 32], "dtype": "int8", "kind": "input"}
                        for n in ("x", "y")},
            "nodes": [{"id": "relu", "op": "Relu", "inputs": ["y"], "output": "z"}],
            "outputs": ["x", "z"],
        })
        target = load_profile("fpga_zu15eg_balanced")
        plan, _, commands = make_pipeline_plan(graph, target, fuse=True, reuse=True,
                                               overlap=True, device_policy="gpu-only")
        base = plan["ddr_allocations"]["x"]["offset"]
        for at in (base, base + 32):
            matching = [c["direction"] for c in commands if c["type"] == "dma" and
                        c["ddr_offset"] <= at < c["ddr_offset"] + c["bytes"]]
            self.assertEqual(matching, ["load", "store"])

    def test_dma_slice_size_uses_beat_not_scratch_alignment(self):
        graph = normalize(parse_model(vision(2)))
        target = load_profile("fpga_zu15eg_balanced")
        target["sram"]["alignment"] = 64
        plan, _, commands = make_pipeline_plan(graph, target, fuse=True, reuse=True,
                                               overlap=True, device_policy="auto")
        base = plan["ddr_allocations"]["pixel_logits"]["offset"]
        stops = [(c["ddr_offset"], c["bytes"]) for c in commands if c["type"] == "dma"
                 and c["direction"] == "store" and base <= c["ddr_offset"] < base + 192]
        self.assertEqual(sorted(stops), [(base, 96), (base + 96, 96)])

    def test_overlap_scheduler_respects_two_single_owner_halves(self):
        graph = normalize(parse_model(vision(4)))
        target = load_profile("fpga_zu15eg_balanced")
        for overlap in (False, True):
            plan, _, commands = make_pipeline_plan(graph, target, fuse=True, reuse=True,
                                                   overlap=overlap, device_policy="auto")
            stages = {r["id"]: r for r in plan["stages"]}
            active, peak = {}, 0
            kinds = set()
            for cmd in commands:
                typ = cmd["type"]
                if typ == "fence":
                    active.pop(cmd["wait_for"])
                elif typ == "dma-fence":
                    active.pop("dma")
                else:
                    name = "dma" if typ == "dma" else cmd["stage"]
                    engine = "DMA" if typ == "dma" else (
                        "GPU" if stages[name]["device"] == "GPU" else "TENSOR")
                    self.assertNotIn(engine, [r[0] for r in active.values()])
                    self.assertNotIn(cmd["zone"], [r[1] for r in active.values()])
                    active[name] = engine, cmd["zone"]
                    peak = max(peak, len(active))
                    if len(active) == 2:
                        kinds.add(tuple(sorted(r[0] for r in active.values())))
            self.assertFalse(active)
            self.assertEqual(peak, 2 if overlap else 1)
            if overlap:
                self.assertIn(("GPU", "TENSOR"), kinds)
                self.assertTrue(any("DMA" in pair for pair in kinds))


if __name__ == "__main__":
    unittest.main()
