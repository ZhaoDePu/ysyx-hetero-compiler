import unittest

from examples.c4a_streams.generate import attention, vision
from hetero_compiler.ir import normalize
from hetero_compiler.onnx import parse_model
from hetero_compiler.pipeline import _simplify_tile_layouts, make_pipeline_plan
from hetero_compiler.profile import load_profile
from hetero_compiler.reference import evaluate
from tools.c4a_imem import ReplayPrograms, _can_relocate


class C4APipelineTests(unittest.TestCase):
    def test_nhwc_boundary_copies_fold_without_affecting_raw_graph(self):
        graph = normalize(parse_model(vision(2)))
        reduced = _simplify_tile_layouts(graph)
        self.assertTrue(any(node["op"] == "Transpose" for node in graph["nodes"]))
        self.assertFalse(any(node["op"] == "Transpose" for node in reduced["nodes"]))
        self.assertEqual(graph["inputs"], reduced["inputs"])
        self.assertEqual(graph["outputs"], reduced["outputs"])
        original, optimized = evaluate(graph), evaluate(reduced)
        for name in graph["outputs"]:
            self.assertEqual(original[name], optimized[name])

    def test_oldest_tile_first_and_per_half_fences(self):
        graph = normalize(parse_model(vision(4)))
        target = load_profile("fpga_zu15eg_balanced")
        target["pipeline_scheduler"] = "bank-aware"
        plan, _, commands = make_pipeline_plan(graph, target, fuse=True, reuse=True,
                                               overlap=True, device_policy="auto")
        self.assertEqual(plan["pipeline_scheduler"], "bank-aware")
        stages = {s["id"]: s for s in plan["stages"]}
        active, overlap, order = {}, set(), {"GPU": [], "NPU": []}
        for cmd in commands:
            kind, zone = cmd["type"], cmd["zone"]
            if kind in {"fence", "dma-fence"}:
                name = cmd["wait_for"] if kind == "fence" else "dma"
                self.assertEqual(active.pop(name)[1], zone)
                continue
            if kind == "launch":
                name, engine = cmd["stage"], cmd["device"]
                order[engine].append(stages[name]["tile"])
            else:
                name, engine = "dma", "DMA"
            self.assertNotIn(engine, {value[0] for value in active.values()})
            self.assertNotIn(zone, {value[1] for value in active.values()})
            active[name] = engine, zone
            if len(active) > 1:
                overlap.add(frozenset(value[0] for value in active.values()))
        self.assertFalse(active)
        self.assertEqual(sorted(order["NPU"]), order["NPU"])
        self.assertEqual(sorted(order["GPU"]), order["GPU"])
        self.assertIn(frozenset(("NPU", "GPU")), overlap)
        self.assertTrue(any("DMA" in pair for pair in overlap))

    def test_attention_retains_standard_qdq_causal_semantics(self):
        graph = normalize(parse_model(attention(2)))
        softmax = [node for node in graph["nodes"] if node["op"] == "QSoftmax"]
        self.assertEqual(len(softmax), 2)
        for node in softmax:
            self.assertEqual(graph["tensors"][node["inputs"][0]]["shape"], [2, 48])
            self.assertEqual(node["attrs"]["causal_offset"], 46)

    def test_upper_qsoftmax_layout_is_scoped_to_bank_aware_reuse(self):
        graph = normalize(parse_model(attention(2)))
        target = load_profile("fpga_zu15eg_balanced")
        page = target["gpu"]["window_bytes"]
        for scheduler, reuse in (("bank-aware", True), ("bank-aware", False), ("c3", True)):
            target["pipeline_scheduler"] = scheduler
            plan, _, _ = make_pipeline_plan(graph, target, fuse=reuse, reuse=reuse,
                                              overlap=False, device_policy="auto")
            for stage in plan["stages"]:
                if stage["op"] != "QSoftmax":
                    continue
                source, destination = [plan["allocations"][name] for name in stage["buffers"]]
                pair = (source["offset"] % page, destination["offset"] % page)
                upper = (page - source["stored_size"] - destination["stored_size"],
                         page - destination["stored_size"])
                if scheduler == "bank-aware" and reuse:
                    self.assertEqual(pair, upper)
                    self.assertEqual(source["reserved_last"], destination["reserved_last"])
                else:
                    self.assertNotEqual(pair, upper)

    def test_imem_reuses_safe_images_and_rejects_relocation_of_pc_values(self):
        nop, jump_forward, ebreak = 0x13, 0x0040006F, 0x00100073
        self.assertFalse(_can_relocate([0xCC3032F3, ebreak]))
        self.assertFalse(_can_relocate([ebreak, nop]))
        self.assertFalse(_can_relocate([jump_forward]))
        self.assertFalse(_can_relocate([0x004000EF, ebreak]))
        self.assertFalse(_can_relocate([0x17, ebreak]))
        images = {"a": [jump_forward, ebreak], "b": [nop, ebreak], "same": [nop, ebreak]}
        cache = ReplayPrograms(images)
        jobs = [{"entry": 0, "gpu_job_reg": 3}]
        self.assertEqual(cache.commands("b", jobs), cache.commands("same", jobs))
        self.assertNotEqual(cache.commands("a", jobs), cache.commands("b", jobs))
        self.assertEqual(len(cache.preload), 4)
        uncached = ReplayPrograms({"unknown": [0x17]})
        self.assertEqual(len(uncached.commands("unknown", jobs)), 2)
        self.assertEqual(len(uncached.commands("unknown", jobs)), 1)


if __name__ == "__main__":
    unittest.main()
