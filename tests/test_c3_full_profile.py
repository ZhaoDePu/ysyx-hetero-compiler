import tempfile
import unittest
from pathlib import Path

from hetero_compiler.compiler import compile_bundle
from hetero_compiler.profile import load_profile
from hetero_compiler.yhb import inspect_yhb


class FullBoardProjection(unittest.TestCase):
    def test_launch_waves_only_use_the_four_abi_slots(self):
        graph = {"tensors": {"x": {"shape": [72], "dtype": "int8"}},
                 "nodes": [{"id": "relu", "op": "Relu", "inputs": ["x"], "output": "y"}],
                 "outputs": ["y"]}
        full = load_profile("fpga_zu15eg_full")
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "gpu"
            manifest = compile_bundle(graph, full, path, device_policy="gpu-only")
            backend = manifest["backends"]["gpu"]
            self.assertEqual((backend["warps"], backend["launch_warps"]), (8, 4))
            self.assertEqual(backend["dispatch_abi"], "z15-mmio-v1")
            jobs = backend["stages"][0]["jobs"]
            self.assertEqual([job["warp"] for job in jobs], [0,1,2,3,0,1,2,3,0])
            self.assertEqual([job["wave"] for job in jobs], [0,0,0,0,1,1,1,1,2])
            self.assertTrue(all(job["gpu_job_reg"] >> 26 == 0 for job in jobs))
            self.assertEqual(inspect_yhb((path / "program.yhb").read_bytes())["target"], 2)
