"""Regression tests for observed overlap and classifier spatial padding."""

import unittest

import numpy as np

from hetero_compiler.c9_spatial import pixel_map
from tools.c9_ledger import derive


class C9DataTest(unittest.TestCase):
    def test_wall_partition_respects_triple_overlap(self):
        row = dict(workload="example", variant="checked", source="rtl.log",
                   npu_mhz=125, gpu_mhz=100, wall_cycles=100, work_units=1,
                   measurement_kind="rtl_end_to_end", exact=True, error=0, slverr=0, invalid_dma=0,
                   native_busy_cycles=60, npu_busy_cycles=40, vpu_busy_cycles=20,
                   gpu_busy_cycles=25, dma_busy_cycles=20, native_gpu_overlap_cycles=10,
                   compute_dma_overlap_cycles=12, triple_overlap_cycles=3, logical_ops=1000, dma_bytes=128)
        data = derive(row)
        self.assertEqual([data[key] for key in ("single_compute_cycles", "native_gpu_without_dma_cycles",
                                                "dma_only_cycles", "compute_dma_overlap_cycles",
                                                "idle_from_union_cycles")], [56, 7, 8, 12, 17])
        self.assertEqual(data["gops_at_measured_pair"], 1.25)
        self.assertIsNone(data["latency_ms_200_150"])

    def test_pixel_map_uses_end_padding_without_moving_input_bytes(self):
        result = pixel_map(2, 2, 1, 1, 4, 3, 2, 1)
        np.testing.assert_array_equal(result, [[16, 16, 16, 16, 0, 4, 16, 8, 12]])


if __name__ == "__main__":
    unittest.main()
