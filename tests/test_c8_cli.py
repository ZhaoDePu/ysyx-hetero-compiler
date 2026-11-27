"""Verify the YOLO CLI counter conversion without a board or an RTL replay."""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
COUNTER_FIXTURE = r"""
#include "ysyx_yolo26n_api.h"

struct ysyx_yolo26n_ctx { int sim, phase17; };
static struct ysyx_yolo26n_ctx ctx;

ysyx_yolo26n_ctx_t *ysyx_yolo26n_open(const char *path, int sim)
{
    (void)path;
    ctx.sim = sim;
    ctx.phase17 = 0;
    return &ctx;
}

ysyx_yolo26n_ctx_t *ysyx_yolo26n_open_p17(const char *path, int sim)
{
    ysyx_yolo26n_ctx_t *handle = ysyx_yolo26n_open(path, sim);
    handle->phase17 = 1;
    return handle;
}

int ysyx_yolo26n_infer_rgb320(ysyx_yolo26n_ctx_t *handle, const uint8_t *rgb,
                            float conf, ysyx_det_t *detections, int max_dets, ysyx_perf_t *perf)
{
    (void)rgb;
    (void)conf;
    (void)detections;
    (void)max_dets;
    *perf = (ysyx_perf_t){
        .wall_ticks = handle->phase17 ? 9129685 : 25000000,
        .hardware_counters_valid = !handle->sim,
        .io_and_compute_ms = 250.0,
        .estimated_ms_at_125_mhz = 200.0,
        .estimated_ms_at_164_555_mhz = 151.924897,
    };
    return 0;
}

void ysyx_yolo26n_close(ysyx_yolo26n_ctx_t *handle)
{
    (void)handle;
}
"""


@unittest.skipUnless(shutil.which("cc"), "a C99 compiler is required")
class YoloCliClockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        out = Path(directory.name)
        fixture = out / "counter_fixture.c"
        fixture.write_text(COUNTER_FIXTURE, encoding="ascii")
        cls.cli = out / "ysyx_yolo26n_cli"
        cls.rgb = out / "frame.rgb"
        cls.rgb.write_bytes(bytes(320 * 320 * 3))
        subprocess.run(
            ["cc", "-std=c99", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "include"),
             str(ROOT / "runtime/ysyx_yolo26n_cli.c"), str(fixture), "-o", str(cls.cli)],
            check=True,
        )

    def invoke(self, *options):
        return subprocess.run([self.cli, "fixture.yhb", "--rgb", self.rgb, *options],
                              capture_output=True, text=True)

    def result(self, *options):
        result = self.invoke(*options)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_default_uses_the_existing_json_schema(self):
        result = self.result()
        self.assertEqual(set(result), {"detections", "perf"})
        self.assertEqual(set(result["perf"]), {
            "wall_ticks", "npu_ticks", "gpu_ticks", "overlap_ticks", "dma_bytes", "dma_operations",
            "abi_error", "invalid_dma_descriptors", "hardware_counters_valid", "io_and_compute_ms",
            "estimated_ms_at_125_mhz", "estimated_ms_at_164_555_mhz",
        })

    def test_requested_frequency_adds_an_estimate_alongside_125_mhz(self):
        default = self.result()["perf"]
        perf = self.result("--npu-mhz", "200")["perf"]
        self.assertEqual(perf.pop("npu_mhz"), 200)
        self.assertEqual(perf.pop("estimated_ms_at_npu_mhz"), 125.0)
        self.assertEqual(perf.pop("estimated_fps_at_npu_mhz"), 8.0)
        self.assertEqual(perf.pop("estimated_fps_at_125_mhz"), 5.0)
        self.assertEqual(perf, default)

    def test_fractional_frequency_is_retained(self):
        perf = self.result("--npu-mhz", "199.9980015")["perf"]
        self.assertEqual(perf["npu_mhz"], 199.9980015)
        self.assertAlmostEqual(perf["estimated_ms_at_npu_mhz"], 125.001249, places=6)

    def test_unavailable_counters_produce_a_null_estimate(self):
        perf = self.result("--sim", "--npu-mhz", "200")["perf"]
        self.assertFalse(perf["hardware_counters_valid"])
        self.assertIsNone(perf["estimated_ms_at_npu_mhz"])
        self.assertIsNone(perf["estimated_fps_at_npu_mhz"])
        self.assertIsNone(perf["estimated_fps_at_125_mhz"])

    def test_explicit_phase17_uses_its_own_backend(self):
        perf = self.result("--target", "p17", "--sim")["perf"]
        self.assertEqual(perf["wall_ticks"], 9129685)
        self.assertFalse(perf["hardware_counters_valid"])
        self.assertEqual(self.invoke("--target", "old").returncode, 2)

    def test_invalid_frequency_fails_with_usage(self):
        for options in [("--npu-mhz",), *[("--npu-mhz", value) for value in
                                         ("0", "-1", "inf", "nan", "1e999", "200MHz", "")],
                        ("--npu-mhz", "125", "--npu-mhz", "200")]:
            with self.subTest(options=options):
                result = self.invoke(*options)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertIn("usage:", result.stderr)
