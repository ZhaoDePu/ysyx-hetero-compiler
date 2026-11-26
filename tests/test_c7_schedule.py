"""Owner and DDR dependency checks for overlapping Z15 tiles."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from hetero_compiler.c6_gpu import build_gpu_stage
from hetero_compiler.c6_memory import Allocation, HALF_BYTES
from hetero_compiler.c6_schedule import emit_program
from hetero_compiler.c6_tensor import NativeStage, Tile, Transfer


def native(name, zone, load, store):
    start = zone * HALF_BYTES
    stage = NativeStage(name, "VPU", ((0xA0, start), (0xA8, start + 32),
                                      (0xB0, 32), (0xBC, 8), (0x10, 64)))
    return Tile(zone, (Transfer("load", load, start, 32),), (stage,),
                (Transfer("store", store, start + 32, 32),), 64)


def gpu(name, zone, load, store):
    start = zone * HALF_BYTES

    def kernel(p):
        p.load(6, 5, "int8")
        p.store(6, 5, "int8", 32)

    stage = build_gpu_stage(name, start // 4096, 32, kernel)
    return Tile(zone, (Transfer("load", load, start, 32),), (stage,),
                (Transfer("store", store, start + 32, 32),), 64)


def fused(name, zone, load, store):
    source = gpu(f"{name}/sum", zone, load, store)
    start = zone * HALF_BYTES
    raw = NativeStage(f"{name}/raw", "NPU", (
        (0x60, 1 | 32 << 16), (0x64, 4 | 128 << 16),
        (0x68, start), (0x6C, start + 32), (0x70, start + 160), (0x74, 0),
        (0x78, 1 << 9 | 1 << 13 | 1 << 14), (0x7C, 1 | 1 << 9 | 4 << 18),
        (0x80, 1 << 16), (0x84, 0), (0x10, 8)))
    epilogue = NativeStage(f"{name}/epilogue", "NPU", (
        (0xC4, 0), (0xC8, 1), (0x10, 128),
        (0x60, 1 | 32 << 16), (0x64, 4 | 16 << 16),
        (0x68, start + 288), (0x6C, start + 292), (0x70, start + 32),
        (0x74, start + 160), (0x78, 1 << 1 | 1 << 2 | 1 << 15 | 1 << 17),
        (0x7C, 0), (0x80, 0), (0x84, 0), (0x10, 8)))
    return replace(source, stages=(raw, *source.stages, epilogue), scratch_bytes=448)


class C7ScheduleTests(unittest.TestCase):
    def emit(self, layers, *, overlap, **options):
        with tempfile.TemporaryDirectory() as temp:
            slots = {name: Allocation(at, 32) for name, at in zip("abcde", range(0, 160, 32))}
            manifest = emit_program(Path(temp), layers, {}, [], [], slots,
                                    {name: bytes(32) for name in slots}, overlap=overlap, **options)
        self.assertLessEqual(manifest["plan"]["scratch_peak"], 2 * HALF_BYTES)
        return manifest["execution"]

    def test_independent_other_half_can_launch_before_prior_fence(self):
        layers = ((native("a", 0, 0, 64),), (gpu("b", 1, 32, 96),))
        fast = self.emit(layers, overlap=True)
        baseline = self.emit(layers, overlap=False)
        launch = {"type": "launch", "stage": "b"}
        fence = {"type": "fence", "wait_for": "a"}
        self.assertLess(fast.index(launch), fast.index(fence))
        self.assertGreater(baseline.index(launch), baseline.index(fence))

    def test_shared_ddr_requires_the_producer_dma_fence(self):
        layers = ((native("a", 0, 0, 64),), (gpu("b", 1, 64, 96),))
        sequence = self.emit(layers, overlap=True)
        store = next(i for i, step in enumerate(sequence)
                     if step.get("direction") == "store" and step["ddr_offset"] == 64)
        load = next(i for i, step in enumerate(sequence)
                    if step.get("direction") == "load" and step["ddr_offset"] == 64)
        self.assertEqual(sequence[store + 1], {"type": "dma-fence"})
        self.assertLess(store + 1, load)

    def test_shared_half_remains_exclusive(self):
        layers = ((native("a", 0, 0, 64),), (gpu("b", 0, 32, 96),))
        sequence = self.emit(layers, overlap=True)
        self.assertLess(sequence.index({"type": "fence", "wait_for": "a"}),
                        sequence.index({"type": "launch", "stage": "b"}))

    def test_shared_tensor_configuration_keeps_native_launch_order(self):
        layers = (
            (gpu("prelude", 0, 0, 64),),
            (native("first", 0, 64, 96), native("second", 1, 32, 64)),
        )
        sequence = self.emit(layers, overlap=True)
        self.assertLess(sequence.index({"type": "fence", "wait_for": "first"}),
                        sequence.index({"type": "launch", "stage": "second"}))


    def test_padded_ddr_extent_holds_the_second_half(self):
        first = native("a", 0, 0, 64)
        first = Tile(0, first.loads, first.stages, (Transfer("store", 64, 32, 64),), 96)
        layers = ((first,), (gpu("b", 1, 96, 128),))
        sequence = self.emit(layers, overlap=True)
        a = sequence.index({"type": "fence", "wait_for": "a"})
        b = sequence.index({"type": "launch", "stage": "b"})
        self.assertLess(a, b)

    def test_misassigned_dma_is_rejected_before_scheduling(self):
        first = native("a", 0, 0, 96)
        first = Tile(0, (Transfer("load", 0, HALF_BYTES, 32),), first.stages, first.stores, 64)
        with self.assertRaisesRegex(ValueError, "DMA ownership half"):
            self.emit(((first,),), overlap=True)

    def test_cross_k_starts_other_half_raw_before_gpu_fence(self):
        layers = ((fused("a", 0, 0, 64), fused("b", 1, 32, 96)),)
        old = self.emit(layers, overlap=True)
        fast = self.emit(layers, overlap=True, stream_overlap=True)
        raw = {"type": "launch", "stage": "b/raw"}
        fence = {"type": "fence", "wait_for": "a/sum"}
        self.assertGreater(old.index(raw), old.index(fence))
        self.assertLess(fast.index(raw), fast.index(fence))
        self.assertLess(fast.index({"type": "fence", "wait_for": "a/epilogue"}),
                        fast.index({"type": "launch", "stage": "b/epilogue"}))

    def test_multiple_epilogues_preserve_their_order_after_raw_overlap(self):
        a, b = fused("a", 0, 0, 64), fused("b", 1, 32, 96)
        tail = replace(a.stages[-1], id="a/epilogue2", writes=a.stages[-1].writes[3:])
        a = replace(a, stages=(*a.stages, tail))
        fast = self.emit(((a, b),), overlap=True, stream_overlap=True)
        native = [cmd["stage"] for cmd in fast
                  if cmd["type"] == "launch" and not cmd["stage"].endswith("/sum")]
        self.assertEqual(native, ["a/raw", "b/raw", "a/epilogue", "a/epilogue2", "b/epilogue"])

    def test_cross_k_preserves_ddr_producer_fence(self):
        # A producer/consumer edge belongs between passes. Within a
        # pass, both halves read different slices of the same layer input.
        a, b = fused("a", 0, 0, 64), fused("b", 1, 64, 96)
        fast = self.emit(((a,), (b,)), overlap=True, stream_overlap=True)
        store = fast.index({"type": "dma", "direction": "store", "ddr_offset": 64,
                            "scratch_offset": 32, "bytes": 32})
        self.assertEqual(fast[store + 1], {"type": "dma-fence"})
        self.assertLess(store + 1, fast.index({"type": "launch", "stage": "b/raw"}))

    def test_cross_k_preserves_native_order_for_raw_stage_that_writes_table(self):
        a, b = fused("a", 0, 0, 64), fused("b", 1, 32, 96)
        raw, *tail = b.stages
        raw = replace(raw, writes=((0xC4, 0), (0xC8, 2), (0x10, 128), *raw.writes))
        b = replace(b, stages=(raw, *tail))
        old = self.emit(((a, b),), overlap=True)
        self.assertEqual(old, self.emit(((a, b),), overlap=True, stream_overlap=True))
