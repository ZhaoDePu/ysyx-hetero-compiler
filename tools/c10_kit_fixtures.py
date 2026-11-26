"""Stage the four C10 board cases and exact-clock C9 comparison rows offline."""
import argparse
import csv
import json
from pathlib import Path
import shutil

from hetero_compiler.compiler import tensor_bytes
from hetero_compiler.tinylm import PITCH, PROMPT, compile_tinylm, read_onnx
from hetero_compiler.yhb import inspect_yhb
from runtime.verify_z15_abi import corrupt_words
from tools.run_c5_tinylm import fixed_reference

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = ROOT.parent / "artifacts"
SCRATCH = Path("/dev/shm/ysyx_compiler_c10")
C7 = ARCHIVES / "compiler-phase-c7-yolo26n-fast-20261126-140127"
C8 = ARCHIVES / "compiler-phase-c8-yolo26n-20261126-162320"
C9 = ARCHIVES / "compiler-phase-c9-iccd-20261126-174150"
PROFILES = ((125, 100), (200, 150))
MODELS = (
    ("yolo26n", "YOLO26n", "C8"),
    ("tinylm", "TinyLM-Chat", "A3"),
    ("resnet18", "ResNet-18", "A3"),
    ("mobilenetv2", "MobileNetV2", "A3"),
)
COMPARISON_COLUMNS = (
    "model", "freq_profile", "unit", "work_units", "sim_wall_cycles", "sim_pl_ms",
    "sim_throughput", "sim_source", "board_status", "board_samples",
    "board_wall_cycles", "board_pl_ms", "board_host_ms", "board_execute_host_ms",
    "board_ps_outside_exec_ms", "board_native_gpu_union_cycles",
    "board_native_vpu_busy_cycles", "board_gpu_busy_cycles", "board_native_gpu_overlap_cycles",
    "board_npu_busy_cycles", "board_vpu_busy_cycles", "board_dma_busy_cycles",
    "board_dma_bytes", "board_dma_descriptors", "board_throughput",
    "board_over_sim_wall_cycles", "board_vs_sim_wall_delta_pct",
    "board_wall_agreement_pct", "board_ps_outside_exec_pct",
)


class Case:
    def __init__(self, folder):
        self.folder = folder
        folder.mkdir(parents=True)
        self.rows = []

    def add(self, direction, frame, symbol, data):
        if not data:
            raise ValueError(f"empty binding {frame}/{symbol}")
        name = f"{frame}.{symbol}.bin"
        (self.folder / name).write_bytes(data)
        self.rows.append((direction, frame, symbol, name))

    def finish(self):
        with (self.folder / "io.tsv").open("w", newline="") as handle:
            csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(self.rows)


def stage_yolo(folder, vectors, bundle):
    case = Case(folder)
    shutil.copyfile(bundle, folder / "program.yhb")
    for frame in ("bus", "zidane"):
        rgb = (vectors / f"{frame}.rgb_nhwc_u8.bin").read_bytes()
        quant = (vectors / f"{frame}.input_nhwc_s8.bin").read_bytes()
        if len(rgb) != 320 * 320 * 3 or len(quant) != len(rgb):
            raise ValueError(f"not a 320x320 RGB fixture: {frame}")
        # C8 consumes packed NHWC4, with a zero fourth lane.
        packed = bytearray(320 * 320 * 4)
        for channel in range(3):
            packed[channel::4] = quant[channel::3]
        case.add("input", frame, "images", packed)
        (folder / f"{frame}.rgb").write_bytes(rgb)
        for index, (stride, prefix) in enumerate(
                (stride, prefix) for stride in (8, 16, 32) for prefix in ("boxes", "scores")):
            gold = (vectors / f"{frame}.pl_out_{index}.bin").read_bytes()
            channels = 4 if prefix == "boxes" else 80
            if len(gold) != (320 // stride) ** 2 * channels:
                raise ValueError(f"unexpected {frame} stride {stride} {prefix} layout")
            case.add("output", frame, f"{prefix}_stride{stride}", gold)
    case.finish()


def stage_classifier(folder, source, prefix, oracle):
    case = Case(folder)
    low = source / "cases" / f"{prefix}-A3-125"
    high = source / "cases" / f"{prefix}-A3-200"
    for name in ("program.yhb", "input-5.bin", "output-7.bin"):
        if (low / name).read_bytes() != (high / name).read_bytes():
            raise ValueError(f"{prefix}: clock-profile fixtures disagree for {name}")
    gold = oracle.read_bytes()
    if gold != (low / "output-7.bin").read_bytes() or len(gold) != 1000:
        raise ValueError(f"{prefix}: C9 replay and oracle disagree")
    shutil.copyfile(low / "program.yhb", folder / "program.yhb")
    case.add("input", "frame0", "images", (low / "input-5.bin").read_bytes())
    case.add("output", "frame0", "logits", gold)
    case.finish()


def stage_tinylm(folder, build, release):
    case = Case(folder)
    example = ROOT / "examples/c5_tinylm"
    training = json.loads((example / "training.json").read_text())
    prompts = training["prompts"]
    if len(prompts) != 3 or any(len(p) != PROMPT for p in prompts):
        raise ValueError("C9 TinyLM benchmark requires three four-token prompts")
    model = example / "TinyLM-Chat.onnx"
    manifest = compile_tinylm(model, build, prompts, phase="A3", vpu=True)
    workload = manifest["tinylm"]
    if not workload["effective_native_attention"] or workload["decode_steps"] != PITCH - PROMPT:
        raise ValueError("compiled TinyLM does not match C9 A3-on")
    reference = fixed_reference(read_onnx(model)[0], prompts, True)
    if reference.keys() != manifest["output_bindings"].keys():
        raise ValueError("TinyLM compiled output symbols differ from the independent reference")
    shutil.copyfile(build / "program.yhb", folder / "program.yhb")
    for lane, tokens in enumerate(prompts):
        case.add("input", "prompt0", f"prompt_{lane}", tensor_bytes(tokens, "int64"))
    for name, binding in manifest["output_bindings"].items():
        data = tensor_bytes(reference[name], binding["dtype"])
        if len(data) != binding["size"]:
            raise ValueError(f"TinyLM reference has a wrong size for {name}")
        case.add("output", "prompt0", name, data)
    if len(reference) != 78 or sum(v["size"] for v in manifest["output_bindings"].values()) != 3096:
        raise ValueError("unexpected TinyLM C9 output coverage")
    timing = release / "m3/frequency-replays/tinylm/A3-125-100/c9-metric.json"
    if not timing.is_file():
        raise ValueError("C9 TinyLM provenance receipt is missing")
    (folder / "prompts.json").write_text(json.dumps({
        "prompts": prompts,
        "prompt_texts": [" ".join(training["vocab"][token] for token in ids) for ids in prompts],
        "prefill_positions": PROMPT,
        "decode_steps": PITCH - PROMPT,
        "gold_tokens": [[reference[f"p{lane}_token_{i}"][0] for i in range(PITCH - PROMPT + 1)]
                        for lane in range(len(prompts))],
        "c9_rtl_receipt": str(timing),
    }, ensure_ascii=False, indent=2) + "\n")
    case.finish()


def stage_rejections(folder, model):
    folder.mkdir()
    valid = model.read_bytes()
    dma = inspect_yhb(valid)["sections"]["dma"]
    if not dma["count"]:
        raise ValueError("negative fixture requires a checked DMA descriptor")
    damaged = bytearray(valid)
    damaged[-1] ^= 1
    (folder / "bad_crc.yhb").write_bytes(damaged)
    corrupt_words(valid, folder, "bad_dma", dma["offset"], {3: 0})


def stage_comparison(folder, c9):
    available = json.loads((c9 / "m3/iccd_master_metrics.json").read_text())["rows"]
    records = []
    for model, workload, variant in MODELS:
        for npu, gpu in PROFILES:
            matches = [r for r in available if (r["workload"], r["stage"], r["npu_mhz"],
                                                r["gpu_mhz"]) == (workload, variant, npu, gpu)]
            if len(matches) != 1:
                raise ValueError(f"expected one C9 exact-clock row for {model} {npu}/{gpu}")
            row = matches[0]
            if row["measurement_kind"] != "rtl_end_to_end" or not row["exact"]:
                raise ValueError(f"unexpected C9 measurement scope for {model}")
            wall = row["wall_cycles"]
            units = row["work_units"]
            throughput = units * npu * 1_000_000 / wall
            records.append(dict(model=model, freq_profile=f"{npu}_{gpu}", unit=row["unit"],
                                work_units=units, sim_wall_cycles=wall,
                                sim_pl_ms=f"{wall / npu / 1000:.6f}",
                                sim_throughput=f"{throughput:.6f}", sim_source=row["source"],
                                board_status="not_run"))
    for name, columns in (("sim_baselines.csv", COMPARISON_COLUMNS[:8]),
                          ("sim_vs_board_comparison.csv", COMPARISON_COLUMNS)):
        with (folder / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, columns, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="fresh kit root under C10 scratch")
    parser.add_argument("--build", type=Path, required=True, help="fresh TinyLM bundle root under C10 scratch")
    args = parser.parse_args()
    out, build = args.out.resolve(), args.build.resolve()
    if any(SCRATCH not in path.parents or path.exists() for path in (out / "models", build)):
        parser.error("staging needs fresh model/build directories inside the C10 scratch tree")
    models = out / "models"
    stage_yolo(models / "yolo26n", C7 / "m2/vectors", C8 / "m3/yolo26n_320_int8.yhb")
    stage_tinylm(models / "tinylm", build, C9)
    source = C9 / "m2"
    stage_classifier(models / "resnet18", source, "resnet", source / "resnet/64/logits.bin")
    stage_classifier(models / "mobilenetv2", source, "mobilenet", source / "mobilenet/64/gold-logits.bin")
    stage_rejections(models / "reject", models / "tinylm/program.yhb")
    stage_comparison(out, C9)


if __name__ == "__main__":
    main()
