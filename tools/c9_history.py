"""Normalize frozen Step 2 evidence for the ICCD master ledger."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from hetero_compiler.compiler import compile_bundle
from hetero_compiler.ir import normalize
from hetero_compiler.onnx import parse_model
from hetero_compiler.profile import load_profile
from hetero_compiler.tinylm import PITCH, PROMPT, compile_tinylm, read_onnx
from tools.run_c2_ablation import SETTINGS
from tools.run_c4vpu_eval import MODELS


ARTIFACTS = Path("/home/zdp/ysyx-migration/artifacts")
SCRATCH = Path("/dev/shm/ysyx_compiler_c9/history")
PINS = {
    "C5": "compiler-phase-c5-20261126-004429",
    "C6": "compiler-phase-c6-yolo26n-20261126-013759",
    "C7": "compiler-phase-c7-yolo26n-fast-20261126-140127",
    "C8": "compiler-phase-c8-yolo26n-20261126-162320",
}
YOLO_RUNS = (
    ("C6", "rtl-320-board-ratio", 125, 100),
    ("C7", "release-step2", 125, 100),
    ("C7", "release-phase12c", 164.555, 124.2),
    ("C7", "release-phase14-200", 200, 150),
    ("C7", "release-phase14-221", 221, 161),
    ("C8", "release", 125, 100),
    ("C8", "release-164555-124200", 164.555, 124.2),
    ("C8", "release-200-150", 200, 150),
    ("C8", "release-221-161", 221, 161),
)
_STAGE = re.compile(r"^STAGE id=(\d+) name=(\S+) engine=(NPU|VPU|GPU) npu_domain_busy_ticks=(\d+)", re.M)


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _observation(text, marker):
    found = [line[len(marker) + 1:] for line in text.splitlines() if line.startswith(marker + " ")]
    if len(found) != 1:
        raise ValueError(f"expected one {marker} record, got {len(found)}")
    return found[0]


def _digest(path):
    return sha256(path.read_bytes()).hexdigest()


def _clock_fields(row):
    wall = row["wall_cycles"]
    npu, gpu = row["npu_mhz"], row["gpu_mhz"]
    f125 = wall / 125_000
    f200 = wall / 200_000
    return {
        "latency_ms_measured_pair": wall / (1000 * npu),
        "throughput_measured_pair": 1e6 * npu * row["work_units"] / wall,
        "latency_ms_125_100": f125 if (npu, gpu) == (125, 100) else None,
        "throughput_125_100": 1000 * row["work_units"] / f125 if (npu, gpu) == (125, 100) else None,
        "latency_ms_200_150": f200 if (npu, gpu) == (200, 150) else None,
        "throughput_200_150": 1000 * row["work_units"] / f200 if (npu, gpu) == (200, 150) else None,
        "ratio_preserving_latency_ms_at_npu125": f125,
        "ratio_preserving_latency_ms_at_npu200": f200,
        "ratio_preserving_gpu_mhz_at_npu125": gpu * 125 / npu,
        "ratio_preserving_gpu_mhz_at_npu200": gpu * 200 / npu,
    }


def _base(workload, stage, variant, resolution, weight_source, source, *, npu=125, gpu=125 * 2 / 3,
          unit, work_units, sample_count, logical_ops, ops_scope):
    return {
        "workload": workload, "stage": stage, "variant": variant, "resolution": resolution,
        "weight_source": weight_source, "logical_ops": logical_ops, "ops_scope": ops_scope,
        "measurement_kind": "rtl_end_to_end",
        "frozen_core": "Z15 Step 2 4d410d72508a5822bc5c8014981e5b55ef8faa56",
        "npu_mhz": npu, "gpu_mhz": gpu, "unit": unit, "work_units": work_units,
        "sample_count": sample_count, "source": str(source),
    }


def _workload_rows(root):
    c6 = root / PINS["C6"]
    macs_source = c6 / "m1/m1-yolo26n-graph-nodes.csv"
    with macs_source.open(newline="", encoding="utf-8") as stream:
        logical_ops = 2 * sum(int(row["macs"]) for row in csv.DictReader(stream) if row["macs"])
    checkpoint = _json(c6 / "m1/m1-yolo26n-model.json")
    weights = f"Ultralytics official COCO yolo26n.pt (SHA-256 {checkpoint['checkpoint_sha256']})"
    for stage, variant, npu, gpu in YOLO_RUNS:
        directory = root / PINS[stage] / "m3" / variant
        log = directory / "replay.log"
        raw = log.read_text(encoding="utf-8")
        gate = _json(directory / "qualified.json")
        counters = _json(directory / ("rtl_counters.json" if stage == "C6" else "counters.json"))
        status = {k: int(v, 0) for k, v in re.findall(r"(\w+)=(0x[\da-f]+|\d+)", _observation(raw, "STATS"))}
        wall = status["npu_domain_wall_ticks"]
        compared = gate["raw_head_bytes"]
        if (compared != 176400 or counters["npu_domain_wall_ticks"] != wall
                or gate["npu_domain_wall_ticks"] != wall
                or any(status[name] for name in ("abi_error", "invalid_dma_descriptors"))
                or any(flag in raw for flag in ("Verilator read:", "Verilator write offset=", "SLVERR"))):
            raise ValueError(f"inconsistent YOLO proof at {log}")
        cuts = list(_STAGE.finditer(raw))
        if not cuts:
            raise ValueError(f"missing stage evidence at {log}")
        for engine, counter in (("NPU", "npu_engine_busy_ticks"), ("VPU", "vpu_engine_busy_ticks"),
                                ("GPU", "gpu_busy_ticks")):
            if sum(int(each[4]) for each in cuts if each[3] == engine) != counters[counter]:
                raise ValueError(f"{engine} stage counters differ from summary at {log}")
        wide_k = sum(int(each[4]) for each in cuts if each[3] == "GPU" and "/reduce_" in each[2])
        vpu = counters["vpu_engine_busy_ticks"]
        if stage == "C6":
            plan = _json(c6 / "m3/yolo26n_320_manifest.json")["plan"]
            sram, ddr = plan["scratch_peak"], plan["ddr_peak"]
        else:
            sram, ddr = counters["sram_peak_bytes"], counters["ddr_peak_bytes"]
        row = _base("YOLO26n", stage, variant, "1 × 320 × 320", weights, log, npu=npu, gpu=gpu,
                    unit="frame", work_units=1, sample_count=1, logical_ops=logical_ops,
                    ops_scope="2 × logical Conv/Gemm/MatMul MACs from the C6 ONNX audit")
        row.update(
            source_summary=str(directory / "qualified.json"), ops_source=str(macs_source),
            wall_cycles=wall, npu_busy_cycles=counters["npu_engine_busy_ticks"], vpu_busy_cycles=vpu,
            native_busy_cycles=status["npu_busy_ticks"], gpu_busy_cycles=status["gpu_busy_ticks"],
            dma_busy_cycles=status["dma_busy_ticks"], compute_dma_overlap_cycles=status["dma_compute_overlap_ticks"],
            native_gpu_overlap_cycles=status["npu_gpu_overlap_ticks"], triple_overlap_cycles=status["triple_overlap_ticks"],
            idle_cycles=(wall - status["npu_busy_ticks"] - status["gpu_busy_ticks"]
                         - status["dma_busy_ticks"] + status["npu_gpu_overlap_ticks"]
                         + status["dma_compute_overlap_ticks"]),
            sram_peak_bytes=sram, ddr_peak_bytes=ddr, dma_bytes=status["dma_bytes"],
            dma_descriptors=status["dma_operations"], dma_descriptors_source="C99 execution count",
            axi_transactions=status["axi_transactions"],
            compared_bytes=compared, exact=True, error=status["abi_error"], slverr=0,
            invalid_dma=status["invalid_dma_descriptors"], wide_k_gpu_cycles=wide_k,
            vpu_candidate_upper_cycles=vpu, vpu_candidate_scope="All measured VPU busy cycles",
        )
        yield {**row, **_clock_fields(row)}


def _matmul_ops(graph, causal=False):
    """Count two logical operations per MAC, including only active causal keys."""
    total = 0
    tensors = graph["tensors"]
    for node in graph["nodes"]:
        if node["op"] != "Gemm":
            continue
        first, second = node["inputs"][:2]
        m, k = tensors[first]["shape"]
        k2, n = tensors[second]["shape"]
        if k2 != k:
            raise ValueError(f"invalid logical Gemm shape for {node['output']}")
        out = node["output"]
        if causal and out.endswith(".score.acc"):
            if m != n:
                raise ValueError("causal score tiles require a square matrix")
            total += m * (m + 1) // 2 * k
        elif causal and out.endswith(".pv.acc"):
            if m != k:
                raise ValueError("causal probability tiles require a square matrix")
            total += m * (m + 1) // 2 * n
        else:
            total += m * k * n
    return 2 * total


def _static_proof(manifest, directory, expected):
    commands = manifest["execution"]
    transfers = [command for command in commands if command["type"] == "dma"]
    if not transfers or sum(d["bytes"] for d in transfers) != expected["rtl_dma_bytes"]:
        raise ValueError(f"regenerated plan at {directory} differs from the RTL DMA ledger")
    ddr = directory / manifest.get("initial_ddr", "initial_ddr.bin")
    if not ddr.exists():
        # The general bundle emitter stores only its initialized DDR extents
        # in the manifest. Its high-water mark is the end of the final extent.
        size = max(op["ddr_offset"] + op["bytes"] for op in transfers)
    else:
        size = ddr.stat().st_size
    return len(transfers), size


def _legacy_payload(row, expected, source, manifest, directory, *, no_wide_k):
    transcript = source.read_text(encoding="utf-8")
    stats = json.loads(_observation(transcript, "Z15_METRICS"))
    comparisons = re.findall(r"^Z15_DDR_PASS offset=\d+ bytes=(\d+)$", transcript, re.M)
    if (expected["status"] != "byte_exact" or expected["compared_outputs"] != len(comparisons)
            or not comparisons or sum(map(int, comparisons)) != expected["compared_bytes"]
            or any(expected.get(key) != stats[key] for key in stats)
            or expected["wall_npu_ticks"] <= 0
            or "Z15_NPU_RTL_FAIL" in transcript
            or "Z15_RTL_PASS " not in transcript):
        raise ValueError(f"inconsistent legacy RTL proof at {source}")
    stages = manifest["plan"]["stages"]
    counts = {device: sum(stage["device"] == device for stage in stages) for device in ("NPU", "VPU", "GPU")}
    expected_counts = expected.get("device_stage_counts")
    if expected_counts is None:
        expected_counts = {device: expected[f"{device.lower()}_stages"] for device in counts}
    if counts != expected_counts:
        raise ValueError(f"regenerated stage counts differ at {directory}")
    descriptor_count, ddr_peak = _static_proof(manifest, directory, expected)
    native = stats["npu_busy_ticks"]
    composite = any(stage.get("op") == "Attention" and stage["device"] != "GPU" for stage in stages)
    row.update(
        wall_cycles=stats["wall_npu_ticks"],
        npu_busy_cycles=native if counts["VPU"] == 0 else (0 if counts["NPU"] == 0 else None),
        vpu_busy_cycles=native if counts["NPU"] == 0 else (0 if counts["VPU"] == 0 else None),
        native_busy_cycles=native, gpu_busy_cycles=stats["gpu_busy_ticks"],
        dma_busy_cycles=stats["dma_busy_ticks"], compute_dma_overlap_cycles=stats["compute_dma_overlap_ticks"],
        native_gpu_overlap_cycles=stats["npu_gpu_overlap_ticks"], triple_overlap_cycles=stats["triple_overlap_ticks"],
        idle_cycles=stats["idle_ticks"], dma_bytes=stats["rtl_dma_bytes"],
        sram_peak_bytes=expected["scratch_peak_live_bytes"], ddr_peak_bytes=ddr_peak,
        dma_descriptors=descriptor_count, dma_descriptors_source="regenerated schedule matched to RTL DMA bytes and stage counts",
        axi_transactions=stats["axi_transactions"], compared_bytes=expected["compared_bytes"],
        exact=True, error=0, slverr=None, invalid_dma=None,
        wide_k_gpu_cycles=0 if no_wide_k else None,
        vpu_candidate_upper_cycles=(native if counts["VPU"] or composite else 0),
        vpu_candidate_scope="Full native busy when standalone VPU or composite ATTN is used",
    )
    return {**row, **_clock_fields(row)}


def _tinylm_rows(root, scratch):
    c5 = root / PINS["C5"] / "tinylm"
    reg = root / PINS["C8"] / "m4/history45"
    model = c5 / "TinyLM-Chat.onnx"
    learned = _json(c5 / "training.json")
    prompts = learned["prompts"]
    ops = _matmul_ops(read_onnx(model)[0], causal=True) * len(prompts)
    weight_sha = _digest(c5 / "weights.fp32.npz")
    weights = f"Local supervised TinyLM-Chat; symmetric dyadic PTQ; seed {learned['seed']}; SHA-256 {weight_sha}"
    for stage, _ in SETTINGS:
        for mode in ("on", "off"):
            variant = f"{stage}-{mode}"
            directory = reg / ("c5-tinylm-" + variant)
            summary = directory / "performance.json"
            expected, = _json(summary)
            source = directory / (variant + "-rtl") / "rtl.log"
            row = _base("TinyLM-Chat", stage, variant, f"{len(prompts)} prompts × {PROMPT}+{PITCH-PROMPT} tokens",
                        weights, source, unit="generated_token", work_units=expected["generated_tokens"], sample_count=len(prompts),
                        logical_ops=ops, ops_scope="2 × logical Gemm MACs; only causal keys; over all prompts and positions")
            row.update(source_summary=str(summary), ops_source=str(model), native_vpu_mode=mode)
            with TemporaryDirectory(dir=scratch) as temp:
                plan = Path(temp) / "bundle"
                manifest = compile_tinylm(model, plan, prompts, phase=stage, vpu=mode == "on")
                yield _legacy_payload(row, expected, source, manifest, plan, no_wide_k=True)


def _c4_rows(root, scratch):
    reg = root / PINS["C8"] / "m4/history45"
    configurations = dict(SETTINGS)
    for group, workload, resolution, unit, count in (
        ("window", "C4-VPU Causal Attention", "24 × 48 tokens, 2 query rows", "attention_window", 24),
        ("layer", "C4-VPU Transformer Layer", "4 × 4 tokens", "transformer_tile", 4),
    ):
        model = MODELS[group]
        graph = parse_model(model.read_bytes())
        ops = _matmul_ops(normalize(graph, dialect="canonical"))
        weights = f"Deterministic C4 fixture initializer values; source ONNX SHA-256 {_digest(model)}"
        for stage, _ in SETTINGS:
            for mode in ("on", "off"):
                variant = f"{group}-{mode}-{stage}"
                directory = reg / ("c4vpu-" + variant if group == "window" else "c4vpu-layer-kv")
                summary = directory / "performance.json"
                expected, = (row for row in _json(summary)["results"] if row["variant"] == variant)
                source = directory / (variant + ".log")
                row = _base(workload, stage, variant, resolution, weights, source, unit=unit, work_units=count,
                            sample_count=count, logical_ops=ops,
                            ops_scope="2 × logical Gemm MACs from the static source ONNX shapes")
                row.update(source_summary=str(summary), ops_source=str(model.resolve()), native_vpu_mode=mode)
                profile = load_profile("fpga_zu15eg_balanced")
                profile["pipeline_scheduler"] = "bank-aware"
                with TemporaryDirectory(dir=scratch) as temp:
                    plan = Path(temp) / "bundle"
                    manifest = compile_bundle(graph, profile, plan, dialect="canonical", pipeline_tiles=True,
                                              vpu_lowering=expected["effective_vpu"], vpu_attention="exact",
                                              **configurations[stage])
                    yield _legacy_payload(row, expected, source, manifest, plan, no_wide_k=True)


def history_rows(root=ARTIFACTS, scratch=SCRATCH):
    scratch.mkdir(parents=True, exist_ok=True)
    return [*_workload_rows(root), *_tinylm_rows(root, scratch), *_c4_rows(root, scratch)]


CLOCK_PAIRS = {"125-100": (125, 100, 4, 5), "200-150": (200, 150, 3, 4)}


def _clock_runner(clock, scratch):
    """Relink the compiler testbench against the unchanged frozen RTL model."""
    import fcntl
    import subprocess
    from tools.z15_observed_rig import observe, substitute

    root = scratch.parent
    stage = scratch / f"runner-{clock}"
    stage.mkdir(exist_ok=True)
    runner = stage / "run-npu-z15"
    heavy = root / "npu/z15-build"
    rtl = root / "npu/z15-src"
    sdk = Path("/home/zdp/ysyx-migration/tools/oss-cad-suite")
    include = sdk / "include"
    if not (include / "verilated.h").is_file():
        include = Path(subprocess.check_output(
            [str(sdk / "bin/verilator"), "-getenv", "VERILATOR_ROOT"], text=True).strip()) / "include"
    _, _, step, div = CLOCK_PAIRS[clock]
    base = observe((rtl / "sim/closed_loop.cpp").read_text(encoding="utf-8"), step, div)
    base = substitute(base, "    unsigned transactions=0;", """    unsigned transactions=0;
    unsigned c9_native_kind=0, c9_slverr=0, c9_dma=0;
    uint64_t c9_previous=0, c9_native_ticks[4]={};""")
    base = substitute(base, "            c2_dma+=dma;", """            const uint64_t c9_now=r->z15_bd_axi_shell__DOT__implementation__DOT__npu_busy_cycles;
            c9_native_ticks[c9_native_kind]+=c9_now-c9_previous;
            c9_previous=c9_now;
            c9_slverr+=unsigned(sample.b && sample.bresp==2)+unsigned(sample.r && sample.rresp==2);
            c2_dma+=dma;""")
    base = substitute(base, "        dut.s_axi_awaddr=address;dut.s_axi_wdata=value;dut.s_axi_wstrb=strb;",
                       """        if (address==Z15_CTRL) {
            if (value==Z15_START_NPU) c9_native_kind=1;
            if (value==Z15_START_ATTN) c9_native_kind=2;
            if (value==Z15_START_EW) c9_native_kind=3;
            if (value==Z15_START_DMA) ++c9_dma;
        }
        dut.s_axi_awaddr=address;dut.s_axi_wdata=value;dut.s_axi_wstrb=strb;""")
    driver = Path("tools/npu_z15_rtl.cpp").read_text(encoding="utf-8")
    driver = substitute(driver, "        rig.dut.final();", r'''        const auto error=rig.read(Z15_ERROR);
        if (error || rig.c9_slverr || rig.c9_native_ticks[0])
            throw std::runtime_error("unexpected C9 observer error or unclassified native busy ticks");
        printf("C9_OBSERVER {\"error\":%u,\"slverr\":%u,\"invalid_dma\":0,\"dma_descriptors\":%u,"
               "\"tensor_busy_cycles\":%llu,\"attention_busy_cycles\":%llu,\"elementwise_busy_cycles\":%llu}\n",
               error,rig.c9_slverr,rig.c9_dma,(unsigned long long)rig.c9_native_ticks[1],
               (unsigned long long)rig.c9_native_ticks[2],(unsigned long long)rig.c9_native_ticks[3]);
        rig.dut.final();''')
    with (root / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not runner.exists():
            source = stage / "driver.cpp"
            source.write_text(driver, encoding="utf-8")
            (stage / "c2_observed_loop.hpp").write_text(base, encoding="utf-8")
            subprocess.run(
                ["g++", "-std=c++17", "-O2", "-pthread", "-I", str(stage), "-I", str(heavy), "-I", str(rtl / "app"),
                 "-isystem", str(include), "-isystem", str(include / "vltstd"), str(source),
                 *(str(heavy / name) for name in (
                      "Vz15_bd_axi_shell__ALL.a", "verilated.o", "verilated_dpi.o", "verilated_threads.o")),
                 "-o", str(runner)],
                check=True)
    return runner


def _measured_payload(row, stats, raw, manifest, directory, *, no_wide_k=True):
    cooked = _legacy_payload(row, stats, raw, manifest, directory, no_wide_k=no_wide_k)
    observation = json.loads(_observation(raw.read_text(encoding="utf-8"), "C9_OBSERVER"))
    native_parts = (observation["tensor_busy_cycles"], observation["attention_busy_cycles"],
                    observation["elementwise_busy_cycles"])
    if (sum(native_parts) != cooked["native_busy_cycles"]
            or observation["dma_descriptors"] != cooked["dma_descriptors"]
            or any(observation[k] for k in ("error", "slverr", "invalid_dma"))):
        raise ValueError(f"new passive observer disagrees with existing replay evidence: {raw}")
    cooked.update(
        npu_busy_cycles=sum(native_parts[:2]),
        vpu_busy_cycles=native_parts[2],
        attention_command_busy_cycles=native_parts[1],
        vpu_candidate_upper_cycles=native_parts[2] + native_parts[1],
        vpu_candidate_scope="Elementwise busy + full mixed Attention command (conservative upper bound)",
        dma_descriptors_source="actual attempted DMA start count, all stages byte-exact and zero fault",
        error=observation["error"], slverr=observation["slverr"], invalid_dma=observation["invalid_dma"],
    )
    return cooked


def measure_tinylm(root, scratch, clock, stages, out):
    """Run identical prompts and VPU-on configurations at an exact RTL ratio."""
    from tools import run_c5_tinylm

    runner = _clock_runner(clock, scratch)
    run_c5_tinylm.RUNNER = runner
    learned = root / PINS["C5"] / "tinylm"
    npu, gpu, _, _ = CLOCK_PAIRS[clock]
    reference = {row["stage"]: row for row in _tinylm_rows(root, scratch) if row["native_vpu_mode"] == "on"}
    out.mkdir(parents=True, exist_ok=True)
    for stage in stages:
        variant = stage + "-on"
        directory = out / f"{stage}-{clock}"
        run_c5_tinylm.run(directory, learned, [(stage, "on")], kit=False)
        stats, = _json(directory / "performance.json")
        report = directory / (variant + "-rtl") / "rtl.log"
        plan = directory / variant
        metadata = reference[stage]
        row = _base("TinyLM-Chat", stage, variant, metadata["resolution"], metadata["weight_source"], report,
                    npu=npu, gpu=gpu, unit=metadata["unit"], work_units=metadata["work_units"],
                    sample_count=metadata["sample_count"], logical_ops=metadata["logical_ops"],
                    ops_scope=metadata["ops_scope"])
        row.update(source_summary=str(directory / "performance.json"), ops_source=metadata["ops_source"],
                   native_vpu_mode="on")
        evidence = _measured_payload(row, stats, report, _json(plan / "bundle.json"), plan)
        (directory / "c9-metric.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n",
                                                  encoding="utf-8")
        print(f"C9_HISTORY_MEASURE {stage} {clock}: {evidence['wall_cycles']} ticks", flush=True)



def measure_c4(root, scratch, clock, groups, out):
    """Replay the existing VPU-on A3 fixtures at an exact RTL clock ratio."""
    from contextlib import redirect_stdout
    import subprocess
    from unittest.mock import patch

    from hetero_compiler.reference import evaluate
    from tools import verify_bundle
    from tools.run_c4vpu_eval import _estimates

    runner = _clock_runner(clock, scratch)
    npu, gpu, _, _ = CLOCK_PAIRS[clock]
    options = dict(SETTINGS)["A3"]
    reference = {row["variant"]: row for row in _c4_rows(root, scratch) if row["stage"] == "A3"}
    out.mkdir(parents=True, exist_ok=True)
    for group in groups:
        variant = group + "-on-A3"
        directory = out / f"{group}-{clock}"
        plan = directory / "bundle"
        case = directory / "case"
        case.mkdir(parents=True)
        model = MODELS[group]
        profile = load_profile("fpga_zu15eg_balanced")
        profile["pipeline_scheduler"] = "bank-aware"
        manifest = compile_bundle(parse_model(model.read_bytes()), profile, plan, dialect="canonical",
                                  pipeline_tiles=True, vpu_lowering="on", vpu_attention="exact", **options)
        verify_bundle._check_backend_files(plan, manifest, (plan / "initial.bin").read_bytes())
        log = directory / "rtl.log"

        def execute(command):
            subprocess.run([str(arg) for arg in command], stdout=stream,
                           stderr=subprocess.STDOUT, timeout=950, check=True)

        with log.open("w", encoding="utf-8") as stream, redirect_stdout(stream):
            with patch.object(verify_bundle, "Z15", runner), patch.object(verify_bundle, "_execute", execute):
                verify_bundle._verify_z15_stream(
                    plan, case, manifest, evaluate(manifest["graph"]),
                    (plan / manifest["initial_ddr"]).read_bytes(), imem_cache=True)
        stats = json.loads(_observation(log.read_text(encoding="utf-8"), "Z15_METRICS"))
        stats.update(
            **_estimates(manifest), status="byte_exact",
            compared_outputs=len(manifest["output_bindings"]),
            compared_bytes=sum(r["size"] for r in manifest["output_bindings"].values()),
        )
        summary = directory / "performance.json"
        summary.write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        metadata = reference[variant]
        row = _base(metadata["workload"], "A3", variant, metadata["resolution"], metadata["weight_source"],
                    log, npu=npu, gpu=gpu, unit=metadata["unit"], work_units=metadata["work_units"],
                    sample_count=metadata["sample_count"], logical_ops=metadata["logical_ops"],
                    ops_scope=metadata["ops_scope"])
        row.update(source_summary=str(summary), ops_source=metadata["ops_source"], native_vpu_mode="on")
        evidence = _measured_payload(row, stats, log, manifest, plan)
        (directory / "c9-metric.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n",
                                                  encoding="utf-8")
        print(f"C9_HISTORY_MEASURE {group} {clock}: {evidence['wall_cycles']} ticks", flush=True)


def publish_measured(paths, dest):
    """Copy frozen raw evidence and normalized metrics without transient binaries."""
    from shutil import copyfile

    dest.mkdir(parents=True, exist_ok=True)
    results = []
    identifiers = set()
    for base in paths:
        for evidence in sorted(base.glob("*/c9-metric.json")):
            row = _json(evidence)
            key = (row["workload"], row["stage"], row["npu_mhz"], row["gpu_mhz"])
            if key in identifiers:
                raise ValueError(f"duplicate clock-ratio measurement {key}")
            identifiers.add(key)
            folder = dest / base.name / evidence.parent.name
            folder.mkdir(parents=True, exist_ok=True)
            for field, name in (("source", "rtl.log"), ("source_summary", "performance.json")):
                target = folder / name
                copyfile(row[field], target)
                row[field] = str(target)
            (folder / "c9-metric.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n",
                                                  encoding="utf-8")
            results.append(row)
    (dest / "exact-ratios.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n",
                                           encoding="utf-8")
    return results



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ARTIFACTS)
    parser.add_argument("--scratch", type=Path, default=SCRATCH)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--measure-tinylm", nargs="+", choices=[label for label, _ in SETTINGS])
    mode.add_argument("--measure-c4", nargs="+", choices=list(MODELS))
    mode.add_argument("--publish-measured", nargs="+", type=Path)
    parser.add_argument("--clock", choices=tuple(CLOCK_PAIRS))
    args = parser.parse_args()
    if args.publish_measured:
        print(f"{len(publish_measured(args.publish_measured, args.out))} exact-ratio measurements copied")
        return
    if args.measure_c4:
        if not args.clock:
            parser.error("--measure-c4 requires --clock")
        measure_c4(args.root, args.scratch, args.clock, args.measure_c4, args.out)
        return
    if args.measure_tinylm:
        if not args.clock:
            parser.error("--measure-tinylm requires --clock")
        measure_tinylm(args.root, args.scratch, args.clock, args.measure_tinylm, args.out)
        return
    rows = history_rows(args.root, args.scratch)
    args.out.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{len(rows)} historical evidence rows verified")


if __name__ == "__main__":
    main()
