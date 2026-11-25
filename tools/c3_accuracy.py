"""Replay independent labeled FP32/INT8 model validation through the C99 Z15 runtime."""

import argparse
import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from examples.c3_training.export import quantize
from examples.c3_training.fp32_onnx import Fp32Onnx
from examples.c3_training.train import FMAX_MHZ, SHAPES, make_data
from hetero_compiler.compiler import compile_bundle, tensor_bytes
from hetero_compiler.onnx import parse_model
from hetero_compiler.profile import load_profile
from hetero_compiler.reference import evaluate
from hetero_compiler.yhb import inspect_yhb

SCRATCH = Path("/dev/shm/ysyx_compiler_c3/training")
RUNNER = SCRATCH.parent / "runtime/verilator/ysyx_run"
PRIORITY = ["nice", "-n", "10", "ionice", "-c2", "-n7"]
NUMPY_DTYPES = {"int8": "i1", "uint8": "u1", "int32": "<i4", "int64": "<i8"}


def _hardware_input(x, task):
    return x.transpose(2, 0, 1)[None] if task == "vision" else x


def _sample_indices(task, y, count):
    if task == "vision" and count % 4 == 0:
        # Predetermined stratification by label, independent of predictions.
        return np.sort(np.concatenate([np.flatnonzero(y == c)[:count//4] for c in range(4)]))
    return np.arange(count)


def _crc_guard(case, program, runner):
    encoded = program.read_bytes()
    inspect_yhb(encoded)
    corrupted = case / "corrupted.yhb"
    corrupted.write_bytes(encoded[:-1] + bytes((encoded[-1] ^ 0x80,)))
    completed = subprocess.run(PRIORITY + [str(runner), str(corrupted), "--inspect"],
                               capture_output=True, text=True, timeout=30)
    (case / "crc.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode == 0 or "CRC32" not in completed.stderr:
        raise RuntimeError("C99 loader did not reject the corrupted CRC32")


def _test_trial(case, runner, program, inputs, input_specs, outputs, values):
    case.mkdir()
    command = PRIORITY + [str(runner), str(program), "--backend", "verilator", "--dump-stats"]
    for name, vector in inputs.items():
        src = case / (name + ".input.bin")
        src.write_bytes(tensor_bytes(vector, input_specs[name]["dtype"]))
        command.extend(("--input", f"{name}={src}"))
    result = {}
    for name, spec in outputs.items():
        path = case / (name + ".output.bin")
        command.extend(("--output", f"{name}={path}"))
        result[name] = (path, spec)
    completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
    (case / "runtime.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"{case.name}: C99 replay failed with exit {completed.returncode}")
    native, count = {}, 0
    for name, (path, spec) in result.items():
        got = path.read_bytes()
        expect = tensor_bytes(values[name], spec["dtype"])
        (case / (name + ".golden.bin")).write_bytes(expect)
        if expect != got:
            raise RuntimeError(f"{case.name}: {name} differs from independent fixed-point reference")
        native[name] = np.frombuffer(got, dtype=NUMPY_DTYPES[spec["dtype"]]).reshape(spec["shape"])
        count += len(got)
    line, = (x for x in completed.stdout.splitlines() if x.startswith("STATS "))
    raw = dict(field.split("=", 1) for field in line.split()[1:])
    if raw.get("backend") != "verilator" or int(raw["npu_domain_wall_ticks"]) <= 0:
        raise RuntimeError(f"{case.name}: not a measured Verilator RTL replay")
    return native, count, {k: int(v) if v.isdecimal() else v for k, v in raw.items()}


def _summary(task, model, fp32_results, experiment, report, program):
    all_fp32, all_labels = fp32_results
    stats = experiment["rows"]
    f, q = np.concatenate(experiment["fp32"]), np.concatenate(experiment["int8"])
    delta = np.abs(f - q)
    scale = report["logit_scale"]
    pred_float = np.concatenate([r["fp32_prediction"] for r in stats])
    pred_int8 = np.concatenate([r["rtl_prediction"] for r in stats])
    labels = np.concatenate([r["labels"] for r in stats])
    wall = np.asarray([r["npu_domain_wall_ticks"] for r in stats])
    return {
        "model": model, "heldout_dataset_examples": len(all_labels), "heldout_rtl_examples": len(stats),
        "scored_rtl_predictions": len(labels), "fp32_full_heldout_top1": float((all_fp32 == all_labels).mean()),
        "fp32_rtl_subset_top1": float((pred_float == labels).mean()),
        "rtl_int8_top1": float((pred_int8 == labels).mean()),
        "fp32_rtl_prediction_agreement": float((pred_float == pred_int8).mean()),
        "logit_cosine": float(np.dot(f, q)/(np.linalg.norm(f)*np.linalg.norm(q))),
        "logits_mae": float(delta.mean()), "logits_max_abs_err": float(delta.max()),
        "logit_scale": scale, "yhb_bytes": program.stat().st_size,
        "checked_output_bytes": sum(row["bytes_checked"] for row in stats), "byte_errors": 0,
        "rtl_npu_domain_wall_ticks_mean": float(wall.mean()),
        "rtl_npu_domain_wall_ticks_min": int(wall.min()), "rtl_npu_domain_wall_ticks_max": int(wall.max()),
        "phase12c_npu_fmax_mhz": FMAX_MHZ, "equivalent_latency_us_mean": float(wall.mean()/FMAX_MHZ),
        "measurement": "frozen Balanced Verilator C99 MMIO",
        "prediction_scope": "causal positions 1,2,3 (zero-based)" if task == "transformer" else "all images",
        "crc_rejected": True, "input_runtime": "fresh CLI run for each held-out tensor"
    }


def evaluate_trained(task, models, destination, runner, output_count, policy):
    root = destination / task
    root.mkdir()
    report = json.loads((models / f"{task}.training.json").read_text(encoding="utf-8"))
    path_float = models / report["models"]["fp32"]
    path_int8 = models / report["models"]["int8"]
    f32_model = Fp32Onnx(path_float.read_bytes())
    input_name, = f32_model.inputs
    x, y = make_data(task, "test")
    results_float = [f32_model.evaluate(**{input_name: _hardware_input(v, task)}) for v in x]
    observed_float = np.stack([r["prediction"].reshape(-1) for r in results_float])
    if task == "transformer":
        all_fp32, all_labels = observed_float[:, 1:], y[:, 1:]
    else:
        all_fp32, all_labels = observed_float[:, 0], y
    if (all_fp32 == all_labels).mean() < .95:
        raise RuntimeError(f"{task}: actual exported FP32 ONNX fails Top-1 gate")
    if not 16 <= output_count <= len(x):
        raise ValueError("number of RTL samples must be between 16 and the test split size")
    manifest = compile_bundle(parse_model(path_int8.read_bytes()), load_profile("fpga_zu15eg_balanced"),
                              root / "bundle", dialect="canonical", device_policy=policy)
    program = root / "bundle" / manifest["program"]
    graph, input_specs = manifest["graph"], manifest["input_bindings"]
    if graph["inputs"] != [input_name] or set(graph["outputs"]) != {"logits", "prediction"}:
        raise RuntimeError(f"{task}: INT8 ONNX changes I/O symbols")
    if task == "transformer" and not any(s["op"] == "QSoftmax" and s["device"] in ("GPU", "VPU")
                                         for s in manifest["plan"]["stages"]):
        raise RuntimeError("trained causal model does not exercise Q/DQ Softmax hardware lowering")
    _crc_guard(root, program, runner)
    targets = _sample_indices(task, y, output_count)
    rows, scored_float, scored_int8 = [], [], []
    specs = {name: graph["tensors"][name] for name in graph["outputs"]}
    for sample in targets:
        case = root / f"sample-{sample:03d}"
        raw_x = _hardware_input(x[sample], task)
        inputs = {input_name: quantize(raw_x, report["input_scale"]).flatten().tolist()}
        values = evaluate(graph, inputs)
        native, checked, counters = _test_trial(case, runner, program, inputs, input_specs, specs, values)
        pos = slice(1, None) if task == "transformer" else slice(None)
        golden = results_float[sample]
        scored_float.append(golden["logits"][pos].astype(np.float64).flatten())
        scored_int8.append(native["logits"][pos].astype(np.float64).flatten() * report["logit_scale"])
        label = np.asarray(y[sample]).reshape(-1)[pos]
        rows.append({"sample": int(sample), "labels": label.tolist(),
                     "fp32_prediction": golden["prediction"][pos].tolist(),
                     "rtl_prediction": native["prediction"][pos].tolist(), "bytes_checked": checked, **counters})
        print(f"{task} sample={sample} RTL ticks={counters['npu_domain_wall_ticks']} bytes={checked} PASS",
              flush=True)
    outcome = _summary(task, path_int8.name, (all_fp32, all_labels),
                       {"rows": rows, "fp32": scored_float, "int8": scored_int8}, report, program)
    stages = manifest["plan"]["stages"]
    outcome["device_policy"] = policy
    outcome["npu_stages"] = sum(stage["device"] == "NPU" for stage in stages)
    outcome["gpu_stages"] = sum(stage["device"] == "GPU" for stage in stages)
    outcome["vpu_stages"] = sum(stage["device"] == "VPU" for stage in stages)
    if policy == "npu-greedy" and not (outcome["npu_stages"] and outcome["gpu_stages"]):
        raise RuntimeError(f"{task}: expected both native NPU and GPU stages")
    (root / "result.json").write_text(json.dumps({"summary": outcome, "trials": rows}, indent=2) + "\n")
    print(json.dumps(outcome, sort_keys=True), flush=True)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runner", type=Path, default=RUNNER)
    parser.add_argument("--vectors", type=int, default=16)
    parser.add_argument("--only", choices=SHAPES)
    parser.add_argument("--device-policy", choices=("auto", "npu-greedy"), default="npu-greedy")
    args = parser.parse_args()
    dest = args.out.resolve()
    if SCRATCH.resolve() != SCRATCH or SCRATCH not in dest.parents:
        parser.error("output can only be within C3 training scratch")
    dest.mkdir()
    rows = [evaluate_trained(task, args.models, dest, args.runner, args.vectors, args.device_policy)
            for task in ([args.only] if args.only else SHAPES)]
    (dest / "accuracy.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (dest / "accuracy.csv").open("w", newline="") as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=rows[0])
        csv_writer.writeheader()
        csv_writer.writerows(rows)


if __name__ == "__main__":
    main()
