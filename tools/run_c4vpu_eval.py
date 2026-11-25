"""Measure the frozen Z15 C4-VPU ablations and publish an independent accuracy ledger."""

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from examples.c4vpu.reference import causal_kv_fp32, long_prefix_fp32
from examples.c4vpu.transformer import evaluate_fp32
from hetero_compiler.compiler import compile_bundle
from hetero_compiler.kv_schedule import compile_kv_bundle
from hetero_compiler.onnx import parse_model
from hetero_compiler.profile import load_profile
from tools.run_c2_ablation import SETTINGS
from tools.run_c4a_kv import _fixed_reference, export_case, replay


SCRATCH = Path("/dev/shm/ysyx_compiler_c5")
CLOCKS = {"phase12c": 164.555, "step2": 125.00}
MODELS = {
    "window": Path("examples/c4a_streams/LongPrefix_Stream_C4A.onnx"),
    "layer": Path("examples/c4vpu/TransformerLayer_INT8_C4VPU.onnx"),
}
KV = Path("examples/c4a_kv/causal_kv32.json")
_BASELINES = {"off": "gpu_fallback", "on": "vpu_lowered"}


def _dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _log_observation(log, marker):
    matches = [line.removeprefix(marker + " ") for line in log.splitlines()
               if line.startswith(marker + " ")]
    if len(matches) != 1:
        raise ValueError(f"expected one {marker} observation, got {len(matches)}")
    return json.loads(matches[0])


def _latencies(row):
    for label, freq in CLOCKS.items():
        row[f"wall_us_{label}"] = round(row["wall_npu_ticks"] / freq, 6)


def _estimates(manifest):
    summary = manifest["estimates"]
    keys = ("scratch_peak_live_bytes", "scratch_reserved_bytes", "external_transfer_bytes",
            "scheduled_stage_count", "ir_operator_count")
    result = {k: summary[k] for k in keys if k in summary}
    by_device = Counter(stage["device"] for stage in manifest["plan"]["stages"])
    result.update({f"{d.lower()}_stages": by_device[d] for d in ("GPU", "NPU", "VPU")})
    native_ops = Counter(stage["op"] for stage in manifest["plan"]["stages"] if stage["device"] == "VPU")
    result["vpu_ops"] = dict(native_ops)
    actions = {8: "tensor", 32: "attention", 64: "elementwise",
               128: "scale_commit", 256: "activation_commit"}
    counts = Counter(actions[value] for command in manifest["backends"]["npu"]["commands"]
                     for address, value in command["writes"] if address == 0x10 and value in actions)
    result["native_ctrl_actions"] = dict(counts)
    return result


def _model_variant(root, group, raw, level, options, mode, attention):
    name = f"{group}-{mode}-{level}"
    row = {"group": group, "variant": name, "ablation": level, "requested_vpu": mode}
    effective = "on" if mode == "on" and level in ("A2", "A3") else "off"
    row["effective_vpu"] = effective
    bundle = root / name
    profile = load_profile("fpga_zu15eg_balanced")
    profile["pipeline_scheduler"] = "bank-aware"
    try:
        manifest = compile_bundle(raw, profile, bundle, dialect="canonical", pipeline_tiles=True,
                                  vpu_lowering=effective, vpu_attention=attention, **options)
        row.update(_estimates(manifest))
        if effective == "on":
            expected_attention = 24 if group == "window" else 4
            if row["native_ctrl_actions"].get("attention", 0) != expected_attention:
                raise ValueError(f"expected {expected_attention} native attention doorbells")
            if group == "layer":
                required = {"Z15RMSNorm": 8, "Lookup": 4, "Z15Mul": 4, "Add": 8}
                if any(row["vpu_ops"].get(op) != count for op, count in required.items()):
                    raise ValueError("expected native RMSNorm, SwiGLU and residual Add in every tile")
        log = root / f"{name}.log"
        case = root / f"{name}.rtl"
        cmd = [sys.executable, "-B", "-m", "tools.verify_bundle", "--bundle", str(bundle),
               "--case", str(case), "--execute-plan", "--imem-cache"]
        with log.open("w", encoding="utf-8") as output:
            subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=970,
                           env={**os.environ, "YSYX_REPLAY_TIMEOUT": "950", "TMPDIR": str(SCRATCH)})
        transcript = log.read_text(encoding="utf-8")
        row.update(_log_observation(transcript, "Z15_METRICS"))
        row["imem_cache"] = _log_observation(transcript, "Z15_IMEM_CACHE")
        fixed = _log_observation(transcript, "BUNDLE_RTL_PASS")
        bindings = manifest["output_bindings"]
        if row["rtl_dma_bytes"] != row["external_transfer_bytes"]:
            raise ValueError("actual AXI payload bytes disagree with scheduled bytes")
        if set(fixed) != set(bindings) or transcript.count("Z15_DDR_PASS ") != len(bindings):
            raise ValueError("RTL did not compare every output binding")
        # Each reported value is backed by a full byte comparison against RTL DDR.
        row["compared_outputs"] = len(bindings)
        row["compared_bytes"] = sum(binding["size"] for binding in bindings.values())
        row["status"] = "byte_exact"
        _latencies(row)
        _dump(root / f"{name}.outputs.json", fixed)
        return row, fixed
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        row.update(status="error", reason=str(exc))
        return row, None


def _kv_variant(root, spec, mode, schedule, attention):
    name = f"kv-{mode}-{schedule}"
    row = {"group": "kv", "variant": name, "schedule": schedule, "requested_vpu": mode,
           "effective_vpu": mode}
    try:
        bundle = root / name
        manifest = compile_kv_bundle(spec, bundle, mode=schedule, vpu_lowering=mode,
                                     vpu_attention=attention)
        row.update(_estimates(manifest))
        if mode == "on":
            expected = spec["prefill_rows"] // spec["tile_rows"] + spec["decode_steps"]
            if row["native_ctrl_actions"].get("attention", 0) != expected:
                raise ValueError(f"expected {expected} native KV attention doorbells")
        case = root / f"{name}.rtl"
        row.update(replay(bundle, manifest, case))
        _latencies(row)
        fixed = _fixed_reference(manifest)
        if set(fixed) != set(manifest["output_bindings"]):
            raise ValueError("RTL did not compare every KV output binding")
        _dump(root / f"{name}.outputs.json", fixed)
        if mode == "on" and schedule == "cached":
            export_case(bundle, manifest, root / "kv-vpu-kit-case")
        return row, fixed
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        row.update(status="error", reason=str(exc))
        return row, None


def import_window_baseline(roots):
    """Load explicit byte-exact off evidence from fresh C4-A-compatible replay slices."""
    variants = {}
    for root in roots:
        rows = json.loads((root / "ablation.json").read_text(encoding="utf-8"))["results"]
        for source in rows:
            label = source["ablation"]
            if label in variants or source["model"] != MODELS["window"].stem or source["status"] != "byte_exact":
                raise ValueError(f"duplicate or invalid C4-A window baseline at {root}")
            log = root / f"{source['model']}-{label}.log"
            transcript = log.read_text(encoding="utf-8")
            values = _log_observation(transcript, "BUNDLE_RTL_PASS")
            expected = _log_observation(transcript, "Z15_METRICS")
            if any(source.get(k) != v for k, v in expected.items()):
                raise ValueError(f"baseline counters disagree with RTL log at {root}")
            base = root / f"{source['model']}-{label}"
            manifest = json.loads((base / "bundle.json").read_text(encoding="utf-8"))
            # This path accepts the exact old off shape, never an on graph.
            if manifest["graph"]["name"] != MODELS["window"].stem or "VPU" in {
                    stage["device"] for stage in manifest["plan"]["stages"]}:
                raise ValueError("historical fallback evidence unexpectedly contains native VPU")
            count = manifest["output_bindings"]
            if values.keys() != count.keys() or transcript.count("Z15_DDR_PASS ") != len(count):
                raise ValueError("baseline replay omitted a published output")
            row = {**_estimates(manifest), **source, "group": "window", "variant": f"window-off-{label}",
                   "requested_vpu": "off", "effective_vpu": "off",
                   "compared_outputs": len(count),
                   "compared_bytes": sum(f["size"] for f in count.values()),
                   "reused_evidence_log": str(log)}
            _latencies(row)
            variants[label] = (row, values)
    return variants


def import_kv_baseline(root):
    """Use a fresh legacy C4-A cached/recompute replay as matching off evidence."""
    variants = {}
    for source in json.loads((root / "kv-compare.json").read_text(encoding="utf-8"))["results"]:
        schedule = source["mode"]
        if schedule not in ("cached", "recompute") or schedule in variants or source["status"] != "byte_exact":
            raise ValueError(f"duplicate or invalid C4-A KV baseline at {root}")
        manifest = json.loads((root / schedule / "bundle.json").read_text(encoding="utf-8"))
        if manifest["kv"].get("vpu_lowering", "off") != "off":
            raise ValueError("legacy KV baseline unexpectedly enabled native VPU")
        log = root / f"{schedule}-rtl" / "rtl.log"
        transcript = log.read_text(encoding="utf-8")
        counts = _log_observation(transcript, "Z15_METRICS")
        if any(source.get(k) != v for k, v in counts.items()):
            raise ValueError(f"baseline counters disagree with RTL log at {root}")
        fixed = _fixed_reference(manifest)
        if (set(fixed) != set(manifest["output_bindings"]) or
                transcript.count("Z15_DDR_PASS ") != len(fixed)):
            raise ValueError("legacy KV replay omitted a published output")
        row = {**_estimates(manifest), **source, "group": "kv", "variant": f"kv-off-{schedule}",
               "schedule": schedule, "requested_vpu": "off", "effective_vpu": "off",
               "reused_evidence_log": str(log)}
        _latencies(row)
        variants[schedule] = (row, fixed)
    if set(variants) != {"cached", "recompute"}:
        raise ValueError("legacy KV evidence must contain both schedules")
    return variants


def _metrics(left, right):
    if left.shape != right.shape or left.ndim != 1 or not left.size:
        raise ValueError("accuracy operands must be nonempty matching vectors")
    error = np.abs(left - right)
    product = float(np.linalg.norm(left) * np.linalg.norm(right))
    cosine = (float(np.clip(np.dot(left, right) / product, -1, 1))
              if product else float(np.array_equal(left, right)))
    return {"max_abs_err": float(error.max()), "mae": float(error.mean()), "cosine_similarity": cosine}


def _is_token(name):
    return any(token in name for token in ("_token", "_predict"))


def accuracy_rows(group, label, gpu, vpu, fp32):
    if gpu.keys() != vpu.keys() or gpu.keys() != fp32.keys():
        raise ValueError(f"{group} reference projects different outputs from frozen RTL")
    base = {"group": group, "comparison_set": label, "fixed_point_rtl_checked": "byte_exact_each_mode"}
    rows = []
    scale = 1 if group == "kv" else 1/16
    for pair in ("vpu_vs_gpu", "gpu_vs_fp32", "vpu_vs_fp32"):
        lhs, rhs = pair.split("_vs_")
        tokens, logits = [[], []], [[], []]
        for symbol in gpu:
            is_token = _is_token(symbol)
            unit = 1 if is_token else scale
            values = [np.asarray(t[symbol], np.float64).ravel() * factor
                      for t, factor in ((gpu, unit), (vpu, unit), (fp32, 1))]
            a, b = [{"gpu": values[0], "vpu": values[1], "fp32": values[2]}[key] for key in (lhs, rhs)]
            row = {**base, "pair": pair, "scope": symbol, "sample_count": a.size}
            if is_token:
                if a.shape != b.shape or not a.size:
                    raise ValueError("incompatible token samples")
                row.update(top1_agreement=float((a == b).mean()), top1_matches=int((a == b).sum()))
                tokens[0].append(a)
                tokens[1].append(b)
            else:
                row.update(_metrics(a, b))
                if not symbol.endswith("_cache"):
                    logits[0].append(a)
                    logits[1].append(b)
            rows.append(row)
        a, b = [np.concatenate(parts) for parts in logits]
        t_a, t_b = [np.concatenate(parts) for parts in tokens]
        rows.append({**base, "pair": pair, "scope": "logits_and_token_summary", "sample_count": a.size,
                     **_metrics(a, b), "top1_agreement": float((t_a == t_b).mean()),
                     "top1_matches": int((t_a == t_b).sum()), "token_count": t_a.size})
        if group == "kv":
            decode = [name for name in gpu if name.startswith("decode_") and _is_token(name)]
            d_src = {"gpu": gpu, "vpu": vpu, "fp32": fp32}
            d_a, d_b = [[int(d_src[side][name][0]) for name in decode] for side in (lhs, rhs)]
            rows.append({**base, "pair": pair, "scope": "decode_token_summary", "token_count": len(decode),
                         "top1_matches": sum(x == y for x, y in zip(d_a, d_b, strict=True)),
                         "top1_agreement": float(np.mean(np.equal(d_a, d_b)))})
    return rows


def _take(result, rows, outputs):
    row, values = result
    print(row["variant"], row["status"], row.get("wall_npu_ticks", row.get("reason", "")), flush=True)
    rows.append(row)
    if values is not None:
        outputs[row["variant"]] = values


def _record_ratios(rows):
    good = {row["variant"]: row for row in rows if row["status"] == "byte_exact"}
    for row in good.values():
        group, mode = row["group"], row["requested_vpu"]
        key = f"{group}-{mode}-" + ("recompute" if group == "kv" else "A0")
        if key in good:
            row["speedup_vs_baseline"] = round(good[key]["wall_npu_ticks"]/row["wall_npu_ticks"], 8)
        if mode == "on":
            match = row["variant"].replace("-on-", "-off-")
            if match in good:
                row["speedup_vs_vpu_off"] = round(good[match]["wall_npu_ticks"]/row["wall_npu_ticks"], 8)


def _equivalence_gates(outputs):
    pairs = [(f"kv-{mode}-cached", f"kv-{mode}-recompute") for mode in _BASELINES]
    pairs.extend((f"{group}-{mode}-A2", f"{group}-{mode}-A3")
                 for group in MODELS for mode in _BASELINES)
    pairs.extend((f"{group}-off-{label}", f"{group}-on-{label}")
                 for group in MODELS for label in ("A0", "A1"))
    return {f"{left} == {right}": outputs[left] == outputs[right]
            for left, right in pairs if left in outputs and right in outputs}


def _ledger(outputs, spec):
    rows = []
    pairs = (("window", ("A2", "A3")), ("layer", ("A2", "A3")), ("kv", ("recompute", "cached")))
    for group, labels in pairs:
        for label in labels:
            gpu = outputs.get(f"{group}-off-{label}")
            vpu = outputs.get(f"{group}-on-{label}")
            if gpu is None or vpu is None:
                continue
            fp32 = {"kv": lambda: causal_kv_fp32(spec), "window": long_prefix_fp32,
                    "layer": evaluate_fp32}[group]()
            rows.extend(accuracy_rows(group, label, gpu, vpu, fp32))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="Fresh destination in C4-VPU RAM")
    p.add_argument("--groups", choices=("window", "layer", "kv"), nargs="+",
                   default=["window", "layer", "kv"])
    p.add_argument("--modes", choices=tuple(_BASELINES), nargs="+", default=list(_BASELINES))
    p.add_argument("--only", choices=[label for label, _ in SETTINGS], nargs="+",
                   help="Restrict ONNX ablation slices, without altering KV schedules")
    p.add_argument("--window-off-evidence", type=Path, nargs="+", default=[])
    p.add_argument("--kv-off-evidence", type=Path)
    p.add_argument("--attention", choices=("exact", "fast"), default="exact")
    args = p.parse_args()
    root = args.out.resolve()
    if SCRATCH not in root.parents or SCRATCH.resolve() != SCRATCH:
        p.error("output must be a fresh directory under compiler C4-VPU RAM")
    if any(len(x) != len(set(x)) for x in (args.groups, args.modes)):
        p.error("select each group and mode once")
    root.mkdir()
    rows, outputs = [], {}
    spec = json.loads(KV.read_text(encoding="utf-8"))
    reuse = import_window_baseline(args.window_off_evidence) if args.window_off_evidence else {}
    kv_reuse = import_kv_baseline(args.kv_off_evidence) if args.kv_off_evidence else {}
    for group in args.groups:
        if group == "kv":
            for mode in args.modes:
                for schedule in ("recompute", "cached"):
                    existing = kv_reuse.get(schedule) if mode == "off" else None
                    result = existing or _kv_variant(root, spec, mode, schedule, args.attention)
                    if existing:
                        _dump(root / f"kv-off-{schedule}.outputs.json", result[1])
                    _take(result, rows, outputs)
            continue
        raw = parse_model(MODELS[group].read_bytes())
        for mode in args.modes:
            for level, options in SETTINGS:
                if args.only is not None and level not in args.only:
                    continue
                inherited = reuse.get(level) if group == "window" else None
                if inherited is not None and (mode == "off" or level in ("A0", "A1")):
                    row, values = inherited
                    row = {**row, "requested_vpu": mode, "variant": f"{group}-{mode}-{level}"}
                    _dump(root / f"{row['variant']}.outputs.json", values)
                    _take((row, values), rows, outputs)
                else:
                    _take(_model_variant(root, group, raw, level, options, mode, args.attention), rows, outputs)
    _record_ratios(rows)
    equivalence = _equivalence_gates(outputs)
    _dump(root / "performance.json", {"measurement": "frozen Z15 Balanced Verilator MMIO",
                                      "clock_conversion_mhz": CLOCKS, "attention": args.attention,
                                      "equivalence_gates": equivalence, "results": rows})
    _csv(root / "performance.csv", rows)
    accuracy = _ledger(outputs, spec)
    if accuracy:
        _dump(root / "accuracy.json", {"reference": "independent NumPy continuous FP32, no integer graph evaluation",
                                       "fixed_point": "all published bytes validated by frozen RTL replay",
                                       "units": "Q4-real (window/layer); continuous code units (KV)",
                                       "results": accuracy})
        _csv(root / "accuracy.csv", accuracy)
    return int(any(row["status"] != "byte_exact" for row in rows) or not all(equivalence.values()))


if __name__ == "__main__":
    raise SystemExit(main())
