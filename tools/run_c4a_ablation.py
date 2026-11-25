"""Run C4-A exact frozen Z15 ablations with common safe program residency."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from hetero_compiler.compiler import compile_bundle
from hetero_compiler.onnx import parse_model
from hetero_compiler.profile import load_profile
from tools.run_c2_ablation import SETTINGS

SCRATCH = Path("/dev/shm/ysyx_compiler_c4a")
FMAX_MHZ = 164.555
MINIMUM_A3_GAIN_PCT = 10.0


def _json_line(lines, tag):
    matched = [ln.removeprefix(tag + " ") for ln in lines if ln.startswith(tag + " ")]
    if len(matched) != 1:
        raise ValueError(f"expected one {tag} observation, got {len(matched)}")
    return json.loads(matched[0])


def _one(root, path, raw, label, options):
    profile = load_profile("fpga_zu15eg_balanced")
    profile["pipeline_scheduler"] = "bank-aware"
    stem = f"{path.stem}-{label}"
    result = {"model": path.stem, "ablation": label}
    try:
        bundle = root / stem
        manifest = compile_bundle(raw, profile, bundle, dialect="canonical",
                                  pipeline_tiles=True, **options)
        report = manifest["estimates"]
        result.update({key: report[key] for key in ("scratch_peak_live_bytes", "scratch_reserved_bytes",
                      "external_transfer_bytes", "scheduled_stage_count", "ir_operator_count")})
        result.update({f"{name.lower()}_stages": count
                       for name, count in report["device_stage_counts"].items()})
        nice_adjustment = max(0, 10 - os.nice(0))
        cmd = ["nice", "-n", str(nice_adjustment), sys.executable, "-B", "-m", "tools.verify_bundle",
               "--bundle", str(bundle), "--case", str(root / f"{stem}.rtl"),
               "--execute-plan", "--imem-cache"]
        log = root / f"{stem}.log"
        with log.open("w", encoding="utf-8") as out:
            subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, check=True,
                           timeout=960, env={**os.environ, "YSYX_REPLAY_TIMEOUT": "930",
                                             "TMPDIR": str(SCRATCH), "OMP_NUM_THREADS": "1"})
        lines = log.read_text(encoding="utf-8").splitlines()
        result.update(_json_line(lines, "Z15_METRICS"))
        result["imem_cache"] = _json_line(lines, "Z15_IMEM_CACHE")
        if result["rtl_dma_bytes"] != report["external_transfer_bytes"]:
            raise ValueError("scheduled DMA size disagrees with AXI beat count")
        if sum(ln.startswith("BUNDLE_RTL_PASS ") for ln in lines) != 1:
            raise ValueError("missing exact byte gate")
        result.update(status="byte_exact", fmax_latency_us=round(result["wall_npu_ticks"] / FMAX_MHZ, 6))
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        result.update(status="error", reason=str(exc))
    print(path.stem, label, result["status"], result.get("wall_npu_ticks", result.get("reason", "")),
          flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", nargs="+", choices=[key for key, _ in SETTINGS])
    p.add_argument("models", nargs="+", type=Path)
    args = p.parse_args()
    root = args.out.resolve()
    if SCRATCH not in root.parents or SCRATCH.resolve() != SCRATCH:
        p.error("output must be under C4-A RAM")
    root.mkdir()
    results, gates = [], {}
    for path in args.models:
        raw = parse_model(path.read_bytes())
        rows = [_one(root, path, raw, label, options) for label, options in SETTINGS
                if not args.only or label in args.only]
        exact = all(r["status"] == "byte_exact" for r in rows)
        if exact and not args.only:
            ticks = [r["wall_npu_ticks"] for r in rows]
            for row, cycle in zip(rows, ticks, strict=True):
                row["speedup_vs_a0"] = round(ticks[0]/cycle, 6)
            gates[path.stem] = {"strict_ladder": all(a > b for a, b in zip(ticks, ticks[1:])),
                                "a3_gain_pct": (ticks[2] - ticks[3]) / ticks[2] * 100,
                                "overlap": rows[-1]["compute_dma_overlap_ticks"] > 0 or
                                           rows[-1]["npu_gpu_overlap_ticks"] > 0}
        results.extend(rows)
    target = len(gates) < 2 or any(g["a3_gain_pct"] >= MINIMUM_A3_GAIN_PCT for g in gates.values())
    package = {"measurement": "frozen Z15 Balanced Verilator MMIO", "fmax_npu_mhz": FMAX_MHZ,
               "gates": gates, "two_model_a3_target_met": target if len(gates) >= 2 else None,
               "results": results}
    (root / "ablation.json").write_text(json.dumps(package, indent=2) + "\n")
    fields = list(dict.fromkeys(key for row in results for key in row))
    with (root / "ablation.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    if any(row["status"] != "byte_exact" for row in results):
        return 1
    return int(not args.only and (not target or any(not g["strict_ladder"] or not g["overlap"]
                                                    for g in gates.values())))


if __name__ == "__main__":
    raise SystemExit(main())
