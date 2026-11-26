"""Produce IEEE double-column graphics and tables from the ICCD master JSON."""

import argparse
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


FAMILIES = ("YOLO26n", "TinyLM-Chat", "ResNet-18", "MobileNetV2")
LABELS = ("YOLO26n", "TinyLM", "ResNet-18", "MobileNetV2")
COLORS = ("#315674", "#A15238", "#3C795C", "#9C6F25")
CLOCKS = ((125, 100), (200, 150))
FIGS = ("fig_c1_ablation_speedup", "fig_c2_exec_breakdown_overlap",
        "fig_c3_memory_dma_descriptors", "fig_c4_roofline_codesign")
STAGES = {"YOLO26n": ("C6", "C7", "C8"),
          **dict.fromkeys(FAMILIES[1:], ("A0", "A1", "A2", "A3"))}


def select(rows, family, stage, pair):
    matches = [r for r in rows if r["workload"] == family and r["stage"] == stage
               and (r["npu_mhz"], r["gpu_mhz"]) == pair
               and r["measurement_kind"] in ("rtl_end_to_end", "rtl_layer_aggregate")]
    measured = [r for r in matches if r["measurement_kind"] == "rtl_end_to_end"]
    matches = measured or matches
    return next((r for r in matches if r["variant"].endswith("-on")), matches[0] if matches else None)


def pick(rows, family, pair):
    return select(rows, family, STAGES[family][-1], pair)


def finish(fig, out, name):
    fig.savefig(out / f"{name}.pdf", metadata={"Title": name, "CreationDate": None})
    fig.savefig(out / f"{name}.svg", metadata={"Date": None})
    fig.savefig(out / f"{name}.png", dpi=600)
    plt.close(fig)


def stylize(ax, title, label, *, horizontal=False):
    ax.set_title(title, loc="left", fontweight="bold", fontsize=8.5)
    ax.set_axisbelow(True)
    axis = "x" if horizontal else "y"
    ax.grid(axis=axis, color="#E4E8E9", linewidth=0.55)
    if horizontal:
        ax.set_xlabel(label)
    else:
        ax.set_ylabel(label)


def ablation(rows, out):
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.85), layout="constrained")
    for ax, family, color in zip(axes.flat, FAMILIES, COLORS, strict=True):
        stages = STAGES[family]
        records = [select(rows, family, stage, CLOCKS[0]) for stage in stages]
        slots = np.arange(len(stages))
        values = [r["latency_ms_measured_pair"] if r else np.nan for r in records]
        artists = ax.bar(slots, values, color=color, width=.56)
        for i, (rec, artist) in enumerate(zip(records, artists)):
            if not rec:
                ax.text(i, .03, "—", transform=ax.get_xaxis_transform(), ha="center")
                continue
            aggregate = rec["measurement_kind"] == "rtl_layer_aggregate"
            if aggregate:
                artist.set_hatch("////")
                artist.set_edgecolor("#253846")
            speed = rec["speedup_vs_same_ratio_baseline"]
            marker = "†" if rec["baseline_measurement_kind"] == "rtl_layer_aggregate" else ""
            value = f"{values[i]:.2f}" + (f" ms\n{speed:.2f}×{marker}" if speed else " ms")
            ax.annotate(value, (i, values[i]), xytext=(0, 4), textcoords="offset points", ha="center")
        vals = [v for v in values if np.isfinite(v)]
        if any(value <= 0 for value in vals):
            raise ValueError(f"{family}: latency must be positive")
        wide_range = len(vals) > 1 and max(vals) / min(vals) > 12
        if wide_range:
            ax.set_yscale("log")
            ax.set_ylim(.65 * min(vals), 2.15 * max(vals))
        else:
            ax.set_ylim(0, 1.33 * max(vals, default=1))
        ax.set_xticks(slots, stages)
        if not wide_range:
            ax.yaxis.set_major_locator(MaxNLocator(4))
        unit = "ms / session" if family == "TinyLM-Chat" else "ms / frame"
        stylize(ax, family, unit + (" (log)" if wide_range else ""))
    finish(fig, out, FIGS[0])


def execution(rows, out):
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.16, 3.18), layout="constrained")
    active = [pick(rows, family, CLOCKS[0]) for family in FAMILIES]
    slots = np.arange(len(active))
    partitions = (("Single compute", "single_compute_cycles", "#315674"),
                  ("Native & GPU", "native_gpu_without_dma_cycles", "#7FA58B"),
                  ("DMA only", "dma_only_cycles", "#DDB86D"),
                  ("Compute & DMA", "compute_dma_overlap_cycles", "#A15238"),
                  ("None active", "idle_from_union_cycles", "#B3B4B7"))
    bottom = np.zeros(len(active))
    for name, key, color in partitions:
        amounts = [100 * r[key] / r["wall_cycles"] if r and r.get(key) is not None else 0 for r in active]
        left.bar(slots, amounts, bottom=bottom, color=color, label=name, width=.63)
        bottom += np.asarray(amounts)
    for ix, r in enumerate(active):
        if not r or r.get("idle_from_union_cycles") is None:
            left.text(ix, 2, "—", ha="center")
    for j, (short, key, color) in enumerate((("NPU", "npu", "#315674"), ("VPU", "vpu", "#91B3C8"),
                                           ("GPU", "gpu", "#7FA58B"), ("DMA", "dma", "#DDB86D"))):
        values = [100 * r[f"{key}_wall_fraction"] if r and r.get(f"{key}_wall_fraction") is not None
                  else np.nan for r in active]
        right.bar(slots + (j-1.5)*.18, values, width=.16, label=short, color=color)
    for ax in (left, right):
        ax.set_xticks(slots, LABELS, rotation=12, ha="right")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.19), ncol=3)
        ax.set_ylim(0, 104)
    stylize(left, "a  Exclusive wall-time partition", "% of wall time")
    stylize(right, "b  Engine busy fractions", "% of wall time")
    finish(fig, out, FIGS[1])


def memory(rows, out):
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.15), layout="constrained")
    slots = np.arange(len(FAMILIES))
    ladder = [("Peak SRAM", "sram_peak_bytes", 1024, "KiB"),
              ("DDR traffic", "dma_bytes", 1024, "KiB"),
              ("DMA descriptors", "dma_descriptors", 1, "count")]
    baseline = [select(rows, f, "C6" if f == "YOLO26n" else "A1", CLOCKS[0]) for f in FAMILIES]
    end = [pick(rows, f, CLOCKS[0]) for f in FAMILIES]
    for ax, (title, key, scale, ylabel) in zip(axes, ladder, strict=True):
        for j, (name, data, color) in enumerate((("C6 / A1", baseline, "#B3B4B7"), ("C8 / A3", end, "#315674"))):
            vals = [r[key] / scale if r and r.get(key) is not None else np.nan for r in data]
            if any(np.isfinite(v) and v <= 0 for v in vals):
                raise ValueError(f"{name}: a log-scaled memory metric must be positive")
            ax.bar(slots + (j-.5)*.32, vals, width=.28, color=color, label=name)
        ax.set_yscale("log")
        ax.set_xticks(slots, LABELS, rotation=25, ha="right")
        stylize(ax, title, ylabel)
    axes[0].legend(loc="upper center", bbox_to_anchor=(.55, -.30), ncol=2)
    finish(fig, out, FIGS[2])


def roofline(rows, roofs, out):
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.16, 3.52), layout="constrained")
    optimal = [(family, color, pick(rows, family, pair), pair) for family, color in zip(FAMILIES, COLORS)
               for pair in CLOCKS]
    values = [r["ops_per_dma_byte"] for _, _, r, _ in optimal if r and r["ops_per_dma_byte"]]
    x = np.geomspace(min(values, default=1)/2, max(values, default=32)*2, 200)
    for pair, dash in zip(CLOCKS, ("--", ":"), strict=True):
        key = "_".join(map(str, pair))
        data = roofs[key]
        left.plot(x, np.minimum(data["array_peak_gops"], x * data["interface_peak_gb_s"]),
                  linestyle=dash, color="#575E62", label=f"NPU ideal {pair[0]}/{pair[1]}")
        bw = data["best_observed_dma_gb_s"]
        if bw:
            left.plot(x, np.minimum(data["array_peak_gops"], x * bw), linestyle=dash,
                      color="#91B3C8", alpha=.9, label="Observed DMA roof" if pair == CLOCKS[0] else None)
    for family, color, row, pair in optimal:
        if row and row["ops_per_dma_byte"] and row["gops_at_measured_pair"]:
            left.scatter(row["ops_per_dma_byte"], row["gops_at_measured_pair"], color=color, s=30,
                         marker="o" if pair == CLOCKS[0] else "s",
                         label=family if pair == CLOCKS[0] else None)
    left.set(xscale="log", yscale="log", xlabel="Logical ops / DMA byte")
    stylize(left, "a  Frozen Step 2 roofline", "Effective throughput (Gop/s)")
    left.legend(loc="upper center", bbox_to_anchor=(.55, -.23), ncol=2, columnspacing=.8)
    opts = [pick(rows, family, CLOCKS[0]) for family in FAMILIES]
    slots = np.arange(len(opts))
    for i, (title, key, color) in enumerate((("Wide-K GPU", "wide_k_max_busy_share", "#315674"),
                                            ("VPU", "vpu_max_busy_share", "#A15238"))):
        vals = [100 * r[key] if r and r.get(key) is not None else np.nan for r in opts]
        right.barh(slots + (i-.5)*.32, vals, height=.28, label=title, color=color)
    right.set_yticks(slots, LABELS)
    right.invert_yaxis()
    stylize(right, "b  Phase 16 candidates", "Maximum addressable busy time (%)", horizontal=True)
    right.legend(loc="upper center", bbox_to_anchor=(.55, -.19), ncol=2)
    finish(fig, out, FIGS[3])


def _fmt(value, places=2):
    return "--" if value is None else f"{value:,.{places}f}"


def _start(title, fmt, heads):
    return "\n".join((r"\begin{table*}[t]", r"\centering\scriptsize", rf"\caption{{{title}}}",
                      rf"\begin{{tabular}}{{{fmt}}}", r"\toprule", " & ".join(heads) + r" \\",
                      r"\midrule"))


def _weights(row):
    if not row:
        return "--"
    raw = row["weight_source"]
    version = re.search(r"image\s*net[-_ ]?1k[-_ ]+v(\d+)", raw, re.IGNORECASE)
    if version:
        return "ImageNet V" + version[1]
    if "official" in raw.lower() and "coco" in raw.lower():
        return "Official COCO"
    if "trained" in raw.lower() or "local supervised" in raw.lower():
        return "Locally trained"
    return "See ledger"



def tables(rows, out):
    footer = "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
    baseline_lines = []
    for family in FAMILIES:
        base, high = (pick(rows, family, pair) for pair in CLOCKS)
        values = [family, base["resolution"].replace("×", r"\(\times\)") if base else "--", _weights(base),
                  str(base["compared_bytes"]) if base else "--",
                  _fmt(base["latency_ms_measured_pair"] if base else None),
                  _fmt(high["latency_ms_measured_pair"] if high else None),
                  _fmt(high["throughput_measured_pair"] if high else None)]
        baseline_lines.append(" & ".join(values) + r" \\")
    title = "Frozen Step 2 benchmark workloads at their best compiled settings."
    text = _start(title, "llllrrr", ("Workload", "Input", "Weights", "Exact bytes", "125/100 (ms)", "200/150 (ms)", "200/150 rate"))
    note = (r"\par\smallskip\footnotesize Rate is frames/s for vision and generated tokens/s for TinyLM. "
            r"Clock pairs are in MHz. Both pairs use frozen Step 2 RTL. Dashes indicate unavailable qualified RTL data.")
    (out / "table_iccd_benchmarks.tex").write_text(text+"\n"+"\n".join(baseline_lines)+footer+note+"\n"+r"\end{table*}"+"\n", encoding="utf-8")
    lines = []
    for family in FAMILIES:
        for stage in STAGES[family]:
            row = select(rows, family, stage, CLOCKS[0])
            if row:
                kind = "Layer aggregation" if row["measurement_kind"] == "rtl_layer_aggregate" else "End-to-end RTL"
                reference = (r"$^{\dagger}$" if row["baseline_measurement_kind"] == "rtl_layer_aggregate"
                             and row["speedup_vs_same_ratio_baseline"] is not None else "")
                vals = [family, stage, kind, *[_fmt(row[k], 0) for k in ("wall_cycles", "gpu_busy_cycles", "dma_bytes", "dma_descriptors")],
                        _fmt(row["speedup_vs_same_ratio_baseline"]) + reference]
            else:
                vals = [family, stage, "Unavailable", "--", "--", "--", "--", "--"]
            lines.append(" & ".join(vals) + r" \\")
    title = "Compiler ablations at a measured 125/100 MHz ratio."
    text = _start(title, "lllrrrrr", ("Workload", "Stage", "Evidence", "Cycles", "GPU cycles", "DMA bytes", "DMA desc.", "Speedup"))
    note = (r"\par\smallskip\footnotesize Speedup uses the same-ratio C6 or A0 baseline. "
            r"$\dagger$ uses a geometry-weighted isolated-layer baseline. Each representative is a whole operator. "
            r"Classifier inputs are resampled for RTL validation.")
    (out / "table_iccd_ablation.tex").write_text(text+"\n"+"\n".join(lines)+footer+note+"\n"+r"\end{table*}"+"\n", encoding="utf-8")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("ledger", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    data = json.loads(args.ledger.read_text(encoding="utf-8"))
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .8, "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    ablation(data["rows"], out)
    execution(data["rows"], out)
    memory(data["rows"], out)
    roofline(data["rows"], data["rooflines"], out)
    tables(data["rows"], out)


if __name__ == "__main__":
    main()
