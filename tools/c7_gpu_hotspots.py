"""Join C6 measured GPU stages with ONNX geometry and interpreter job counts."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import struct

from hetero_compiler.yhb import inspect_yhb


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(archive, instructions):
    steps = read_csv(archive / "m3/rtl-320-board-ratio/rtl_stages.csv")
    counters = json.loads((archive / "m3/rtl-320-board-ratio/rtl_counters.json").read_text())
    manifest = json.loads((archive / "m3/yolo26n_320_manifest.json").read_text())
    quant = json.loads((archive / "m2-320/quant.json").read_text())
    onnx = {row["name"]: row for row in read_csv(archive / "m1/m1-yolo26n-graph-nodes.csv")}
    shapes = {row["name"]: json.loads(row["shape"]) for row in read_csv(
        archive / "m1/m1-yolo26n-graph-tensors.csv")}
    ir = {node["id"]: node for node in quant["nodes"]}
    machine = {stage["id"]: stage["jobs"] for stage in manifest["backends"]["gpu"]["stages"]}
    blob = (archive / "m3/yolo26n_320_int8.yhb").read_bytes()
    stage_table = inspect_yhb(blob)["sections"]["stages"]["offset"]
    decoded = iter(tuple(map(int, line.split()[1:])) for line in instructions.read_text().splitlines())
    grouped = defaultdict(Counter)
    for row in steps:
        if row["engine"] != "GPU":
            continue
        stage = row["stage"]
        engine, code_words = struct.unpack_from("<I4xI", blob, stage_table + 32 * int(row["index"]) + 8)
        if engine != 1:
            raise ValueError(f"{stage}: inconsistent stage engine")
        group = grouped[row["node"]]
        group.update(ticks=int(row["busy_ticks"]), uploaded_words=code_words, stages=1)
        for job in machine[stage]:
            pc, cfg, issued = next(decoded)
            if pc != job["entry"] or cfg != job["gpu_job_reg"]:
                raise ValueError(f"{stage}: inconsistent interpreter job order")
            group.update(jobs=1, lane_instructions=issued)
    if next(decoded, None) is not None:
        raise ValueError("unconsumed interpreter jobs")
    gpu = counters["gpu_busy_ticks"]
    wall = counters["npu_domain_wall_ticks"]
    if sum(row["ticks"] for row in grouped.values()) != gpu:
        raise ValueError("GPU stage ticks do not sum to the physical register")
    rows = []
    for name, row in sorted(grouped.items(), key=lambda item: -item[1]["ticks"]):
        node = ir[name]
        original = onnx[name]
        ins = json.loads(original["inputs"])
        outs = json.loads(original["outputs"])
        shape = f"{shapes[ins[0]]} → {shapes[outs[0]]}"
        if node["op"] == "Conv":
            kernel = node["attrs"]["kernel"]
            shape += f", K={shapes[ins[0]][1] * kernel[0] * kernel[1]}"
        rows.append(dict(node=name, onnx_op=original["op_type"], shape=shape, **row,
                         gpu_percent=100 * row["ticks"] / gpu, frame_percent=100 * row["ticks"] / wall))
    return rows, gpu, wall


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "instructions", "out", "report"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    args = cli.parse_args()
    rows, gpu, wall = summarize(args.archive, args.instructions)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    by_op = Counter()
    for row in rows:
        by_op[row["onnx_op"]] += row["ticks"]
    lines = [
        "# C7 M1 冻结 Z15 的 GPU 热点",
        "",
        f"以 C6 的 YOLO26n-320 bus.jpg 整帧为基线。共 {wall:,} 个 NPU 域拍，GPU 忙 {gpu:,} 拍。",
        f"宽 K Conv 的 INT32 部分和归约与宽 Requant 占 {by_op['Conv']:,} 拍（{100*by_op['Conv']/gpu:.2f}% GPU）。"
        f"两处 Q23 Softmax 占 {by_op['Softmax']:,} 拍（{100*by_op['Softmax']/gpu:.2f}% GPU）。",
        "",
        "形状来自静态 FP32 ONNX，按 NCHW 或原始注意力轴显示。"
        "指令数是在同一张图上由 C99 解释器记录的各活跃 lane 动态指令总数，包含 EBREAK。"
        "RTL 忙拍来自 Step 2 125/100 MHz 时钟比下的阶段计数，按 NPU 时钟域计数。",
        "",
        "| 排名 | ONNX 算子与节点 | 输入→输出 | 动态 lane 指令 | GPU 忙拍 | 整帧占比 | GPU 占比 |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for index, r in enumerate(rows[:10], 1):
        lines.append(f"| {index} | {r['onnx_op']} \`{r['node']}\` | {r['shape']} | {r['lane_instructions']:,} | "
                     f"{r['ticks']:,} | {r['frame_percent']:.2f}% | {r['gpu_percent']:.2f}% |")
    lines += [
        "",
        "## 优化顺序",
        "",
        "1. 宽 K 卷积尽量只在 GPU 合并 INT32 部分和。用 NPU 原有逐通道缩放、饱和与 SiLU 处理最终结果，"
        "避免在每条 GPU lane 上执行两次软件宽积和裁剪分支。",
        "2. 保留 Q23 指数和。对逐元素归一化减少 DIVU／REMU。每个替代算法先跑四张真图的 C99 完整模型，"
        "双阈值通过后才安排硬件长跑。",
        "3. 在另一 SRAM 半区预取或发射独立引擎工作。保留重叠前后图的 DDR 依赖和引擎寄存器顺序，"
        "每个步骤单独留存周期与精度消融。",
        "",
    ]
    args.report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
