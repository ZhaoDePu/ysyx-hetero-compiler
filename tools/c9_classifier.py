"""Compile and replay a C9 classifier on the unchanged Z15 Step 2 shell."""

import argparse
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from hetero_compiler.c6_compile import _conv, compile_graph, fuse_conv_edge_luts, physical_array
from hetero_compiler.c6_memory import align_dma
from hetero_compiler.c6_quant import QuantGraph
from hetero_compiler.c9_gpu_conv import lower_gpu_conv
from hetero_compiler.c9_gpu_misc import lower_gpu_depthwise, lower_gpu_partition, lower_gpu_vector
from hetero_compiler.c9_spatial import average_tiles, max_pool_tiles, pixel_map, stem_tiles
from tools.c6_accounting import runtime_ledger, write_csv


def _shape(ctx, node):
    source, target = node["inputs"][0], node["outputs"][0]
    _, h, w, ci = ctx.tensors[source]["physical_shape"]
    _, oh, ow, co = ctx.tensors[target]["physical_shape"]
    return source, target, h, w, ci, oh, ow, co


def _stem(node, ctx):
    name, attrs = node["id"], node["attrs"]
    src, dst, h, w, ci, oh, ow, co = _shape(ctx, node)
    if ci != 4 or attrs["kernel"] != [7, 7] or attrs["pads"] != [3] * 4 or attrs["strides"] != [2, 2]:
        raise ValueError(f"{name}: expected stride-two padded 7x7 RGB stem")
    weights = np.zeros((co, 7, 7, 4), dtype=np.int8)
    weights[..., :3] = ctx.constants[attrs["weight"]].transpose(3, 0, 1, 2)
    w_name = ctx.constant(f"{name}/@kernel", weights.tobytes())
    b_name = ctx.constant(f"{name}/@bias", ctx.constants[attrs["bias"]].astype("<i4").tobytes())
    offsets = pixel_map(h, w, oh, ow, ci, 7, 2, 3).tobytes()
    map_name = ctx.constant(f"{name}/@map", offsets.ljust(align_dma(len(offsets)) + 32, b"\0"))

    def schedule(slots):
        return (stem_tiles(name, slots[src].offset, slots[w_name].offset, slots[b_name].offset,
                           slots[map_name].offset, slots[dst].offset,
                           height=h, width=w, oh=oh, ow=ow, channels=co,
                           mul=attrs["mul"], shift=attrs["shift"], lut=attrs.get("lut")),)
    return schedule


def _max_pool(node, ctx):
    name, attrs = node["id"], node["attrs"]
    src, dst, h, w, ci, oh, ow, co = _shape(ctx, node)
    if ci != co or attrs["kernel_shape"] != [3, 3] or attrs["strides"] != [2, 2] or attrs["pads"] != [1]*4:
        raise ValueError(f"{name}: expected signed 3x3, stride-two max pooling")
    offsets = pixel_map(h, w, oh, ow, ci, 3, 2, 1).tobytes()
    mapping = ctx.constant(f"{name}/@map", offsets.ljust(align_dma(len(offsets)), b"\0") +
                           bytes([128]) * align_dma(ci))

    def schedule(slots):
        return (max_pool_tiles(name, slots[src].offset, slots[mapping].offset,
                               slots[dst].offset, height=h, width=w, oh=oh, ow=ow, channels=ci),)
    return schedule


def _average(node, ctx):
    src, dst, h, w, ci, oh, ow, co = _shape(ctx, node)
    if (oh, ow, co) != (1, 1, ci):
        raise ValueError(f"{node['id']}: expected global spatial mean")

    def schedule(slots):
        return (average_tiles(node["id"], slots[src].offset, slots[dst].offset, channels=ci, pixels=h*w),)
    return schedule


def build(model, out, level, selected=None):
    source = QuantGraph(model)
    graph = fuse_conv_edge_luts(source.graph) if level in ("A2", "A3") else source.graph
    seeded = None
    if selected is not None:
        # Selected operators expose their original graph values as input/output.
        graph = source.graph
        selected_nodes = [node for node in graph["nodes"] if node["id"] in selected]
        produced = {name for node in selected_nodes for name in node["outputs"]}
        external = {name for node in selected_nodes for name in node["inputs"]
                    if name and name not in produced and name not in source.constants}
        with np.load(model / "captures.npz", allow_pickle=False) as captures:
            seeded = {name: captures[name] for name in sorted(external)}
    optimized = level in ("A2", "A3")

    def conv(node, ctx):
        if level == "A0":
            lower = lower_gpu_conv if node["attrs"]["groups"] == 1 else lower_gpu_depthwise
            return lower(node, ctx)
        if node["attrs"]["kernel"] == [7, 7]:
            return _stem(node, ctx)
        return _conv(node, ctx, fast_reduce=optimized, reduce_in_scratch=level == "A3")

    lowerers = {"Conv": conv, "MaxPool": _max_pool, "GlobalAveragePool": _average}
    if level in ("A0", "A1"):
        lowerers.update(dict.fromkeys(("EWAdd", "RequantLUT", "Lookup"), lower_gpu_vector))
        lowerers.update(dict.fromkeys(("Split", "Concat"), lower_gpu_partition))
    manifest, tensors, _ = compile_graph(out, graph, source.constants, classifier=True,
                                         reuse_ddr=optimized, lowerers=lowerers,
                                         overlap=level == "A3", stream_overlap=level == "A3",
                                         selected=selected, seeded=seeded)
    return manifest, tensors


def replay(model, out, level, executor, npu_mhz, gpu_mhz, timeout, selected=None):
    manifest, tensors = build(model, out, level, selected)
    log = out / "rtl.log"
    cmd = [str(executor), str(out / "program.yhb"), "--backend", "verilator", "--dump-stats"]
    with np.load(model / "captures.npz", allow_pickle=False) as captures:
        for symbol, binding in manifest["input_bindings"].items():
            if not binding["needs_data"]:
                continue
            data = physical_array(symbol, captures[symbol], tensors)
            path = out / f"input-{len(cmd)}.bin"
            path.write_bytes(data)
            cmd.extend(["--input", f"{symbol}={path}"])
        outputs = {}
        for symbol in manifest["output_bindings"]:
            path = out / f"output-{len(cmd)}.bin"
            outputs[symbol] = path
            cmd.extend(["--output", f"{symbol}={path}"])
        with log.open("w") as stream:
            subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=timeout)
        compared = 0
        for symbol, path in outputs.items():
            expected = physical_array(symbol, captures[symbol], tensors)
            if path.read_bytes() != expected:
                raise ValueError(f"{symbol}: frozen RTL output differs from the independent golden")
            compared += len(expected)
    stages, metrics = runtime_ledger(manifest, log)
    if re.search(r"Verilator (read:|write offset=)|SLVERR", log.read_text()):
        raise ValueError("AXI-Lite diagnostic on positive replay")
    if metrics["dma_bytes"] != manifest["plan"]["dma_bytes"]:
        raise ValueError("planned and real DMA traffic differ")
    row = {**metrics, "stage": level, "npu_mhz": npu_mhz, "gpu_mhz": gpu_mhz,
           "compared_bytes": compared, "slverr": 0,
           "sram_peak_bytes": manifest["plan"]["scratch_peak"],
           "ddr_peak_bytes": manifest["plan"]["ddr_peak"]}
    write_csv(out / "rtl-stages.csv", stages)
    (out / "qualified.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--level", choices=("A0", "A1", "A2", "A3"), required=True)
    cli.add_argument("--executor", type=Path)
    cli.add_argument("--npu-mhz", type=float, default=125)
    cli.add_argument("--gpu-mhz", type=float, default=100)
    cli.add_argument("--timeout", type=int, default=2400)
    cli.add_argument("--op", action="append")
    args = cli.parse_args()
    selected = set(args.op) if args.op else None
    if args.executor:
        result = replay(args.model, args.out, args.level, args.executor,
                        args.npu_mhz, args.gpu_mhz, args.timeout, selected)
    else:
        manifest, _ = build(args.model, args.out, args.level, selected)
        result = {k: v for k, v in manifest["plan"].items() if k not in ("allocations", "stages")}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
