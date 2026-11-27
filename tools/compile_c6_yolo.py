#!/usr/bin/env python3
"""Emit a Step 2 YHB1 program or a Phase 17 RTL replay from a C6 PTQ model."""

import argparse
import json
from pathlib import Path

from hetero_compiler.c6_compile import compile_graph, fuse_conv_edge_luts
from hetero_compiler.c6_schedule import emit_rtl_control
from hetero_compiler.c6_quant import QuantGraph
from hetero_compiler.c8_vector import fuse_vector_partitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=("step2", "p17"), default="step2")
    parser.add_argument("--bus-vectors", type=Path,
                        help="C6 bus.input_nhwc_s8.bin and bus.pl_out_*.bin; emit RTL replay")
    parser.add_argument("--fast-reduce", action="store_true",
                        help="use aligned GPU sums and the native NPU requant epilogue")
    parser.add_argument("--overlap", action="store_true",
                        help="schedule independent DMA and GPU/native work across ownership halves")
    parser.add_argument("--fuse-edge-luts", action="store_true",
                        help="compose private requant LUTs into Conv output epilogues")
    parser.add_argument("--reduce-bank-skew", action="store_true",
                        help="rotate SIMT lane addresses across the four G3 SRAM banks")
    parser.add_argument("--reduce-in-scratch", action="store_true",
                        help="keep wide-reduction INT32 partials in their ownership half")
    parser.add_argument("--stream-overlap", action="store_true",
                        help="overlap same-layer raw stages with the other half's GPU sum")
    parser.add_argument("--fuse-vector-partitions", action="store_true",
                        help="fuse split/LUT/concat and remove inverse PV transposes")
    args = parser.parse_args()
    source = QuantGraph(args.quant)
    graph = fuse_vector_partitions(source.graph) if args.fuse_vector_partitions else source.graph
    if args.fuse_edge_luts:
        graph = fuse_conv_edge_luts(graph)
    manifest, tensors, initial = compile_graph(args.output, graph, source.constants,
                                   fast_reduce=args.fast_reduce, overlap=args.overlap,
                                   reduce_bank_skew=args.reduce_bank_skew,
                                   reduce_in_scratch=args.reduce_in_scratch,
                                   stream_overlap=args.stream_overlap, target=args.target)
    if args.bus_vectors:
        image, = manifest["graph"]["inputs"]
        raw = (args.bus_vectors / "bus.input_nhwc_s8.bin").read_bytes()
        n, h, w, c = tensors[image]["physical_shape"]
        if c != 4 or len(raw) != n * h * w * 3:
            raise ValueError("expected packed RGB input with three channels")
        seed = bytearray(n * h * w * c)
        for i in range(3):
            seed[i::4] = raw[i::3]
        golden = {}
        names = (f"{role}_stride{scale}" for scale in (8, 16, 32) for role in ("boxes", "scores"))
        for i, name in enumerate(names):
            data = (args.bus_vectors / f"bus.pl_out_{i}.bin").read_bytes()
            if len(data) != manifest["output_bindings"][name]["size"]:
                raise ValueError(f"unexpected golden vector length for {name}")
            golden[name] = data
        emit_rtl_control(args.output, manifest, {**initial, image: seed}, golden)
    print(json.dumps({"program": str(args.output / ("program.yhb" if args.target == "step2" else "manifest.json")),
                      "hardware_ops": len(manifest["c6"]["hardware_ops"]),
                      "ddr_bytes": manifest["plan"]["ddr_peak"],
                      "scratch_bytes_per_half": manifest["plan"]["scratch_peak_by_half"],
                      "dma_bytes_per_frame": manifest["plan"]["dma_bytes"],
                      "stages": len(manifest["plan"]["stages"])}))


if __name__ == "__main__":
    main()
