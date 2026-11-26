#!/usr/bin/env python3
"""Emit a reusable Z15 YHB1 program from an official C6 PTQ model directory."""

import argparse
import json
from pathlib import Path

from hetero_compiler.c6_compile import compile_graph, fuse_conv_edge_luts
from hetero_compiler.c6_quant import QuantGraph
from hetero_compiler.c8_vector import fuse_vector_partitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    manifest, _, _ = compile_graph(args.output, graph, source.constants,
                                   fast_reduce=args.fast_reduce, overlap=args.overlap,
                                   reduce_bank_skew=args.reduce_bank_skew,
                                   reduce_in_scratch=args.reduce_in_scratch,
                                   stream_overlap=args.stream_overlap)
    print(json.dumps({"program": str(args.output / "program.yhb"),
                      "hardware_ops": len(manifest["c6"]["hardware_ops"]),
                      "ddr_bytes": manifest["plan"]["ddr_peak"],
                      "scratch_bytes_per_half": manifest["plan"]["scratch_peak_by_half"],
                      "dma_bytes_per_frame": manifest["plan"]["dma_bytes"],
                      "stages": len(manifest["plan"]["stages"])}))


if __name__ == "__main__":
    main()
