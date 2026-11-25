#!/usr/bin/env python3
"""Emit a reusable Z15 YHB1 program from an official C6 PTQ model directory."""

import argparse
import json
from pathlib import Path

from hetero_compiler.c6_compile import compile_graph
from hetero_compiler.c6_quant import QuantGraph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = QuantGraph(args.quant)
    manifest, _, _ = compile_graph(args.output, source.graph, source.constants)
    print(json.dumps({"program": str(args.output / "program.yhb"),
                      "hardware_ops": len(manifest["c6"]["hardware_ops"]),
                      "ddr_bytes": manifest["plan"]["ddr_peak"],
                      "scratch_bytes_per_half": manifest["plan"]["scratch_peak_by_half"],
                      "dma_bytes_per_frame": manifest["plan"]["dma_bytes"],
                      "stages": len(manifest["plan"]["stages"])}))


if __name__ == "__main__":
    main()
