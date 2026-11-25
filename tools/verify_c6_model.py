#!/usr/bin/env python3
"""Compile and replay selected official C6 graph layers on the Step 2 shell."""

import argparse
from fnmatch import fnmatchcase
import json
from pathlib import Path
import subprocess

import numpy as np

from hetero_compiler.c6_compile import compile_graph, physical_array
from hetero_compiler.c6_quant import QuantGraph
from hetero_compiler.c6_schedule import emit_rtl_control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant", type=Path, required=True)
    parser.add_argument("--input-nhwc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--layer", action="append", metavar="GLOB",
                        help="select official graph node IDs for a representative replay")
    args = parser.parse_args()

    ref = QuantGraph(args.quant)
    selected = (None if args.layer is None else
                {node["id"] for node in ref.graph["nodes"] if node["device"] != "PS" and
                 any(fnmatchcase(node["id"], pattern) for pattern in args.layer)})
    if selected == set():
        parser.error("no hardware layer matched the selection")

    image = ref.graph["inputs"][0]
    image_desc = ref.tensors[image]
    codes = np.fromfile(args.input_nhwc, dtype=np.int8).reshape(image_desc["physical_shape"])
    nchw = codes.transpose(0, 3, 1, 2)
    captures = {image: nchw}
    if selected is not None:
        active = [node for node in ref.graph["nodes"] if node["id"] in selected]
        needed = {name for node in active for name in node["inputs"] + node["outputs"]}
        produced = {name for node in active for name in node["outputs"]}
        boundary_inputs = needed - produced - ref.constants.keys() - {""}
    else:
        active = [node for node in ref.graph["nodes"] if node["device"] == "PS"]
        needed = {name for node in active for name in node["inputs"]}
        boundary_inputs = {image}

    def capture(node, name, data):
        if name in needed:
            captures[name] = data.copy()

    ref.infer(nchw.astype(np.float32) * np.float32(image_desc["scale"]), capture=capture)
    manifest, tensors, initial = compile_graph(
        args.output, ref.graph, ref.constants, selected=selected,
        seeded={name: captures[name] for name in boundary_inputs})
    aliases = manifest.get("c6", {}).get("symbols", {})
    outputs = {name: physical_array(aliases.get(name, name), captures[aliases.get(name, name)], tensors)
               for name in manifest["graph"]["outputs"] if name not in {"c6_head_scales", "c6_score_lut"}}
    control = emit_rtl_control(args.output, manifest, initial, outputs)
    plan = manifest["plan"]
    print(json.dumps({"layers": len(manifest["c6"]["hardware_ops"]),
                      "calls": len(plan["stages"]), "ddr_peak": plan["ddr_peak"],
                      "scratch_peak": plan["scratch_peak_by_half"], "dma_bytes": plan["dma_bytes"],
                      "output_bytes": sum(map(len, outputs.values()))}), flush=True)
    if args.runner:
        subprocess.run([args.runner, control], check=True)


if __name__ == "__main__":
    main()
