"""Compare C8 and P17 spatial/channel K-chain plans layer by layer."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from hetero_compiler.c14_conv import can_chain_conv
from hetero_compiler.c14_spatial import panels, row_plan, split_1x1
from hetero_compiler.c6_compile import fuse_conv_edge_luts, prepare_graph
from hetero_compiler.c6_quant import QuantGraph
from hetero_compiler.c6_tensor import Conv
from hetero_compiler.c8_vector import fuse_vector_partitions


def scheduled(graph, constants, target):
    _, _, _, _, _, layers, _, info = prepare_graph(
        graph, constants, target=target, fast_reduce=True, reduce_bank_skew=True, reduce_in_scratch=True)
    all_layers = iter(layers)
    result = {}
    for entry in info["hardware_ops"]:
        tiles = [tile for _ in entry["passes"] for tile in next(all_layers)]
        transfers = [dma for tile in tiles for dma in (*tile.loads, *tile.stores)]
        selected = [s.id for tile in tiles for s in tile.stages if "/chain" in s.id]
        result[entry["id"]] = {
            "dma_bytes": sum(dma.size for dma in transfers),
            "descriptors": len(transfers),
            "peak_half0": max((tile.scratch_bytes for tile in tiles if tile.zone == 0), default=0),
            "peak_half1": max((tile.scratch_bytes for tile in tiles if tile.zone == 1), default=0),
            "chain_stages": len(selected),
            "chain_tiles": len({s.rsplit("/chain", 1)[0] for s in selected}),
        }
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("quant", type=Path)
    cli.add_argument("out", type=Path)
    args = cli.parse_args()
    model = QuantGraph(args.quant)
    graph = fuse_conv_edge_luts(fuse_vector_partitions(model.graph))
    baseline, p17 = (scheduled(graph, model.constants, target) for target in ("step2", "p17"))
    rows = []
    for node in graph["nodes"]:
        if node["op"] != "Conv" or node["attrs"]["groups"] != 1:
            continue
        x = graph["tensors"][node["inputs"][0]]["shape"]
        y = graph["tensors"][node["outputs"][0]]["shape"]
        params = node["attrs"]
        spec = Conv(node["id"], x[2], x[3], x[1], y[1], params["kernel"][0],
                    params["strides"][0], params["pads"][0])
        if spec.depth <= 511:
            continue
        op = node["id"]
        old, new = baseline[op], p17[op]
        oc = panels(spec)
        tiles = row_plan(spec, 4 * y[1] + y[1] * spec.depth) if new["chain_tiles"] and not can_chain_conv(spec) else [(0, y[2])]
        coverage = bool(new["chain_tiles"])
        rows.append({"name": op, "input_channels": x[1], "m": y[2] * y[3], "n": y[1], "k": spec.depth,
                     "covered": coverage, "macro_rows": json.dumps(tiles) if coverage else "",
                     "widths": json.dumps([min(16, y[3]-i) for i in range(0, y[3], 16)]) if coverage else "",
                     "oc_panels": json.dumps(oc) if coverage else "",
                     "chain_tiles": new["chain_tiles"], "chain_stages": new["chain_stages"],
                     "half0_bytes": new["peak_half0"], "half1_bytes": new["peak_half1"],
                     "c8_dma_bytes": old["dma_bytes"], "p17_dma_bytes": new["dma_bytes"],
                     "c8_descriptors": old["descriptors"], "p17_descriptors": new["descriptors"],
                     "mode": ("split_gemm" if split_1x1(spec) else "implicit_conv") if coverage else "c8_fallback"})
    with args.out.open("w", newline="") as out:
        writer = csv.DictWriter(out, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name, data in (("C8", baseline), ("C14-v2", p17)):
        totals = Counter()
        for r in data.values():
            totals.update({k: r[k] for k in ("dma_bytes", "descriptors")})
        print(name, dict(totals))
    print(f"covered={sum(row['covered'] for row in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
