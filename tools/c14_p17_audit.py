"""Measure C12 W4 scale compatibility and YOLO native Conv chain coverage."""

import argparse
import json
from pathlib import Path
from struct import iter_unpack


def qwen_scale_audit(layout, model):
    """A common shift can change units but preserves unequal channel scales."""
    weights = json.loads(layout.read_text())
    group = weights["group"]
    rows = []
    with model.open("rb") as image:
        for name, tensor in weights["tensors"].items():
            if tensor["bits"] != 4:
                continue
            n, k = tensor["shape"]
            if k % group or n % 16:
                raise ValueError(f"{name}: tensor shape is incompatible with the probe geometry")
            image.seek(tensor["scale_offset"])
            data = image.read(4 * n * (k // group))
            if len(data) != 4 * n * (k // group):
                raise ValueError(f"{name}: truncated scale table")
            uniform_tiles = distinct_tiles = exact_q08_units = 0
            for tile in iter_unpack("<16I", data):
                uniform_tiles += len(set(tile)) == 1
                distinct_tiles += len(set(tile)) == 16
                exact_q08_units += sum(scale % 65536 == 0 and scale // 65536 <= 256
                                       for scale in tile)
            rows.append({"tensor": name, "groups": k // group, "channels": n,
                         "tiles": len(data) // 64,
                         "uniform_tiles": uniform_tiles, "all_distinct_tiles": distinct_tiles,
                         "scales": len(data) // 4, "scales_exact_q08_same_units": exact_q08_units})
    return rows


def yolo_conv_audit(path):
    """A direct Phase 17 implicit Conv continuation holds a whole MxN tile."""
    graph = json.loads(path.read_text())
    rows = []
    for node in graph["nodes"]:
        if node["op"] != "Conv" or node["attrs"]["groups"] != 1:
            continue
        _, ci, _, _ = graph["tensors"][node["inputs"][0]]["shape"]
        _, co, oh, ow = graph["tensors"][node["outputs"][0]]["shape"]
        ky, kx = node["attrs"]["kernel"]
        k = ky * kx * ((ci + 3) & -4)
        if k <= 511:
            continue
        rows.append({"node": node["id"], "output_hw": [oh, ow], "m_min": ow,
                     "n": co, "k": k, "direct_chain_shape": ow <= 16 and co <= 16,
                     "needs_horizontal_tile": ow > 16, "needs_channel_tile": co > 16})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    args = parser.parse_args()
    result = {"qwen": qwen_scale_audit(args.layout, args.model), "yolo": yolo_conv_audit(args.yolo)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
