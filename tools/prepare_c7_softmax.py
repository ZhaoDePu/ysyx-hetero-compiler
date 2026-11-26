"""Generate fixed-point C7 ablation cases using the audited C6 fixtures."""

import argparse
import json
from math import log, log2
from pathlib import Path
import shutil

import numpy as np

from hetero_compiler.c6_quant import QuantGraph, requant
from tools.c6_ptq import aggregate, compare, detections, multiplier_shifts

_MODES = ("q23", "vpu", "vpu-raw", "log2", "log4", "log8")
_LETTERBOX = ("gain", "pad_left", "pad_top", "source_width", "source_height")


def apply_modes(graph, modes):
    attention = [node for node in graph["nodes"] if node["op"] == "Q23Softmax"]
    if len(attention) != len(modes):
        raise ValueError("need one mode per audited attention layer")
    for node, mode in zip(attention, modes, strict=True):
        attrs = node["attrs"]
        scale = graph["tensors"][node["inputs"][0]]["scale"]
        if mode.startswith("log"):
            node["op"] = "C7LogSoftmax"
            attrs.update(scale=scale, steps=int(mode[3:]))
        elif mode in ("vpu", "vpu-raw"):
            if mode == "vpu-raw":
                score = node["inputs"][0]
                dot, = (n for n in graph["nodes"] if n["outputs"] == [score])
                if dot["op"] != "MatMul" or graph["use_counts"][score] != 1:
                    raise ValueError("native raw softmax requires an exclusive MatMul score tensor")
                dot["attrs"]["raw_output"] = True
                scale *= dot["attrs"]["mul"] / (1 << dot["attrs"]["shift"])
                tensor = graph["tensors"][score]
                tensor.update(dtype="int32", scale=scale, bytes=tensor["bytes"] * 4)
            node["op"], node["device"] = "VpuSoftmax", "VPU"
            node["attrs"] = {"axis": attrs["axis"], "shift": round(log2(log(2) / scale))}
    graph["softmax_mode"] = list(modes)


def write_vectors(folder, stem, output, captures, contracts):
    np.save(folder / f"{stem}.npy", output)
    for name, suffix, dtype, values in (
        ("coordinates", "coords_q16_16", "<i4", np.rint(output[:, :4].astype(np.float64) * 65536)),
        ("scores", "scores_q16_16", "<i4", np.rint(output[:, 4].astype(np.float64) * 65536)),
        ("classes", "classes_i64", "<i8", output[:, 5]),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite {name} in {stem}")
        values.astype(dtype).tofile(folder / f"{stem}.{suffix}.bin")
    if set(captures) != set(contracts):
        raise ValueError(f"{stem}: missing a detection head")
    for tensor, desc in contracts.items():
        array = captures[tensor]
        if desc["layout"] == "NHWC":
            array = array.transpose(0, 2, 3, 1)
        array.astype(desc["dtype"]).tofile(folder / (stem + desc["suffix"]))


def _calibration_images(path):
    from PIL import Image

    inputs = []
    for file in sorted(path.glob("*.jpg")):
        with Image.open(file) as original:
            rgb = original.convert("RGB")
        width, height = rgb.size
        gain = min(320 / width, 320 / height)
        resized_width, resized_height = round(width * gain), round(height * gain)
        canvas = Image.new("RGB", (320, 320), (114, 114, 114))
        offset = (round((320 - resized_width) / 2 - .1), round((320 - resized_height) / 2 - .1))
        canvas.paste(rgb.resize((resized_width, resized_height), Image.Resampling.BILINEAR), offset)
        inputs.append(np.asarray(canvas, dtype=np.float32)[None].transpose(0, 3, 1, 2) / np.float32(255))
    if not inputs:
        raise ValueError("archived natural calibration inputs are required")
    return inputs


def calibrate_raw_pv(ref, path, q23):
    """Fit one native PV gain per attention, in topological order."""
    frames = _calibration_images(path)
    nodes = {n["id"]: n for n in ref.graph["nodes"]}
    records = []
    for node_id, attrs in q23:
        softmax = nodes[node_id]
        if softmax["op"] != "VpuSoftmax" or ref.tensors[softmax["inputs"][0]]["dtype"] != "int32":
            raise ValueError("PV gain calibration requires native raw scores at both blocks")
        prefix = node_id.removesuffix("Softmax")
        dot, pv = (nodes[prefix + suffix] for suffix in ("MatMul", "MatMul_1"))
        score, probs, value = dot["outputs"][0], softmax["outputs"][0], pv["inputs"][0]
        wanted = {score, probs, value}
        weight_lut = np.asarray(attrs["lut"], dtype=np.float64)
        statistics = np.zeros(4, dtype=np.float64)
        for frame in frames:
            tensors = {}

            def capture(node, name, array):
                if name in wanted:
                    tensors[name] = array.copy()

            ref.infer(frame, capture=capture)
            old_scores = requant(tensors[score], dot["attrs"]["mul"], dot["attrs"]["shift"])
            delta = old_scores.max(-1, keepdims=True).astype(np.int16) - old_scores
            weights = weight_lut[delta]
            expected = np.rint(weights * 256 / weights.sum(-1, keepdims=True)).clip(0, 255).astype(np.int64)
            native = tensors[probs].astype(np.int64)
            val = tensors[value].astype(np.int64)
            y_expected = val @ expected.swapaxes(-1, -2)
            y_native = val @ native.swapaxes(-1, -2)
            statistics += np.array([np.sum(y_expected * y_native), np.sum(y_native * y_native),
                                    expected.sum(), native.sum()])
        if statistics[1] == 0 or statistics[3] == 0:
            raise ValueError(f"{node_id}: degenerate native calibration response")
        gain = statistics[0] / statistics[1]
        params = pv["attrs"]
        mul, shift = multiplier_shifts(gain * params["mul"] / (1 << params["shift"]))
        params.update(mul=int(mul), shift=int(shift))
        records.append({"node": pv["id"], "gain": gain, "mul": int(mul), "shift": int(shift),
                        "probability_mass_ratio": statistics[2] / statistics[3]})
    return {"images": len(frames), "method": "native PV least squares against Q23 responses",
            "blocks": records}


def prepare(archive, destination, modes, *, save=True, calibrate_pv=False):
    source = archive / "m2-320"
    fp32 = json.loads((archive / "m1/fp32_320.json").read_text())
    ref = QuantGraph(source)
    q23 = [(n["id"], dict(n["attrs"])) for n in ref.graph["nodes"] if n["op"] == "Q23Softmax"]
    apply_modes(ref.graph, modes)
    calibration = calibrate_raw_pv(ref, archive / "m1/calibration", q23) if calibrate_pv else None
    contracts = json.loads((source / "vectors/pl_outputs.json").read_text())
    if save:
        destination.mkdir(parents=True, exist_ok=False)
        vectors = destination / "vectors"
        vectors.mkdir()
        (destination / "quant.json").write_text(json.dumps(ref.graph, indent=2) + "\n")
        (vectors / "pl_outputs.json").write_text(json.dumps(contracts, indent=2) + "\n")
        shutil.copyfile(source / "weights.npz", destination / "weights.npz")
    ledger = {}
    for image, expected in fp32.items():
        scene = Path(image).stem
        rgb_path = source / "vectors" / (scene + ".rgb_nhwc_u8.bin")
        rgb = np.fromfile(rgb_path, dtype=np.uint8).reshape(1, 320, 320, 3)
        inputs = rgb.transpose(0, 3, 1, 2).astype(np.float32) / np.float32(255)
        outputs = {}

        def capture(node, name, value):
            if name in contracts:
                outputs[name] = value

        result = next(iter(ref.infer(inputs, capture=capture).values())).reshape(-1, 6)
        meta = tuple(expected["letterbox"][key] for key in _LETTERBOX)
        int8 = detections(result, meta, .25)
        ledger[image] = {
            "fp32": expected["detections"],
            "int8": int8,
            "at_0_25": compare(expected["detections"], int8, .25),
            "at_0_40": compare(expected["detections"], int8, .40),
        }
        if save:
            write_vectors(vectors, scene, result, outputs, contracts)
            for suffix in ("rgb_nhwc_u8.bin", "input_nhwc_s8.bin", "ppm"):
                path = source / "vectors" / f"{scene}.{suffix}"
                shutil.copyfile(path, vectors / path.name)
    totals = {key: aggregate([row[key] for row in ledger.values()]) for key in ("at_0_25", "at_0_40")}
    report = {"modes": list(modes), "scenes": ledger, "aggregate": totals}
    if calibration is not None:
        report["calibration"] = calibration
    if save:
        (destination / "accuracy.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--modes", required=True, choices=_MODES, nargs=2)
    parser.add_argument("--calibrate-pv", action="store_true",
                        help="fit both native PV gains on the C6 training-only calibration images")
    args = parser.parse_args()
    if args.calibrate_pv and args.modes != ["vpu-raw", "vpu-raw"]:
        parser.error("PV gain calibration requires both attention blocks set to vpu-raw")
    result = prepare(args.archive, args.output, args.modes, calibrate_pv=args.calibrate_pv)
    print(json.dumps(result["aggregate"]))


if __name__ == "__main__":
    main()
