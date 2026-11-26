"""Calibrate official MobileNetV2, export standard Q/DQ and integer Z15 fixtures."""

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from examples.generate_quantized_onnx import FLOAT, INT8, Model, field, varint, value_info


WEIGHTS_URL = "https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth"
WEIGHTS_SHA256 = "7ebf99e03e254b273379b23edca7ec0da9f48273b23a332b93c1c99d49e86e8f"
CHANNEL_BLOCK = 256
CODES = np.arange(256, dtype=np.uint8).view(np.int8).astype(np.int16)


def digest(path):
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def signed_codes(values):
    return np.clip(np.rint(values), -128, 127).astype(np.int8)


def scale(values):
    return max(float(np.max(np.abs(values)) / 127), 1e-8)


def calibrate_multiplier(ratio):
    """Approximate a positive ratio by the frozen u16 multiplier and u5 shift."""
    ratio = np.asarray(ratio, dtype=np.float64)
    if np.any(~np.isfinite(ratio)) or np.any(ratio <= 0) or np.any(ratio > 65535):
        raise ValueError("unsupported native requantization ratio")
    shifts = np.clip(np.floor(np.log2(65535 / ratio)), 0, 31).astype(np.int64)
    factors = np.clip(np.rint(np.ldexp(ratio, shifts)), 1, 65535).astype(np.int64)
    return factors, shifts


def resize_codes(value):
    """Pack an already normalized/preprocessed RGB code tensor for Step 2."""
    if value.ndim != 4 or value.shape[:2] != (1, 3) or value.dtype != np.int8:
        raise ValueError("expected one signed RGB code tensor")
    return np.ascontiguousarray(np.pad(value.transpose(0, 2, 3, 1), ((0, 0), (0, 0), (0, 0), (0, 1))))


def fixed_conv(value, weights, bias, stride, pad, groups):
    """Independent integer NCHW/OIHW convolution without compiler or HAL calls."""
    if not (groups == 1 or groups == value.shape[1] == weights.shape[0]):
        raise ValueError("expected dense or multiplier-one depthwise convolution")
    height, width = weights.shape[-2:]
    padded = np.pad(value, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    patches = np.lib.stride_tricks.sliding_window_view(padded, (height, width), axis=(-2, -1))
    patches = patches[:, :, ::stride, ::stride].astype(np.int64)
    weights = weights.astype(np.int64)
    if groups == 1:
        batch, _, rows, columns, _, _ = patches.shape
        dot = patches.transpose(0, 2, 3, 1, 4, 5).reshape(batch * rows * columns, -1)
        result = (dot @ weights.reshape(len(weights), -1).T).reshape(batch, rows, columns, -1)
        result = result.transpose(0, 3, 1, 2)
    else:
        result = np.einsum("nchwkl,ckl->nchw", patches, weights[:, 0], optimize=False)
    result += bias.reshape(1, -1, 1, 1)
    if result.min() < -(1 << 31) or result.max() >= 1 << 31:
        raise ValueError("oracle convolution exceeds the signed INT32 accumulator")
    return result


@dataclass(frozen=True)
class Step:
    name: str
    op: str
    inputs: tuple
    weights: np.ndarray | None = None
    bias: np.ndarray | None = None
    stride: int = 1
    pad: int = 0
    groups: int = 1


def torchvision_steps(model):
    """Walk the 17 inverted residual blocks and fold each inference BatchNorm."""
    steps = []

    def block(name, source, conv, bn, activation):
        gain = bn.weight.numpy().astype(np.float64) / np.sqrt(
            bn.running_var.numpy().astype(np.float64) + bn.eps)
        weights = (conv.weight.numpy() * gain[:, None, None, None]).astype(np.float32)
        bias = (bn.bias.numpy() - bn.running_mean.numpy() * gain).astype(np.float32)
        steps.append(Step(name, "Conv", (source,), weights, bias, conv.stride[0], conv.padding[0], conv.groups))
        if activation:
            steps.append(Step(name + "/relu6", "Clip", (name,)))
        return steps[-1].name

    stem = model.features[0]
    current = block("features/0", "images", stem[0], stem[1], True)
    for index, feature in enumerate(model.features[1:-1], 1):
        ops = list(feature.conv.children())
        residual = current
        for offset, nested in enumerate(ops[:-2]):
            current = block(f"features/{index}/{offset}", current, nested[0], nested[1], True)
        current = block(f"features/{index}/projection", current, *ops[-2:], False)
        if feature.use_res_connect:
            name = f"features/{index}/residual"
            steps.append(Step(name, "Add", (residual, current)))
            current = name
    final = model.features[-1]
    current = block("features/18", current, final[0], final[1], True)
    steps.append(Step("avgpool", "GlobalAveragePool", (current,)))
    classifier = model.classifier[1]
    steps.append(Step("logits", "Gemm", ("avgpool",), classifier.weight.numpy().reshape(1000, 1280, 1, 1),
                      classifier.bias.numpy()))
    if len([step for step in steps if step.op == "Conv"]) != 52:
        raise ValueError("unexpected official MobileNetV2 convolution structure")
    if len([step for step in steps if step.op == "Add"]) != 10:
        raise ValueError("unexpected official MobileNetV2 residual structure")
    return steps


def load_frame(path, image_size):
    """Use the IMAGENET1K_V2 mean/std and 232:224 resize/crop ratio."""
    from PIL import Image
    import torch
    from torchvision.transforms import functional as tf
    from torchvision.transforms import InterpolationMode

    with Image.open(path) as image:
        image = tf.resize(image.convert("RGB"), [round(image_size * 232 / 224)],
                          InterpolationMode.BILINEAR, antialias=True)
        image = tf.center_crop(image, [image_size, image_size])
        value = tf.pil_to_tensor(image).to(torch.float32) / 255
    return tf.normalize(value, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])[None]


def calibrate(steps, image):
    import torch
    from torch.nn import functional as F

    values = {"images": image}
    with torch.inference_mode():
        for step in steps:
            a, *others = (values[name] for name in step.inputs)
            if step.op in ("Conv", "Gemm"):
                out = F.conv2d(a, torch.from_numpy(step.weights.copy()), torch.from_numpy(step.bias.copy()),
                               stride=step.stride, padding=step.pad, groups=step.groups)
            elif step.op == "Clip":
                out = torch.clamp(a, 0, 6)
            elif step.op == "Add":
                out = a + others[0]
            else:
                out = a.mean(dim=(-2, -1), keepdim=True)
            values[step.name] = out
    return values


def hardware_parameters(step, source_scale, target_scale):
    """Per-output-channel signed symmetric 8-bit weights and scaled INT32 bias."""
    weights = step.weights.astype(np.float64)
    minimum_scale = np.maximum(np.abs(step.bias) / (source_scale * (1 << 22)),
                               np.ldexp(target_scale / source_scale, -31))
    weight_scales = np.maximum(np.max(np.abs(weights), axis=(1, 2, 3)) / 127, minimum_scale)
    weights = signed_codes(weights / weight_scales[:, None, None, None])
    bias = np.rint(step.bias / (source_scale * weight_scales))
    if bias.min() < -(1 << 31) or bias.max() >= 1 << 31:
        raise ValueError(f"{step.name} quantized bias exceeds INT32")
    mul, shift = calibrate_multiplier(source_scale * weight_scales / target_scale)
    return weights, weight_scales.astype(np.float32), bias.astype(np.int32), mul, shift


class QdqModel(Model):
    """Standard opset 18 ONNX with packed raw fields for large initializers."""

    def array(self, name, value, dtype):
        value = np.asarray(value, dtype={FLOAT: "<f4", INT8: "i1"}[dtype])
        dims = field(1, b"".join(varint(d) for d in value.shape)) if value.ndim else b""
        self.tensors[name] = dims + field(2, dtype) + field(8, name) + field(9, value.tobytes())
        return name

    def activate(self, name, producer, factor):
        sc = self.array(name + "/activation-scale", factor, FLOAT)
        self.op(name + "/Q", "QuantizeLinear", [producer, sc, "zero"], name + "/code")
        self.op(name + "/DQ", "DequantizeLinear", [name + "/code", sc, "zero"], name)
        return name


def standard_qdq(steps, shapes, factors, parameters):
    model = QdqModel("MobileNetV2_IMAGENET1K_V2_symmetric_INT8_QDQ")
    model.inputs.append(value_info("images", shapes["images"], FLOAT))
    model.array("zero", 0, INT8)
    model.array("clip-min", 0.0, FLOAT)
    model.array("clip-max", 6.0, FLOAT)
    mapping = {"images": model.activate("normalized", "images", factors["images"])}
    for step in steps:
        name = step.name
        result = name + "/float"
        ins = [mapping[item] for item in step.inputs]
        if step.op in ("Conv", "Gemm"):
            weights, weights_scale, bias, _, _ = parameters[name]
            if step.op == "Gemm":
                weights = weights[:, :, 0, 0]
                ins[0] = model.op(name + "/flatten", "Flatten", ins, name + "/flat", axis=1)
            data = model.array(name + "/weights", weights, INT8)
            ws = model.array(name + "/weights-scale", weights_scale, FLOAT)
            zero = model.array(name + "/weights-zero", np.zeros((len(weights),), dtype=np.int8), INT8)
            w = model.op(name + "/weights-DQ", "DequantizeLinear", [data, ws, zero], name + "/weight-fp", axis=0)
            b = model.array(name + "/bias", bias * factors[step.inputs[0]] * weights_scale, FLOAT)
            attrs = ({"kernel_shape": list(weights.shape[-2:]), "strides": [step.stride] * 2,
                      "pads": [step.pad] * 4, "dilations": [1, 1], "group": step.groups}
                     if step.op == "Conv" else {"transB": 1})
            model.op(name, step.op, [*ins, w, b], result, **attrs)
        else:
            if step.op == "Clip":
                ins += ["clip-min", "clip-max"]
            model.op(name, step.op, ins, result)
        mapping[name] = model.activate(name, result, factors[name])
    model.output("logits", [1, 1000], FLOAT)
    return model.encode()


class IntegerGraph:
    """Emit C6 integer IR, including width legalization only on the target side."""

    def __init__(self, shapes, factors):
        self.tensors = {}
        self.constants = {}
        self.nodes = []
        self.shape = shapes
        self.factors = factors
        self.tensor("images", shapes["images"], factors["images"])

    def tensor(self, name, shape, factor):
        batch, channels, height, width = shape
        self.tensors[name] = {"shape": list(shape), "physical_shape": [batch, height, width, channels],
                              "layout": "NHWC", "dtype": "int8", "scale": factor}

    def node(self, name, op, ins, outs, attrs, device="VPU", macs=0):
        self.nodes.append({"id": name, "op": op, "inputs": list(ins), "outputs": list(outs), "attrs": attrs,
                           "device": device, "estimated_macs": macs})

    def convolution(self, step, parameters):
        name = step.name
        source, = step.inputs
        weights, _, bias, mul, shift = parameters
        count = len(weights)
        widths = [min(CHANNEL_BLOCK, count - first) for first in range(0, count, CHANNEL_BLOCK)] if count > 511 else [count]
        in_shape = self.shape[source]
        is_depthwise = step.groups > 1
        sections = [source] * len(widths)
        if is_depthwise and len(widths) > 1:
            sections = [f"{name}/split-{i}" for i in range(len(widths))]
            for part, width in zip(sections, widths, strict=True):
                self.tensor(part, [in_shape[0], width, *in_shape[2:]], self.factors[source])
            self.node(name + "/split", "Split", [source], sections, {"axis": 1, "split": widths})

        outputs = []
        first = 0
        for index, (part, width) in enumerate(zip(sections, widths, strict=True)):
            branch = name if len(widths) == 1 else f"{name}/part-{index}"
            outputs.append(branch)
            self.tensor(branch, [1, width, *self.shape[name][2:]], self.factors[name])
            weight_id, bias_id = branch + "/hwio", branch + "/bias32"
            end = first + width
            self.constants[weight_id] = np.ascontiguousarray(weights[first:end].transpose(2, 3, 1, 0))
            self.constants[bias_id] = bias[first:end].copy()
            macs = int(np.prod(self.shape[name][2:])) * int(weights[first:end].size)
            self.node(branch, "Conv", [part], [branch], {
                "strides": [step.stride] * 2, "pads": [step.pad] * 4, "dilations": [1, 1],
                "groups": width if is_depthwise else 1, "kernel": list(weights.shape[-2:]),
                "weight": weight_id, "bias": bias_id, "mul": mul[first:end].tolist(), "shift": shift[first:end].tolist(),
            }, "NPU", macs)
            first = end

        if len(outputs) > 1:
            self.tensor(name, self.shape[name], self.factors[name])
            self.node(name + "/join", "Concat", outputs, [name], {"axis": 1})

    def rest(self, step):
        name = step.name
        source, *other = step.inputs
        self.tensor(name, self.shape[name], self.factors[name])
        if step.op == "Clip":
            maximum = min(int(np.rint(6.0 / self.factors[name])), 127)
            table = np.clip(CODES, 0, maximum).tolist()
            self.node(name, "RequantLUT", [source], [name], {"lut": table})
        elif step.op == "Add":
            operands = []
            for index, operand in enumerate((source, *other)):
                symbol = f"{name}/scale-{index}"
                self.tensor(symbol, self.shape[operand], self.factors[name])
                table = signed_codes(CODES * self.factors[operand] / self.factors[name]).tolist()
                self.node(symbol, "RequantLUT", [operand], [symbol], {"lut": table})
                operands.append(symbol)
            self.node(name, "EWAdd", operands, [name], {})
        else:
            self.node(name, "GlobalAveragePool", [source], [name], {})

    def save(self, output, onnx_sha):
        uses = Counter(name for node in self.nodes for name in node["inputs"])
        graph = {"nodes": self.nodes, "tensors": self.tensors, "inputs": ["images"], "outputs": ["logits"],
                 "use_counts": dict(uses), "source_onnx_sha256": onnx_sha,
                 "logical_macs": sum(node["estimated_macs"] for node in self.nodes)}
        np.savez_compressed(output / "weights.npz", **self.constants)
        (output / "quant.json").write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
        return graph


def evaluate(graph, constants, codes):
    """Independently execute target IR and retain the oracle at every boundary."""
    captures = {"images": codes}
    for node in graph["nodes"]:
        attrs = node["attrs"]
        source, *others = (captures[operand] for operand in node["inputs"])
        op = node["op"]
        if op == "Conv":
            inputs = constants[attrs["weight"]].transpose(3, 2, 0, 1)
            actual = fixed_conv(source, inputs, constants[attrs["bias"]], attrs["strides"][0], attrs["pads"][0],
                                attrs["groups"])
            m = np.asarray(attrs["mul"]).reshape(1, -1, 1, 1)
            s = np.asarray(attrs["shift"]).reshape(1, -1, 1, 1)
            result = np.clip(actual * m >> s, -128, 127).astype(np.int8)
        elif op == "RequantLUT":
            result = np.asarray(attrs["lut"], dtype=np.int8)[source.view(np.uint8)]
        elif op == "EWAdd":
            result = np.clip(source.astype(np.int16) + others[0], -128, 127).astype(np.int8)
        elif op == "GlobalAveragePool":
            area = source.shape[-1] * source.shape[-2]
            result = (source.sum(axis=(-1, -2), keepdims=True, dtype=np.int64) // area).astype(np.int8)
        elif op == "Concat":
            result = np.concatenate([source, *others], axis=1)
        elif op == "Split":
            result = tuple(np.split(source, np.cumsum(attrs["split"][:-1]), axis=1))
        else:
            raise ValueError(f"unsupported reference operation {op}")
        results = result if isinstance(result, tuple) else (result,)
        for name, value in zip(node["outputs"], results, strict=True):
            if list(value.shape) != graph["tensors"][name]["shape"]:
                raise ValueError(f"reference shape mismatch for {name}")
            captures[name] = np.ascontiguousarray(value)
    return captures


def run(args):
    import torch
    import torchvision

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.size < 32 or args.size % 32:
        raise ValueError("choose a static resolution that is a positive multiple of 32")
    if digest(args.checkpoint) != WEIGHTS_SHA256:
        raise ValueError("official torchvision checkpoint SHA-256 mismatch")
    model = torchvision.models.mobilenet_v2(weights=None).eval().requires_grad_(False)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    steps = torchvision_steps(model)
    calibration_paths = sorted(args.calibration.glob("*.jpg"))
    if not calibration_paths:
        raise ValueError("calibration directory must contain natural JPG images")
    calibration_hashes = {path.name: digest(path) for path in calibration_paths}
    if digest(args.image) in calibration_hashes.values():
        raise ValueError("the independent evaluation image also occurs in calibration")
    calibration = torch.cat([load_frame(path, args.size) for path in calibration_paths])
    values = calibrate(steps, calibration)
    shapes = {name: [1, *value.shape[1:]] for name, value in values.items()}
    factors = {name: scale(value.numpy()) for name, value in values.items()}
    for step in steps:
        if step.op in ("Clip", "GlobalAveragePool"):
            factors[step.name] = factors[step.inputs[0]]
    # Saturating the raw Conv to the calibrated ReLU6 range is exact for the
    # following clipping step, and matching scales lets the Z15 fuse it.
    for step in steps:
        if step.op == "Clip":
            factors[step.inputs[0]] = factors[step.name] = scale(values[step.name].numpy())
    for step in steps:
        if step.op == "GlobalAveragePool":
            factors[step.name] = factors[step.inputs[0]]
    parameters = {step.name: hardware_parameters(step, factors[step.inputs[0]], factors[step.name])
                  for step in steps if step.op in ("Conv", "Gemm")}

    with torch.inference_mode():
        sample = load_frame(args.image, args.size)
        official_fp32 = model(sample).numpy()
        folded_fp32 = calibrate(steps, sample)["logits"].numpy().reshape(1, -1)
    np.testing.assert_allclose(folded_fp32, official_fp32, rtol=1e-4, atol=2e-4)

    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / "mobilenetv2-int8-qdq.onnx"
    onnx_path.write_bytes(standard_qdq(steps, shapes, factors, parameters))
    executable = IntegerGraph(shapes, factors)
    for step in steps:
        if step.op in ("Conv", "Gemm"):
            executable.convolution(step, parameters[step.name])
        else:
            executable.rest(step)
    graph = executable.save(args.out, digest(onnx_path))
    codes = signed_codes(sample.numpy() / factors["images"])
    captures = evaluate(graph, executable.constants, codes)
    np.savez_compressed(args.out / "captures.npz", **captures)
    np.save(args.out / "fp32-logits.npy", official_fp32, allow_pickle=False)
    (args.out / "images.bin").write_bytes(resize_codes(codes).tobytes())
    logits = captures["logits"].reshape(-1)
    (args.out / "gold-logits.bin").write_bytes(logits.tobytes())
    fp = official_fp32.reshape(-1)
    int8_fp = logits.astype(np.float64) * factors["logits"]
    report = {
        "model": "MobileNetV2", "input_resolution": [args.size, args.size],
        "weight_source": "torchvision MobileNet_V2_Weights.IMAGENET1K_V2 (official pretrained)",
        "checkpoint_url": WEIGHTS_URL, "weights_sha256": WEIGHTS_SHA256,
        "onnx_sha256": digest(onnx_path), "calibration_images": calibration_hashes,
        "evaluation_image": args.image.name, "evaluation_image_sha256": digest(args.image),
        "preprocessing": "IMAGENET1K_V2 normalization, 232:224 resize/crop ratio scaled to test resolution",
        "onnx_arithmetic": "standard Q/DQ QuantizeLinear rounds to nearest-even",
        "fixed_oracle_arithmetic": "independent NumPy integer Conv and floor native requant; signed INT32 accumulators",
        "fp32_top1_class": int(np.argmax(fp)), "fixed_int8_top1_class": int(np.argmax(logits)),
        "fp32_int8_logits_cosine": float(np.dot(fp, int8_fp) / (np.linalg.norm(fp) * np.linalg.norm(int8_fp))),
        "logical_macs": graph["logical_macs"], "logical_ops_two_per_mac": 2 * graph["logical_macs"],
        "int8_output_bytes": int(logits.nbytes), "physical_legalized_ops": len(graph["nodes"]),
        "standard_topology_conv": sum(step.op == "Conv" for step in steps),
        "standard_topology_depthwise_conv": sum(step.op == "Conv" and step.groups > 1 for step in steps),
        "standard_topology_residual_add": sum(step.op == "Add" for step in steps),
    }
    (args.out / "provenance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "calibration_images"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "calibration", "image", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--size", type=int, default=64)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
