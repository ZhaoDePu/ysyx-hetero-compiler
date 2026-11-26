"""Export and calibrate official torchvision ResNet-18 for the C9 experiments."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from examples.generate_quantized_onnx import FLOAT, INT8, INT32, Model, field, value_info, varint


SOURCE = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
DIGEST = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


class QDQ(Model):
    """Use standard ONNX operators with array-based protobuf initializers."""

    def array(self, name, array):
        array = np.asarray(array)
        dtype = {"float32": FLOAT, "int8": INT8, "int32": INT32}[array.dtype.name]
        dims = field(1, b"".join(varint(n) for n in array.shape)) if array.shape else b""
        self.tensors[name] = (dims + field(2, dtype) + field(8, name)
                              + field(9, array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes()))
        return name

    def params(self, name, scale, dtype="int8"):
        scale = np.asarray(scale, dtype=np.float32)
        return [self.array(name + ".scale", scale),
                self.array(name + ".zero", np.zeros(scale.shape, dtype=dtype))]

    def quant(self, name, source, scale):
        params = self.params(name, scale)
        code = self.op(name + ".Q", "QuantizeLinear", [source, *params], name + ".q")
        return self.op(name + ".DQ", "DequantizeLinear", [code, *params], name + ".dq")

    def dequant(self, name, values, scale, axis=0):
        code = self.array(name + ".q", values)
        params = self.params(name, scale, str(values.dtype))
        return self.op(name + ".DQ", "DequantizeLinear", [code, *params], name + ".dq", axis=axis)


def load_image(path, size):
    from PIL import Image

    with Image.open(path) as source:
        rgb = source.convert("RGB")
    target = round(size * 256 / 224)
    shape = tuple(round(n * target / min(rgb.size)) for n in rgb.size)
    rgb = rgb.resize(shape, Image.Resampling.BILINEAR)
    x, y = ((extent - size) // 2 for extent in shape)
    crop = np.asarray(rgb.crop((x, y, x + size, y + size)), dtype=np.float32)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return ((crop / 255 - mean) / std).transpose(2, 0, 1)


def scale_for(values):
    return max(float(values.abs().max().item()) / 127, 1e-8)


def requant_lut(source, target, relu=False):
    codes = np.arange(256, dtype=np.uint8).view(np.int8).astype(np.float64)
    if relu:
        codes = np.maximum(codes, 0)
    return np.rint(codes * (source / target)).clip(-128, 127).astype(np.int8).tolist()


def fixed_multiplier(ratio):
    shift = np.clip(np.floor(np.log2(65535 / np.maximum(ratio, 1e-30))), 0, 31).astype(np.int64)
    mul = np.rint(np.ldexp(ratio, shift)).clip(0, 65535).astype(np.int64)
    return mul.tolist(), shift.tolist()


def packed_desc(cal_values, scale):
    _, channels, height, width = cal_values.shape
    return {"shape": [1, channels, height, width], "physical_shape": [1, height, width, channels],
            "dtype": "int8", "layout": "NHWC", "scale": scale}


def fold_bn(conv, bn):
    weight = conv.weight.detach().double()
    scale = bn.weight.detach().double() / (bn.running_var.detach().double() + bn.eps).sqrt()
    bias = conv.bias.detach().double() if conv.bias is not None else 0
    return ((weight * scale[:, None, None, None]).float(),
            ((bias - bn.running_mean) * scale + bn.bias.detach()).float())


class CalibratedResNet:
    def __init__(self, torch, calibration):
        self.torch = torch
        self.fun = torch.nn.functional
        self.onnx = QDQ("ResNet18_ImageNet1K_V1_QDQ")
        self.values = {"images": calibration}
        self.tensors = {"images": packed_desc(calibration, scale_for(calibration))}
        self.onnx.inputs.append(value_info("images", self.tensors["images"]["shape"], FLOAT))
        self.qnames = {"images": self.onnx.quant("images", "images", self.scale("images"))}
        self.nodes = []
        self.weights = {}

    def scale(self, name):
        return self.tensors[name]["scale"]

    def result(self, name, op, inputs, cal_value, scale=None, device="VPU", **attrs):
        self.tensors[name] = packed_desc(cal_value, scale_for(cal_value) if scale is None else scale)
        self.values[name] = cal_value
        self.nodes.append({"id": name, "inputs": list(inputs), "outputs": [name], "op": op,
                           "device": device, "attrs": attrs})
        return name

    def conv_data(self, source, weight, bias, output_scale):
        unquantized = weight.detach().numpy()
        ws = np.maximum(np.max(np.abs(unquantized), axis=(1, 2, 3)) / np.float32(127), 1e-12)
        qw = np.rint(unquantized / ws[:, None, None, None]).clip(-127, 127).astype(np.int8)
        bs = np.asarray(ws * self.scale(source), dtype=np.float32)
        qb = np.rint(bias.detach().numpy().astype(np.float64) / bs)
        if np.any(np.abs(qb) > (1 << 31) - 1):
            raise ValueError("folded bias does not fit INT32")
        mul, shift = fixed_multiplier(bs.astype(np.float64) / output_scale)
        return qw, qb.astype(np.int32), ws, bs, mul, shift

    def conv_node(self, name, source, value, qw, qb, mul, shift, scale, strides, pads):
        w, b = name + ".weight", name + ".bias"
        self.weights[w] = np.ascontiguousarray(qw.transpose(2, 3, 1, 0))
        self.weights[b] = qb
        return self.result(name, "Conv", [source, w, b], value, scale, "NPU", weight=w, bias=b,
                           kernel=list(qw.shape[2:]), groups=1, strides=strides,
                           pads=pads, dilations=[1, 1], mul=mul, shift=shift)

    def conv(self, name, source, conv, bn):
        w, b = fold_bn(conv, bn)
        strides, pads = list(conv.stride), list(conv.padding) * 2
        cal = self.fun.conv2d(self.values[source], w, b, stride=tuple(strides), padding=conv.padding)
        scale = scale_for(cal)
        qw, qb, ws, bs, mul, shift = self.conv_data(source, w, b, scale)
        if qw.shape[0] <= 511:
            self.conv_node(name, source, cal, qw, qb, mul, shift, scale, strides, pads)
        else:
            parts = []
            for start in range(0, qw.shape[0], 128):
                stop = start + 128
                part = f"{name}.part{start // 128}"
                parts.append(self.conv_node(part, source, cal[:, start:stop], qw[start:stop], qb[start:stop],
                                            mul[start:stop], shift[start:stop], scale, strides, pads))
            self.result(name, "Concat", parts, cal, scale, axis=1)
        weight = self.onnx.dequant(name + ".weight", qw, ws)
        bias = self.onnx.dequant(name + ".bias", qb, bs)
        result = self.onnx.op(name, "Conv", [self.qnames[source], weight, bias], name + ".fp",
                              kernel_shape=list(conv.kernel_size), pads=pads, strides=strides,
                              dilations=[1, 1], group=1)
        self.qnames[name] = self.onnx.quant(name, result, scale)
        return name

    def relu(self, name, source):
        cal = self.values[source].relu()
        scale = scale_for(cal)
        self.result(name, "RequantLUT", [source], cal, scale, lut=requant_lut(self.scale(source), scale, True))
        result = self.onnx.op(name, "Relu", [self.qnames[source]], name + ".fp")
        self.qnames[name] = self.onnx.quant(name, result, scale)
        return name

    def residual(self, name, first, second):
        a, b = self.values[first], self.values[second]
        cal = a + b
        scale = max(scale_for(a), scale_for(b), scale_for(cal))
        arms = []
        for index, source in enumerate((first, second)):
            if self.scale(source) == scale:
                arms.append(source)
                continue
            edge = f"{name}.qedge{index}"
            self.result(edge, "RequantLUT", [source], self.values[source], scale,
                        lut=requant_lut(self.scale(source), scale))
            self.qnames[edge] = self.onnx.quant(edge, self.qnames[source], scale)
            arms.append(edge)
        self.result(name, "EWAdd", arms, cal, scale)
        result = self.onnx.op(name, "Add", [self.qnames[edge] for edge in arms], name + ".fp")
        self.qnames[name] = self.onnx.quant(name, result, scale)
        return name

    def pool(self, source):
        value = self.fun.max_pool2d(self.values[source], 3, 2, 1)
        self.result("maxpool", "MaxPool", [source], value, self.scale(source), kernel_shape=[3, 3],
                    pads=[1, 1, 1, 1], strides=[2, 2], dilations=[1, 1], ceil_mode=0)
        self.qnames["maxpool"] = self.onnx.op("maxpool", "MaxPool", [self.qnames[source]], "maxpool.fp",
                                             kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2])
        return "maxpool"

    def classifier(self, source, fc):
        height, width = self.values[source].shape[-2:]
        value = self.values[source].mean(dim=(-2, -1), keepdim=True)
        gap = self.result("avgpool", "GlobalAveragePool", [source], value, self.scale(source),
                          axes=[2, 3], kernel_shape=[height, width], keepdims=1)
        result = self.onnx.op(gap, "GlobalAveragePool", [self.qnames[source]], gap + ".fp")
        flat = self.onnx.op("flatten", "Flatten", [self.onnx.quant(gap, result, self.scale(gap))],
                            "flatten.fp", axis=1)
        logits = self.fun.linear(value.flatten(1), fc.weight, fc.bias)
        value = logits[:, :, None, None]
        scale = scale_for(value)
        qw, qb, ws, bs, mul, shift = self.conv_data(gap, fc.weight[:, :, None, None], fc.bias, scale)
        parts = []
        start = 0
        for index, width in enumerate((256, 256, 256, 224, 8)):
            stop = start + width
            name = f"fc.part{index}"
            self.conv_node(name, gap, value[:, start:stop], qw[start:stop], qb[start:stop],
                           mul[start:stop], shift[start:stop], scale, [1, 1], [0, 0, 0, 0])
            parts.append(name)
            start = stop
        if start != fc.out_features:
            raise ValueError("expected exactly 1,000 official logits")
        self.result("logits", "Concat", parts, value, scale, axis=1)
        weight = self.onnx.dequant("fc.weight", qw[:, :, 0, 0], ws)
        bias = self.onnx.dequant("fc.bias", qb, bs)
        result = self.onnx.op("fc", "Gemm", [flat, weight, bias], "fc.fp", transB=1)
        result = self.onnx.quant("output", result, scale)
        self.onnx.op("logits", "Identity", [result], "logits")
        self.onnx.output("logits", [1, fc.out_features], FLOAT)

    def build(self, model):
        source = self.relu("stem.relu", self.conv("stem.conv", "images", model.conv1, model.bn1))
        source = self.pool(source)
        for group in range(1, 5):
            for index, block in enumerate(getattr(model, f"layer{group}")):
                name = f"layer{group}.{index}"
                skip = source
                first = self.relu(name + ".relu1", self.conv(name + ".conv1", source, block.conv1, block.bn1))
                second = self.conv(name + ".conv2", first, block.conv2, block.bn2)
                if block.downsample:
                    skip = self.conv(name + ".projection", skip, *block.downsample)
                source = self.relu(name + ".relu2", self.residual(name + ".add", second, skip))
        self.classifier(source, model.fc)

    def graph(self, onnx_sha256):
        counts = Counter(name for node in self.nodes for name in node["inputs"])
        for node in self.nodes:
            if node["op"] == "Conv":
                kh, kw, ci, co = self.weights[node["attrs"]["weight"]].shape
                h, w = self.tensors[node["id"]]["shape"][2:]
                node["estimated_macs"] = kh * kw * ci * co * h * w
        return {"inputs": ["images"], "outputs": ["logits"], "nodes": self.nodes,
                "tensors": self.tensors, "use_counts": counts, "source_onnx_sha256": onnx_sha256}


def fixed_reference(torch, graph, weights, image):
    """Independent integer oracle using exact-precision FP64 convolution arithmetic."""
    scale = graph["tensors"]["images"]["scale"]
    inputs = np.rint(image.astype(np.float64) / scale).clip(-128, 127).astype(np.int8)[None]
    values = {"images": inputs}
    for node in graph["nodes"]:
        op, attrs = node["op"], node["attrs"]
        conv = op == "Conv"
        args = [values[key] for key in node["inputs"][:1 if conv else None]]
        if conv:
            w = weights[attrs["weight"]].transpose(3, 2, 0, 1)
            b = weights[attrs["bias"]]
            # Each dot is bounded by K*128**2 plus INT32 bias, below 2**53.
            accum = torch.nn.functional.conv2d(
                torch.from_numpy(args[0]).double(), torch.from_numpy(w).double(), torch.from_numpy(b).double(),
                stride=tuple(attrs["strides"]), padding=tuple(attrs["pads"][:2]),
            ).numpy().astype(np.int64)
            mul, shift = (np.asarray(attrs[k], dtype=np.int64)[None, :, None, None] for k in ("mul", "shift"))
            out = (accum * mul) // np.left_shift(1, shift)
        elif op == "RequantLUT":
            out = np.asarray(attrs["lut"], dtype=np.int8)[args[0].view(np.uint8)]
        elif op == "EWAdd":
            out = args[0].astype(np.int16) + args[1].astype(np.int16)
        elif op == "MaxPool":
            out = torch.nn.functional.max_pool2d(torch.from_numpy(args[0]).double(), 3, 2, 1).numpy()
        elif op == "GlobalAveragePool":
            value, = args
            out = value.sum(axis=(2, 3), dtype=np.int64, keepdims=True) // (value.shape[2] * value.shape[3])
        elif op == "Concat":
            out = np.concatenate(args, axis=attrs["axis"])
        else:
            raise ValueError(f"unsupported reference operator {op}")
        out = out.clip(-128, 127).astype(np.int8)
        name, = node["outputs"]
        if list(out.shape) != graph["tensors"][name]["shape"]:
            raise ValueError(f"{name} shape mismatch: {out.shape}")
        values[name] = out
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "calibration", "image", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--size", type=int, default=64, choices=(32, 64, 128, 224))
    parser.add_argument("--samples", type=int, default=8, choices=range(1, 33))
    args = parser.parse_args()
    digest = sha256(args.weights.read_bytes()).hexdigest()
    if digest != DIGEST:
        raise ValueError(f"unexpected official state_dict SHA-256: {digest}")
    import torch
    from torchvision.models import resnet18

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source = resnet18(weights=None).eval()
    source.load_state_dict(torch.load(args.weights, weights_only=True, map_location="cpu"))
    samples = sorted(args.calibration.glob("*.jpg"))[:args.samples]
    image_digest = sha256(args.image.read_bytes()).hexdigest()
    sample_digests = {image.name: sha256(image.read_bytes()).hexdigest() for image in samples}
    if len(samples) != args.samples or image_digest in sample_digests.values():
        raise ValueError("calibration images are missing or include the RTL input")
    calibration = torch.from_numpy(np.stack([load_image(path, args.size) for path in samples]))
    image = load_image(args.image, args.size)
    with torch.inference_mode():
        model = CalibratedResNet(torch, calibration)
        model.build(source)
        folded = model.values["logits"].reshape(len(samples), -1)
        stock = source(calibration)
        np.testing.assert_allclose(folded.numpy(), stock.numpy(), rtol=2e-4, atol=2e-4)
        fp32 = source(torch.from_numpy(image[None])).numpy()
        onnx = model.onnx.encode()
        graph = model.graph(sha256(onnx).hexdigest())
        captures = fixed_reference(torch, graph, model.weights, image)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "quant.json").write_text(json.dumps(graph, indent=2) + "\n")
    np.savez_compressed(args.out / "weights.npz", **model.weights)
    np.savez_compressed(args.out / "captures.npz", **captures)
    inp = captures["images"].transpose(0, 2, 3, 1)
    np.pad(inp, ((0, 0), (0, 0), (0, 0), (0, 1))).tofile(args.out / "images.bin")
    captures["logits"].transpose(0, 2, 3, 1).tofile(args.out / "logits.bin")
    np.save(args.out / "fp32_logits.npy", fp32, allow_pickle=False)
    (args.out / "resnet18_qdq.onnx").write_bytes(onnx)
    metadata = {"model": "ResNet-18", "resolution": args.size, "classes": int(fp32.shape[1]),
                "weight_source": SOURCE, "weight_name": "torchvision ImageNet-1K V1 (official pretrained)",
                "weight_sha256": digest, "onnx_sha256": graph["source_onnx_sha256"],
                "calibration_sha256": sample_digests, "input_image": args.image.name,
                "input_image_sha256": image_digest,
                "logical_macs": sum(node.get("estimated_macs", 0) for node in model.nodes),
                "runtime_nodes": len(model.nodes), "reference_output_bytes": captures["logits"].nbytes,
                "fp32_fold_max_abs_error": float((folded - stock).abs().max().item()),
                "fixedpoint_contract": "signed INT8, per-channel weights, floor native requant and GAP",
                "onnx_contract": "standard opset-18 Q/DQ, QuantizeLinear uses ONNX round-to-nearest-even"}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
