"""Calibrate the public ViT-Tiny weights and export a full standard Q/DQ graph."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

WEIGHT_SHA256 = "fecf81b492bd13ee7a5297cb74d1d417aac8bf7e1b7d96aed89c4691984587ed"
TINY, HEADS, TOKENS = 192, 3, 197
ERF = np.frompyfunc(math.erf, 1, 1)


def load_weights(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != WEIGHT_SHA256:
        raise ValueError("unexpected ViT-Tiny checkpoint")
    header_len = int.from_bytes(raw[:8], "little")
    weights = {}
    for name, item in json.loads(raw[8:8 + header_len]).items():
        if name == "__metadata__":
            continue
        if item["dtype"] != "F32":
            raise ValueError(f"unexpected weight type for {name}")
        lo, hi = item["data_offsets"]
        weights[name] = np.frombuffer(raw[8 + header_len + lo:8 + header_len + hi],
                                      dtype="<f4").reshape(item["shape"])
    return weights


def image_input(path, config):
    h, w = config["input_size"][-2:]
    pct = config["crop_pct"]
    with Image.open(path) as source:
        image = source.convert("RGB")
        size = int(min(h, w) / pct)
        x, y = image.size
        x, y = (size, int(y * size / x)) if x <= y else (int(x * size / y), size)
        image = image.resize((x, y), Image.Resampling.BICUBIC)
        left, top = round((x - w) / 2), round((y - h) / 2)
        rgb = np.asarray(image.crop((left, top, left + w, top + h)), dtype=np.float32) / 255
    norm = (rgb - np.asarray(config["mean"], dtype=np.float32)) / np.asarray(config["std"], dtype=np.float32)
    return norm.transpose(2, 0, 1)[None]


def w8(weight):
    rows = weight.reshape(weight.shape[0], -1)
    scale = np.maximum(np.max(np.abs(rows), axis=1) / np.float32(127), np.float32(1e-8))
    broad = scale.reshape((len(scale),) + (1,) * (weight.ndim - 1))
    return np.rint(weight / broad).clip(-127, 127).astype(np.int8), scale


def forward(x, weights, activate):
    def linear(z, name):
        z = activate(name, z)
        w = weights[name + ".weight"]
        return z @ w.T + weights[name + ".bias"]

    def norm(z, name):
        centered = z - z.mean(-1, keepdims=True)
        variance = np.mean(centered * centered, axis=-1, keepdims=True)
        return (centered / np.sqrt(variance + np.float32(1e-6))
                * weights[name + ".weight"] + weights[name + ".bias"])

    patches = activate("patch_embed.proj", x).reshape(1, 3, 14, 16, 14, 16)
    patches = patches.transpose(0, 2, 4, 1, 3, 5).reshape(1, 196, 768)
    z = patches @ weights["patch_embed.proj.weight"].reshape(TINY, 768).T
    z += weights["patch_embed.proj.bias"]
    z = np.concatenate((weights["cls_token"], z), axis=1) + weights["pos_embed"]
    for idx in range(12):
        name = f"blocks.{idx}."
        qkv = linear(norm(z, name + "norm1"), name + "attn.qkv")
        q, k, v = qkv.reshape(1, TOKENS, 3, HEADS, 64).transpose(2, 0, 3, 1, 4)
        score = q @ k.swapaxes(-2, -1) * np.float32(0.125)
        p = np.exp(score - score.max(axis=-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        u = (p @ v).transpose(0, 2, 1, 3).reshape(1, TOKENS, TINY)
        z = z + linear(u, name + "attn.proj")
        u = linear(norm(z, name + "norm2"), name + "mlp.fc1")
        u = np.float32(0.5) * u * (np.float32(1) + ERF(u * np.float32(2 ** -0.5)).astype(np.float32))
        z = z + linear(u, name + "mlp.fc2")
    return linear(norm(z, "norm")[:, 0], "head")


def export_onnx(path, weights, scales=None):
    import onnx
    from onnx import helper as h
    from onnx import numpy_helper

    nodes, init = [], {}

    def constant(name, val):
        if name not in init:
            init[name] = numpy_helper.from_array(np.asarray(val), name)
        return name

    def op(kind, name, *args, **attrs):
        nodes.append(h.make_node(kind, list(args), [name], name=name, **attrs))
        return name

    def reshape(x, name, shape):
        return op("Reshape", name, x, constant(name + "/shape", np.asarray(shape, dtype=np.int64)))

    def act(x, name):
        if scales is None:
            return x
        scale = constant(name + "/a.scale", np.float32(scales[name]))
        zero = constant("signed_zero", np.int8(0))
        return op("DequantizeLinear", name + "/a.dq", op("QuantizeLinear", name + "/a.q", x, scale, zero),
                  scale, zero)

    def param(name):
        w = weights[name + ".weight"]
        if scales is None:
            return constant(name + ".weight", w)
        q, s = w8(w)
        return op("DequantizeLinear", name + "/w.dq", constant(name + "/w.q", q),
                  constant(name + "/w.scale", s), constant(name + "/w.zero", np.zeros(s.size, np.int8)),
                  axis=0)

    def linear(x, name, out):
        w = op("Transpose", name + "/w.T", param(name), perm=[1, 0])
        product = op("MatMul", name + "/mm", act(x, name), w)
        return op("Add", out, product, constant(name + ".bias", weights[name + ".bias"]))

    def norm(x, name):
        return op("LayerNormalization", name, x,
                  constant(name + ".weight", weights[name + ".weight"]),
                  constant(name + ".bias", weights[name + ".bias"]), axis=-1, epsilon=1e-6)

    patch = "patch_embed.proj"
    x = op("Conv", "patch.conv", act("image", patch), param(patch),
           constant(patch + ".bias", weights[patch + ".bias"]), strides=[16, 16])
    x = reshape(x, "patch.flat", [1, TINY, 196])
    x = op("Transpose", "patch.tokens", x, perm=[0, 2, 1])
    x = op("Concat", "with_cls", constant("cls_token", weights["cls_token"]), x, axis=1)
    x = op("Add", "embedding", x, constant("pos_embed", weights["pos_embed"]))
    for idx in range(12):
        name = f"blocks.{idx}."
        qkv = linear(norm(x, name + "norm1"), name + "attn.qkv", name + "qkv")
        qkv = reshape(qkv, name + "qkv.reshape", [1, TOKENS, 3, HEADS, 64])
        qkv = op("Transpose", name + "qkv.heads", qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = [op("Gather", name + out, qkv, constant(f"qkv_index_{i}", np.int64(i)), axis=0)
                   for i, out in enumerate(("q", "k", "v"))]
        kt = op("Transpose", name + "k.T", k, perm=[0, 1, 3, 2])
        score = op("MatMul", name + "qk", q, kt)
        score = op("Mul", name + "qk.scaled", score, constant("qk_scale", np.float32(0.125)))
        p = op("Softmax", name + "prob", score, axis=-1)
        z = op("MatMul", name + "pv", p, v)
        z = op("Transpose", name + "pv.tokens", z, perm=[0, 2, 1, 3])
        z = reshape(z, name + "pv.flat", [1, TOKENS, TINY])
        z = linear(z, name + "attn.proj", name + "attention")
        x = op("Add", name + "skip1", x, z)
        z = linear(norm(x, name + "norm2"), name + "mlp.fc1", name + "gate")
        erf = op("Erf", name + "erf", op("Mul", name + "erf.arg", z, constant("inv_sqrt2", np.float32(2 ** -0.5))))
        erf = op("Add", name + "cdf", erf, constant("one", np.float32(1)))
        z = op("Mul", name + "gelu", op("Mul", name + "gelu.input", z, constant("half", np.float32(0.5))), erf)
        z = linear(z, name + "mlp.fc2", name + "mlp")
        x = op("Add", name + "output", x, z)
    x = op("Gather", "cls", norm(x, "norm"), constant("qkv_index_0", np.int64(0)), axis=1)
    linear(x, "head", "logits")
    graph = h.make_graph(nodes, "vit_tiny_patch16_224",
                         [h.make_tensor_value_info("image", onnx.TensorProto.FLOAT, [1, 3, 224, 224])],
                         [h.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, [1, 1000])],
                         list(init.values()))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 19)], ir_version=8, producer_name="Valence-C")
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--weights", type=Path, required=True)
    cli.add_argument("--config", type=Path, required=True)
    cli.add_argument("--cal", type=Path, nargs="+", required=True)
    cli.add_argument("--test", type=Path, nargs="+", required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    if len(args.test) < 5 or set(map(Path.resolve, args.cal)) & set(map(Path.resolve, args.test)):
        cli.error("use at least five independent test images")
    args.out.mkdir()
    w = load_weights(args.weights)
    meta = json.loads(args.config.read_text())
    config = meta.get("pretrained_cfg", meta)
    peaks = {}

    def observe(name, value):
        peaks[name] = max(peaks.get(name, 0.0), float(np.max(np.abs(value))))
        return value

    for image in args.cal:
        forward(image_input(image, config), w, observe)
    scale = {name: np.float32(max(peak / 127, 1e-8)) for name, peak in peaks.items()}
    qweights = dict(w)
    for name in scale:
        key = name + ".weight"
        q, s = w8(w[key])
        qweights[key] = q.astype(np.float32) * s.reshape((s.size,) + (1,) * (q.ndim - 1))

    def quant(name, value):
        return np.rint(value / scale[name]).clip(-128, 127) * scale[name]

    rows = []
    for image in args.test:
        x = image_input(image, config)
        fp32 = forward(x, w, lambda _, data: data).ravel()
        a8 = forward(x, qweights, quant).ravel()
        ftop, qtop = np.argsort(-fp32, kind="stable")[:5], np.argsort(-a8, kind="stable")[:5]
        rows.append({"image": image.name, "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                     "fp32_top5": ftop.tolist(), "w8a8_top5": qtop.tolist(),
                     "top1_match": bool(ftop[0] == qtop[0]), "fp32_top1_in_w8a8_top5": bool(ftop[0] in qtop),
                     "cosine": float(np.dot(fp32, a8) / (np.linalg.norm(fp32) * np.linalg.norm(a8))),
                     "mae": float(np.mean(np.abs(fp32 - a8)))})
        np.savez(args.out / f"{image.stem}.npz", image=x, fp32=fp32, w8a8=a8)
    (args.out / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.out / "calibration.json").write_text(json.dumps(
        {"weights": WEIGHT_SHA256, "scales": {k: float(v) for k, v in scale.items()},
         "images": {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in args.cal}}, indent=2) + "\n")
    export_onnx(args.out / "vit_tiny_fp32.onnx", w)
    export_onnx(args.out / "vit_tiny_w8a8.onnx", w, scale)
    print(json.dumps({"samples": len(rows), "matched_top1": sum(x["top1_match"] for x in rows),
                      "mean_cosine": sum(x["cosine"] for x in rows) / len(rows)}))


if __name__ == "__main__":
    main()
