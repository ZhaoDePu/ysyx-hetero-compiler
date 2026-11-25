#!/usr/bin/env python3
"""Generate deterministic quantized ONNX graphs using only protobuf wire primitives."""

import math
from pathlib import Path
import struct

FLOAT, UINT8, INT8, INT32, INT64 = 1, 2, 3, 6, 7
DOMAIN = "com.ysyx"


def varint(value):
    value &= (1 << 64) - 1
    data = bytearray()
    while value >= 128:
        data.append((value & 127) | 128)
        value >>= 7
    data.append(value)
    return bytes(data)


def field(tag, value):
    if isinstance(value, int):
        return varint(tag << 3) + varint(value)
    if isinstance(value, str):
        value = value.encode()
    return varint(tag << 3 | 2) + varint(len(value)) + value


def value_info(name, shape, dtype):
    dims = b"".join(field(1, field(1, d)) for d in shape)
    return field(1, name) + field(2, field(1, field(1, dtype) + field(2, dims)))


def attribute(name, value):
    if isinstance(value, int):
        param = field(3, value) + field(20, 2)
    elif isinstance(value, str):
        param = field(4, value) + field(20, 3)
    else:
        param = field(8, b"".join(varint(v) for v in value)) + field(20, 7)
    return field(5, field(1, name) + param)


class Model:
    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.tensors = {}
        self.inputs = []
        self.outputs = []

    def tensor(self, name, shape, dtype, values):
        form = {FLOAT: "f", UINT8: "B", INT8: "b", INT32: "i", INT64: "q"}[dtype]
        if len(values) != math.prod(shape) or name in self.tensors:
            raise ValueError(f"{name}: invalid initializer")
        dims = field(1, b"".join(varint(d) for d in shape)) if shape else b""
        raw = struct.pack(f"<{len(values)}{form}", *values)
        self.tensors[name] = dims + field(2, dtype) + field(8, name) + field(9, raw)
        return name

    def input(self, name, shape, dtype, defaults):
        self.inputs.append(value_info(name, shape, dtype))
        return self.tensor(name, shape, dtype, defaults)

    def output(self, name, shape, dtype):
        self.outputs.append(value_info(name, shape, dtype))

    def op(self, name, op, ins, out, domain="", **attrs):
        outputs = [out] if isinstance(out, str) else out
        message = (b"".join(field(1, i) for i in ins) + b"".join(field(2, o) for o in outputs)
                   + field(3, name) + field(4, op))
        message += b"".join(attribute(k, v) for k, v in attrs.items())
        if domain:
            message += field(7, domain)
        self.nodes.append(field(1, message))
        return out

    def qparams(self):
        self.tensor("q4", [], FLOAT, [1/16])
        self.tensor("unit", [], FLOAT, [1])
        self.tensor("z", [], INT8, [0])

    def qlinear(self, name, op, a, b, out, bias=(), attrs=None):
        return self.op(name, op, [a, "q4", "z", b, "unit", "z", "q4", "z", *bias],
                       out, **(attrs or {}))

    def encode(self):
        graph = b"".join(self.nodes) + field(2, self.name)
        graph += b"".join(field(5, t) for t in self.tensors.values())
        graph += b"".join(field(11, i) for i in self.inputs)
        graph += b"".join(field(12, o) for o in self.outputs)
        return (field(1, 8) + field(2, "ysyx-hetero-compiler")
                + field(7, graph) + field(8, field(2, 18))
                + field(8, field(1, DOMAIN) + field(2, 1))
                + field(8, field(1, "com.microsoft") + field(2, 1)))


def image_values(seed):
    return [((i*11 + (i//5)*7 + seed*(i % 5 + 1)) % 31) - 15 for i in range(64)]


def token_values(seed):
    return [((i*13 + (i//4)*3 + seed*(i % 5 + 1)) % 23) - 11 for i in range(16)]


def vision():
    m = Model("ysyx_quantized_yolo_feature_block")
    image = m.input("image", [1, 4, 4, 4], INT8, image_values(0))
    m.qparams()
    w = m.tensor("spatial_filter", [4, 4, 3, 3], INT8,
                 [1 if i % 9 == 4 and (i//9)%4 == i//36
                  else (1 if (i*17+3) % 19 == 0 else -1 if i % 23 == 0 else 0)
                  for i in range(144)])
    bias = m.tensor("spatial_bias", [4], INT32, [-3, 0, 7, 2])
    spatial = m.qlinear("spatial_conv", "QLinearConv", image, w, "spatial",
                        [bias], {"pads": [1, 1, 1, 1]})
    active = m.op("spatial_silu", "SiLU", [spatial], "activated", DOMAIN)
    sizes = m.tensor("split_sizes", [2], INT64, [2, 2])
    left, right = m.op("channels", "Split", [active, sizes], ["left", "right"], axis=1)
    skip = m.op("reorder_channels", "Concat", [right, left], "skip", axis=1)
    w2 = m.tensor("point_filter", [4, 4, 1, 1], INT8, [1,0,1,0,-1,1,0,0,0,0,1,-1,0,1,0,1])
    bias2 = m.tensor("point_bias", [4], INT32, [0, -2, 1, 3])
    point = m.qlinear("point_conv", "QLinearConv", skip, w2, "pointwise", [bias2])
    residual = m.op("feature_residual", "QLinearAdd",
                    [point, "q4", "z", active, "q4", "z", "q4", "z"], "residual", "com.microsoft")
    pooled = m.op("large_receptive_field", "MaxPool", [residual], "pooled",
                  kernel_shape=[5, 5], pads=[2, 2, 2, 2], strides=[1, 1])
    scales = m.tensor("scale2", [4], FLOAT, [1, 1, 2, 2])
    pixel_logits = m.op("upsample_logits", "Resize", [pooled, "", scales], "pixel_logits",
                        mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor")
    m.op("candidate_classes", "ArgMax", [pixel_logits], "classes", axis=1, keepdims=0)
    m.output("pixel_logits", [1, 4, 8, 8], INT8)
    m.output("classes", [1, 8, 8], INT64)
    return m.encode()


def transformer():
    m = Model("ysyx_quantized_causal_transformer_layer")
    x = m.input("tokens", [4, 4], INT8, token_values(0))
    m.qparams()

    def projection(name, source, ci, co, seed):
        data = [((r*7+c*3+seed+(r*c)%3) % 5) - 2 for r in range(ci) for c in range(co)]
        w = m.tensor(name + "_weight", [ci, co], INT8, data)
        return m.qlinear(name, "QLinearMatMul", source, w, name + "_out")

    def residual(name, a, b):
        return m.op(name, "QLinearAdd", [a, "q4", "z", b, "q4", "z", "q4", "z"],
                    name + "_out", "com.microsoft")

    norm = m.op("rms_before_attention", "RMSNorm", [x], "normalized", DOMAIN)
    q = projection("query", norm, 4, 4, 1)
    k = projection("key", norm, 4, 4, 2)
    v = projection("value", norm, 4, 4, 3)
    kt = m.op("keys_transposed", "Transpose", [k], "kt", perm=[1, 0])
    scores32 = m.op("dot_qk", "MatMulInteger", [q, kt], "scores32")
    scores = m.op("scale_scores", "Requant", [scores32], "scores", DOMAIN, shift=6)
    probs = m.op("causal_probabilities", "Softmax", [scores], "probs", DOMAIN, causal=1, axis=1)
    attn32 = m.op("dot_pv", "MatMulInteger", [probs, v], "weighted_values32")
    attn = m.op("scale_values", "Requant", [attn32], "weighted_values", DOMAIN, shift=8)
    out = projection("output_projection", attn, 4, 4, 4)
    pre_ff = residual("attention_residual", out, x)
    rms_ff = m.op("rms_before_swiglu", "RMSNorm", [pre_ff], "ff_normalized", DOMAIN)
    gate = projection("ff_gate", rms_ff, 4, 8, 4)
    up = projection("ff_up", rms_ff, 4, 8, 7)
    act = m.op("gate_activation", "SiLU", [gate], "silu_gate", DOMAIN)
    act32 = m.op("extend_gate", "Cast", [act], "gate32", to=INT32)
    up32 = m.op("extend_up", "Cast", [up], "up32", to=INT32)
    hidden32 = m.op("swiglu_multiply", "Mul", [act32, up32], "hidden32")
    hidden = m.op("swiglu_q4", "Requant", [hidden32], "hidden", DOMAIN, shift=4)
    down = projection("ff_down", hidden, 8, 4, 17)
    layers = residual("ff_residual", down, pre_ff)
    logits = projection("vocabulary", layers, 4, 8, 2)
    m.op("select_tokens", "ArgMax", [logits], "next_tokens", axis=1, keepdims=0)
    m.output("next_tokens", [4], INT64)
    m.output(logits, [4, 8], INT8)
    return m.encode()


def main():
    here = Path(__file__).resolve().parent
    (here / "yolo_block_int8.onnx").write_bytes(vision())
    (here / "transformer_layer_int8.onnx").write_bytes(transformer())


if __name__ == "__main__":
    main()
