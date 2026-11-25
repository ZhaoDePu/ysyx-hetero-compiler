"""Export the trained networks as standard-domain FP32 and static INT8 ONNX."""

import hashlib
import math

import numpy as np

from examples.generate_quantized_onnx import FLOAT, INT8, INT32, INT64, UINT8, Model, field, value_info

from .train import SHAPES


class StandardModel(Model):
    def add_input(self, name, shape, dtype):
        self.inputs.append(value_info(name, shape, dtype))
        return name

    def encode(self):
        graph = b"".join(self.nodes) + field(2, self.name)
        graph += b"".join(field(5, t) for t in self.tensors.values())
        graph += b"".join(field(11, i) for i in self.inputs)
        graph += b"".join(field(12, o) for o in self.outputs)
        return (field(1, 8) + field(2, "ysyx-c3-numpy-training")
                + field(7, graph) + field(8, field(2, 18)))


def quantize(x, scale):
    return np.clip(np.rint(x / scale), -128, 127).astype(np.int8)


def _choose_scale(x, positive_limit=120):
    amplitude = max(float(np.abs(x).max()), 2**-20)
    return 2.0 ** math.ceil(math.log2(amplitude / positive_limit))


def _initializer(model, name, values, dtype):
    arr = np.asarray(values)
    return model.tensor(name, arr.shape, dtype, arr.flatten().tolist())


def _scale(model, name, value):
    return model.tensor(name + "_scale", (), FLOAT, [value])


def _mask(model, t=4):
    return _initializer(model, "causal_mask", np.where(np.tri(t, dtype=bool), 0., -np.inf), FLOAT)


def _vision(params, scale, use_int8):
    m = StandardModel("trained_oriented_strokes_cnn_" + ("int8" if use_int8 else "fp32"))
    x = m.add_input("image", SHAPES["vision"], INT8 if use_int8 else FLOAT)
    if use_int8:
        m.tensor("zero", (), INT8, [0])
        source_scale = _scale(m, "input", scale["input"])
        previous_scale = scale["input"]
    for idx, (stride, pad) in enumerate(((1, 1), (2, 1), (1, 0)), 1):
        key = f"c{idx}"
        w = params[key].transpose(3, 2, 0, 1)  # NHWC/HWIO training uses OIHW on the wire.
        b = params[f"b{idx}"]
        geometry = {"strides": [stride, stride], "pads": [pad]*4}
        y = f"conv{idx}_out"
        if use_int8:
            weight_scale = _choose_scale(w, 127)
            weight = _initializer(m, f"w{idx}", quantize(w, weight_scale), INT8)
            bias = _initializer(m, f"b{idx}", np.rint(b / (previous_scale*weight_scale)).astype(np.int32), INT32)
            ws, ys = _scale(m, f"w{idx}", weight_scale), _scale(m, key, scale[key])
            m.op(key, "QLinearConv", [x, source_scale, "zero", weight, ws, "zero", ys, "zero", bias],
                 y, **geometry)
            source_scale, previous_scale = ys, scale[key]
        else:
            weight = _initializer(m, f"w{idx}", w, FLOAT)
            bias = _initializer(m, f"b{idx}", b, FLOAT)
            m.op(key, "Conv", [x, weight, bias], y, **geometry)
        x = m.op(f"relu{idx}", "Relu", [y], f"act{idx}") if idx < 3 else y
    shape = m.tensor("logits_shape", (2,), INT64, [1, 4])
    m.op("flatten_logits", "Reshape", [x, shape], "logits")
    m.op("classify", "ArgMax", ["logits"], "prediction", axis=1, keepdims=0)
    m.output("logits", (1, 4), INT8 if use_int8 else FLOAT)
    m.output("prediction", (1,), INT64)
    return m


def _transformer(params, scales, use_int8):
    m = StandardModel("trained_lagged_token_transformer_" + ("int8" if use_int8 else "fp32"))
    x = m.add_input("tokens", SHAPES["transformer"], INT8 if use_int8 else FLOAT)
    mask = _mask(m)
    if use_int8:
        m.tensor("zero", (), INT8, [0])
        m.tensor("zero_u", (), UINT8, [0])
        scale_names = {k: _scale(m, k, v) for k, v in scales.items()}
        scale_names["prob"] = _scale(m, "prob", 1/256)
        score_k = _scale(m, "key_attention", scales["key"] * .5)

    def matmul(name, a, b, out, sa, sb, sy, zero="zero"):
        if use_int8:
            return m.op(name, "QLinearMatMul",
                        [a, sa, zero, b, sb, "zero", sy, "zero"], out)
        return m.op(name, "MatMul", [a, b], out)

    def project(key, a, source, out=None):
        weights = params[key]
        if use_int8:
            step = _choose_scale(weights, 127)
            bound = _scale(m, key + "_weight", step)
            weight = _initializer(m, key + "_weights", quantize(weights, step), INT8)
            return matmul(key, a, weight, out or key + "_out",
                          scale_names[source], bound, scale_names[key])
        weight = _initializer(m, key + "_weights", weights, FLOAT)
        return matmul(key, a, weight, out or key + "_out", None, None, None)

    def dq(name, source, scale_key):
        return m.op(name, "DequantizeLinear", [source, scale_names[scale_key], "zero"], name + "_out")

    def residual(name, a, akey, b, bkey):
        if not use_int8:
            return m.op(name, "Add", [a, b], name + "_out")
        left, right = dq(name + "_a_dq", a, akey), dq(name + "_b_dq", b, bkey)
        summed = m.op(name, "Add", [left, right], name + "_float")
        return m.op(name + "_q", "QuantizeLinear", [summed, scale_names[name], "zero"], name + "_out")

    q, k, v = (project(name, x, "input") for name in ("query", "key", "value"))
    kt = m.op("transpose_key", "Transpose", [k], "key_t", perm=[1, 0])
    if use_int8:
        scores = matmul("score_qk_half", q, kt, "scores_out",
                        scale_names["query"], score_k, scale_names["scores"])
        score_float = dq("scores_dq", scores, "scores")
    else:
        raw = matmul("score_qk", q, kt, "raw_scores", None, None, None)
        factor = m.tensor("head_scale", (), FLOAT, [.5])
        score_float = m.op("rescale_head", "Mul", [raw, factor], "scores_out")
    masked = m.op("causal_mask_add", "Add", [score_float, mask], "masked")
    p = m.op("causal_softmax", "Softmax", [masked], "float_prob", axis=-1)
    if use_int8:
        pq = m.op("prob_q", "QuantizeLinear", [p, scale_names["prob"], "zero_u"], "prob")
        context = matmul("attend", pq, v, "context", scale_names["prob"],
                         scale_names["value"], scale_names["context"], "zero_u")
    else:
        context = matmul("attend", p, v, "context", None, None, None)
    project_out = project("output", context, "context")
    skip = residual("attn_skip", x, "input", project_out, "output")
    ff = project("ff_up", skip, "attn_skip")
    active = m.op("ff_relu", "Relu", [ff], "ff_active")
    ff_out = project("ff_down", active, "ff_up")
    ff_skip = residual("ff_skip", skip, "attn_skip", ff_out, "ff_down")
    project("classifier", ff_skip, "ff_skip", "logits")
    m.op("predict_token", "ArgMax", ["logits"], "prediction", axis=1, keepdims=0)
    m.output("logits", (4, 4), INT8 if use_int8 else FLOAT)
    m.output("prediction", (4,), INT64)
    return m


def export_models(task, params, calibration, destination):
    scale = {name: _choose_scale(x) for name, x in calibration.items()}
    if task == "transformer":
        scale["scores"] = max(scale["scores"], 1/16)
        if scale["scores"] not in (1/16, 1/8):
            raise ValueError("calibrated score scale is outside the validated Softmax range")
    graph = _vision if task == "vision" else _transformer
    names = {"fp32": f"{task}_fp32.onnx", "int8": f"{task}_int8.onnx"}
    hashes = {}
    for mode, filename in names.items():
        model = graph(params, scale, use_int8=(mode == "int8"))
        encoded = model.encode()
        (destination / filename).write_bytes(encoded)
        hashes[mode] = hashlib.sha256(encoded).hexdigest()
    return {"models": names, "sha256": hashes, "activation_scales": scale,
            "input_scale": scale["input"], "logit_scale": scale["c3" if task == "vision" else "classifier"]}
