"""Generate the C4-VPU standard ONNX Transformer and evaluate its FP32 graph."""

from pathlib import Path

import numpy as np

from examples.c4a_streams.generate import _pattern
from examples.generate_quantized_onnx import FLOAT, INT8, INT64, UINT8, Model


TILES = 4
TOKENS = 4
WIDTH = 4
HIDDEN = 8
VOCAB = 4
EPSILON = 2**-10
SCALE = 1/16

_PROJECTIONS = {
    "query": (WIDTH, WIDTH), "key": (WIDTH, WIDTH), "value": (WIDTH, WIDTH),
    "out_proj": (WIDTH, WIDTH), "ff_gate": (WIDTH, HIDDEN), "ff_up": (WIDTH, HIDDEN),
    "ff_down": (HIDDEN, WIDTH), "vocab": (WIDTH, VOCAB),
}
_GAMMA = {"attention_norm": [16, 14, 18, 17], "ff_norm": [15, 18, 16, 14]}


def _parameters():
    return {name: _pattern(ci*co, 11 + i, 1)
            for i, (name, (ci, co)) in enumerate(_PROJECTIONS.items())}


def _input(tile):
    return _pattern(TOKENS*WIDTH, 53 + tile*6, 12)


def encode():
    model = Model("C4VPU_Transformer_Layer")
    model.qparams()
    model.tensor("prob_scale", [], FLOAT, [1/256])
    model.tensor("score_scale32", [], FLOAT, [SCALE/8])
    model.tensor("pv_scale32", [], FLOAT, [SCALE/256])
    model.tensor("zero_u", [], UINT8, [0])
    model.tensor("axes", [1], INT64, [-1])
    model.tensor("rms_epsilon", [], FLOAT, [EPSILON])
    model.tensor("causal_mask", [TOKENS, TOKENS], FLOAT,
                 [0 if k <= q else -float("inf") for q in range(TOKENS) for k in range(TOKENS)])
    weights = {name: model.tensor(name+"_weight", _PROJECTIONS[name], INT8, values)
               for name, values in _parameters().items()}
    gamma = {name: model.tensor(name + "_gamma", [WIDTH], INT8, values)
             for name, values in _GAMMA.items()}

    for i in range(TILES):
        prefix = f"layer{i:02d}_"

        def op(label, kind, args, **attrs):
            return model.op(prefix+label, kind, args, prefix+label+"_out", **attrs)

        def dq(label, tensor, scale="q4", point="z"):
            return op(label, "DequantizeLinear", [tensor, scale, point] if point else [tensor, scale])

        def quant(label, value, scale="q4", point="z"):
            return op(label, "QuantizeLinear", [value, scale, point])

        def proj(name, source):
            return model.qlinear(prefix+name, "QLinearMatMul", source, weights[name], prefix+name+"_out")

        def rms(name, value):
            x = dq(name+"_input_dq", value)
            gain = dq(name+"_gamma_dq", gamma[name])
            squared = op(name+"_square", "Mul", [x, x])
            mean = op(name+"_mean", "ReduceMean", [squared, "axes"], keepdims=1)
            denominator = op(name+"_sqrt", "Sqrt", [op(name+"_epsilon", "Add", [mean, "rms_epsilon"])])
            norm = op(name+"_divide", "Div", [x, denominator])
            return quant(name, op(name+"_scale", "Mul", [norm, gain]))

        def binary(name, operation, a, b):
            lhs = dq(name+"_a_dq", a)
            rhs = dq(name+"_b_dq", b)
            return quant(name, op(name+"_f", operation, [lhs, rhs]))

        x = model.input(prefix+"tokens", [TOKENS, WIDTH], INT8, _input(i))
        nx = rms("attention_norm", x)
        q, k, v = [proj(name, nx) for name in ("query", "key", "value")]
        kt = op("key_transpose", "Transpose", [k], perm=[1, 0])
        score = op("qk", "MatMulInteger", [q, kt])
        score_q = quant("score_quant", dq("score_i32_dq", score, "score_scale32", None))
        masked = op("mask", "Add", [dq("score_dq", score_q), "causal_mask"])
        probability = quant("probability", op("softmax", "Softmax", [masked], axis=-1),
                            "prob_scale", "zero_u")
        weighted = op("pv", "MatMulInteger", [probability, v])
        attn = quant("context", dq("pv_i32_dq", weighted, "pv_scale32", None))
        residual = binary("attention_residual", "Add", x, proj("out_proj", attn))
        nres = rms("ff_norm", residual)
        gate, up = [proj(name, nres) for name in ("ff_gate", "ff_up")]
        gate_f = dq("gate_dq", gate)
        sigmoid = op("sigmoid", "Sigmoid", [gate_f])
        activated = quant("silu", op("silu_f", "Mul", [gate_f, sigmoid]))
        hidden = binary("swiglu", "Mul", activated, up)
        layer = binary("ff_residual", "Add", residual, proj("ff_down", hidden))
        logits = proj("vocab", layer)
        token = op("predict", "ArgMax", [logits], axis=-1, keepdims=0)
        model.output(logits, [TOKENS, VOCAB], INT8)
        model.output(token, [TOKENS], INT64)
    return model.encode()


def _softmax(scores, visible):
    bounded = np.where(visible, scores, np.float32(-np.inf))
    weights = np.exp(bounded - np.max(bounded, axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


def evaluate_fp32():
    """Evaluate the unquantized graph with the ONNX fixture's dyadic parameters."""
    weights = {name: np.asarray(values, np.float32).reshape(_PROJECTIONS[name])
               for name, values in _parameters().items()}
    gamma = {name: np.asarray(values, np.float32) * np.float32(SCALE)
             for name, values in _GAMMA.items()}

    def rms(x, name):
        return x * gamma[name] / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + np.float32(EPSILON))

    result = {}
    col = np.arange(TOKENS)
    for i in range(TILES):
        prefix = f"layer{i:02d}_"
        x = np.asarray(_input(i), np.float32).reshape(TOKENS, WIDTH) * np.float32(SCALE)
        normalized = rms(x, "attention_norm")
        q, k, v = [normalized @ weights[name] for name in ("query", "key", "value")]
        # QK codes use input scales 1/16 and received real scale 1/128.
        scores = np.float32(2) * (q @ k.T)
        probability = _softmax(scores, col[None, :] <= col[:, None])
        residual = x + (probability @ v) @ weights["out_proj"]
        normalized = rms(residual, "ff_norm")
        gate, up = [normalized @ weights[name] for name in ("ff_gate", "ff_up")]
        hidden = gate / (1 + np.exp(-gate)) * up
        logits = (residual + hidden @ weights["ff_down"]) @ weights["vocab"]
        result[prefix+"vocab_out"] = logits.ravel()
        result[prefix+"predict_out"] = np.argmax(logits, axis=-1)
    return result


if __name__ == "__main__":
    (Path(__file__).resolve().parent / "TransformerLayer_INT8_C4VPU.onnx").write_bytes(encode())
