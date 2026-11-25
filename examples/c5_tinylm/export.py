"""Export standard static INT8 ONNX from the learned NumPy parameters."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from examples.generate_quantized_onnx import FLOAT, INT8, INT32, INT64, UINT8, Model
from .train import DECODE, EPSILON, FF, HEADS, HEAD_WIDTH, LAYERS, PROMPT, VOCAB, WIDTH

PITCH = PROMPT + DECODE
Q_ACT = 3
Q_GAMMA = 4
Q_LOGIT = 2
Q_SCORE = 4


def export(weights, example):
    model = Model("TinyLM-Chat")
    model.input("tokens", [PITCH], INT64, example)
    scales = {}

    def dyadic(exp):
        if exp not in scales:
            name = f"scale_{exp:+d}"
            scales[exp] = model.tensor(name, [], FLOAT, [2.0 ** -exp])
        return scales[exp]

    zero8 = model.tensor("zero8", [], INT8, [0])
    zero32 = model.tensor("zero32", [], INT32, [0])
    zerou8 = model.tensor("zerou8", [], UINT8, [0])
    eps = model.tensor("epsilon", [], FLOAT, [EPSILON])
    axes = model.tensor("last_axis", [1], INT64, [-1])
    mask = model.tensor("causal_mask", [PITCH, PITCH], FLOAT,
                        [0 if j <= i else -math.inf for i in range(PITCH) for j in range(PITCH)])

    def op(name, kind, inputs, **attrs):
        return model.op(name, kind, inputs, name, **attrs)

    def parameter(key, exp):
        values = np.clip(np.rint(weights[key] * 2.0**exp), -128, 127).astype(np.int8)
        return model.tensor(key + ".weight", list(values.shape), INT8, values.ravel().tolist())

    def dq(name, source, exp, zero=zero8):
        return op(name, "DequantizeLinear", [source, dyadic(exp), zero])

    def q(name, source, exp, zero=zero8):
        return op(name, "QuantizeLinear", [source, dyadic(exp), zero])

    def dot(name, a, b, exp, out_exp):
        acc = op(name + ".acc", "MatMulInteger", [a, b])
        x = dq(name + ".real", acc, exp, zero32)
        return q(name + ".out", x, out_exp)

    def project(name, x, out_exp=Q_ACT):
        peak = float(np.max(np.abs(weights[name])))
        exp = min(12, math.floor(math.log2(127 / peak))) if peak else 6
        w = parameter(name, exp)
        return dot(name, x, w, Q_ACT + exp, out_exp)

    def norm(name, x):
        a = dq(name + ".real", x, Q_ACT)
        sq = op(name + ".square", "Mul", [a, a])
        mean = op(name + ".mean", "ReduceMean", [sq, axes], keepdims=1)
        eps_sum = op(name + ".eps", "Add", [mean, eps])
        root = op(name + ".root", "Sqrt", [eps_sum])
        normalized = op(name + ".norm", "Div", [a, root])
        direct = parameter(name, Q_GAMMA)
        gamma = dq(name + ".gamma", direct, Q_GAMMA)
        scaled = op(name + ".scaled", "Mul", [normalized, gamma])
        return q(name + ".out", scaled, Q_ACT)

    def residual(name, a, b):
        x = dq(name + ".a", a, Q_ACT)
        y = dq(name + ".b", b, Q_ACT)
        return q(name + ".out", op(name + ".sum", "Add", [x, y]), Q_ACT)

    x = op("embedded", "Gather", [parameter("embedding", Q_ACT), "tokens"], axis=0)
    for layer in range(LAYERS):
        base = f"l{layer}."
        normalized = norm(base + "gamma1", x)
        contexts = []
        for head in range(HEADS):
            prefix = base + f"h{head}."
            query, key, value = [project(prefix + proj, normalized) for proj in ("q", "k", "v")]
            kt = op(prefix + "kt", "Transpose", [key], perm=[1, 0])
            score = dot(prefix + "score", query, kt, 2 * Q_ACT + 1, Q_SCORE)
            real_score = dq(prefix + "soft.in", score, Q_SCORE)
            masked = op(prefix + "soft.masked", "Add", [real_score, mask])
            probability = op(prefix + "soft.real", "Softmax", [masked], axis=1)
            p = q(prefix + "p", probability, 8, zerou8)
            contexts.append(dot(prefix + "pv", p, value, 8 + Q_ACT, Q_ACT))
        heads = op(base + "heads", "Concat", contexts, axis=1)
        residual1 = residual(base + "res1", x, project(base + "out", heads))
        normalized2 = norm(base + "gamma2", residual1)
        gate = dq(base + "silu.in", project(base + "gate", normalized2), Q_ACT)
        g = op(base + "silu.sigmoid", "Sigmoid", [gate])
        act = q(base + "silu.out", op(base + "silu.scaled", "Mul", [gate, g]), Q_ACT)
        up = dq(base + "mod.up", project(base + "up", normalized2), Q_ACT)
        silu = dq(base + "mod.gate", act, Q_ACT)
        glu = q(base + "mod.out", op(base + "mod.multiply", "Mul", [silu, up]), Q_ACT)
        x = residual(base + "res2", residual1, project(base + "down", glu))
    logits = project("lm", x, Q_LOGIT)
    token = op("token", "ArgMax", [logits], axis=1, keepdims=0)
    model.output(logits, [PITCH, VOCAB], INT8)
    model.output(token, [PITCH], INT64)
    return model.encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    training = json.loads((args.weights / "training.json").read_text())
    with np.load(args.weights / "weights.fp32.npz", allow_pickle=False) as weights:
        blob = export(weights, training["corpus"][0][:PITCH])
    args.out.write_bytes(blob)


if __name__ == "__main__":
    main()
