"""Deterministic ONNX throughput fixtures partitionable by data dependence."""

import argparse
from pathlib import Path

from examples.generate_quantized_onnx import DOMAIN, Model, INT8, INT32, INT64


def vision(batch=24):
    m = Model("StreamVision_MultiTile")
    samples = [((i*17 + (i//9)*11 + (i//192)*9) % 25)-12 for i in range(batch*4*6*8)]
    image = m.input("image", [batch, 4, 6, 8], INT8, samples)
    m.qparams()
    w = m.tensor("spatial_weight", [4, 4, 3, 3], INT8,
                 [1 if i % 9 == 4 and (i//9) % 4 == i//36 else
                  -1 if i % 17 == 5 else int(i % 13 == 8) for i in range(144)])
    bias = m.tensor("spatial_bias", [4], INT32, [0, -2, 3, 1])
    x = m.qlinear("spatial", "QLinearConv", image, w, "spatial_q4", [bias])
    act = m.op("activation", "SiLU", [x], "silu_q4", DOMAIN)
    w = m.tensor("point_weight", [4, 4, 1, 1], INT8,
                 [1,0,1,0,-1,1,0,0,0,0,1,-1,0,1,0,1])
    bias = m.tensor("point_bias", [4], INT32, [-2, 1, 3, 0])
    point = m.qlinear("pointwise", "QLinearConv", act, w, "projection", [bias])
    residual = m.op("skip", "Add", [act, point], "pixel_logits")
    m.op("classes", "ArgMax", [residual], "classes", axis=1, keepdims=0)
    m.output(residual, [batch, 4, 4, 6], INT8)
    m.output("classes", [batch, 4, 6], INT64)
    return m.encode()


def transformer(contexts=36):
    s, d = 8, 4
    m = Model("StreamLLM_MultiTile")
    m.qparams()
    for number, name in enumerate(("q", "k", "v", "project", "gate", "up", "down", "vocab"), 1):
        m.tensor(name + "_weight", [d, d], INT8,
                 [((i*7 + j*3 + number + (i*j)%3) % 5) - 2 for i in range(d) for j in range(d)])
    for index in range(contexts):
        prefix = f"c{index:02d}_"

        def op(label, kind, ins, domain="", **attrs):
            return m.op(prefix+label, kind, ins, prefix+label+"_out", domain, **attrs)

        def proj(label, source):
            return m.qlinear(prefix+label, "QLinearMatMul",
                             source, label+"_weight", prefix+label+"_out")

        values = [((i*13 + (i//8)*7 + index*(i%5+1)) % 23)-11 for i in range(s*d)]
        x = m.input(prefix+"tokens", [s, d], INT8, values)
        n = op("rms1", "RMSNorm", [x], DOMAIN)
        q, k, v = (proj(label, n) for label in ("q", "k", "v"))
        kt = op("transpose", "Transpose", [k], perm=[1, 0])
        scores = op("qk", "MatMulInteger", [q, kt])
        score8 = op("score_scale", "Requant", [scores], DOMAIN, shift=5)
        probs = op("causal", "Softmax", [score8], DOMAIN, axis=1, causal=1)
        pv = op("pv", "MatMulInteger", [probs, v])
        values = op("value_scale", "Requant", [pv], DOMAIN, shift=8)
        projected = proj("project", values)
        residual = op("attention_residual", "Add", [x, projected])
        n2 = op("rms2", "RMSNorm", [residual], DOMAIN)
        gate, up = proj("gate", n2), proj("up", n2)
        silu = op("activation", "SiLU", [gate], DOMAIN)
        gate32 = op("gate_cast", "Cast", [silu], to=INT32)
        up32 = op("up_cast", "Cast", [up], to=INT32)
        product = op("swiglu", "Mul", [gate32, up32])
        hidden = op("hidden_scale", "Requant", [product], DOMAIN, shift=4)
        down = proj("down", hidden)
        out = op("ffn_residual", "Add", [down, residual])
        logits = proj("vocab", out)
        ids = op("next_ids", "ArgMax", [logits], axis=1, keepdims=0)
        m.output(logits, [s, d], INT8)
        m.output(ids, [s], INT64)
    return m.encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--vision-tiles", type=int, default=24)
    parser.add_argument("--llm-tiles", type=int, default=36)
    args = parser.parse_args()
    if min(args.vision_tiles, args.llm_tiles) < 2:
        parser.error("each fixture needs at least two tiles")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "StreamVision_MultiTile.onnx").write_bytes(vision(args.vision_tiles))
    (args.out / "StreamLLM_MultiTile.onnx").write_bytes(transformer(args.llm_tiles))


if __name__ == "__main__":
    main()
