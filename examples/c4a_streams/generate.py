"""Export deterministic wide-receptive-field vision and long-prefix Attention ONNX."""

import argparse
from pathlib import Path

from examples.generate_quantized_onnx import DOMAIN, FLOAT, INT8, INT32, INT64, UINT8, Model


def _pattern(count, seed, span=3):
    return [((i * 7 + (i//7) * 11 + seed * (i%13+1)) % (2*span+1)) - span for i in range(count)]


def vision(batch=16, depth=4):
    m = Model("EdgeConv_Stream_C4A")
    h = w = 10
    camera = m.input("camera_tiles", [batch, h, w, 8], INT8, _pattern(batch*h*w*8, 7, 7))
    leading = m.op("camera_nchw", "Transpose", [camera], "camera_nchw_view", perm=[0, 3, 1, 2])
    m.qparams()
    wa = m.tensor("stem_filter", [4, 8, 3, 3], INT8, _pattern(288, 3, 1))
    wb = m.tensor("body_filter", [4, 4, 3, 3], INT8, _pattern(144, 5, 1))
    bias = m.tensor("stem_bias", [4], INT32, [0, -2, 3, 1])
    stem = m.qlinear("stem", "QLinearConv", leading, wa, "stem_q4", [bias], {"strides": [2, 2]})
    x = m.op("stem_relu", "Relu", [stem], "stem_active")
    shortcut = x
    for i in range(depth):
        p = f"body{i:02d}_"
        z = m.qlinear(p + "spatial", "QLinearConv", x, wb, p + "conv", attrs={"pads": [1, 1, 1, 1]})
        x = m.op(p + "relu", "Relu", [z], p + "active")
    shortcut = m.op("skip_nhwc", "Transpose", [shortcut], "skip_nhwc_view", perm=[0, 2, 3, 1])
    x = m.op("logits_nhwc", "Transpose", [x], "logits_nhwc_view", perm=[0, 2, 3, 1])
    residual = m.op("skip", "Add", [shortcut, x], "pixel_logits")
    m.op("argmax", "ArgMax", [residual], "classes", axis=-1, keepdims=0)
    m.output("pixel_logits", [batch, 4, 4, 4], INT8)
    m.output("classes", [batch, 4, 4], INT64)
    return m.encode()


def attention(tiles=24, sequence=48, queries=2, width=4):
    m = Model("LongPrefix_Stream_C4A")
    m.qparams()
    m.tensor("zero_u", [], UINT8, [0])
    m.tensor("prob_scale", [], FLOAT, [1/256])
    starts = m.tensor("query_start", [1], INT64, [sequence - queries])
    ends = m.tensor("query_end", [1], INT64, [sequence])
    axes = m.tensor("query_axes", [1], INT64, [0])
    mask = m.tensor("causal_mask", [queries, sequence], FLOAT,
                    [0. if s <= sequence-queries+q else -float("inf")
                     for q in range(queries) for s in range(sequence)])
    weights = {tag: m.tensor(f"{tag}_weight", [width, width], INT8, _pattern(width*width, i, 1))
               for i, tag in enumerate(("query", "key", "value", "project", "up", "down", "vocab"), 1)}
    for i in range(tiles):
        p = f"c{i:02d}_"

        def op(label, kind, ins, domain="", **attrs):
            return m.op(p+label, kind, ins, p+label+"_out", domain, **attrs)

        def proj(tag, inp):
            return m.qlinear(p+tag, "QLinearMatMul", inp, weights[tag], p+tag+"_out")

        context = m.input(p+"tokens", [sequence, width], INT8, _pattern(sequence*width, i+1, 7))
        suffix = op("query_slice", "Slice", [context, starts, ends, axes])
        q, k, v = proj("query", suffix), proj("key", context), proj("value", context)
        kt = op("keys_t", "Transpose", [k], perm=[1, 0])
        score32 = op("qk", "MatMulInteger", [q, kt])
        score8 = op("score_scale", "Requant", [score32], DOMAIN, shift=3)
        score_f = op("dq_scores", "DequantizeLinear", [score8, "q4", "z"])
        masked = op("mask", "Add", [score_f, mask])
        p_f = op("softmax", "Softmax", [masked], axis=-1)
        probs = op("prob_quant", "QuantizeLinear", [p_f, "prob_scale", "zero_u"])
        context32 = op("pv", "MatMulInteger", [probs, v])
        ctx = op("context_scale", "Requant", [context32], DOMAIN, shift=8)
        x = proj("project", ctx)
        skip = op("attention_skip", "Add", [x, suffix])
        hidden = op("ffn_active", "Relu", [proj("up", skip)])
        ff = op("ffn_skip", "Add", [proj("down", hidden), skip])
        logits = proj("vocab", ff)
        indices = op("predict", "ArgMax", [logits], axis=1, keepdims=0)
        m.output(logits, [queries, width], INT8)
        m.output(indices, [queries], INT64)
    return m.encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--vision-tiles", type=int, default=16)
    parser.add_argument("--vision-depth", type=int, default=4)
    parser.add_argument("--attention-tiles", type=int, default=24)
    parser.add_argument("--sequence-length", type=int, default=48)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "EdgeConv_Stream_C4A.onnx").write_bytes(vision(args.vision_tiles, args.vision_depth))
    (args.out / "LongPrefix_Stream_C4A.onnx").write_bytes(attention(args.attention_tiles, args.sequence_length))


if __name__ == "__main__":
    main()
