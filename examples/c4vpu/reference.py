"""Unquantized FP32 references for C4-VPU's unchanged C4-A fixtures."""

import numpy as np

from examples.c4a_streams.generate import _pattern

from .transformer import _softmax


def long_prefix_fp32(tiles=24, sequence=48, queries=2, width=4):
    names = ("query", "key", "value", "project", "up", "down", "vocab")
    weights = {tag: np.asarray(_pattern(width*width, i, 1), np.float32).reshape(width, width)
               for i, tag in enumerate(names, 1)}
    visible = np.arange(sequence)[None, :] <= np.arange(sequence-queries, sequence)[:, None]
    outputs = {}
    for i in range(tiles):
        x = np.asarray(_pattern(sequence*width, i+1, 7), np.float32).reshape(sequence, width) / 16
        suffix = x[-queries:]
        q = suffix @ weights["query"]
        k, v = [x @ weights[tag] for tag in ("key", "value")]
        p = _softmax(np.float32(2) * (q @ k.T), visible)
        skip = suffix + (p @ v) @ weights["project"]
        hidden = np.maximum(0, skip @ weights["up"])
        logits = (skip + hidden @ weights["down"]) @ weights["vocab"]
        outputs[f"c{i:02d}_vocab_out"] = logits.ravel()
        outputs[f"c{i:02d}_predict_out"] = np.argmax(logits, axis=-1)
    return outputs


def causal_kv_fp32(model):
    """Evaluate the causal graph in continuous code units without intermediate quantization."""
    width = model["width"]
    seq = model["prefill_rows"]
    x = np.asarray(model["prompt"], np.float32).reshape(seq, width)
    weights = {k: np.asarray(v, np.float32).reshape(width, width) for k, v in model["weights"].items()}
    gain = np.asarray(model["token_gain"], np.float32)
    bias = np.asarray(model["token_bias"], np.float32)

    def project(inputs, name, shift):
        return (inputs @ weights[name]) / np.float32(2**shift)

    def block(inputs, start, keys, values):
        queries = project(inputs, "q", 1)
        scores = (queries @ keys.T) / np.float32(64)
        visible = np.arange(len(keys))[None, :] <= np.arange(start, start+len(inputs))[:, None]
        attn = _softmax(scores, visible) @ values
        residual = inputs + project(attn, "out", 1)
        hidden = np.maximum(0, project(residual, "ff1", 2))
        return project(residual + project(hidden, "ff2", 2), "lm", 2)

    keys, values = [project(x, name, 1) for name in ("k", "v")]
    logits = block(x, 0, keys, values)
    ids = np.argmax(logits, axis=1)
    result = {"prefill_logits": logits.ravel(), "prefill_tokens": ids}
    token = ids[-1]
    for step in range(1, model["decode_steps"]+1):
        x = (gain * np.float32(token) + bias).reshape(1, width)
        keys = np.concatenate([keys, project(x, "k", 1)])
        values = np.concatenate([values, project(x, "v", 1)])
        logits = block(x, seq+step-1, keys, values)
        token = np.argmax(logits[0])
        result[f"decode_{step}_logits"] = logits.ravel()
        result[f"decode_{step}_token"] = np.asarray([token])
    result["k_cache"] = keys.ravel()
    result["v_cache"] = values.ravel()
    return result
