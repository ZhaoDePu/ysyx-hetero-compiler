"""Independent NumPy oracle for the bounded causal KV transformer."""

from decimal import Decimal, localcontext
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _weights():
    with localcontext() as ctx:
        ctx.prec = 60
        return np.array([int((Decimal(1 << 23)*(-Decimal(d)/Decimal(8)).exp()).to_integral_value())
                         for d in range(256)], dtype=np.int64)


def _s8(x):
    return np.clip(x,-128,127).astype(np.int64)


def evaluate_kv(spec, prompt=None, step_overrides=None, *, vpu_lowering="off", vpu_attention="exact"):
    """Evaluate one whole causal prefill then actual greedy autoregression."""
    if vpu_lowering not in ("off", "on") or vpu_attention not in ("exact", "fast"):
        raise ValueError("invalid KV VPU lowering or attention mode")
    width = spec["width"]
    sequence = spec["prefill_rows"]
    prompts = np.asarray(spec["prompt"] if prompt is None else prompt,dtype=np.int64).reshape(sequence,width)
    if np.any(prompts < -128) or np.any(prompts > 127):
        raise ValueError("prompt entries must fit signed byte")
    gain = np.asarray(spec["token_gain"],dtype=np.int64)
    bias = np.asarray(spec["token_bias"],dtype=np.int64)
    weights = {key:np.asarray(w,dtype=np.int64).reshape(width,width) for key,w in spec["weights"].items()}
    overrides = step_overrides or {}

    def project(inp,name,shift):
        return _s8(np.right_shift(inp@weights[name],shift))

    def block(x,start,key,value):
        queries = project(x,"q",1)
        score = queries @ key.T
        if vpu_lowering == "off":
            score = _s8(score >> 3)
        probability = np.zeros_like(score)
        for row in range(len(x)):
            causal = score[row,:start+row+1]
            if vpu_lowering == "on":
                distance = np.minimum((causal.max() - causal) >> 5, 9)
                weights_for_row = 256 >> distance
                total = weights_for_row.sum()
                pq = ((weights_for_row * (65280 // total)) >> 8 if vpu_attention == "fast"
                      else (weights_for_row * 255) // total)
            else:
                exp = _weights()[causal.max()-causal]
                den = exp.sum()
                q,rem = np.divmod(exp*128,den)
                pq = q+((2*rem > den)|((2*rem == den)&(q%2 != 0)))
            probability[row,:len(causal)] = pq
        attn = _s8(np.right_shift(probability@value,8 if vpu_lowering == "on" else 7))
        residual = _s8(x+project(attn,"out",1))
        inner = np.maximum(0,project(residual,"ff1",2))
        return project(_s8(residual+project(inner,"ff2",2)),"lm",2)

    keys = project(prompts,"k",1)
    values = project(prompts,"v",1)
    logits = block(prompts,0,keys,values)
    tokens = np.argmax(logits,axis=1)
    result = {"prefill_logits":logits.ravel().tolist(),"prefill_tokens":tokens.tolist()}
    prev = int(tokens[-1])
    for i in range(1,spec["decode_steps"]+1):
        token = overrides.get(i,prev)
        if type(token) is not int or not 0 <= token < width:
            raise ValueError(f"decode step {i} overrides token outside vocabulary")
        x = (gain*token+bias).reshape(1,width)
        keys = np.vstack((keys,project(x,"k",1)))
        values = np.vstack((values,project(x,"v",1)))
        logits = block(x,sequence+i-1,keys,values)
        prev = int(np.argmax(logits[0]))
        result[f"decode_{i}_logits"] = logits.ravel().tolist()
        result[f"decode_{i}_token"] = [prev]
    result["k_cache"],result["v_cache"] = keys.ravel().tolist(),values.ravel().tolist()
    return result
