"""Generate the fixed-shape C4-A single-sequence decoder example."""
import json
from pathlib import Path
from random import Random

from hetero_compiler.kv_reference import evaluate_kv


def fixture(seed):
    rng=Random(seed)
    return {"schema":"ysyx-causal-kv-v1","name":"SingleSequence_KV32x8_Decode4",
            "width":4,"prefill_rows":32,"tile_rows":8,"decode_steps":4,
            "prompt":[rng.randint(-14,14) for _ in range(128)],
            "weights":{key:[rng.randint(-3,3) for _ in range(16)]
                       for key in ("q","k","v","out","ff1","ff2","lm")},
            "token_gain":[3,-2,2,-3],"token_bias":[-5,4,0,7]}


if __name__=="__main__":
    for seed in range(100):
        model=fixture(seed)
        result=evaluate_kv(model)
        ids=[result[f"decode_{n}_token"][0] for n in range(1,5)]
        if len(set(ids))>=3:
            break
    else:
        raise RuntimeError("no informative decoder sequence")
    Path(__file__).with_name("causal_kv32.json").write_text(json.dumps(model,indent=2)+"\n")
    print(seed,ids)
