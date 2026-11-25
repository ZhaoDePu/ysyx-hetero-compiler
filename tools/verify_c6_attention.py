"""Check dynamic attention products and Q23 GPU softmax in frozen RTL."""

import argparse
from pathlib import Path
import subprocess

import numpy as np

from hetero_compiler.c6_attention import lower_matmul, lower_q23_softmax, lower_vpu_softmax
from hetero_compiler.c6_memory import Lifetime, reserve_ddr
from hetero_compiler.c6_schedule import emit_program, emit_rtl_control


def replay(directory, runner, arrays, expected, kernel):
    initial = {key: value.tobytes() for key, value in arrays.items()}
    slots, _ = reserve_ddr([
        Lifetime(name, data.nbytes, 0, 1) for name, data in (arrays | {"result": expected}).items()
    ])
    offsets = {name: slot.offset for name, slot in slots.items()}
    test_inputs = [name for name in arrays if name != "lut"]
    tensors = {
        name: {"shape": list(data.shape), "dtype": data.dtype.name}
        for name, data in (arrays | {"result": expected}).items() if name != "lut"
    }
    manifest = emit_program(directory, [kernel(offsets)], tensors, test_inputs, ["result"],
                            slots, initial, external_inputs=False)
    control = emit_rtl_control(directory, manifest, initial, {"result": expected.tobytes()})
    subprocess.run([runner, control], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--case", choices=("signed", "unsigned", "q23", "vpu"), required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(61028)
    if args.case == "vpu":
        scores = rng.integers(-65, 60, (2, args.tokens, args.tokens), dtype=np.int8)
        shift = 3
        delta = scores.max(-1, keepdims=True).astype(np.int16) - scores
        weight = 256 >> np.minimum(delta >> shift, 9)
        expected = ((255 * weight.astype(np.int32)) // weight.sum(-1, keepdims=True)).astype(np.uint8)
        replay(args.output, args.runner, {"source": scores}, expected,
               lambda at: lower_vpu_softmax("vpu", at["source"], at["result"], scores.shape, shift))
    elif args.case == "q23":
        scores = rng.integers(-65, 60, (2, args.tokens, args.tokens), dtype=np.int8)
        lut = np.rint((1 << 23) * np.exp(-0.098 * np.arange(256))).astype("<u4")
        delta = scores.max(-1, keepdims=True).astype(np.int16) - scores
        weight = lut[delta].astype(np.uint64)
        denom = weight.sum(-1, keepdims=True)
        base, remainder = np.divmod(256 * weight, denom)
        expected = np.minimum(255, base + ((remainder * 2 > denom) |
                           ((remainder * 2 == denom) & (base % 2 == 1)))).astype(np.uint8)
        replay(args.output, args.runner, {"source": scores, "lut": lut}, expected,
               lambda at: lower_q23_softmax("q23", at["source"], at["result"], at["lut"], scores.shape))
    else:
        unsigned = args.case == "unsigned"
        m, k, n = (64, args.tokens, args.tokens) if unsigned else (args.tokens, 32, args.tokens)
        a = rng.integers(-80, 80, (2, m, k), dtype=np.int8)
        b = (rng.integers(0, 256, (2, k, n), dtype=np.uint8) if unsigned
             else rng.integers(-50, 50, (2, k, n), dtype=np.int8))
        mul, shift = 27437, 25
        result = a.astype(np.int64) @ b.astype(np.int64)
        expected = np.clip(result * mul >> shift, -128, 127).astype(np.int8)
        replay(args.output, args.runner, {"left": a, "right": b}, expected,
               lambda at: lower_matmul("matmul", at["left"], at["right"], at["result"],
                                       a.shape, b.shape, mul, shift, unsigned_rhs=unsigned))


if __name__ == "__main__":
    main()
