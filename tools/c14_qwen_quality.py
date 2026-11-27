"""Compare experimental P17 block-16 images to the pinned official FP32 prefill."""

import argparse
import ast
import contextlib
import csv
import json
import mmap
import os
from pathlib import Path
import re
import struct
import subprocess

import numpy as np

from tools.c11_qwen_export import HEAD_DIM, HEADS, HIDDEN, INTER, KV_HEADS, LAYERS, MODEL_SHA256, VOCAB, Tensors, fp32, sha256
from tools.c11_qwen_quality import compare


class Official:
    def __init__(self, image):
        self.tensors = Tensors(image)

    def weights(self, name, shape):
        return fp32(self.tensors.bf16(f"model.{name}.weight", shape))

    def norm(self, value, name):
        gamma = self.weights(name, (HIDDEN,))
        denominator = np.sqrt(np.mean(value * value, axis=-1, keepdims=True, dtype=np.float32) + np.float32(1e-6))
        return (value / denominator) * gamma

    def forward(self, ids):
        """Single-threaded FP32 eager evaluation. The tied head is streamed."""
        count = len(ids)
        emb = self.tensors.bf16("model.embed_tokens.weight", (VOCAB, HIDDEN))
        x = fp32(emb[np.asarray(ids)])
        inv_freq = np.float32(1) / np.power(np.float32(1000000), np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM)
        angle = np.arange(count, dtype=np.float32)[:, None] * inv_freq[None, :]
        angle = np.concatenate((angle, angle), axis=1)[:, None, :]
        cosine = np.cos(angle)
        sine = np.sin(angle)
        mask = np.arange(count)[None, :] > np.arange(count)[:, None]

        def rotate(a):
            first, second = np.split(a, 2, axis=-1)
            return cosine * a + sine * np.concatenate((-second, first), axis=-1)

        for idx in range(LAYERS):
            name = f"layers.{idx}."
            norm = self.norm(x, name + "input_layernorm")
            tensors = []
            for key, heads in (("q", HEADS), ("k", KV_HEADS), ("v", KV_HEADS)):
                pname = name + f"self_attn.{key}_proj"
                rows = heads * HEAD_DIM
                bias = fp32(self.tensors.bf16(f"model.{pname}.bias", (rows,)))
                tensors.append((norm @ self.weights(pname, (rows, HIDDEN)).T + bias).reshape(count, heads, HEAD_DIM))
            q, k, v = tensors
            q = rotate(q).transpose(1, 0, 2)
            k = np.repeat(rotate(k), HEADS // KV_HEADS, axis=1).transpose(1, 2, 0)
            v = np.repeat(v, HEADS // KV_HEADS, axis=1).transpose(1, 0, 2)
            scores = (q @ k) / np.sqrt(np.float32(HEAD_DIM))
            scores[:, mask] = -np.inf
            probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True)
            context = (probs @ v).transpose(1, 0, 2).reshape(count, HIDDEN)
            x = x + context @ self.weights(name + "self_attn.o_proj", (HIDDEN, HIDDEN)).T
            norm = self.norm(x, name + "post_attention_layernorm")
            gate = norm @ self.weights(name + "mlp.gate_proj", (INTER, HIDDEN)).T
            up = norm @ self.weights(name + "mlp.up_proj", (INTER, HIDDEN)).T
            activated = gate / (1 + np.exp(-np.clip(gate, -80, 80))) * up
            x = x + activated @ self.weights(name + "mlp.down_proj", (HIDDEN, INTER)).T
        x = self.norm(x, "norm")
        logits = np.empty((count, VOCAB), dtype=np.float32)
        for first in range(0, VOCAB, 4096):
            logits[:, first:first+4096] = x @ fp32(emb[first:first+4096]).T
        return logits


def decode(tokenizer, ids):
    if tokenizer[:8] != b"YSQBPE1\0":
        raise ValueError("expected the archived Qwen tokenizer")
    ordinary, n = struct.unpack_from("<II", tokenizer, 8)
    strings = 48 + 4 * (n + 1)
    pieces = []
    for i in ids:
        if i >= ordinary:  # special tokens are suppressed in text
            continue
        a, b = struct.unpack_from("<II", tokenizer, 48 + i * 4)
        pieces.append(tokenizer[strings+a:strings+b])
    return b"".join(pieces).decode("utf-8", "replace")


def run_fixed(cpu, model, ids, steps, minimum, scratch):
    seed, trace = scratch / "input-ids.txt", scratch / "logits.q16"
    seed.write_text("".join(f"{t}\n" for t in ids))
    env = {**os.environ, "YSYX_QWEN_MIN_NEW_TOKENS": str(minimum), "YSYX_QWEN_KV_BITS": "8"}
    result = subprocess.check_output([cpu, model, seed, str(steps), trace], env=env, text=True)
    return [int(i) for i in result.split()], trace


def check_fib(text, scratch):
    code = next(iter(re.findall(r"\x60{3}(?:python)?\s*\n(.*?)\x60{3}", text, re.S)), "")
    if not code:
        return 0, "missing complete code block"
    allowed = {ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.If, ast.Compare, ast.Eq,
               ast.Lt, ast.LtE, ast.Return, ast.Constant, ast.Name, ast.Load, ast.Store,
               ast.Assign, ast.AugAssign, ast.For, ast.Call, ast.Tuple, ast.BinOp, ast.Add, ast.Sub}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0, "syntax error"
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    statements = [node for node in tree.body if not isinstance(node, ast.FunctionDef)]
    def example(node):
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            return False
        call = node.value
        if not isinstance(call.func, ast.Name) or call.func.id != "print" or call.keywords or len(call.args) != 1:
            return False
        test = call.args[0]
        return (isinstance(test, ast.Call) and isinstance(test.func, ast.Name) and test.func.id == "fib"
                and not test.keywords and len(test.args) == 1
                and isinstance(test.args[0], ast.Constant) and type(test.args[0].value) is int)

    if len(functions) != 1 or not all(example(node) for node in statements):
        return 0, "definition falls outside the restricted evaluator"
    fib = functions[0]
    if (fib.name != "fib" or fib.decorator_list
            or any(type(node) not in allowed for node in ast.walk(fib))
            or any(not isinstance(node.func, ast.Name) or node.func.id != "range"
                   for node in ast.walk(fib) if isinstance(node, ast.Call))):
        return 0, "definition falls outside the restricted evaluator"
    code = ast.get_source_segment(code, fib)
    runner = scratch / "fib-check.py"
    runner.write_text("ns = {'__builtins__': {}, 'range': range}\n"
                      f"exec({code!r}, ns)\n"
                      "print([ns['fib'](i) for i in (0, 1, 2, 5, 10, 15)])\n")
    try:
        result = subprocess.check_output(["python3", "-I", "-B", runner], text=True, timeout=2, stderr=subprocess.DEVNULL)
        values = ast.literal_eval(result)
        return sum(a == b for a, b in zip(values, (0, 1, 1, 5, 55, 610), strict=True)), "ran six cases"
    except (subprocess.SubprocessError, ValueError, TypeError):
        return 0, "function failed"


def eval_suite(args, official):
    cases = dict(x.split("=", 1) for x in args.case)
    old = json.loads(args.prompts.read_text())
    direct = json.loads(args.fib.read_text())
    tokenizer = args.tokenizer.read_bytes()
    prompts, per_prompt, summary = [], [], []
    for prompt in old["prompts"]:
        tokens = prompt["input_ids"]
        scores = official.forward(tokens) if official else None
        fp32_gold = prompt["outputs"]["fp32"]
        row = {"prompt": prompt["label"], "input": prompt["prompt"], "outputs": {"fp32": fp32_gold}}
        for case, image in cases.items():
            got, path = run_fixed(args.cpu, image, tokens, 64, 64, args.scratch)
            if case == "c12" and prompt["outputs"]["c12_kv8"]["tokens"] != got:
                raise ValueError(f"{prompt['label']}: C12 generation replay changed")
            metrics = {}
            if scores is not None:
                fixed = np.memmap(path, mode="r", dtype="<i4", shape=(len(tokens), VOCAB))
                metrics = compare(scores, fixed)
                del fixed
            metrics.update({"prompt": prompt["label"], "variant": case, "generated_tokens": len(got),
                            "decode_top1_matches": sum(a == b for a, b in zip(got, fp32_gold["tokens"])),
                            "decode_top1_total": len(got)})
            per_prompt.append(metrics)
            text = decode(tokenizer, got)
            row["outputs"][case] = {"tokens": got, "text": text}
            print(json.dumps(metrics), flush=True)
        prompts.append(row)
    fib, = [p for p in direct["prompts"] if p["label"] == "fib_zh"]
    result = {"fp32": fib["outputs"]["fp32"]}
    for case, image in cases.items():
        steps = args.fib_steps or direct["max_new_tokens"]
        got, _ = run_fixed(args.cpu, image, fib["input_ids"], steps, direct["min_new_tokens"], args.scratch)
        text = decode(tokenizer, got)
        n, why = check_fib(text, args.scratch)
        result[case] = {"text": text, "tokens": got, "passed": n, "reason": why}
        subset = [row for row in per_prompt if row["variant"] == case]
        summary.append({"variant": case,
                        "prefill_mean_cosine": np.mean([r["prefill_mean_cosine"] for r in subset]) if official else None,
                        "prefill_top1_matches": sum(r.get("prefill_top1_matches", 0) for r in subset) if official else None,
                        "prefill_top1_total": sum(r.get("prefill_top1_total", 0) for r in subset) if official else None,
                        "decode_top1_matches": sum(r["decode_top1_matches"] for r in subset),
                        "decode_top1_total": sum(r["decode_top1_total"] for r in subset), "fib_passed": n})
    for name, rows in (("metrics", per_prompt), ("ablation", summary)):
        with (args.out / f"{name}.csv").open("w", newline="") as file:
            csvdata = csv.DictWriter(file, list(rows[0]))
            csvdata.writeheader()
            csvdata.writerows(rows)
    (args.out / "dialogues.json").write_text(json.dumps({"prompts": prompts, "fib": result}, ensure_ascii=False, indent=2) + "\n")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("cpu", "prompts", "fib", "tokenizer", "scratch", "out"):
        cli.add_argument(f"--{key}", type=Path, required=True)
    cli.add_argument("--official", type=Path)
    cli.add_argument("--fib-steps", type=int, choices=range(64, 257))
    cli.add_argument("--case", action="append", required=True, metavar="NAME=PATH")
    args = cli.parse_args()
    if not args.official:
        eval_suite(args, None)
    else:
        if sha256(args.official) != MODEL_SHA256:
            raise ValueError("official raw model SHA-256 differed from pinned digest")
        with args.official.open("rb") as f, contextlib.closing(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as image:
            eval_suite(args, Official(image))


if __name__ == "__main__":
    main()
