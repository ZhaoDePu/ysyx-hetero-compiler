"""Measure fixed C12 images against FP32, using an independent alpha holdout."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from tools.c11_qwen_export import MODEL_SHA256, REVISION, VOCAB
from tools.c11_qwen_quality import PROMPTS, compare, official
from tools.c12_smoothquant import VALIDATION, chat_ids, load_official


DIRECT_PROMPTS = (
    ("fpga_zh", "用通俗中文解释 FPGA 的可重构逻辑、它与 CPU 的区别，并举例说明一个用途。先说结论。"),
    ("array_en", "In hardware, what is a systolic array of multiply-accumulate processing elements? "
                 "Explain how values flow between neighboring processing elements during matrix multiplication."),
    ("fib_zh", "用 Python 编写 fib(n)，其中 F(0)=0、F(1)=1。用两个变量迭代。"
               "回答先给完整函数代码，然后用一句中文解释。"),
)


def run_fixed(reference, weights, ids, steps, minimum, scratch, kv):
    with tempfile.TemporaryDirectory(dir=scratch) as folder:
        root = Path(folder)
        seed, trace = root / "tokens.txt", root / "logits.q16"
        seed.write_text("".join(f"{token}\n" for token in ids), encoding="ascii")
        options = {**os.environ, "YSYX_QWEN_MIN_NEW_TOKENS": str(minimum)}
        if kv:
            options["YSYX_QWEN_KV_BITS"] = str(kv)
        start = time.perf_counter()
        text = subprocess.check_output(
            [str(reference), str(weights), str(seed), str(steps), str(trace)],
            env=options, text=True)
        logits = np.fromfile(trace, dtype="<i4").reshape(len(ids), VOCAB)
        return logits, [int(piece) for piece in text.split()], time.perf_counter() - start


def parse_cases(args):
    cases = []
    names = set()
    for item in args.case:
        label, binary, image, bits = item.split(":", 3)
        if label in names or bits not in {"-", "4", "8"}:
            raise ValueError("Case names must be unique and KV must be -, 4, or 8")
        names.add(label)
        cases.append((label, Path(binary), Path(image), None if bits == "-" else int(bits)))
    return cases


def stage(args):
    torch, tokenizer, model = load_official(args.original, args.threads)
    comparisons = []
    prompts = []
    evaluation = parse_cases(args)
    if args.validation:
        dataset = tuple((f"validation_{i}", x) for i, x in enumerate(VALIDATION))
    else:
        dataset = DIRECT_PROMPTS if args.prompt_set == "direct" else PROMPTS
    for label, prompt in dataset:
        ids = chat_ids(tokenizer, prompt)
        maximum = 1 if args.validation else args.steps
        minimum = 0 if args.validation else args.min_new_tokens
        print(f"{label}: FP32", flush=True)
        elapsed_start = time.perf_counter()
        expected, fp32_tokens = official(model, ids, maximum, minimum)
        fp32_seconds = time.perf_counter() - elapsed_start
        fp32 = compare(expected, expected.astype(np.float64) * 65536)
        comparisons.append({
            "prompt": label, "variant": "fp32", **fp32,
            "decode_top1_matches_free_run": len(fp32_tokens),
            "decode_top1_total_free_run": len(fp32_tokens),
            "generated_tokens": len(fp32_tokens), "host_seconds": fp32_seconds,
        })
        prompt_result = {
            "label": label, "prompt": prompt, "input_ids": ids,
            "outputs": {"fp32": {"tokens": fp32_tokens, "text": tokenizer.decode(
                fp32_tokens, skip_special_tokens=False)}},
        }
        for name, binary, image, kv in evaluation:
            print(f"{label}: {name}", flush=True)
            got, tokens, elapsed = run_fixed(binary, image, ids, maximum, minimum, args.scratch, kv)
            stats = compare(expected, got)
            stats.update(
                decode_top1_matches_free_run=sum(a == b for a, b in zip(fp32_tokens, tokens)),
                decode_top1_total_free_run=min(len(fp32_tokens), len(tokens)),
                generated_tokens=len(tokens), host_seconds=elapsed)
            comparisons.append({"prompt": label, "variant": name, **stats})
            prompt_result["outputs"][name] = {
                "tokens": tokens, "text": tokenizer.decode(tokens, skip_special_tokens=False)}
            print(f"{label}: {name} cosine {stats['prefill_mean_cosine']:.7f}", flush=True)
        prompts.append(prompt_result)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "quality.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(comparisons[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparisons)
    with (args.out / "dialogues.json").open("w", encoding="utf-8") as f:
        json.dump({
            "revision": REVISION, "original_sha256": MODEL_SHA256, "torch": torch.__version__,
            "evaluation": "alpha_holdout" if args.validation else "ChatML",
            "prompt_set": "holdout" if args.validation else args.prompt_set,
            "max_new_tokens": maximum, "min_new_tokens": minimum,
            "prompts": prompts,
        }, f, ensure_ascii=False, indent=2)
        f.write("\n")
    if args.validation:
        mean = {name: float(np.mean([row["prefill_mean_cosine"] for row in comparisons
                                    if row["variant"] == name])) for name, *_ in evaluation}
        (args.out / "selection.json").write_text(json.dumps({
            "criterion": "equal-prompt mean cosine on independent holdout",
            "scores": mean, "selected": max(mean, key=mean.get),
        }, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for attr in ("original", "out", "scratch"):
        parser.add_argument("--" + attr, type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--prompt-set", choices=("legacy", "direct"), default="legacy")
    parser.add_argument("--steps", type=int, choices=range(64, 257), default=64)
    parser.add_argument("--min-new-tokens", type=int, choices=range(0, 257), default=64)
    parser.add_argument("--case", action="append", required=True, metavar="NAME:REFERENCE:IMAGE:KV")
    args = parser.parse_args()
    if args.min_new_tokens > args.steps:
        parser.error("minimum generation length must fit the decode budget")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    stage(args)


if __name__ == "__main__":
    main()
