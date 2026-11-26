"""Compare official PyTorch greedy generation with the integer W4A8 reference."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from tools.c11_qwen_export import MODEL_SHA256, REVISION, VOCAB, sha256


PROMPTS = (
    ("fpga_zh", "请用两段通俗的中文解释什么是 FPGA、它和 CPU 的区别，并给出一个实际用途。"),
    ("array_en", "Explain what a systolic array is in two complete sentences. Include how data moves and why it is efficient for matrix multiplication."),
    ("fib_zh", "用 Python 写一个计算第 n 个斐波那契数的迭代函数。请给出完整代码和一句中文解释。"),
)


def official(model, tokens, steps):
    import torch

    eos = set(model.generation_config.eos_token_id)
    ids = torch.tensor([tokens], dtype=torch.long)
    with torch.inference_mode():
        first = model(ids, use_cache=True)
        prefill = first.logits[0].float().numpy().copy()
        past = first.past_key_values
        last = first.logits[:, -1, :]
        generated = []
        for step in range(steps):
            token = last.argmax(dim=-1, keepdim=True)
            found = int(token.item())
            generated.append(found)
            if found in eos or step + 1 == steps:
                break
            result = model(token, past_key_values=past, use_cache=True, logits_to_keep=1)
            past, last = result.past_key_values, result.logits[:, -1, :]
    return prefill, generated


def compare(official_logits, w4_logits):
    fp32 = official_logits.astype(np.float64)
    fixed = w4_logits.astype(np.float64) / 65536
    l2 = np.linalg.norm(fp32, axis=1) * np.linalg.norm(fixed, axis=1)
    cosine = np.sum(fp32 * fixed, axis=1) / l2
    expected = np.argmax(fp32, axis=1)
    actual = np.argmax(w4_logits, axis=1)
    return {
        "prefill_mean_cosine": float(np.mean(cosine)),
        "prefill_min_cosine": float(np.min(cosine)),
        "prefill_last_cosine": float(cosine[-1]),
        "prefill_top1_matches": int(np.sum(expected == actual)),
        "prefill_top1_total": len(expected),
        "prefill_last_top1_equal": bool(expected[-1] == actual[-1]),
    }


def run(args):
    import torch
    import transformers

    if sha256(args.original / "model.safetensors") != MODEL_SHA256:
        raise ValueError("Checkpoint does not match the official LFS sha256")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    tok = transformers.AutoTokenizer.from_pretrained(args.original, local_files_only=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.original, torch_dtype=torch.float32, local_files_only=True).eval()
    args.out.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    dataset = []
    for label, prompt in PROMPTS:
        print(f"{label}: official greedy generation", flush=True)
        tokens = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                         tokenize=True, add_generation_prompt=True)
        start = time.perf_counter()
        official_logits, raw = official(model, tokens, args.steps)
        official_elapsed = time.perf_counter() - start
        print(f"{label}: W4A8 C99 greedy generation", flush=True)
        seed = args.scratch / f"{label}.tokens"
        seed.write_text("".join(f"{token}\n" for token in tokens))
        trace = args.scratch / f"{label}.logits.q16"
        start = time.perf_counter()
        process = subprocess.run([str(args.reference), str(args.weights), str(seed),
                                  str(args.steps), str(trace)], text=True, check=True,
                                 stdout=subprocess.PIPE)
        fixed_elapsed = time.perf_counter() - start
        fixed_tokens = [int(value) for value in process.stdout.split()]
        fixed_logits = np.memmap(trace, dtype="<i4", mode="r", shape=(len(tokens), VOCAB))
        metrics = compare(official_logits, fixed_logits)
        del fixed_logits
        trace.unlink()
        pairs = min(len(raw), len(fixed_tokens))
        matches = sum(a == b for a, b in zip(raw, fixed_tokens))
        metrics.update({"decode_top1_matches_free_run": matches,
                        "decode_top1_total_free_run": pairs,
                        "fp32_generated": len(raw), "w4a8_generated": len(fixed_tokens)})
        record = {"label": label, "prompt": prompt, "input_ids": tokens,
                  "fp32_tokens": raw, "w4a8_tokens": fixed_tokens,
                  "fp32_text": tok.decode(raw, skip_special_tokens=True),
                  "w4a8_text": tok.decode(fixed_tokens, skip_special_tokens=True),
                  "fp32_host_seconds": official_elapsed, "w4a8_host_seconds": fixed_elapsed,
                  "metrics": metrics}
        dataset.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        (args.out / "dialogue.json").write_text(json.dumps({
            "revision": REVISION, "torch": torch.__version__, "transformers": transformers.__version__,
            "fp32_execution": "Official BF16 checkpoint values loaded as FP32",
            "w4a8_execution": "C99 fixed Q16 state and Q24 W4 scales",
            "prompts": dataset,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    entries = [{"prompt": row["label"], **row["metrics"]} for row in dataset]
    with (args.out / "quality.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, list(entries[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("original", "weights", "reference", "scratch", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=48, choices=range(32, 49))
    parser.add_argument("--threads", type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    run(args)


if __name__ == "__main__":
    main()
