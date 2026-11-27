"""Check complete prompt logits and generated IDs against Valence C12 KV8."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


MODES = (("c12_kv8", 0), ("c13_rms", 1), ("c13_swiglu", 3), ("c13_attention", 7))


def run(reference, weights, tokens, logits, steps, minimum, mask):
    env = {**os.environ, "YSYX_QWEN_KV_BITS": "8", "YSYX_QWEN_MIN_NEW_TOKENS": str(minimum)}
    env.pop("VALENCE_C13_CPU_OPS", None)
    if mask:
        env["VALENCE_C13_CPU_OPS"] = str(mask)
    result = subprocess.check_output(
        [str(reference), str(weights), str(tokens), str(steps), str(logits)], env=env, text=True)
    with logits.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    return {"ids": [int(item) for item in result.split()], "logits_sha256": digest,
            "logits_bytes": logits.stat().st_size}


def stage(args):
    archive = args.c12_archive
    args.out.mkdir(parents=True, exist_ok=True)
    ledger = []
    with tempfile.TemporaryDirectory(dir=args.scratch) as folder:
        tokens = Path(folder) / "tokens.txt"
        logits = Path(folder) / "logits.q16"
        for dataset in ("final-legacy", "final-direct"):
            data = json.loads((archive / "m1" / dataset / "dialogues.json").read_text(encoding="utf-8"))
            for prompt in data["prompts"]:
                if dataset == "final-direct" and prompt["label"] != "fib_zh":
                    continue
                tokens.write_text("".join(f"{token}\n" for token in prompt["input_ids"]), encoding="ascii")
                base = None
                for label, mask in MODES:
                    found = run(args.reference, args.weights, tokens, logits,
                                data["max_new_tokens"], data["min_new_tokens"], mask)
                    if base is None:
                        base = found
                        expected = prompt["outputs"]["c12_kv8"]["tokens"]
                        if found["ids"] != expected:
                            raise ValueError(f"Archived C12 tokens differ for {prompt['label']}")
                    if found != base:
                        raise ValueError(f"C13 results differ for {prompt['label']} / {label}")
                    row = {"dataset": dataset, "prompt": prompt["label"], "variant": label,
                           "mode_mask": mask, "status": "emulated_c_exact", **found}
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    ledger.append(row)
    (args.out / "emulator-comparison.json").write_text(
        json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    baseline = list(csv.DictReader((archive / "m1" / "comparison-64.csv").open()))
    core = next(row for row in baseline if row["variant"] == "c12_kv8")
    for row in baseline:
        row["evidence"] = "C12 archived FP32 comparison"
        row["fib_pass"] = "6/6" if row["variant"] in ("c12_kv8", "fp32") else (
            "syntax_error" if row["variant"] == "c12_kv4" else "not_run")
    for label, _ in MODES[1:]:
        baseline.append({**core, "variant": label, "evidence": "C kernel replay exact to C12",
                         "fib_pass": "6/6 identical generated code"})
    with (args.out / "quality.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(baseline[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(baseline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("c12-archive", "reference", "weights", "out", "scratch"):
        parser.add_argument("--" + name, type=Path, required=True)
    stage(parser.parse_args())


if __name__ == "__main__":
    main()
