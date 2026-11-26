"""Cross-check the pure-C Qwen tokenizer against pinned official tokenizers."""

import argparse
from pathlib import Path
import random
import subprocess

from transformers import AutoTokenizer

from tools.c11_qwen_quality import PROMPTS


CASES = [
    "", " I'm we'll don't It's he's I'LL and 'ſ", "abc中文123٤⑤",
    "a\r\n  b\n  c", " \t\t\n  \u00a0 ", "puppies    ✨ 🧬❤️",
    "cafe\u0301 ◌́ Ångström 汉字 ﬁ K\u0301", "각과 각 한글",
    "日本語 한국어 français العربية १२३ meow", "print('你好!')",
    "<|im_start|>assistant\n你a好<|im_end|>", " … punctuation!?<>_+=\r\n",
    "s\u0323\u0307 E\u0302\u0300 \u1100\u1161\u11a8",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("original", "tokenizer", "runner"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(args.original, local_files_only=True)

    def encode(command, text):
        result = subprocess.run([str(args.runner), "--tokenizer", str(args.tokenizer), command, text],
                                capture_output=True, text=True, check=True)
        return [int(value) for value in result.stdout.split()]

    rng = random.Random(511)
    samples = CASES + [ "".join(rng.choice(
        "汉字abcXYZ123?!：；✦　　 \r\n\t \u00a0\xe9\u0315\u0300。(),.'™𝟬안")
        for _ in range(rng.randrange(1, 96))) for _ in range(64)]
    for item in samples:
        expected = tok.encode(item, add_special_tokens=False)
        found = encode("--encode", item)
        if found != expected:
            raise AssertionError(f"token IDs differ for {item!r}: expected {expected}, found {found}")
    for _, prompt in PROMPTS:
        expected = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           tokenize=True, add_generation_prompt=True)
        found = encode("--encode-chat", prompt)
        if found != expected:
            raise AssertionError(f"ChatML IDs differ for {prompt!r}: expected {expected}, found {found}")
    print(f"C11_BPE_OFFICIAL_EXACT text_cases={len(samples)} chatml=3")


if __name__ == "__main__":
    main()
