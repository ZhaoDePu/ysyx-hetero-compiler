"""Recheck a P19A offline pipeline plan."""

import argparse
import json
from pathlib import Path

from hetero_compiler.p19 import reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    report, words = reference(plan)
    if [int(w, 16) for w in plan["transfer_barrier_words_hex"]] != words:
        raise ValueError("serialized commands disagree with the audited schedule")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
