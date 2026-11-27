"""Wrap a frozen Phase 17 YOLO plan for explicit deployment via the C14 CLI."""

import argparse
import json
from pathlib import Path
import tempfile

from hetero_compiler.yhb import emit_yhb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="Phase 17 compilation directory")
    parser.add_argument("destination", type=Path, help="output .yhp container")
    args = parser.parse_args()
    manifest = json.loads((args.plan / "manifest.json").read_text())
    if manifest.get("rtl_target") != "p17" or manifest["backends"]["gpu"]["stages"]:
        parser.error("expected the native Phase 17 YOLO26n plan")
    if args.destination.suffix != ".yhp":
        parser.error("Phase 17 containers must use the .yhp suffix")
    initial = manifest.get("initial_ddr", manifest["initial_image"])
    if Path(initial).name != initial:
        parser.error("expected a local initial DDR image")
    with tempfile.TemporaryDirectory(dir=args.destination.parent) as scratch:
        bank = Path(scratch)
        image = (args.plan / initial).resolve(strict=True)
        (bank / image.name).symlink_to(image)
        raw = emit_yhb(bank, manifest).read_bytes()
        if raw[:4] != b"YHB1":
            raise ValueError("unexpected compiler output format")
        container = bank / args.destination.name
        container.write_bytes(b"YHP1" + raw[4:])
        container.replace(args.destination)


if __name__ == "__main__":
    main()
