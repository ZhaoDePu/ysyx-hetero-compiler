import argparse
import json
from pathlib import Path

from .compiler import compile_bundle
from .onnx import parse_model
from .profile import load_profile


def main():
    parser = argparse.ArgumentParser(description="Compile an integer graph for YSYX hardware.")
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--format", choices=("auto", "onnx", "json", "legacy"), default="auto")
    parser.add_argument("--profile", required=True, help="Built-in target name or profile JSON path")
    parser.add_argument("--out", required=True, type=Path, help="New directory for the executable bundle")
    parser.add_argument("-O", dest="optimization_level", type=int, choices=(3,), default=3,
                        help="Default optimization level; selectively disable passes with flags")
    parser.add_argument("--device-policy", choices=("auto", "gpu-only", "npu-greedy"), default="auto")
    parser.add_argument("--no-fusion", "--no-fuse", dest="no_fuse", action="store_true")
    parser.add_argument("--no-overlap", action="store_true")
    parser.add_argument("--stream-ddr", action="store_true",
                        help="Use the DDR-backed Z15 scratch schedule even when the graph fits resident")
    parser.add_argument("--no-mem-reuse", "--no-reuse", dest="no_reuse", action="store_true")
    args = parser.parse_args()
    model_format = args.format
    if model_format == "auto":
        model_format = "onnx" if args.graph.suffix.lower() == ".onnx" else "auto"
    try:
        raw = (parse_model(args.graph.read_bytes()) if model_format == "onnx"
               else json.loads(args.graph.read_text(encoding="utf-8")))
        dialect = {"legacy": "legacy", "auto": "auto"}.get(model_format, "canonical")
        result = compile_bundle(raw, load_profile(args.profile), args.out, dialect=dialect,
                                fuse=not args.no_fuse, reuse=not args.no_reuse,
                                overlap=not args.no_overlap, device_policy=args.device_policy,
                                stream_ddr=args.stream_ddr)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"compiler error: {exc}\n")
    plan = result["plan"]
    print(f"{plan['profile']}: {len(plan['stages'])} stages, "
          f"{plan['peak_live_bytes']} B peak live scratch, bundle {args.out}")


if __name__ == "__main__":
    main()
