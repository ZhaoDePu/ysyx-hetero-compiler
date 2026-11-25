#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
scratch=/dev/shm/ysyx_compiler_c5
root=${1:?new evidence directory under compiler C2 RAM required}
[[ $(realpath -e -- "$scratch") == "$scratch" && -O "$scratch" ]]
[[ $(realpath -m -- "$root") == "$root" && "$root" == "$scratch/"* && ! -e "$root" ]]
mkdir -p "$root"/{logs,bundles,replay,models}
cd "$repo"
export TMPDIR="$scratch" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 LC_ALL=C
low=(nice -n 10 ionice -c2 -n7)
run() {
    local label=$1
    shift
    "${low[@]}" "$@" > "$root/logs/$label.log" 2>&1 || { cat "$root/logs/$label.log"; return 1; }
}
compile() {
    local label=$1 target=$2 model=$3
    run "$label.compile" python3 -B -m hetero_compiler --graph "$model" \
        --profile "$target" --out "$root/bundles/$label" --vpu-lowering off
}
replay() {
    local label=$1
    shift
    run "$label.rtl" python3 -B -m tools.verify_bundle --bundle "$root/bundles/$label" \
        --case "$root/replay/$label" "$@"
}
build_npu() {
    local name=$1
    [[ -x "$scratch/npu/$name-build/run-npu-$name" ]] ||
        run "$name.build" bash "tools/build_npu_${name}_rtl.sh"
}
run software python3 -B -m unittest discover -s tests -v
build_npu p8
build_npu z15
for variant in v6 g4-16t4w g4-8t8w; do
    run "gpu-$variant" python3 -B tools/validate_gpu_backends.py "$variant"
    [[ $(grep -c '^PASS ' "$root/logs/gpu-$variant.log") == 14 ]]
done
for target in p8 z15; do
    run "npu-$target" python3 -B -m tools.verify_npu_rtl "$target" \
        "$scratch/npu/$target-build/run-npu-$target" "$root/replay/native-$target"
done
[[ $(grep -h -c 'STAGE_PASS' "$root/logs/npu-p8.log") == 8 ]]
[[ $(grep -h -c 'STAGE_PASS' "$root/logs/npu-z15.log") == 5 ]]

for target in fpga_zu15eg_balanced fpga_zu15eg_full asic_nangate45_p8 asic_nangate45_ip asic_nangate45_ip_compat4k; do
    compile "$target" "$target" examples/mixed.json
    run "$target.yhb" python3 -B - "$root/bundles/$target/program.yhb" <<'PY'
from pathlib import Path
import sys
from hetero_compiler.yhb import inspect_yhb
path = Path(sys.argv[1])
print(inspect_yhb(path.read_bytes()))
PY
done
replay fpga_zu15eg_balanced --execute-plan
replay asic_nangate45_p8
for example in conv_epilogue causal_attention; do
    compile "$example" fpga_zu15eg_balanced "examples/$example.json"
    replay "$example"
done
printf '%s\n' '{"x":[5,-7]}' > "$root/models/onnx-inputs.json"
compile c1-onnx asic_nangate45_p8 examples/matmul_integer.onnx
replay c1-onnx --inputs "$root/models/onnx-inputs.json"
run independent-zones.fixture python3 -B - "$root/models/independent-zones.json" <<'PYFIXTURE'
import json, math, sys
from pathlib import Path

def tensor(shape, seed, kind):
    return {"shape": shape, "dtype": "int8", "kind": kind,
            "data": [(i*53 + i*i*seed//7) % 129 - 64 for i in range(math.prod(shape))]}
raw = {
    "name": "two_independent_ownership_halves",
    "tensors": {"a": tensor([273], 3, "input"), "b": tensor([273], 5, "input"),
                "x": tensor([16, 32], 7, "input"), "w": tensor([32, 8], 11, "constant")},
    "nodes": [{"id": "gpu_wave_add", "op": "Add", "inputs": ["a", "b"],
               "output": "gpu_out", "attrs": {"narrowing": "wrap"}},
              {"id": "npu_matmul", "op": "Gemm", "inputs": ["x", "w"], "output": "acc"},
              {"id": "npu_requant", "op": "Requant", "inputs": ["acc"],
               "output": "npu_out", "attrs": {"shift": 6}}],
    "outputs": ["gpu_out", "npu_out"],
}
Path(sys.argv[1]).write_text(json.dumps(raw), encoding="ascii")
PYFIXTURE
compile independent-zones fpga_zu15eg_balanced "$root/models/independent-zones.json"
replay independent-zones --execute-plan
grep -Eq 'overlap_npu_ticks=[1-9][0-9]*' "$root/logs/independent-zones.rtl.log"
hardware=/home/zdp/ysyx-migration/repos/ysyx-npu-ext
pin=2741540496f6f683843a231361bfa88743d7fabe
git -C "$hardware" archive "$pin" tiny-gpu/compiler/models | tar -xf - -C "$root/models"
for graph in "$root/models/tiny-gpu/compiler/models/"*.json; do
    name="legacy-$(basename "$graph" .json)"
    compile "$name" asic_nangate45_p8 "$graph"
    replay "$name"
done
[[ $(grep -l 'BUNDLE_RTL_PASS' "$root/logs/legacy-"*.rtl.log | wc -l) == 13 ]]
[[ $(grep -l 'BUNDLE_RTL_PASS' "$root/logs/"*.rtl.log | wc -l) == 19 ]]
printf 'C2 regression passed. Evidence: %s\n' "$root"
