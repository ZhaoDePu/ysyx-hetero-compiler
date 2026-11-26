#!/usr/bin/env bash
set -euo pipefail
runtime=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$runtime/../tools/c5_resources.sh"
compiler_build_slots
root="$compiler_scratch"
case "${YSYX_Z15_SIM_GPU_RATIO:-2:3}" in
  2:3) ratio=2:3; out="$root/runtime/verilator" ;;
  4:5) ratio=4:5; out="$root/runtime/verilator-125-100" ;;
  24840:32911) ratio=24840:32911; out="$root/runtime/verilator-164555-124200" ;;
  3:4) ratio=3:4; out="$root/runtime/verilator-200-150" ;;
  161:221) ratio=161:221; out="$root/runtime/verilator-221-161" ;;
  *) echo "unsupported GPU/NPU clock ratio" >&2; exit 2 ;;
esac
build="$root/npu/z15-build"
src="$root/npu/z15-src"
sdk=/home/zdp/ysyx-migration/tools/oss-cad-suite
[[ "$(realpath -e "$root")" == "$root" && -O "$root" ]]
[[ -f "$build/Vz15_bd_axi_shell__ALL.a" && -f "$src/sim/closed_loop.cpp" ]]
test ! -L "$root/runtime" && test ! -L "$out"
mkdir -p "$out"
export TMPDIR="$out" CCACHE_DISABLE=1 OMP_NUM_THREADS=1
low=(nice -n 10 ionice -c2 -n7)
"${low[@]}" python3 -B "$runtime/../tools/z15_observed_rig.py" "$src/sim/closed_loop.cpp" "$out/observed_rig.inc" --gpu-clock-ratio "$ratio"
"${low[@]}" make -C "$runtime" -f verilator.mk -j"$compiler_jobs" OUT="$out" RTL_BUILD="$build" RTL_SRC="$src" SDK="$sdk"
file "$out/ysyx_run"
