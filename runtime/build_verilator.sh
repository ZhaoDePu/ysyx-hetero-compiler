#!/usr/bin/env bash
set -euo pipefail
runtime=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=/dev/shm/ysyx_compiler_c4vpu
out="$root/runtime/verilator"
build="$root/npu/z15-build"
src="$root/npu/z15-src"
sdk=/home/zdp/ysyx-migration/tools/oss-cad-suite
[[ "$(realpath -e "$root")" == "$root" && -O "$root" ]]
[[ -f "$build/Vz15_bd_axi_shell__ALL.a" && -f "$src/sim/closed_loop.cpp" ]]
test ! -L "$root/runtime" && test ! -L "$out"
mkdir -p "$out"
export TMPDIR="$out" CCACHE_DISABLE=1 OMP_NUM_THREADS=1
low=(nice -n 10 ionice -c2 -n7)
"${low[@]}" python3 -B "$runtime/../tools/z15_observed_rig.py" "$src/sim/closed_loop.cpp" "$out/observed_rig.inc"
"${low[@]}" make -C "$runtime" -f verilator.mk -j4 OUT="$out" RTL_BUILD="$build" RTL_SRC="$src" SDK="$sdk"
file "$out/ysyx_run"
