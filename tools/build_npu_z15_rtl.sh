#!/usr/bin/env bash
set -euo pipefail
work=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$work/tools/c5_resources.sh"
compiler_build_slots
base=/home/zdp/ysyx-migration
board="$base/artifacts/z15-board-bringup-step2-20261125-212110"
scratch="$compiler_scratch/npu"
commit=4d410d72508a5822bc5c8014981e5b55ef8faa56
[[ "$(realpath -e -- "$scratch")" == "$scratch" && -O "$scratch" ]]
mkdir -p -- "$scratch/z15-src"
[[ "$(cat "$board/FROZEN_COMMIT")" == "$commit" ]]
git -C "$base/repos/ysyx-npu-ext" archive "$commit" tiny-gpu/src fpga/vivado2020 | tar -xf - -C "$scratch/z15-src"
tar -xzf "$board/z15-rebuild-source.tar.gz" -C "$scratch/z15-src" rtl sim app
cmp "$board/z15_mmio.h" "$scratch/z15-src/app/z15_mmio.h"
printf 'source_commit=%s\n' "$commit"
cd "$scratch/z15-src"
sha256sum rtl/gen/g3_cluster.v rtl/gen/g3_external.v rtl/z15_hetero_pl_wrapper.sv rtl/z15_bd_axi_shell.v sim/closed_loop.cpp
export TMPDIR="$scratch" CCACHE_DISABLE=1 OMP_NUM_THREADS=1
nice -n 10 ionice -c2 -n7 python3 -B "$work/tools/z15_observed_rig.py" sim/closed_loop.cpp sim/c2_observed_loop.hpp
verilator="$base/tools/oss-cad-suite/bin/verilator"
"$verilator" --version
exec nice -n 10 ionice -c2 -n7 "$verilator" --cc --exe --build -j "$compiler_jobs" -MAKEFLAGS OBJCACHE= -O2 -Wno-fatal --assert \
  --top-module z15_bd_axi_shell --Mdir "$scratch/z15-build" \
  -Ifpga/vivado2020 -Itiny-gpu -Itiny-gpu/src -CFLAGS "-std=c++17 -I$PWD/app -I$PWD/sim -O2" \
  -f fpga/vivado2020/filelist.f rtl/gen/g3_cluster.v rtl/gen/g3_external.v \
  fpga/vivado2020/soc_phase12b_balanced_top.sv rtl/z15_hetero_pl_wrapper.sv rtl/z15_bd_axi_shell.v \
  "$work/tools/npu_z15_rtl.cpp" -o run-npu-z15
